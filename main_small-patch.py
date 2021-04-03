#!/usr/bin/env python
import os
import time
import json
import torch.optim
import torch.nn.parallel
import seed.models as models
import seed.small_patch_builder
import torch.distributed as dist
from tools.opts import parse_opt
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
from tools.logger import setup_logger
from tools.dataset import Small_Patch_TSVDataset
from torch.utils.tensorboard import SummaryWriter
from tools.utils import swav_aug, swav_small_aug, adjust_learning_rate,\
    soft_cross_entropy, AverageMeter, ValueMeter, ProgressMeter,\
    resume_training, load_swav_teacher_encoder, save_checkpoint


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def main(args):

    # set-up the output directory
    os.makedirs(args.output, exist_ok=True)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        cudnn.benchmark = True

        # create logger
        logger = setup_logger(output=args.output, distributed_rank=dist.get_rank(),
                              color=False, name="SEED")

        if dist.get_rank() == 0:
            path = os.path.join(args.output, "config.json")
            with open(path, 'w') as f:
                json.dump(vars(args), f, indent=2)
            logger.info("Full config saved to {}".format(path))

        # save the distributed node machine
        logger.info('world size: {}'.format(dist.get_world_size()))
        logger.info('local_rank: {}'.format(args.local_rank))
        logger.info('dist.get_rank(): {}'.format(dist.get_rank()))

    else:
        # create logger
        logger = setup_logger(output=args.output, color=False, name="SEED")

        logger.info('Single GPU mode for debugging.')

    # create model
    logger.info("=> creating student encoder '{}'".format(args.student_arch))
    logger.info("=> creating teacher encoder '{}'".format(args.teacher_arch))

    # some architectures are not supported yet. It needs to be expanded manually.
    assert args.teacher_arch in models.__dict__

    # hidden_dim: resnet50-2048, resnet50w4-8192, resnet50w5-10240
    if args.teacher_arch == 'swav_resnet50':
        swav_mlp = 2048
    elif args.teacher_arch == 'swav_resnet50w2':
        swav_mlp = 8192
    elif args.teacher_arch == 'swav_resnet50w4':
        swav_mlp = 8192
    elif args.teacher_arch == 'swav_resnet50w5':
        swav_mlp = 10240

    # initialize model object, feed student and teacher into encoders.
    model = seed.small_patch_builder.SEED(models.__dict__[args.student_arch],
                                          models.__dict__[args.teacher_arch],
                                          args.dim,
                                          args.queue,
                                          args.temp,
                                          mlp=args.student_mlp,
                                          temp=args.distill_t,
                                          dist=args.distributed,
                                          swav_mlp=swav_mlp)

    logger.info(model)

    if args.distributed:
        logger.info('Entering distributed mode.')

        model = torch.nn.parallel.DistributedDataParallel(model.cuda(),
                                                          device_ids=[args.local_rank],
                                                          broadcast_buffers=False,
                                                          find_unused_parameters=True)

        logger.info('Model now distributed.')

        args.lr_mult = args.batch_size / 256
        args.warmup_epochs = 5
        optimizer = torch.optim.SGD(model.parameters(),
                                    args.lr_mult * args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        # tensorboard
        if dist.get_rank() == 0:
            summary_writer = SummaryWriter(log_dir=args.output)
        else:
            summary_writer = None

    else:
        args.lr_mult = 1
        args.warmup_epochs = 5

        model = model.cuda()

        optimizer = torch.optim.SGD(model.parameters(), args.lr,  momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        summary_writer = SummaryWriter(log_dir=args.output)

    # load the SSL pre-trained teacher encoder into model.teacher
    if args.distill:
        if os.path.isfile(args.distill):
            model = load_swav_teacher_encoder(args, model, logger, distributed=args.distributed)

            logger.info("=> Teacher checkpoint successfully loaded from '{}'".format(args.distill))
        else:
            logger.info("wrong distillation checkpoint.")

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            model = resume_training(args, model, optimizer, logger)
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # clear unnecessary weights
    torch.cuda.empty_cache()

    # we use 6 small patches by default
    train_dataset = Small_Patch_TSVDataset(os.path.join(args.data, 'train.tsv'),
                                           swav_aug, swav_small_aug, num_patches=6)

    logger.info('TSV Dataset done.')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

        # ensure batch size is dividable by # of GPUs
        assert args.batch_size // dist.get_world_size() == args.batch_size / dist.get_world_size(), \
            'Batch size is not divisible by num of gpus.'

        # create distributed dataloader
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    else:
        # create distributed dataloader
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,
            drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed: train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        loss = train(train_loader, model, soft_cross_entropy, optimizer, epoch, args, logger)

        if summary_writer is not None:
            # Tensor-board logger
            summary_writer.add_scalar('train_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if dist.get_rank() == 0:

            file_str = 'Teacher_{}_T-Epoch_{}_Student_{}_distill-Epoch_{}-checkpoint_{:04d}.pth.tar'\
                .format(args.teacher_ssl, args.epochs, args.student_arch, args.teacher_arch, epoch)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.student_arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(args.output, file_str))

            logger.info('==============> checkpoint saved to {}'.format(os.path.join(args.output, file_str)))


def train(train_loader, model, criterion, optimizer, epoch, args, logger):
    batch_time = AverageMeter('Batch Time', ':5.3f')
    data_time = AverageMeter('Data Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    lr = ValueMeter('LR', ':5.3f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, lr, losses],
        prefix="Epoch: [{}]".format(epoch))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    lr.update(get_learning_rate(optimizer))

    # switch to train mode
    model.train()

    # make key-encoder at eval to freeze BN
    if args.distributed:
        model.module.teacher.eval()

        # check the sanity of key-encoder
        for name, param in model.module.teacher.named_parameters():
            if param.requires_grad:
                logger.info("====================> Key-encoder Sanity Failed, parameters are not frozen.")

    else:
        model.teacher.eval()

        # check the sanity of key-encoder
        for name, param in model.teacher.named_parameters():
           if param.requires_grad:
                logger.info("====================> Key-encoder Sanity Failed, parameters are not frozen.")

    end = time.time()

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for i, (images, small_patches) in enumerate(train_loader):

        if not args.distributed:
            images = images.cuda()
            small_patches = small_patches.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        with torch.cuda.amp.autocast(enabled=True):

            logit, label, s_logit, s_label = model(image=images, small_image=small_patches)
            loss = criterion(logit, label) + criterion(s_logit, s_label)

        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, logger)

    return losses.avg


if __name__ == '__main__':
    main(parse_opt())
