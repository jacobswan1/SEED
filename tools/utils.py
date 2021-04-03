'''
    Helper functions, some inherited from MoCo, Kaiming He.
'''
import math
import torch
import shutil
import random
from PIL import ImageFilter
import torchvision.transforms as transforms


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ValueMeter(object):
    """stores the current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0

    def update(self, val):
        self.val = val

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr * args.lr_mult
    if epoch < args.warmup_epochs:
        # warm up
        lr = args.lr + (args.lr * args.lr_mult - args.lr) / args.warmup_epochs * epoch
    elif args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_moco_teacher_encoder(args, model, logger, distributed=True):
    "Load the pre-trained teacher encoder model.encoder."
    checkpoint = torch.load(args.distill)
    model_checkpoint = model.state_dict()

    if distributed:
        for key in checkpoint['state_dict']:
            if key.startswith('module.encoder_q'):
                model_checkpoint[key.replace('encoder_q', 'teacher')] = checkpoint['state_dict'][key]
                logger.info("Teacher Param {} copyed from =====================> {}"
                            .format(key.replace('encoder_q', 'teacher'), key))
    # single GPU mode, remove module in checkpoint.
    else:
        for key in checkpoint['state_dict']:
            if key.startswith('module.encoder_q'):
                model_checkpoint[key.replace('module.encoder_q', 'teacher')] = checkpoint['state_dict'][key]
                logger.info("Teacher Param {} copyed from =====================> {}"
                            .format(key.replace('module.encoder_q', 'teacher'), key))

    model.load_state_dict(model_checkpoint)
    return model


def load_simclr_teacher_encoder(args, model, logger, distributed=True):
    "Load pre-trained weight from SimCLR"
    checkpoint = torch.load(args.distill)['model']
    model_checkpoint = model.state_dict()

    if distributed:
        for param in checkpoint:
            model_param = param.replace('module.module.module.', '')
            model_checkpoint['module.teacher.module.' + model_param] = checkpoint[param]
            logger.info(' ============> Teacher {} distilled from {}'.format('module.teacher.' + model_param, param))

    # single GPU mode, remove module in checkpoint.
    else:
        for param in checkpoint:
            model_param = param.replace('module.module.module.', '')
            model_checkpoint['teacher.module.' + model_param] = checkpoint[param]
            logger.info(' ============> Teacher {} distilled from {}'.format('module.teacher.' + model_param, param))

    model.load_state_dict(model_checkpoint)
    return model


def load_swav_teacher_encoder(args, model, logger, distributed=True):
    "Load pre-trained weight from SWAV"
    checkpoint = torch.load(args.distill)
    model_checkpoint = model.state_dict()

    if distributed:
        for key in checkpoint:
            # change param name from 'module.conv1*' ==> 'module.encoder_k.conv1*'
            # exclude the prototype branch
            if not key.startswith('module.prototypes'):
                model_key = key.replace('module', 'module.teacher')
                model_checkpoint[model_key] = checkpoint[key]
                logger.info('{} loaded.'.format(model_key))

    # single GPU mode, remove module in checkpoint.
    else:
        for key in checkpoint:
            # change param name from 'module.conv1*' ==> 'module.encoder_k.conv1*'
            # exclude the prototype branch
            if not key.startswith('module.prototypes'):
                model_key = key.replace('module', 'teacher')
                model_checkpoint[model_key] = checkpoint[key]
                logger.info('{} loaded.'.format(model_key))

    model.load_state_dict(model_checkpoint)
    return model


def resume_training(args, model, optimizer, logger):
    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger.info("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
    del checkpoint
    return model


def soft_cross_entropy(student_logit, teacher_logit):
    '''
    :param student_logit: logit of the student arch (without softmax norm)
    :param teacher_logit: logit of the teacher arch (already softmax norm)
    :return: CE loss value.
    '''
    return -(teacher_logit * torch.nn.functional.log_softmax(student_logit, 1)).sum()/student_logit.shape[0]


# ImageNet normalization
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


# SWAV alike augmentation, which uses different scale for cropping
swav_aug = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.14, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize])

swav_small_aug = transforms.Compose([
        transforms.RandomResizedCrop(96, scale=(0.05, 0.14)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

# MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
mocov2_aug = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

# MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
mocov1_aug = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

simclr_aug = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])