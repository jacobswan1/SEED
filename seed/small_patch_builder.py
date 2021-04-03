import torch
import torch.nn as nn
from .utils import concat_all_gather


class SEED(nn.Module):
    """
    Build a SEED model for Self-supervised Distillation: a student encoder, a teacher encoder (stay frozen),
    and an instance queue.
    Adapted from MoCo, He, Kaiming, et al. "Momentum contrast for unsupervised visual representation learning."
    """
    def __init__(self, student, teacher, dim=128, K=65536, t=0.07, mlp=False,
                 temp=1e-4, dist=True, swav_mlp=None):
        """
        dim:        feature dimension (default: 128)
        K:          queue size
        t:          temperature for student encoder
        temp:       distillation temperature
        base_width: width of the base network
        swav_mlp:   MLP length for SWAV resnet, default=None
        """
        super(SEED, self).__init__()

        self.K = K
        self.t = t
        self.temp = temp
        self.dim = dim
        self.dist = dist

        # create the Teacher/Student encoders
        # num_classes is the output fc dimension
        self.student = student(num_classes=dim)

        # teacher encoder, SWAV's resnet50w5 does not use BN.
        self.teacher = teacher(normalize=True, hidden_mlp=swav_mlp, output_dim=dim,
                               batch_norm=not str(teacher).__contains__('resnet50w5'),)

        if mlp:
            dim_mlp = self.student.fc.weight.shape[1]
            self.student.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.student.fc)

        # not update by gradient
        for param_k in self.teacher.parameters():
            param_k.requires_grad = False

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("small_queue", torch.randn(dim, K))
        self.small_queue = nn.functional.normalize(self.small_queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, small_keys, concat=True):

        # gather keys before updating queue in distributed mode
        if concat:
            keys = concat_all_gather(keys)
            small_keys = concat_all_gather(small_keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (de-queue and en-queue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        self.small_queue[:, ptr:ptr + batch_size] = small_keys.transpose(0, 1)

        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    def forward(self, image, small_image):
        """
        Input:
            image: a batch of images
            small_image: a batch of key images: B*6*3*96*96
        Output:
            student logits, teacher logits
        """

        # get the shape of small patches
        B, N, C, W, H = small_image.shape

        # compute query features
        s_emb = self.student(image)  # NxC
        s_emb = nn.functional.normalize(s_emb, dim=1)

        small_image = small_image.view(-1, C, W, H)
        s_small_emb = self.student(small_image)  # queries: NxC
        s_small_emb = nn.functional.normalize(s_small_emb, dim=-1)

        # compute key features
        with torch.no_grad():  # no gradient to keys

            t_emb = self.teacher(image)  # keys: NxC
            t_small_emb = self.teacher(small_image)  # keys: NxC

        # cross-Entropy Loss
        logit_stu = torch.einsum('nc,ck->nk', [s_emb, self.queue.clone().detach()])
        logit_tea = torch.einsum('nc,ck->nk', [t_emb, self.queue.clone().detach()])

        logit_s_p = torch.einsum('nc,nc->n', [s_emb, t_emb]).unsqueeze(-1)
        logit_t_p = torch.einsum('nc,nc->n', [t_emb, t_emb]).unsqueeze(-1)

        logit_stu = torch.cat([logit_s_p, logit_stu], dim=1)
        logit_tea = torch.cat([logit_t_p, logit_tea], dim=1)

        # compute the logit for small-views
        logit_stu_small = torch.einsum('nc,ck->nk', [s_small_emb, self.small_queue.clone().detach()])
        logit_tea_small = torch.einsum('nc,ck->nk', [t_small_emb, self.small_queue.clone().detach()])

        logit_stu_small_pos = torch.einsum('nc,nc->n', [s_small_emb, t_small_emb]).unsqueeze(-1)
        logit_tea_small_pos = torch.einsum('nc,nc->n', [t_small_emb, t_small_emb]).unsqueeze(-1)

        logit_stu_small = torch.cat([logit_stu_small_pos, logit_stu_small], dim=1)
        logit_tea_small = torch.cat([logit_tea_small_pos, logit_tea_small], dim=1)

        # compute soft labels
        logit_stu /= self.t
        logit_tea = nn.functional.softmax(logit_tea/self.temp, dim=1)

        logit_stu_small /= self.t
        logit_tea_small = nn.functional.softmax(logit_tea_small/self.temp, dim=1)

        # use just one of the small patches as enqueue samples
        t_small_emb = t_small_emb.view(B, N, self.dim)

        # de-queue and en-queue
        self._dequeue_and_enqueue(t_emb, t_small_emb[:, 0].contiguous(), concat=self.dist)

        return logit_stu, logit_tea, logit_stu_small, logit_tea_small

