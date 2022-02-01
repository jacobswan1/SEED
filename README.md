
# SEED

Implementations for the [__ICLR-2021 paper: SEED: Self-supervised Distillation For Visual Representation.__](https://arxiv.org/pdf/2101.04731.pdf) 
```
@Article{fang2020seed,
  author  = {Fang, Zhiyuan and Wang, Jianfeng and Wang, Lijuan and Zhang, Lei and Yang, Yezhou and Liu, Zicheng},
  title   = {SEED: Self-supervised Distillation For Visual Representation},
  journal = {International Conference on Learning Representations},
  year    = {2021},
}
```

## Introduction

This paper is concerned with self-supervised learning for small models. <img src="https://user-images.githubusercontent.com/17426159/126873068-ce5ebdce-d821-4a9c-9d94-52585039261e.png" width="330" height="280" align="right"> 
 The 
problem is motivated by our empirical studies that while the widely used contrastive
self-supervised learning method has shown great progress on large model training, 
it does not work well for small models. To address this problem, we propose a
new learning paradigm, named **SE**lf-Sup**E**rvised **D**istillation (**SEED**), where we
leverage a larger network (as Teacher) to transfer its representational knowledge
into a smaller architecture (as Student) in a self-supervised fashion. Instead of
directly learning from unlabeled data, we train a student encoder to mimic the
similarity score distribution inferred by a teacher over a set of instances. We show
that SEED dramatically boosts the performance of small networks on downstream
tasks. Compared with self-supervised baselines, SEED improves the top-1 accuracy
from **42.2%** to **67.6%** on **EfficientNet-B0** and from **36.3%** to **68.2%** on **MobileNetV3-Large** on the ImageNet-1k dataset.
SEED improves the **ResNet-50** from **67.4%** to **74.3%** from the previous MoCo-V2 baseline.
![image](https://user-images.githubusercontent.com/17426159/126872552-a2873b52-a901-435a-a6cc-b8bc1a4e3248.png)

## Preperation
Note: This repository does not contain the ImageNet dataset building, please refer to [MoCo-V2](https://github.com/facebookresearch/moco) for the enviromental setting & dataset preparation. Be careful if you use FaceBook's ImageNet dataset implementation as the provided dataloader here is to handle TSV ImageNet source.

## Self-Supervised Distillation Training

[SWAV](https://github.com/facebookresearch/swav)'s 400_ep ResNet-50 model as Teacher architecture for a Student EfficientNet-b1 model with multi-view strategies. Place the pre-trained checkpoint in <code>./output</code> directory. Remember to change the parameter name in the checkpoint as some module provided by SimCLR, MoCo-V2 and SWAV are inconsistent with regular PyTorch implementations. 
Here we provide the pre-trained SWAV/MoCo-V2/SimCLR Pre-trained checkpoints, but all credits belong to them.

Teacher Arch. | SSL Method |               Teacher SSL-epochs              | Link |
---------|---------|----------------------------------|-------|
ResNet-50  |  MoCo-V1 | 200 | [URL](https://seed.blob.core.windows.net/data/SEED/moco_v1_200ep_pretrain.pth.tar?sv=2020-08-04&st=2021-11-03T22%3A07%3A11Z&se=2031-11-04T22%3A07%3A00Z&sr=b&sp=r&sig=4pVrNIbozl3VXhdUltFCFfd5AiIcHHuwv%2FijbCXWIqE%3D)
ResNet-50  |  SimCLR |    200    |          [URL](https://seed.blob.core.windows.net/data/SEED/simclr_200.pth?sv=2020-08-04&st=2021-11-03T22%3A06%3A55Z&se=2031-11-04T22%3A06%3A00Z&sr=b&sp=r&sig=n6wR%2F22ddPpDpIP2cpw9wJ8Ll4CCpCMaLfRQCgMV5Zc%3D)
ResNet-50  |  MoCo-V2 |    200    |          [URL](https://seed.blob.core.windows.net/data/SEED/jianfw_mocov2_ResNet50_epoch200.pth?sv=2020-08-04&st=2021-11-03T22%3A06%3A28Z&se=2031-11-04T22%3A06%3A00Z&sr=b&sp=r&sig=Ql6sep8UFLDbWYugxaK%2FoUmLTCJhPCpJZfAywS4cu8Q%3D)
ResNet-50  |  MoCo-V2 |    800    |          [URL](https://seed.blob.core.windows.net/data/SEED/moco_v2_800ep_pretrain.pth.tar?sv=2020-08-04&st=2021-11-03T22%3A06%3A03Z&se=2031-11-04T22%3A06%3A00Z&sr=b&sp=r&sig=%2Bu9r3n%2BOuYF5snOL1nqJ4D%2BaEnJbBi1p0IRfhRY0InA%3D)
ResNet-50  |  SWAV |    800    |          [URL](https://seed.blob.core.windows.net/data/SEED/swav_800ep_pretrain.pth.tar?sv=2020-08-04&st=2021-11-03T22%3A05%3A34Z&se=2031-11-04T22%3A05%3A00Z&sr=b&sp=r&sig=gMj1imj4AWNsfz2VQC5ZWQUKKoQo81LHEN5%2FduV9Wrw%3D)
ResNet-101  |  MoCo-V2 |    200    |          [URL](https://seed.blob.core.windows.net/data/SEED/res101-moco-v2-checkpoint_0199.pth.tar?sv=2020-08-04&st=2021-11-03T22%3A05%3A15Z&se=2031-11-04T22%3A05%3A00Z&sr=b&sp=r&sig=QdAtnGB%2B%2Bh9YQBs%2BDlURv42TcoWWIpNDrfHNohadWPU%3D)
ResNet-152  |  MoCo-V2 |    200    |          [URL](https://seed.blob.core.windows.net/data/SEED/200-resnet152-moco-v2-checkpoint_0199.pth.tar?sv=2020-08-04&st=2021-11-03T22%3A04%3A55Z&se=2031-11-04T22%3A04%3A00Z&sr=b&sp=r&sig=CLM0EN7m2yjuXOk1WhdqTH18Nh%2Btj4zfDgHZj9c6iNQ%3D)
ResNet-152  |  MoCo-V2 |    800    |          [URL](https://seed.blob.core.windows.net/data/SEED/800-resnet152-moco-v2-checkpoint_0799.pth.tar?sv=2020-08-04&st=2021-11-03T22%3A04%3A37Z&se=2031-11-04T22%3A04%3A00Z&sr=b&sp=r&sig=Xg1Pf50T9EC9g7b4FbQxt3uC8%2BKuyKZQRF8lt3YbzVE%3D)
ResNet-50X2  |  SWAV |    400    |          [URL](https://seed.blob.core.windows.net/data/SEED/swav_RN50w2_400ep_pretrain.pth.tar?sv=2020-08-04&st=2021-11-03T22%3A04%3A07Z&se=2031-11-04T22%3A04%3A00Z&sr=b&sp=r&sig=AuN3iN6vGZ8H1sLdaCVGiz5LfZIBdRzelehFF8xK0JA%3D)
ResNet-50X4  |  SWAV |    400    |          [URL](https://seed.blob.core.windows.net/data/SEED/swav_RN50w4_400ep_pretrain.pth.tar?sv=2020-08-04&st=2021-11-03T22%3A03%3A49Z&se=2031-11-04T22%3A03%3A00Z&sr=b&sp=r&sig=ywgTDEKuyvN0hjQdq7n3qJVbHJVefs%2FvBNvSgLJk%2BHg%3D)
ResNet-50X5  |  SWAV |    400    |          [URL](https://seed.blob.core.windows.net/data/SEED/swav_RN50w5_400ep_pretrain.pth.tar?sv=2020-08-04&st=2021-11-03T22%3A02%3A37Z&se=2031-11-04T22%3A02%3A00Z&sr=b&sp=r&sig=PVr%2FkyUGN0Fl%2F29z6EZWQvynGJDlP4peF36AcgnVsvg%3D)


To conduct the training one GPU on single Node using Distributed Training: 
```
python -m torch.distributed.launch --nproc_per_node=1 main_small-patch.py \
       -a efficientnet_b1 \
       -k resnet50 \
       --teacher_ssl swav \
       --distill ./output/swav_400ep_pretrain.pth.tar \
       --lr 0.03 \
       --batch-size 16 \
       --temp 0.2 \
       --workers 4 
       --output ./output \
       --data [your TSV imagenet-folder with train folders]
```

Conduct linear evaluations on ImageNet-val split:
```
python -m torch.distributed.launch --nproc_per_node=1  main_lincls.py \
       -a efficientnet_b0 \
       --lr 30 \
       --batch-size 32 \
       --output ./output \ 
       [your TSV imagenet-folder with val folders]
```

## Checkpoints by SEED
Here we provide some pre-trained checkpoints after distillation by SEED. Note: the 800 epcohs one are trained with small-view strategies and have better performances.

Student-Arch. |   Teacher-Arch. |    Teacher SSL |     Student SEED-epochs              | Link |
---------|----------------------------------|-------|-------|-------|
ResNet-18|  ResNet-50 |   MoCo-V2 | 200    |          [URL](https://seed.blob.core.windows.net/data/SEED/seed_model/resnet18_distill_resnet50-moco-v2-checkpoint_0199.pth.tar?sv=2020-08-04&st=2021-11-03T22%3A07%3A40Z&se=2031-11-04T22%3A07%3A00Z&sr=b&sp=r&sig=swrmmIRIQ4RNj%2BWhWsUQ9cbnE0va0hqH0dZ3IH%2Bz8so%3D)
ResNet-18|  ResNet-50W2 |   SWAV | 400    |          [URL](https://seed.blob.core.windows.net/data/SEED/seed_model/AMP_SMALL_PATCH_400_resnet18_distill_resnet50w2-swav-checkpoint_0399.pth.tar?sv=2020-08-04&st=2021-11-03T22%3A08%3A02Z&se=2031-11-04T22%3A08%3A00Z&sr=b&sp=r&sig=AdOBiNemj5P7rqIvL5yM5%2B1C%2BjH6qJltDbPMOgkm0P4%3D)
MobileV3-Large|  ResNet-50 |   MoCo-V2 | 200    |          [URL](https://seed.blob.core.windows.net/data/SEED/seed_model/mobilenetv3_large_distill-moco-v2-checkpoint_0199.pth.tar?sv=2020-08-04&st=2021-11-03T22%3A08%3A22Z&se=2031-11-04T22%3A08%3A00Z&sr=b&sp=r&sig=RUBlaj49a54fanXlTD%2FgRMASSYTpq4FbSQ%2BSpQcRbNI%3D)
EfficientNet-B0|  ResNet-50W4 |   SWAV | 400    |          [URL](https://seed.blob.core.windows.net/data/SEED/seed_model/AMP_SMALL_PATCH_400_efficientnet_b0_distill_resnet50w4-swav-checkpoint_0399.pth.tar?sv=2020-08-04&st=2021-11-03T22%3A08%3A37Z&se=2031-11-04T22%3A08%3A00Z&sr=b&sp=r&sig=GmH6dfLs9E9Rc1U%2FbkyZ7tWyxw6ie%2F2WWOnpe59lrzE%3D)
EfficientNet-B0 | ResNet-50W2 | SWAV | 800 | [URL](https://seed.blob.core.windows.net/data/SEED/seed_model/AMP_SMALL_PATCH_800_efficientnet_b0_distill_resnet50w2-swav-checkpoint_0799.pth.tar?sv=2020-08-04&st=2021-11-03T22%3A08%3A55Z&se=2031-11-04T22%3A08%3A00Z&sr=b&sp=r&sig=1%2F%2Be3xEaU%2FRnNZVDQZPq4RqAqjc5r4uIBED%2FHnocYZk%3D)
EfficientNet-B1|  ResNet-50 |   SWAV | 200    |          [URL](https://seed.blob.core.windows.net/data/SEED/seed_model/resnet18_distill_resnet50-moco-v2-checkpoint_0199.pth.tar?sv=2020-08-04&st=2021-11-03T22%3A12%3A10Z&se=2031-11-04T22%3A12%3A00Z&sr=b&sp=r&sig=m8wkUolK7Er%2B5kbTjv3%2Bv2%2BevpfMxLSl%2BjzDtne7qeI%3D)
EfficientNet-B1|  ResNet-152 |   SWAV | 200    |          [URL](https://seed.blob.core.windows.net/data/SEED/seed_model/AMP_200_efficientnet_b1_distill_resnet152-swav-checkpoint_0199.pth.tar?sv=2020-08-04&st=2021-11-03T22%3A09%3A26Z&se=2031-11-04T22%3A09%3A00Z&sr=b&sp=r&sig=XDqggjKp0fNRgVe0ewnPZr8EeBQU%2B4v0Yao5GdgXHsM%3D)
ResNet-50 | ResNet-50W4 | SWAV | 400 | [URL](https://seed.blob.core.windows.net/data/SEED/seed_model/AMP_SMALL_PATCH_400_resnet50_distill_resnet50w4-swav-checkpoint_0399.pth.tar?sv=2020-08-04&st=2021-11-03T22%3A09%3A46Z&se=2031-11-04T22%3A09%3A00Z&sr=b&sp=r&sig=gPySMEZjnXHt2zCkAKy5JLYw0Ks4UxmtaTkl8ikSCy8%3D)




## Glance of the Performances
ImageNet-1k test accuracy (%) using KNN and linear classification for multiple students and MoCov2 pre-trained deeper teacher architectures. ✗ denotes MoCo-V2 self-supervised learning baselines before
distillation. * indicates using a deeper teacher encoder pre-trained by SWAV, where additional small-patches are
also utilized during distillation and trained for 800 epochs. K denotes Top-1 accuracy using KNN. T-1 and T-5
denote Top-1 and Top-5 accuracy using linear evaluation. First column shows Top-1 Acc. of Teacher network.
First row shows the supervised performances of student networks.
<p align="center">
<img src="https://user-images.githubusercontent.com/17426159/126873030-918a61f0-8cba-4954-a501-ec553dae07a6.png" width="800" align="center"> 
</p>

## Acknowledge
This implementation is largely originated from: [MoCo-V2](https://github.com/facebookresearch/moco).
Thanks [SWAV](https://github.com/facebookresearch/swav) and [SimCLR](https://github.com/google-research/simclr) for the pre-trained SSL checkpoints.

This work is done jointly with [ASU-APG lab](https://yezhouyang.engineering.asu.edu/) and [Microsoft Azure-Florence Group](https://www.microsoft.com/en-us/research/project/azure-florence-vision-and-language). Thanks my collaborators.

## License
SEED is released under the MIT license. 
