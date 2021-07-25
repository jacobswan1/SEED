
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
ResNet-50  |  MoCo-V1 | 200 | [URL](https://vigeast.blob.core.windows.net/data/zfang/outputs/moco_v1_200ep_pretrain.pth.tar?sv=2020-04-08&st=2021-07-25T02%3A56%3A18Z&se=2025-07-26T02%3A56%3A00Z&sr=b&sp=r&sig=Pk4R5ZoA8ikVh4G0Tq%2BzBapVufWgSr5D1gqh4n%2FvQic%3D)
ResNet-50  |  SimCLR |    200    |          [URL](https://vigeast.blob.core.windows.net/data/zfang/outputs/simclr_200.pth?sv=2020-04-08&st=2021-07-25T02%3A55%3A03Z&se=2025-07-26T02%3A55%3A00Z&sr=b&sp=r&sig=DFXKnDCpiTlK0I4ss%2Bp4hjqmLGaUpJAd1vjnfMwVtjI%3D)
ResNet-50  |  MoCo-V2 |    200    |          [URL](https://vigeast.blob.core.windows.net/data/zfang/outputs/jianfw_mocov2_ResNet50_epoch200.pth?sv=2020-04-08&st=2021-07-25T02%3A47%3A33Z&se=2025-07-26T02%3A47%3A00Z&sr=b&sp=r&sig=MI5bGp7gJGke6sT%2BNuxtqa7aASBT6oR7xOwzReWqw3I%3D)
ResNet-50  |  MoCo-V2 |    800    |          [URL](https://vigeast.blob.core.windows.net/data/zfang/outputs/moco_v2_800ep_pretrain.pth.tar?sv=2020-04-08&st=2021-07-25T02%3A50%3A30Z&se=2025-07-26T02%3A50%3A00Z&sr=b&sp=r&sig=PFE5aHskdQNJ5cwMvp5MUKAAuBWsAbiNd5uRmQKQEyU%3D)
ResNet-50  |  SWAV |    800    |          [URL](https://vigeast.blob.core.windows.net/data/zfang/outputs/swav_800ep_pretrain.pth.tar?sv=2020-04-08&st=2021-07-25T02%3A51%3A20Z&se=2025-07-26T02%3A51%3A00Z&sr=b&sp=r&sig=GQ7ZOt61%2FQ2XGSTO%2By4CeAEcSiYZxmalWZyZUFu7GhE%3D)
ResNet-101  |  MoCo-V2 |    200    |          [URL](https://vigeast.blob.core.windows.net/data/zfang/outputs/res101-moco-v2-checkpoint_0199.pth.tar?sv=2020-04-08&st=2021-07-25T02%3A49%3A27Z&se=2025-07-26T02%3A49%3A00Z&sr=b&sp=r&sig=6%2BhWamCRxu9QD83bZu7E52EnTRDHv09u9oyiaySsWgQ%3D)
ResNet-152  |  MoCo-V2 |    200    |          [URL](https://vigeast.blob.core.windows.net/data/zfang/outputs/200-resnet152-moco-v2-checkpoint_0199.pth.tar?sv=2020-04-08&st=2021-07-25T02%3A49%3A50Z&se=2025-07-26T02%3A49%3A00Z&sr=b&sp=r&sig=Fre5Wmer3VzHTNBUHbRijnSYZ05ustM8TsfchQX7Y1w%3D)
ResNet-152  |  MoCo-V2 |    800    |          [URL](https://vigeast.blob.core.windows.net/data/zfang/outputs/800-resnet152-moco-v2-checkpoint_0799.pth.tar?sv=2020-04-08&st=2021-07-25T02%3A50%3A52Z&se=2025-07-26T02%3A50%3A00Z&sr=b&sp=r&sig=ywt6nenAdrgvkW%2BeUz5yOdu6PVckZT7DVh11iSEiXbQ%3D)
ResNet-50X2  |  SWAV |    400    |          [URL](https://vigeast.blob.core.windows.net/data/zfang/outputs/swav_RN50w2_400ep_pretrain.pth.tar?sv=2020-04-08&st=2021-07-25T02%3A51%3A54Z&se=2025-07-26T02%3A51%3A00Z&sr=b&sp=r&sig=GZSdfPrkk4O18UB7ic7wc5GI1MhEyrNcguIPFl8liIc%3D)
ResNet-50X4  |  SWAV |    400    |          [URL](https://vigeast.blob.core.windows.net/data/zfang/outputs/swav_RN50w4_400ep_pretrain.pth.tar?sv=2020-04-08&st=2021-07-25T02%3A52%3A28Z&se=2025-07-26T02%3A52%3A00Z&sr=b&sp=r&sig=HHQNN29%2FPyj4xMmkLy8GCS3uu%2FccVgLTi1OImvYc2OI%3D)
ResNet-50X5  |  SWAV |    200    |          [URL](https://vigeast.blob.core.windows.net/data/zfang/outputs/swav_RN50w5_400ep_pretrain.pth.tar?sv=2020-04-08&st=2021-07-25T02%3A54%3A23Z&se=2025-07-26T02%3A54%3A00Z&sr=b&sp=r&sig=q7Yeoy1N1xmtV6k5w7YrdtnlUIycyEJOf84%2FRWYlkBg%3D)


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
ResNet-18|  ResNet-50 |   MoCo-V2 | 200    |          [URL](https://vigeast.blob.core.windows.net/data/zfang/outputs/resnet18_distill_resnet50-moco-v2-checkpoint_0199.pth.tar?sv=2020-04-08&st=2021-07-25T02%3A59%3A38Z&se=2025-07-26T02%3A59%3A00Z&sr=b&sp=r&sig=brTt6vH4UJk7sMENwBXeZpX%2BSyp6bnp2oTM%2BA45cM7w%3D)
ResNet-18|  ResNet-50W2 |   SWAV | 400    |          [URL](https://vigeast.blob.core.windows.net/data/zfang/outputs/AMP_SMALL_PATCH_400_resnet18_distill_resnet50w2-swav-checkpoint_0399.pth.tar?sv=2020-04-08&st=2021-07-25T03%3A08%3A51Z&se=2025-07-26T03%3A08%3A00Z&sr=b&sp=r&sig=JU1KoWpZ4XSdmTf1qYRNBxGYdLnzgW4%2F7oXZiQvJ69I%3D)
MobileV3-Large|  ResNet-50 |   MoCo-V2 | 200    |          [URL](https://vigeast.blob.core.windows.net/data/zfang/outputs/mobilenetv3_large_distill-moco-v2-checkpoint_0199.pth.tar?sv=2020-04-08&st=2021-07-25T03%3A01%3A55Z&se=2025-07-26T03%3A01%3A00Z&sr=b&sp=r&sig=u9TrucleCJIldYQTNjBEqICVhcL9Pjnqw1Yg9M7M5LM%3D)
EfficientNet-B0|  ResNet-50W4 |   SWAV | 400    |          [URL](https://vigeast.blob.core.windows.net/data/zfang/outputs/AMP_SMALL_PATCH_400_efficientnet_b0_distill_resnet50w4-swav-checkpoint_0399.pth.tar?sv=2020-04-08&st=2021-07-25T03%3A07%3A57Z&se=2025-07-26T03%3A07%3A00Z&sr=b&sp=r&sig=AV4sdyPpd2Nhb55xbKPOI5GDcHCs2K%2Fq2A6V1LUhyQA%3D)
EfficientNet-B0 | ResNet-50W2 | SWAV | 800 | [URL](https://vigeast.blob.core.windows.net/data/zfang/outputs/AMP_SMALL_PATCH_800_efficientnet_b0_distill_resnet50w2-swav-checkpoint_0799.pth.tar?sv=2020-04-08&st=2021-07-25T03%3A10%3A37Z&se=2025-07-26T03%3A10%3A00Z&sr=b&sp=r&sig=ksBSOWukI9qjW3akl7fiwC6zyF52W5teETAuRysblLc%3D)
EfficientNet-B1|  ResNet-152 |   SWAV | 200    |          [URL](https://vigeast.blob.core.windows.net/data/zfang/outputs/AMP_200_efficientnet_b1_distill_resnet152-swav-checkpoint_0199.pth.tar?sv=2020-04-08&st=2021-07-25T03%3A05%3A29Z&se=2025-07-26T03%3A05%3A00Z&sr=b&sp=r&sig=AOlOixFy2ZzLR5xEey7ViCVIjvoIQ%2FnkUF9yHBuP%2Bp8%3D)
EfficientNet-B1|  ResNet-152 |   SWAV | 200    |          [URL](https://vigeast.blob.core.windows.net/data/zfang/outputs/AMP_200_efficientnet_b1_distill_resnet152-swav-checkpoint_0199.pth.tar?sv=2020-04-08&st=2021-07-25T03%3A05%3A29Z&se=2025-07-26T03%3A05%3A00Z&sr=b&sp=r&sig=AOlOixFy2ZzLR5xEey7ViCVIjvoIQ%2FnkUF9yHBuP%2Bp8%3D)
ResNet-50 | ResNet-50W4 | SWAV | 400 | [URL](https://vigeast.blob.core.windows.net/data/zfang/outputs/AMP_SMALL_PATCH_400_resnet50_distill_resnet50w4-swav-checkpoint_0399.pth.tar?sv=2020-04-08&st=2021-07-25T03%3A09%3A33Z&se=2025-07-26T03%3A09%3A00Z&sr=b&sp=r&sig=bDLiB%2FGbczRmXJKrNFys7%2F7TGa7DAUZV3p6ZDGhUdj0%3D)




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
