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


This paper is concerned with self-supervised learning for small models. The
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
Note: This repository does not contain the ImageNet dataset building, please refer to [MoCo-V2](https://github.com/facebookresearch/moco) for the enviromental setting & dataset preparation.

## Self-Supervised Distillation Training

Distributed Training, one GPU on single Node: using [SWAV](https://github.com/facebookresearch/swav)'s 400_ep ResNet-50 model as Teacher architecture for a Student EfficientNet-b1 model. 
```
python -m torch.distributed.launch --nproc_per_node=1 main_small-patch.py \
       -a efficientnet_b1 \
       -k resnet50 \
       --teacher_ssl swav \
       --distill /media/drive2/Unsupervised_Learning/moco_distill/output/swav_400ep_pretrain.pth.tar \
       --lr 0.03 
       --batch-size 16 
       --temp 0.2 \
       --workers 4 --output ./output 
       --data [your imagenet-folder with train and val folders]
```

Conduct linear evaluations on ImageNet-val split:
```
python -m torch.distributed.launch --nproc_per_node=1  main_lincls.py \
       -a efficientnet_b0 
       --lr 30 
       --batch-size 32 
       --output ./output /media/drive2/Data_4TB/imagenet2012
```


