
#  MSVIT: Improving Spiking Vision Transformer Using Multi-scale Attention Fusion ([IJCAI 2025](https://arxiv.org/pdf/2505.14719))

MSVIT has achieved 85.06% top-1 accuracy on ImageNet-1K with 224×224 input size and 4-time steps using direct training from scratch.

## News

[2025.4.29] Accepted by IJCAI 2025.



## Abstact

The combination of Spiking Neural Networks (SNNs) with Vision Transformer architectures has garnered significant attention due to their potential for energy-efficient and high-performance computing paradigms.
However, a substantial performance gap still exists between SNN-based and ANN-based transformer architectures.  While existing methods propose spiking self-attention mechanisms that are successfully combined with SNNs, the overall architectures proposed by these methods suffer from a bottleneck in effectively extracting features from different image scales. In this paper, we address this issue and propose **MSViT**. This novel spike-driven Transformer architecture firstly uses multi-scale spiking attention (MSSA) to enhance the capabilities of spiking attention blocks.
We validate our approach across various main data sets. The experimental results show that MSViT outperforms existing  SNN-based models, positioning itself as a state-of-the-art solution among SNN-transformer architectures. The codes are available at  ([link](https://github.com/Nanhu-AI-Lab/MSViT))

[//]: # (<p align="center">)

[//]: # (<img src="https://github.com/zhouchenlin2096/QKFormer/blob/master/imgs/QKFormer.png">)

[//]: # (</p>)


## Main results on ImageNet-1K


| Method | Spiking | Architecture | Params (M) | Input Size | Time Step | Energy (mJ) | Top-1 Acc. (%) |
|--------|---------|--------------|------------|------------|-----------|-------------|----------------|
| DeiT  | × | DeiT-B | 86.60 | 224x224 | 1 | 80.50 | 81.80 |
| VIT-B/16  | × | ViT-12-768 | 86.59 | $384^2$ | 1 | 254.84 | 77.90 |
| Swin Transformer  | √ | Swin-T | 28.50 | 224x224 | 1 | 70.84 | 81.35 |
| Swin Transformer  | × | Swin-S | 51.00 | 224x224 | 1 | 216.20 | 83.03 |
| Spikformer  | √ | 8-384 | 16.80 | 224x224 | 4 | 5.97 | 70.24 |
| Spikformer  | √ | 8-768 | 66.30 | 224x224 | 4 | 20.0 | 74.81 |
| Spikformer V2  | √ | V2-8-384 | 29.11 | 224x224 | 4 | 4.69 | 78.80 |
| Spikformer V2  | √ | V2-8-512 | 51.55 | 224x224 | 4 | 9.36 | 80.38 |
| Spike-driven  | √ | SDT 8-384 | 16.81 | 224x224 | 4 | 3.90 | 72.28 |
| Spike-driven  | √ | SDT8-512 | 29.68 | 224x224 | 4 | 4.50 | 74.57 |
| Spike-driven  | √ | SDT8-768 | 66.34 | 224x224 | 4 | 6.09 | 77.07 |
| Spike-driven v2  | √ | SDT v2-10-384 | 15.10 | 224x224 | 4 | 16.70 | 74.10 |
| Spike-driven v2  | √ | SDT v2-10-512 | 31.30 | 224x224 | 4 | 32.80 | 77.20 |
| Spike-driven v2  | √ | SDT v2-10-768 | 55.40 | 224x224 | 4 | 52.40 | 80.00 |
| QKFormer  | √ | QK-10-384 | 16.47 | 224x224 | 4 | 15.13 | 78.80 |
| QKFormer  | √ | QK-10-512 | 29.08 | 224x224 | 4 | 21.99 | 82.04 |
| QKFormer  | √ | QK-10-768 | 64.96 | 224x224 | 4 | 38.91 | 84.22 |
| MSViT | √ | MSViT-10-384 | 17.69 | 224x224 | 4 | 16.65 | **80.09** |
| MSViT | √ | MSViT-10-512 | 30.23 | 224x224 | 4 | 24.74 | **82.96** |
| MSViT | √ | MSViT-10-768 | 69.80 | 224x224 | 4 | 45.88 | **85.06** |





## Requirements

```
timm==0.6.12
cupy==11.4.0
torch==1.12.1
spikingjelly==0.0.0.0.12
pyyaml
tensorboard

│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

## Train & Test
### Training  on ImageNet
```
cd imagenet
python -m torch.distributed.launch --nproc_per_node=8 train.py  
```

### Testing ImageNet Val data
Download the trained model first, then:
```
cd imagenet
python test.py
```

### Training  on CIFAR10
Setting hyper-parameters in cifar10.yml
```
cd cifar10
python train.py
```

### Training  on CIFAR100
Setting hyper-parameters in cifar100.yml
```
cd cifar10
python train.py
```

### Training  on DVS128 Gesture
```
cd dvs128-gesture
python train.py
```

### Training  on CIFAR10-DVS
```
cd cifar10-dvs
python train.py
```

## Reference
```
@article{hua2025msvit,
  title={MSVIT: Improving Spiking Vision Transformer Using Multi-scale Attention Fusion},
  author={Hua, Wei and Zhou, Chenlin and Wu, Jibin and Chua, Yansong and Shu, Yangyang},
  journal={arXiv preprint arXiv:2505.14719},
  year={2025}
}
```
If you find this repo useful, please consider citing:
```
@inproceedings{
zhou2024qkformer,
title={{QKF}ormer: Hierarchical Spiking Transformer using Q-K Attention},
author={Chenlin Zhou and Han Zhang and Zhaokun Zhou and Liutao Yu and Liwei Huang and Xiaopeng Fan and Li Yuan and Zhengyu Ma and Huihui Zhou and Yonghong Tian},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=AVd7DpiooC}
}
```


```
@article{zhou2023spikingformer,
  title={Spikingformer: Spike-driven residual learning for transformer-based spiking neural network},
  author={Zhou, Chenlin and Yu, Liutao and Zhou, Zhaokun and Ma, Zhengyu and Zhang, Han and Zhou, Huihui and Tian, Yonghong},
  journal={arXiv preprint arXiv:2304.11954},
  year={2023}
}
```


## Acknowledgement & Contact Information
Related project: [spikformer](https://github.com/ZK-Zhou/spikformer), [spikingformer](https://github.com/zhouchenlin2096/Spikingformer), [QK-Former](https://github.com/zhouchenlin2096/QKFormer), [spikingjelly](https://github.com/fangwei123456/spikingjelly).

For help or issues using this git, please submit a GitHub issue. 

For other communications related to this git, please contact huawei@cnaeit.com, yangyangshu@usw.edu.au or zhouchenlin19@mails.ucas.ac.cn.
