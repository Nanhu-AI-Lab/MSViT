# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
from torch.utils.data import TensorDataset, DataLoader

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import pandas as pd
import os
import torch


def build_dataset_test(is_train, args):
    # 设置数据集的大小和维度
    num_samples = 14197122 if is_train  else 50000  # 假设生成1000000张图像
    image_size = (3, 224, 224)  # ImageNet图像的尺寸
    num_classes = 1000  # ImageNet-1K的类别数
    transform = build_transform(is_train, args)
    # 生成随机图像数据
    images = torch.randn(num_samples, *image_size)

    # 生成随机标签
    labels = torch.randint(0, num_classes, (num_samples,))

    # 将数据封装成TensorDataset
    dataset = TensorDataset(images, labels)

    # 创建DataLoader
    batch_size = args.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 使用DataLoader
    for images, labels in dataloader:
        # 在这里进行模型训练或其他操作
        print(images.shape, labels.shape)

    return dataset

def build_dataset2(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')

    for filename in os.listdir(root):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')):
            image_data = image_data.append({'image_path': os.path.join(root, filename)}, ignore_index=True)

    #dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
