#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Name   : data.py
Author      : wzw
Date Created: 2025/10/31
Description : 图像分类数据处理模块（针对 Tiny ImageNet）
"""

import os
from enum import Enum

import torch
import numpy as np
from torch import nn
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode, Compose

from second.src.utils.log import init_logger
from second.src.utils.exceptions import (
    InvalidDatasetSelection,
    InvalidDatasetClassesNum,
)

# 初始化日志系统
logger = init_logger(name=__name__, module_name="DataProcess", log_dir="logs")


class GaussianBlur:
    """
    高斯模糊数据增强，用于提升模型鲁棒性。
    """

    def __init__(self, kernel_size: int):
        radius = kernel_size // 2
        kernel_size = radius * 2 + 1

        # 定义水平和垂直方向的高斯卷积核
        self.blur_h = nn.Conv2d(
            3, 3, kernel_size=(kernel_size, 1), groups=3, bias=False
        )
        self.blur_v = nn.Conv2d(
            3, 3, kernel_size=(1, kernel_size), groups=3, bias=False
        )

        self.kernel_size = kernel_size
        self.radius = radius
        self.blur = nn.Sequential(nn.ReflectionPad2d(radius), self.blur_h, self.blur_v)

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        """
        应用高斯模糊到图像。
        """
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.radius, self.radius + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        # 更新卷积权重
        self.blur_h.weight.data.copy_(x.view(3, 1, self.kernel_size, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.kernel_size))

        with torch.no_grad():
            img = self.blur(img).squeeze()

        return self.tensor_to_pil(img)


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class ImageClassificationDataset:
    """
    图像分类数据集加载类
    """

    def __init__(self, root_folder: str, num_classes: int = 20) -> None:
        self.root_folder = root_folder
        self.dataset_name = "tinyimagenet"
        self.num_classes = num_classes
        logger.info(f"初始化数据集路径: {root_folder}")

    @staticmethod
    def get_classification_transform(size: int, s: float = 1.0) -> Compose:
        """
        定义 Tiny ImageNet 的数据增强流水线。
        """
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=size, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(kernel_size=int(0.1 * size)),
                transforms.RandomAffine(degrees=20, translate=(0.1, 0.1)),
                transforms.RandomSolarize(threshold=128, p=0.2),
                transforms.RandomEqualize(p=0.2),
                transforms.ToTensor(),
            ]
        )

    def get_subset(self, dataset: ImageFolder) -> Subset:
        """
        获取 Tiny ImageNet 数据集的子集
        """
        if not (1 <= self.num_classes <= 200):
            raise InvalidDatasetClassesNum(
                f"无效的类别数量: {self.num_classes}; 范围应为[1, 200]"
            )

        selected_classes = dataset.classes[: self.num_classes]

        # 获取对应的索引
        selected_indices = [
            i
            for i, (_, label) in enumerate(dataset.samples)
            if dataset.classes[label] in selected_classes
        ]
        subset = Subset(dataset, selected_indices)
        return subset

    def get_dataset(self, split: DatasetSplit) -> torch.utils.data.Dataset:
        """
        获取 Tiny ImageNet 数据集（train 或 val）。
        """
        dataset_path = os.path.join(self.root_folder, "tiny-imagenet-200", split.value)
        if not os.path.exists(dataset_path):
            raise InvalidDatasetSelection(f"未找到数据集路径: {dataset_path}")

        transform = self.get_classification_transform(64)
        dataset = ImageFolder(root=dataset_path, transform=transform)
        dataset = self.get_subset(dataset)

        logger.info(
            f"成功加载 Tiny ImageNet [{split.value}] 数据，共 {self.num_classes} 类，共 {len(dataset)} 张图像。"
        )
        return dataset


if __name__ == "__main__":
    # 测试数据集加载
    dataset = ImageClassificationDataset(root_folder="../datasets", num_classes=20)
    train_dataset = dataset.get_dataset(DatasetSplit.TRAIN)
