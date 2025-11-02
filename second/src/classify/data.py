#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Name   : data.py
Date Created: 2025/10/31
Description : 图像分类数据处理模块（针对 Tiny ImageNet）
"""
import glob
import os
from enum import Enum
from typing import List, Tuple, Dict

import torch
import numpy as np
import torchvision
from torch import nn
from torch.utils.data import Subset, Dataset, ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.transforms import InterpolationMode, Compose

from src.utils.log import init_logger
from src.classify.exceptions import (
    InvalidDatasetSelection,
    InvalidDatasetClassesNum,
)

# 初始化日志系统
logger = init_logger(name=__name__, module_name="DataProcess", log_dir="logs")


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"


class TinyValDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        self.loader = default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class SimpleImageDataset(Dataset):
    """
    轻量数据集：用 (path, label) + classes 列表构造，行为与 ImageFolder 接近，
    便于跨分片合并和重映射。
    """
    def __init__(self, samples: List[Tuple[str, int]], classes: List[str], transform=None, loader=None):
        self.samples = samples
        self.targets = [y for _, y in samples]
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform
        self.loader = loader or torchvision.datasets.folder.default_loader
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = self.loader(p)
        if self.transform: img = self.transform(img)
        return img, y


class ImageClassificationDataset:
    """
    图像分类数据集加载类，支持：
    - tinyimagenet (200 类，可截断)
    - imagenet100 (100 类，可截断)
    """
    def __init__(self, root_folder: str, num_classes: int = 20, dataset_name="tinyimagenet") -> None:
        self.root_folder = root_folder
        self.dataset_name = dataset_name.lower()
        self.num_classes = num_classes

        if self.dataset_name not in {"tinyimagenet", "imagenet100"}:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        print(f"[INFO] 初始化数据集: {self.dataset_name}, 路径: {root_folder}, 选择 {num_classes} 类")

    @staticmethod
    def get_classification_transform(size: int, mode: DatasetSplit = DatasetSplit.TRAIN):
        if mode == DatasetSplit.TRAIN:
            return transforms.Compose([
                transforms.RandomResizedCrop(size, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=7),
                transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), inplace=True),
            ])
        else:
            scale = int(round(size / 0.875))
            return transforms.Compose([
                transforms.Resize(scale, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    # ---------- 公用小工具 ----------
    def _subset_imagefolder(self, ds: ImageFolder) -> Subset:
        """把 ImageFolder 截到前 N 类（按字母序），并保留 transform。"""
        if not (1 <= self.num_classes <= len(ds.classes)):
            raise InvalidDatasetClassesNum(f"num_classes={self.num_classes} 越界，最大 {len(ds.classes)}")
        selected = set(ds.classes[: self.num_classes])
        idxs = [i for i, (_, y) in enumerate(ds.samples) if ds.classes[y] in selected]
        return Subset(ds, idxs)

    def _filter_imagefolder_by_classes(self, ds: ImageFolder, kept_idx: Dict[str, int]) -> SimpleImageDataset:
        """把一个 ImageFolder 过滤到 kept_idx 指定的类集合，并把标签重映射到 [0, N)。"""
        kept = set(kept_idx.keys())
        new_samples = []
        for p, y in ds.samples:
            cls = ds.classes[y]
            if cls in kept:
                new_samples.append((p, kept_idx[cls]))
        classes = [None] * len(kept_idx)
        for cls, i in kept_idx.items():
            classes[i] = cls
        return SimpleImageDataset(new_samples, classes, transform=ds.transform, loader=ds.loader)

    def _merge_sharded_imagefolders(self, base: str, shard_dirs: List[str], split: DatasetSplit, transform):
        """
        把 train.X*/val.X* 多个分片的 ImageFolder 合并，并按“训练集字母序的前 N 类”对齐重映射。
        """
        # 基准类顺序：优先使用标准 train/ 目录；若无，则使用所有分片类名并集
        train_dir = os.path.join(base, DatasetSplit.TRAIN.value)
        if os.path.isdir(train_dir):
            base_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        else:
            s = set()
            for d in shard_dirs:
                ds_part = ImageFolder(d)  # 只为拿 classes
                s.update(ds_part.classes)
            base_classes = sorted(s)

        if not base_classes:
            raise InvalidDatasetSelection("未能确定基准类列表")

        kept = base_classes[: self.num_classes]
        kept_idx = {c: i for i, c in enumerate(kept)}

        # 收集所有分片样本并重映射标签
        all_samples: List[Tuple[str, int]] = []
        loader = None
        for d in shard_dirs:
            ds_part = ImageFolder(d)  # 不传 transform，只读结构
            loader = loader or ds_part.loader
            for p, y in ds_part.samples:
                cls = ds_part.classes[y]
                if cls in kept_idx:
                    all_samples.append((p, kept_idx[cls]))

        classes = [None] * len(kept_idx)
        for cls, i in kept_idx.items():
            classes[i] = cls
        return SimpleImageDataset(all_samples, classes, transform=transform, loader=loader)

    # ---------- Tiny-ImageNet ----------
    def _load_tinyimagenet(self, split: DatasetSplit):
        base = os.path.join(self.root_folder, "tiny-imagenet-200")
        if not os.path.exists(base):
            raise InvalidDatasetSelection(f"未找到 TinyImageNet 路径: {base}")

        if split == DatasetSplit.TRAIN:
            path = os.path.join(base, "train")
            transform = self.get_classification_transform(64, split)
            dataset = ImageFolder(path, transform=transform)
            dataset = self._subset_imagefolder(dataset)
        else:  # VAL
            img_dir = os.path.join(base, "val", "images")
            anno_file = os.path.join(base, "val", "val_annotations.txt")

            train_classes = sorted(os.listdir(os.path.join(base, "train")))[: self.num_classes]
            class_to_idx = {c: i for i, c in enumerate(train_classes)}

            samples: List[Tuple[str, int]] = []
            with open(anno_file) as f:
                for line in f:
                    img, cls, *_ = line.split()
                    if cls in class_to_idx:
                        samples.append((os.path.join(img_dir, img), class_to_idx[cls]))

            transform = self.get_classification_transform(64, DatasetSplit.VAL)
            dataset = TinyValDataset(samples, transform=transform)

        return dataset

    # ---------- ImageNet-100 ----------
    def _load_imagenet100(self, split: DatasetSplit):
        base = os.path.join(self.root_folder, "imagenet-100")
        if not os.path.exists(base):
            raise InvalidDatasetSelection(f"未找到 ImageNet-100 路径: {base}")

        image_size = 224
        tf = self.get_classification_transform(image_size, split)

        std_dir = os.path.join(base, split.value)  # 期望的 train/ 或 val/
        if os.path.isdir(std_dir):
            ds = ImageFolder(std_dir, transform=tf)
            if split == DatasetSplit.TRAIN:
                return self._subset_imagefolder(ds)
            else:
                # 验证集按训练集前 N 类过滤/重映射，确保映射一致
                train_dir = os.path.join(base, DatasetSplit.TRAIN.value)
                if not os.path.isdir(train_dir):
                    raise InvalidDatasetSelection("需要 train/ 目录以对齐验证集类顺序")
                train_classes_all = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
                kept = train_classes_all[: self.num_classes]
                kept_idx = {c: i for i, c in enumerate(kept)}
                return self._filter_imagefolder_by_classes(ds, kept_idx)

        # fallback: 分片目录 train.X* / val.X*
        shard_dirs = sorted(glob.glob(os.path.join(base, f"{split.value}.X*")))
        if shard_dirs:
            return self._merge_sharded_imagefolders(base, shard_dirs, split, tf)

        raise FileNotFoundError(f"未找到 {split.value} 数据。需要 {std_dir} 或 {split.value}.X*")

    # ---------- 入口 ----------
    def get_dataset(self, split: DatasetSplit):
        if self.dataset_name == "tinyimagenet":
            dataset = self._load_tinyimagenet(split)
        elif self.dataset_name == "imagenet100":
            dataset = self._load_imagenet100(split)
        else:
            raise ValueError(f"未知数据集: {self.dataset_name}")

        print(f"[OK] 加载 {self.dataset_name} - {split.value} : {len(dataset)} 张图像")
        return dataset


if __name__ == "__main__":
    # 测试数据集加载
    dataset = ImageClassificationDataset(
        root_folder="../../datasets", dataset_name="imagenet100", num_classes=20
    )
    train_dataset = dataset.get_dataset(DatasetSplit.TRAIN)
    val_dataset = dataset.get_dataset(DatasetSplit.VAL)
