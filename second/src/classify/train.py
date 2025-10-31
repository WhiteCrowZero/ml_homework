#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Name   : train.py
Author      : wzw
Date Created: 2025/10/31
Description : 训练文件，负责训练模型，mian入口
"""
import logging
import os
import shutil
import sys
import csv
import yaml
import math
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import torchvision
from torchvision import datasets, models
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms, InterpolationMode

import torch.backends.cudnn as cudnn

from second.src.utils.log import init_logger

logger = init_logger(__name__, module_name="ModelTrain", log_dir="logs")

def lab_init(seed: int) -> None:
    """
    初始化随机种子，保证实验重复性
    """

    # 设置随机种子，保证实验重复性
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 查看版本和可用设备
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")


def save_checkpoint(
    state: dict, is_best: bool, filename: str = "checkpoint.pth.tar"
) -> None:
    """
    保存模型检查点。
    is_best=True 则额外保存为 model_best.pth.tar。
    """
    torch.save(state, filename)
    if is_best:
        best_path = os.path.join(os.path.dirname(filename), "model_best.pth.tar")
        torch.save(state, best_path)
        logger.info(f"最佳模型已保存到: {best_path}")

def save_config_file(model_checkpoints_folder: str, args: dict) -> None:
    """
    保存训练配置文件为 YAML 格式。
    """
    os.makedirs(model_checkpoints_folder, exist_ok=True)
    config_path = os.path.join(model_checkpoints_folder, "config.yml")
    with open(config_path, "w") as f:
        yaml.dump(args, f, default_flow_style=False)
    logger.info(f"配置文件已保存到: {config_path}")


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> list:
    """
    计算 top-k 准确率。
    """
    with torch.no_grad():
        max_k = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
