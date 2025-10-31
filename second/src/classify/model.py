#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Name   : model.py
Date Created: 2025/10/31
Description : 以ResNet为基础的监督学习分类模型
"""
import os
import random
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torchvision import models

from second.src.utils.log import init_logger
from second.src.classify.exceptions import InvalidBackboneError
from second.src.classify.data import ImageClassificationDataset, DatasetSplit
from second.src.classify.visual import TrainingVisualizer

# 初始化日志系统
logger = init_logger(name=__name__, module_name="ModelLoad", log_dir="logs")


class ResNetClassifier(nn.Module):
    """
    基于 ResNet 的分类模型，可插入自定义卷积层。
        base_model: 'resnet18' / 'resnet34' / 'resnet50'
        num_classes: 输出类别数
        use_custom_conv: 是否添加自定义卷积层（默认False）
    """

    def __init__(
        self, base_model: str, num_classes: int, use_custom_conv: bool = False
    ) -> None:
        super().__init__()

        self.resnet_dict = {
            "resnet18": models.resnet18(weights=None),
            "resnet34": models.resnet34(weights=None),
            "resnet50": models.resnet50(weights=None),
        }

        if base_model not in self.resnet_dict:
            raise InvalidBackboneError(
                f"无效基础模型： '{base_model}'，请选择： resnet18/34/50"
            )

        # 取 backbone
        self.backbone = self.resnet_dict[base_model]
        dim_in = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # 自定义卷积层
        if use_custom_conv:
            self.custom_conv = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True),
            )
        else:
            self.custom_conv = nn.Identity()

        # 分类头（MLP Head）
        self.classifier_head = nn.Sequential(
            nn.Linear(dim_in, dim_in * 2),
            nn.BatchNorm1d(dim_in * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(dim_in * 2, num_classes),
        )

        # 损失函数
        self.cost_function = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        x = self.custom_conv(x)
        features = self.backbone(x)
        logits = self.classifier_head(features)
        return logits, features

    def compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """计算分类损失"""
        return self.cost_function(predictions, targets)


class SupervisedTrainer:
    """
    标准监督学习训练类
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        args: SimpleNamespace,
    ) -> None:
        self.args = args
        self.save_path = args.save_path
        self.model = model.to(args.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.visualizer = TrainingVisualizer()
        self.visual_method=args.visual_method
        self.lab_init(args.seed)

        self.train_loss_history, self.val_loss_history = [], []
        self.train_acc_history, self.val_acc_history = [], []
        self.best_acc = 0.0

    @staticmethod
    def lab_init(seed: int) -> None:
        """
        初始化随机种子，保证实验重复性
        """
        # 设置随机种子，保证实验重复性
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # 查看版本和可用设备
        logger.info(
            f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}"
        )

    def save_checkpoint(self, epoch: int, acc: float, is_best: bool) -> None:
        """
        保存模型检查点。
        is_best=True 则额外保存为 model_best.pth.tar。
        """
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "acc": acc,
        }
        os.makedirs(self.save_path, exist_ok=True)
        ckpt_path = f"{self.save_path}/epoch_{epoch}.pth"
        torch.save(state, ckpt_path)
        if is_best:
            torch.save(state, f"{self.save_path}/model_best.pth")
            logger.info(f"最佳模型已保存到 {self.save_path}/model_best.pth")

    def train_one_epoch(self, dataloader: DataLoader, epoch: int) -> (float, float):
        """单轮训练"""
        self.model.train()
        total_loss, total_correct, total = 0, 0, 0
        for images, targets in dataloader:
            images, targets = images.to(self.args.device), targets.to(self.args.device)
            outputs, _ = self.model(images)
            loss = self.model.compute_loss(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            total_correct += (preds == targets).sum().item()
            total += targets.size(0)

        avg_loss = total_loss / total
        avg_acc = total_correct / total * 100
        return avg_loss, avg_acc

    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """验证集评估"""
        self.model.eval()
        total_loss, total_correct, total = 0, 0, 0

        with torch.no_grad():
            for images, targets in dataloader:
                images, targets = images.to(self.args.device), targets.to(
                    self.args.device
                )
                outputs, _ = self.model(images)
                loss = self.model.compute_loss(outputs, targets)
                total_loss += loss.item() * images.size(0)
                preds = outputs.argmax(1)
                total_correct += (preds == targets).sum().item()
                total += targets.size(0)

        avg_loss = total_loss / total
        acc = total_correct / total * 100
        return avg_loss, acc

    def extract_features(self, dataloader):
        """提取真实特征用于可视化"""
        self.model.eval()
        feats, labels = [], []
        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.args.device)
                _, f = self.model(images)
                feats.append(f.cpu().numpy())
                labels.append(targets.numpy())
        return np.concatenate(feats), np.concatenate(labels)

    def train(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None
    ) -> None:
        """完整训练流程"""
        logger.info("开始监督学习训练！")
        # 训练前可视化
        if val_loader:
            logger.info("训练前特征投影可视化")
            feats, labels = self.extract_features(val_loader)
            self.visualizer.visualize_feature_projection(feats, labels, method=self.visual_method)

        for epoch in range(1, self.args.epochs + 1):
            train_loss, train_acc = self.train_one_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate(val_loader)
            self.scheduler.step()

            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)
            self.val_loss_history.append(val_loss)
            self.val_acc_history.append(val_acc)

            logger.info(
                f"[Epoch {epoch}/{self.args.epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            self.save_checkpoint(epoch, val_acc, is_best=val_acc > self.best_acc)
            self.best_acc = max(self.best_acc, val_acc)

        # 绘制训练曲线
        self.visualizer.plot_training_curves(
            self.train_loss_history,
            self.train_acc_history,
            self.val_loss_history,
            self.val_acc_history,
        )

        # 提取特征再可视化
        if val_loader:
            logger.info("训练后特征投影可视化")
            feats, labels = self.extract_features(val_loader)
            self.visualizer.visualize_feature_projection(feats, labels, method=self.visual_method)

        logger.info("训练完成")


if __name__ == "__main__":
    import torch.optim as optim

    args = SimpleNamespace(
        data_path="../../datasets",
        save_path="checkpoints",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        epochs=3,
        lr=1e-3,
        weight_decay=1e-4,
        log_every_n_steps=50,
        batch_size=64,
        num_classes=20,
        visual_method="tsne",
        seed=0,
    )

    # # 自定义卷积层参数
    # custom_conv_params = {"kernel_size": 3, "out_channels": 32}

    # 模型实例化
    model = ResNetClassifier(
        base_model="resnet18",
        num_classes=20,
        # use_custom_conv=True,
        # custom_conv_params=custom_conv_params,
    )

    # 优化器与调度器
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 数据集接口
    dataset = ImageClassificationDataset(
        root_folder=args.data_path, num_classes=args.num_classes
    )
    train_dataset = dataset.get_dataset(DatasetSplit.TRAIN)
    val_dataset = dataset.get_dataset(DatasetSplit.VAL)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    # 训练
    trainer = SupervisedTrainer(model, optimizer, scheduler, args)
    trainer.train(train_loader, val_loader)
