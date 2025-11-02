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
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from src.utils.log import init_logger
from src.classify.exceptions import InvalidBackboneError
from src.classify.data import ImageClassificationDataset, DatasetSplit
from src.classify.visual import TrainingVisualizer

# 初始化日志系统
logger = init_logger(name=__name__, module_name="ModelLoad", log_dir="logs")

# 卷积加速
torch.backends.cudnn.benchmark = True

import math

import torch
import torch.nn as nn
from typing import Tuple, Dict


def rand_bbox_xyxy(W, H, lam):
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(0, W)
    cy = np.random.randint(0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    lam = 1.0 - ((x2 - x1) * (y2 - y1)) / (W * H + 1e-9)  # 面积修正
    return x1, y1, x2, y2, lam


def build_param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (
            p.ndim == 1
            or n.endswith(".bias")
            or "bn" in n.lower()
            or "norm" in n.lower()
        ):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lr_lambda(e):  # e 从 0 开始
    e += 1
    if e <= warmup:
        return e / warmup
    t = (e - warmup) / (total - warmup)
    return 0.5 * (1 + math.cos(math.pi * t))


class SEBlock(nn.Module):
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        # r 更小，适合轻量模型
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, max(ch // r, 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(ch // r, 4), ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x)
        return x * w


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        norm_layer=nn.BatchNorm2d,
        use_se: bool = True,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = norm_layer(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = norm_layer(out_ch)
        self.use_se = use_se
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_ch),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class CompactResNet(nn.Module):
    """
    轻量 ResNet-like 模型，兼顾表达与参数量（适合 TinyImageNet 类似场景）
    返回 (logits, feat) ，feat 为最后全连接前的向量
    """

    def __init__(
        self,
        num_classes: int = 20,
        base_channels: int = 64,
        use_se: bool = True,
        use_groupnorm: bool = False,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        # 选择归一化层：BN（默认）或 GN（小 batch 更稳）
        if use_groupnorm:
            norm_layer = lambda c: nn.GroupNorm(8, c)
        else:
            norm_layer = nn.BatchNorm2d

        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(base_channels),
            nn.ReLU(inplace=True),
        )

        # stages: channels 64 -> 128 -> 256
        self.layer1 = self._make_stage(
            base_channels,
            base_channels,
            num_blocks=2,
            stride=1,
            norm_layer=norm_layer,
            use_se=use_se,
        )
        self.layer2 = self._make_stage(
            base_channels,
            base_channels * 2,
            num_blocks=2,
            stride=2,
            norm_layer=norm_layer,
            use_se=use_se,
        )
        self.layer3 = self._make_stage(
            base_channels * 2,
            base_channels * 4,
            num_blocks=2,
            stride=2,
            norm_layer=norm_layer,
            use_se=use_se,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = base_channels * 4
        self.fc1 = nn.Linear(feat_dim, 256)
        self.bn_fc = nn.LayerNorm(256) if use_groupnorm else nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(256, num_classes)

        # loss
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # init
        self._init_weights()

        # feature_maps 存放接口（trainer 里可能使用）
        self.feature_maps: Dict[str, torch.Tensor] = {}

    def _make_stage(
        self,
        in_ch: int,
        out_ch: int,
        num_blocks: int,
        stride: int,
        norm_layer,
        use_se: bool,
    ):
        layers = []
        layers.append(
            ResidualBlock(
                in_ch, out_ch, stride=stride, norm_layer=norm_layer, use_se=use_se
            )
        )
        for _ in range(1, num_blocks):
            layers.append(
                ResidualBlock(
                    out_ch, out_ch, stride=1, norm_layer=norm_layer, use_se=use_se
                )
            )
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        if getattr(self, "save_features", False):
            self.feature_maps["layer1"] = x.detach().cpu()
        x = self.layer2(x)
        if getattr(self, "save_features", False):
            self.feature_maps["layer2"] = x.detach().cpu()
        x = self.layer3(x)
        if getattr(self, "save_features", False):
            self.feature_maps["layer3"] = x.detach().cpu()

        pooled = self.avgpool(x)
        pooled = torch.flatten(pooled, 1)  # (B, feat_dim)
        feat = self.fc1(pooled)
        if not isinstance(self.bn_fc, nn.Identity):
            feat = self.bn_fc(feat)
        feat = self.relu(feat)
        feat = self.dropout(feat)
        out = self.classifier(feat)
        return out, feat

    def compute_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(preds, targets)


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
        self.device = args.device
        self.device_str = args.device_str
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.visualizer = TrainingVisualizer()
        self.visual_method = args.visual_method
        self.use_amp = torch.cuda.is_available() and (getattr(args, "use_amp", True))
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        self._init_seed(args.seed)

        self.train_loss_history, self.val_loss_history = [], []
        self.train_acc_history, self.val_acc_history = [], []
        self.best_acc = 0.0
        self.current_epoch = 0

    @staticmethod
    def _init_seed(seed: int) -> None:
        """
        初始化随机种子，保证实验重复性
        """
        # 设置随机种子，保证实验重复性
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 查看版本和可用设备
        logger.info(
            f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}"
        )

    def _save_checkpoint(self, epoch: int, acc: float, is_best: bool) -> None:
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
        self.model.train()
        iters = len(dataloader)
        total_loss, total_acc, total = 0.0, 0.0, 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)

        use_mix = getattr(self.args, "use_mixup", False) or getattr(
            self.args, "use_cutmix", False
        )
        mix_prob = getattr(self.args, "mix_prob", 0.0)
        mixup_alpha = getattr(self.args, "mixup_alpha", 0.4)
        cutmix_alpha = getattr(self.args, "cutmix_alpha", 1.0)

        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()

            # --- 选择是否进行混合增强 ---
            do_mix = use_mix and (np.random.rand() < mix_prob)
            use_cut = False
            lam = 1.0
            targets_a, targets_b = targets, targets

            if do_mix:
                index = torch.randperm(images.size(0), device=self.device)
                if getattr(self.args, "use_cutmix", False) and np.random.rand() < 0.5:
                    # CutMix
                    lam = np.random.beta(cutmix_alpha, cutmix_alpha)
                    x1, y1, x2, y2, lam = rand_bbox_xyxy(
                        images.size(3), images.size(2), lam
                    )
                    images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
                    targets_a, targets_b = targets, targets[index]
                    # 按实际覆盖面积修正 lam
                    lam = 1.0 - (
                        (x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2))
                    )
                    use_cut = True
                else:
                    # Mixup
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                    images = lam * images + (1 - lam) * images[index, :]
                    targets_a, targets_b = targets, targets[index]

            # --- 前向与损失 ---
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs, _ = self.model(images)
                    if do_mix:
                        loss = lam * self.model.compute_loss(outputs, targets_a) + (
                            1 - lam
                        ) * self.model.compute_loss(outputs, targets_b)
                    else:
                        loss = self.model.compute_loss(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs, _ = self.model(images)
                if do_mix:
                    loss = lam * self.model.compute_loss(outputs, targets_a) + (
                        1 - lam
                    ) * self.model.compute_loss(outputs, targets_b)
                else:
                    loss = self.model.compute_loss(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            if isinstance(
                self.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
            ):
                self.scheduler.step(self.current_epoch - 1 + (batch_idx + 1) / iters)

            # --- 维度断言 ---
            if batch_idx == 0:
                if outputs.ndim != 2 or outputs.shape[1] != getattr(
                    self.args, "num_classes", 10
                ):
                    raise AssertionError(
                        f"输出维度异常: {outputs.shape}, expected num_classes={self.args.num_classes}"
                    )

            # --- 准确率：权重化 ---
            preds = outputs.argmax(1)
            if do_mix:
                acc_a = (preds == targets_a).float().mean().item()
                acc_b = (preds == targets_b).float().mean().item()
                batch_acc = lam * acc_a + (1 - lam) * acc_b
            else:
                batch_acc = (preds == targets).float().mean().item()

            bs = targets.size(0)
            total_loss += loss.item() * bs
            total_acc += batch_acc * bs
            total += bs

            # 每 20 epoch 保存特征图（保持原逻辑）
            if (
                batch_idx == 0
                and epoch % 20 == 0
                and hasattr(self.model, "feature_maps")
                and "layer1" in self.model.feature_maps
            ):
                save_dir = "features_map"
                self.visualizer.visualize_feature_maps(
                    self.model.feature_maps["layer1"],
                    save_dir=save_dir,
                    max_maps=8,
                    n_cols=4,
                    idx=epoch,
                )

        avg_loss = total_loss / total
        avg_acc = total_acc / total * 100.0
        self.train_loss_history.append(avg_loss)
        self.train_acc_history.append(avg_acc)
        return avg_loss, avg_acc

    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """验证集评估（兼容 AMP）"""
        self.model.eval()
        total_loss, total_correct, total = 0.0, 0, 0

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, _ = self.model(images)
                        loss = self.model.compute_loss(outputs, targets)
                else:
                    outputs, _ = self.model(images)
                    loss = self.model.compute_loss(outputs, targets)

                # 首批次断言（再次确认）
                if batch_idx == 0:
                    if outputs.ndim != 2 or outputs.shape[1] != getattr(
                        self.args, "num_classes", 10
                    ):
                        raise AssertionError(
                            f"验证输出维度异常: {outputs.shape}, expected num_classes={self.args.num_classes}"
                        )

                total_loss += loss.item() * images.size(0)
                preds = outputs.argmax(1)
                total_correct += (preds == targets).sum().item()
                total += targets.size(0)

        avg_loss = total_loss / total
        acc = total_correct / total * 100
        return avg_loss, acc

    def extract_features(self, dataloader):
        """提取真实特征用于可视化（兼容 AMP）"""
        self.model.eval()
        feats, labels = [], []

        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device)
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        _, f = self.model(images)
                else:
                    _, f = self.model(images)
                feats.append(f.detach().cpu().numpy())
                labels.append(targets.cpu().numpy())

        feats = np.concatenate(feats, axis=0)
        labels = np.concatenate(labels, axis=0)
        return feats, labels

    def train(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None
    ) -> None:
        """完整训练流程"""
        logger.info("开始监督学习训练！")

        for epoch in range(1, self.args.epochs + 1):
            self.current_epoch = epoch
            train_loss, train_acc = self.train_one_epoch(train_loader, epoch)
            val_loss, val_acc = (0.0, 0.0)

            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
                self.val_loss_history.append(val_loss)
                self.val_acc_history.append(val_acc)

            if hasattr(self.scheduler, "step"):
                self.scheduler.step()

            logger.info(
                f"[Epoch {epoch}/{self.args.epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            self._save_checkpoint(epoch, val_acc, is_best=val_acc > self.best_acc)
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
            self.visualizer.visualize_feature_projection(
                feats, labels, method=self.visual_method
            )

        logger.info("训练完成")


if __name__ == "__main__":
    import torch.optim as optim

    args = SimpleNamespace(
        data_path="/home/ubuntu/train/datasets",
        save_path="checkpoints",
        dataset_name="imagenet100",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        device_str="cuda" if torch.cuda.is_available() else "cpu",
        use_amp=True,
        epochs=100,
        lr=1e-3,
        weight_decay=1e-2,
        batch_size=64,
        num_classes=10,
        visual_method="tsne",
        use_mixup=True,
        mixup_alpha=0.4,
        use_cutmix=True,
        cutmix_alpha=1.0,
        mix_prob=0.5,
        seed=0,
    )

    # 模型实例化
    model = CompactResNet(num_classes=args.num_classes)

    optimizer = torch.optim.AdamW(
        build_param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.999)
    )

    warmup, total = 5, args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 数据集接口
    dataset = ImageClassificationDataset(
        root_folder=args.data_path,
        dataset_name=args.dataset_name,
        num_classes=args.num_classes,
    )
    train_dataset = dataset.get_dataset(DatasetSplit.TRAIN)
    val_dataset = dataset.get_dataset(DatasetSplit.VAL)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
    )

    # 训练
    trainer = SupervisedTrainer(model, optimizer, scheduler, args)
    trainer.train(train_loader, val_loader)
