#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Name   : visual.py
Date Created: 2025/10/31
Description : 数据可视化
                1. 模型训练时的折线图，包括loss和acc
                2. 高维数据到二维数据的投影示意图
"""
import math
import os

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import List, Optional
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from src.utils.log import init_logger

# 初始化logger
logger = init_logger(__name__, module_name="DataVisual", log_dir="logs")


class TrainingVisualizer:
    """
    用于训练过程与高维特征的可视化分析。
    包含:
        1. Loss / Acc 曲线绘制
        2. t-SNE / PCA 特征分布可视化
    """

    def __init__(self) -> None:
        matplotlib.use("Agg")
        logger.info("TrainingVisualizer 已初始化，可进行训练过程与特征分布可视化。")

    def plot_training_curves(
        self,
        loss_history: List[float],
        acc_history: Optional[List[float]] = None,
        val_loss_history: Optional[List[float]] = None,
        val_acc_history: Optional[List[float]] = None,
    ) -> None:
        """绘制训练与验证的损失loss和准确率acc变化曲线"""

        logger.info("开始绘制训练过程曲线图...")
        epochs = np.arange(1, len(loss_history) + 1)

        # loss
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, loss_history, label="Train Loss", linewidth=2)
        if val_loss_history:
            plt.plot(epochs, val_loss_history, label="Val Loss", linestyle="--")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("training_loss.png")
        plt.close()

        plt.figure(figsize=(8, 5))
        if acc_history is not None:
            plt.plot(epochs, acc_history, label="Train Accuracy", linewidth=2)
            if val_acc_history is not None:
                plt.plot(epochs, val_acc_history, label="Val Accuracy", linestyle="--")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy (%)")
            plt.title("Training & Validation Accuracy")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig("training_accuracy.png")
            plt.close()
        else:
            logger.warning("未提供准确率数据，仅绘制损失曲线。")

        logger.info("训练过程曲线图已保存。")

    def visualize_feature_projection(
        self,
        feats: np.ndarray,
        labels: np.ndarray,
        method: str = "tsne",
        perplexity: float = 30.0,
        max_iter: int = 500,
    ) -> None:
        """
        使用 t-SNE 或 PCA 对高维特征降维到二维并可视化。

        参数:
            feats: (N, D) 特征矩阵
            labels: (N,) 对应类别标签
            method: "tsne" 或 "pca"
            perplexity: t-SNE 参数
            max_iter: t-SNE 迭代次数
        """
        assert feats.ndim == 2, "输入特征必须是二维的 [样本数, 特征维度]"
        assert len(feats) == len(labels), "特征与标签长度不匹配"

        logger.info(f"开始执行 {method.upper()} 降维可视化...")

        if method.lower() == "tsne":
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                max_iter=max_iter,
                verbose=1,
                init="pca",
                learning_rate="auto",
            )
            emb = tsne.fit_transform(feats)
            title = "t-SNE Feature Projection"
        elif method.lower() == "pca":
            pca = PCA(n_components=2)
            emb = pca.fit_transform(feats)
            title = "PCA Feature Projection"
        else:
            logger.error("method must be 'tsne' or 'pca'")
            raise ValueError("method must be 'tsne' or 'pca'")

        logger.info(f"降维完成，开始绘制 {title} ...")
        plt.figure(figsize=(8, 6))
        num_classes = len(np.unique(labels))
        if num_classes <= 10:
            cmap = plt.get_cmap("tab10")  # 极清晰 10 色
        elif num_classes <= 20:
            cmap = plt.get_cmap("tab20")  # 扩展版
        else:
            cmap = plt.get_cmap("hsv", num_classes)  # 多类动态分布颜色
        plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap=cmap, s=12, alpha=0.8)
        plt.colorbar()
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.savefig("feature_projection.png")

        logger.info(f"{method.upper()} 特征分布可视化完成。")

    def visualize_feature_maps(
        self,
        feature_maps: torch.Tensor,
        save_dir: str = "feature_maps",
        max_maps: int = 8,
        n_cols: int = 4,
        idx: int = 0,
    ):
        """
        可视化特征图，自动拼 grid
        feature_maps: (B, C, H, W)
        """
        os.makedirs(save_dir, exist_ok=True)

        feature_maps = feature_maps.detach().cpu()[0]  # 取 batch 第一张
        C = feature_maps.shape[0]
        num_maps = min(C, max_maps)

        # 计算行数
        n_rows = math.ceil(num_maps / n_cols)

        plt.figure(figsize=(n_cols * 3, n_rows * 3))

        for i in range(num_maps):
            fmap = feature_maps[i].numpy()

            # 子图
            ax = plt.subplot(n_rows, n_cols, i + 1)
            ax.imshow(fmap, cmap="gray")
            ax.axis("off")
            ax.set_title(f"Channel {i}")

        save_path = os.path.join(save_dir, f"feature_grid_{idx}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

        logger.info(f"特征图已保存: {save_path}")


if __name__ == "__main__":
    import numpy as np

    # 模拟训练过程记录
    train_losses = [1.2, 0.9, 0.65, 0.5, 0.35]
    val_losses = [1.1, 0.85, 0.7, 0.55, 0.4]
    train_acc = [45, 60, 72, 80, 87]
    val_acc = [43, 58, 70, 78, 85]

    # 模拟高维特征 (500样本, 128维)
    feats = np.random.randn(500, 128)
    labels = np.random.randint(0, 10, size=500)

    # 创建可视化实例
    visualizer = TrainingVisualizer()

    # 1. 绘制训练过程曲线
    visualizer.plot_training_curves(
        loss_history=train_losses,
        acc_history=train_acc,
        val_loss_history=val_losses,
        val_acc_history=val_acc,
    )

    # 2. 可视化高维特征分布
    visualizer.visualize_feature_projection(feats, labels, method="tsne", perplexity=30)
    visualizer.visualize_feature_projection(feats, labels, method="pca")
