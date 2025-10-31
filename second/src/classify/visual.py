#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Name   : visual.py
Date Created: 2025/10/31
Description : 数据可视化
                1. 模型训练时的折线图，包括loss和acc
                2. 高维数据到二维数据的投影示意图
"""
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from second.src.utils.log import init_logger

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
        self._init_load()
        plt.style.use("seaborn-v0_8")
        logger.info("TrainingVisualizer 已初始化，可进行训练过程与特征分布可视化。")

    @staticmethod
    def _init_load():
        """修复可视化前一些环境变量或依赖问题"""
        # 设置 Matplotlib 使用支持中文的字体
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 黑体（Windows 默认支持）
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

        # 设置多进程环境变量
        os.environ["LOKY_MAX_CPU_COUNT"] = "8"

        # 忽略字体相关警告
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
        warnings.filterwarnings("ignore", message="Glyph .* missing from font")

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
        plt.figure(figsize=(10, 5))

        plt.plot(epochs, loss_history, label="训练损失", linewidth=2)
        if val_loss_history:
            plt.plot(epochs, val_loss_history, label="验证损失", linestyle="--")

        if acc_history:
            plt.plot(epochs, acc_history, label="训练准确率", linewidth=2)
        if val_acc_history:
            plt.plot(epochs, val_acc_history, label="验证准确率", linestyle="--")

        plt.xlabel("训练轮数 (Epoch)")
        plt.ylabel("数值 (Value)")
        plt.title("模型训练过程可视化")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()
        plt.savefig("training_curves.png")

        logger.info("训练与验证曲线绘制完成。")

    def visualize_feature_projection(
        self,
        feats: np.ndarray,
        labels: np.ndarray,
        method: str = "tsne",
        perplexity: float = 30.0,
        n_iter: int = 500,
    ) -> None:
        """
        使用 t-SNE 或 PCA 对高维特征降维到二维并可视化。

        参数:
            feats: (N, D) 特征矩阵
            labels: (N,) 对应类别标签
            method: "tsne" 或 "pca"
            perplexity: t-SNE 参数
            n_iter: t-SNE 迭代次数
        """
        assert feats.ndim == 2, "输入特征必须是二维的 [样本数, 特征维度]"
        assert len(feats) == len(labels), "特征与标签长度不匹配"

        logger.info(f"开始执行 {method.upper()} 降维可视化...")
        if method.lower() == "tsne":
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                n_iter=n_iter,
                verbose=1,
                init="pca",
                learning_rate="auto",
            )
            emb = tsne.fit_transform(feats)
            title = "t-SNE 降维后的特征分布"
        elif method.lower() == "pca":
            pca = PCA(n_components=2)
            emb = pca.fit_transform(feats)
            title = "PCA 降维后的特征分布"
        else:
            logger.error("method 参数错误，应为 'tsne' 或 'pca'")
            raise ValueError("method 必须为 'tsne' 或 'pca'")

        logger.info(f"降维完成，开始绘制 {title} ...")
        plt.figure(figsize=(8, 6))
        plt.scatter(emb[:, 0], emb[:, 1], c=labels, s=6, alpha=0.7, cmap="tab10")
        plt.title(title)
        plt.xlabel("维度 1")
        plt.ylabel("维度 2")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.show()
        plt.savefig("feature_projection.png")
        logger.info(f"{method.upper()} 特征分布可视化完成。")


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
