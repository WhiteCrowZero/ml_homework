#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Name   : model.py
Author      : wzw
Date Created: 2025/9/28
Description : 对数几率回归模型文件，负责模型的加载、训练、保存
"""

import os
import numpy as np
from typing import Optional
from src.utils.log import init_logger


class LogisticRegression:
    def __init__(
        self,
        model_path: Optional[str] = "models",
        learning_rate: float = 0.01,
        epochs: int = 128,
    ):
        self.save_path = None
        self.theta = None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model_path = model_path
        self.ensure_path()
        self.logger = init_logger(
            name=__name__, module_name=self.__class__.__name__, log_dir="logs"
        )

    def init_theta(self, shape: tuple) -> np.ndarray:
        """
        初始化参数
        """
        return np.random.randn(*shape) * 0.01

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        sigmoid 函数
        """
        return 1 / (1 + np.exp(-z))

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算损失
        """
        y_hat = self.sigmoid(X @ self.theta)
        eps = 1e-8
        l = -(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
        return np.mean(l)

    def gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        计算梯度
        """
        y_hat = self.sigmoid(X @ self.theta)
        return X.T @ (y_hat - y) / X.shape[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测，返回值为 0 或 1
        """
        y_hat = self.sigmoid(X @ self.theta)
        return (y_hat >= 0.5).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        评估模型，返回准确率
        """
        y = y.reshape(-1, 1)
        y_pred = self.predict(X)
        return np.mean(y == y_pred)

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        训练模型
        """
        y = y.reshape(-1, 1)
        self.theta = self.init_theta((X.shape[1], 1))

        for epoch in range(self.epochs):
            grad = self.gradient(X, y)
            self.theta -= self.learning_rate * grad

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                loss_val = self.loss(X, y)
                acc_val = self.evaluate(X, y)
                self.logger.info(
                    f"epoch: {epoch + 1:3d}, loss: {loss_val:.4f}, acc: {acc_val:.4f}"
                )

        self.logger.info(f"final loss: {self.loss(X, y):.4f}")
        self.logger.info(f"final acc: {self.evaluate(X, y):.4f}")

    def ensure_path(self):
        """
        确保模型路径存在
        """
        if self.model_path is None:
            raise ValueError("模型路径未指定")
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.save_path = os.path.join(self.model_path, "theta.npy")

    def save_model(self):
        """
        保存模型参数
        """
        if self.theta is None:
            raise ValueError("模型尚未训练，无法保存")
        np.save(self.save_path, self.theta)
        self.logger.info(f"模型参数已保存到 {self.save_path}")

    def load_model(self):
        """
        加载模型参数
        """
        self.theta = np.load(self.save_path)
        self.logger.info(f"模型参数已从 {self.save_path} 加载")


if __name__ == "__main__":
    """
    特别声明：
    下面的代码只测试矩阵维度是否正确，运算是否正常进行，和真实的loss值以及acc值无关
    """

    # 测试数据
    X = np.random.rand(100, 784)  # 100 个样本，每个 28x28 展平
    y = np.random.randint(0, 2, size=(100,))

    from sklearn.model_selection import train_test_split

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    # 初始化模型
    model = LogisticRegression(learning_rate=0.1, epochs=10)
    model.train(X_train, y_train)

    # 评估
    acc = model.evaluate(X_test, y_test)
    model.logger.info(f"Test accuracy: {acc:.4f}")

    # 保存模型
    model.save_model()

    # 加载模型
    new_model = LogisticRegression()
    new_model.load_model()
    new_acc = new_model.evaluate(X_test, y_test)
    new_model.logger.info(f"Test accuracy after loading: {new_acc:.4f}")
