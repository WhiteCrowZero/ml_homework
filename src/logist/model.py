#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Name   : model.py
Author      : wzw
Date Created: 2025/9/28
Description : 对数几率回归模型文件，负责模型的加载、训练、保存
"""
import copy
import os
import warnings

import numpy as np
from typing import Optional, Literal, Any

from matplotlib import pyplot as plt

from src.utils.log import init_logger


class LogisticRegression:
    def __init__(
        self,
        label_map_dict: dict[Literal[0, 1], Any] = None,
        model_path: Optional[str] = "models",
        learning_rate: float = 0.01,
        epochs: int = 128,
        pic_save_path: Optional[str] = "pics",
    ):
        self._epoch_ls = []
        self._acc_ls = []
        self._loss_ls = []
        self._theta = None
        self._save_path = None
        # 构建正向映射（原始 -> 0/1）和反向映射（0/1 -> 原始）
        self._forward_map = {}
        self._reverse_map = {}
        self.epochs = epochs
        self.model_path = model_path
        self.learning_rate = learning_rate
        self._pic_save_path = pic_save_path
        self.label_map_dict = label_map_dict
        self.logger = init_logger(
            name=__name__, module_name=self.__class__.__name__, log_dir="logs"
        )
        self._ensure_path()
        self._load_label_map()

    def _load_label_map(self):
        """
        加载正向映射字典和反向映射字典
        """
        if self.label_map_dict is None:
            self.label_map_dict = {0: 0, 1: 1}
            self.logger.warning("未赋值映射字典，默认使用0/1映射")

        for std_label, raw_values in self.label_map_dict.items():
            if isinstance(raw_values, (list, tuple, set)):
                for v in raw_values:
                    self._forward_map[v] = std_label
            else:
                self._forward_map[raw_values] = std_label

            # # 默认取第一个作为反向映射的代表值
            # if std_label not in self.reverse_map:
            #     self.reverse_map[std_label] = (
            #         raw_values[0]
            #         if isinstance(raw_values, (list, tuple, set))
            #         else raw_values
            #     )

            # 不处理多分类的反向映射，代码更加方便，语义更合适
        self._reverse_map = copy.deepcopy(self.label_map_dict)

    def _label_forward(self, y: np.ndarray) -> np.ndarray:
        """
        将任意原始标签转换为 0/1
        """
        try:
            y = np.array(y).reshape(-1)
            return np.array(
                [self._forward_map[val] for val in y], dtype=np.int64
            ).reshape(-1, 1)
        except KeyError as e:
            raise ValueError(f"标签 {e.args[0]} 未在 label_map_dict 中定义") from None

    def _label_reverse(self, y_pred: np.ndarray) -> np.ndarray:
        """
        将 0/1 的预测结果映射回原始标签
        说明：如果是多分类的映射，不再映射回去

        目前暂时弃用，未来考虑改成对外开放函数，用于呈现测试输出结果，而非单纯 0，1
        """
        warnings.warn(
            "_label_reverse 已废弃，请不要在新代码中使用。未来可能提供新的接口用于呈现测试输出。",
            category=DeprecationWarning,
            stacklevel=2,  # 指向调用处
        )

        if len(self._forward_map) > 2:
            return y_pred
        return np.array([self._reverse_map[val] for val in y_pred.flatten()])

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """
        给特征矩阵添加偏置列
        如果 X 的列数已经比 theta 少 1，说明训练时没加 bias，需要加
        """
        if self._theta is None:
            raise ValueError("theta 未初始化，无法判断是否需要 bias")
        if X.shape[1] == self._theta.shape[0] - 1:
            # 需要加 bias
            return np.hstack([np.ones((X.shape[0], 1)), X])
        elif X.shape[1] == self._theta.shape[0]:
            # 已经加过 bias
            return X
        else:
            raise ValueError(
                f"输入 X 列数 ({X.shape[1]}) 与 theta 行数 ({self._theta.shape[0]}) 不匹配"
            )

    def _init_theta(self, shape: tuple) -> np.ndarray:
        """
        初始化参数
        """
        return np.random.randn(*shape) * 0.01

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        sigmoid 函数
        """
        z = np.asarray(z, dtype=np.float64)
        z = np.clip(z, -500, 500)  # 处理过大或者过小的数据溢出问题
        return 1 / (1 + np.exp(-z))

    def loss(self, X: np.ndarray, y: np.ndarray) -> np.floating:
        """
        计算损失
        """
        y_hat = self.sigmoid(X @ self._theta)
        eps = 1e-8
        l = -(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
        return np.mean(l)

    def gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        计算梯度
        """
        y_hat = self.sigmoid(X @ self._theta)
        return X.T @ (y_hat - y) / X.shape[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        推理，返回值为 0 或 1
        """
        try:
            if self._theta is None:
                raise ValueError("模型尚未训练或未加载参数 theta")

            X = np.asarray(X, dtype=np.float64)
            X = self._add_bias(X)
            y_hat = self.sigmoid(X @ self._theta)
            return (y_hat >= 0.5).astype(int)
        except Exception as e:
            self.logger.error(f"推理失败，错误信息：{e}")
            raise e

    def evaluate(
        self, X: np.ndarray, y: np.ndarray, train_flag: bool = True
    ) -> np.floating:
        """
        评估模型，返回准确率
        train_flag 表示是否处于训练，如果是训练，则标签保持 0/1，
        否则处于预测状态，映射到 0/1 标签，与模型预测结果对应，方便比较
        """
        try:
            y_original = y.reshape(-1, 1)
            y_pred = self.predict(X)
            if not train_flag:
                # 测试时，标签映射到 0 或 1
                y_std = self._label_forward(y_original)
            else:
                # 训练时，标签保持 0 或 1
                y_std = y_original
            return np.mean(y_std == y_pred)
        except Exception as e:
            self.logger.error(f"评估失败，错误信息：{e}")
            raise e

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        训练模型
        """
        try:
            # 处理 y
            y = y.reshape(-1, 1)
            y = self._label_forward(y)  # 自动转成 0/1

            # 处理 X
            self._theta = self._init_theta((X.shape[1] + 1, 1))
            X = self._add_bias(X)
        except Exception as e:
            self.logger.error(f"训练失败，错误信息：{e}")
            raise e

        self.logger.info("开始训练模型...")
        self.logger.info(f"训练集大小：{X.shape[0]}")
        self.logger.info(f"特征维度：{X.shape[1]}")

        for epoch in range(self.epochs):
            grad = self.gradient(X, y)
            self._theta -= self.learning_rate * grad

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                loss_val = self.loss(X, y)
                acc_val = self.evaluate(X, y)
                self._epoch_ls.append(epoch)
                self._loss_ls.append(loss_val)
                self._acc_ls.append(acc_val)
                self.logger.info(
                    f"epoch: {epoch + 1:3d}, loss: {loss_val:.4f}, acc: {acc_val:.4f}"
                )

        self.logger.info(f"最终损失（loss）: {self.loss(X, y):.4f}")
        self.logger.info(f"最终正确率（acc）: {self.evaluate(X, y):.4f}")

    def _ensure_path(self):
        """
        确保模 型路 和 图片 径存在
        """
        # 模型
        if self.model_path is None:
            raise ValueError("模型路径未指定")
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self._save_path = os.path.join(self.model_path, "theta.npy")

        # 图片
        if self._pic_save_path is not None:
            os.makedirs(self._pic_save_path, exist_ok=True)

    def save_model(self):
        """
        保存模型参数
        """
        if self._theta is None:
            raise ValueError("模型尚未训练，无法保存")
        np.save(self._save_path, self._theta)
        self.logger.info(f"模型参数已保存到 {self._save_path}")

    def load_model(self):
        """
        加载模型参数
        """
        self._theta = np.load(self._save_path)
        self.logger.info(f"模型参数已从 {self._save_path} 加载")

    def show_loss_pic(self, name: str = "") -> None:
        """
        显示损失函数曲线
        """
        if self._loss_ls is None:
            raise ValueError("模型尚未训练，无法显示损失函数曲线")
        plt.plot(self._epoch_ls, self._loss_ls)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Pic")
        if self._pic_save_path:
            plt.savefig(os.path.join(self._pic_save_path, f"loss_{name}.png"))
        plt.show()

    def show_acc_pic(self, name: str = "") -> None:
        """
        显示准确率曲线
        """
        if self._acc_ls is None:
            raise ValueError("模型尚未训练，无法显示准确率曲线")
        plt.plot(self._epoch_ls, self._acc_ls)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Pic")
        if self._pic_save_path:
            plt.savefig(os.path.join(self._pic_save_path, f"acc_{name}.png"))
        plt.show()


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

    # 展示图片
    model.show_loss_pic()
    model.show_acc_pic()

    # # 加载模型
    # new_model = LogisticRegression()
    # new_model.load_model()
    # new_acc = new_model.evaluate(X_test, y_test)
    # new_model.logger.info(f"Test accuracy after loading: {new_acc:.4f}")
