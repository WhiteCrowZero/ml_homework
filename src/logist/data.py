#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Name   : data.py
Author      : wzw
Date Created: 2025/9/28
Description : 数据处理文件，负责数据的下载、加载、数据集划分
"""

import os
import gzip
import struct
import requests
import numpy as np
from pathlib import Path
from src.utils.log import init_logger


class DataProcessor:
    base_url = "https://raw.githubusercontent.com/geektutu/tensorflow-tutorial-samples/refs/heads/master/mnist/data_set/{}"
    dataset_name_dict = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            raise ValueError("data_dir不能为空")

        self.data_dir = data_dir
        self.ensure_dir(self.data_dir)
        self.save_dir = Path(data_dir)
        self.logger = init_logger(
            name=__name__, module_name=self.__class__.__name__, log_dir="logs"
        )

        self.logger.info(f"DataProcessor 初始化完毕，data_dir: {data_dir}")

    @staticmethod
    def ensure_dir(dir_path: str) -> None:
        """确保目录存在"""
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def download(self, max_retries=3) -> None:
        """下载MNIST数据集"""

        self.logger.info("开始下载MNIST数据集...")
        for name in self.dataset_name_dict.values():
            save_path = self.save_dir / name
            # 带重试逻辑的下载，默认 3 次
            for i in range(max_retries):
                try:
                    resp = requests.get(self.base_url.format(name))
                    if resp.status_code == 200:
                        with open(save_path, "wb") as f:
                            f.write(resp.content)
                            self.logger.info(f"{name} 下载成功，保存至 {save_path}")
                            break
                    else:
                        self.logger.error(
                            f"{name} 下载失败, 状态码: {resp.status_code}"
                        )
                except Exception as e:
                    self.logger.error(f"第 {i+1} 次尝试， {name} 下载失败: {e}")
            else:
                self.logger.error(
                    f"{name} 下载失败超过最大重试次数 {max_retries}, 放弃下载."
                )
                continue

        self.logger.info("MNIST数据集下载完成")

    @staticmethod
    def load_images(filename: str) -> np.ndarray:
        """加载图片数据"""
        with gzip.open(filename, "rb") as f:
            """
            [0:4] 魔数 (magic number)，应为 2051
            [4:8] 样本数
            [8:12] 行数（28）
            [12:16] 列数（28）
            """
            # 读取头部信息
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise ValueError(f"Magic number mismatch: expected 2051, got {magic}")

            # 读取图像数据
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(num, rows, cols)
            return data

    @staticmethod
    def load_labels(filename: str) -> np.ndarray:
        """加载标签数据"""
        with gzip.open(filename, "rb") as f:
            """
            [0:4] 魔数 (magic number)，应为 2051
            [4:8] 样本数
            """
            magic, num = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise ValueError(f"Magic number mismatch: expected 2049, got {magic}")

            # 读取标签
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels

    def load(self, digits: tuple = (0, 1), test_size: float = 0.5) -> tuple:
        """
        加载 MNIST 数据集，并按需求筛选类别 & 划分比例
        digits : 要选择的数字类别，本实验为 (0, 1)
        test_size : 测试集占比，本实验为 0.5 / 0.3 / 0.1
        """
        # 校验参数
        try:
            digits_set = set(digits)
            valid_digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
            if not digits_set.issubset(valid_digits):
                raise ValueError("digits参数只能为0-9的数字")
            if test_size <= 0 or test_size >= 1:
                raise ValueError("test_size参数必须在0-1之间")
        except Exception as e:
            self.logger.error(f"参数错误: {e}")
            return ()

        # 加载原始数据
        X_train = self.load_images(
            str(self.save_dir / self.dataset_name_dict["train_images"])
        )
        y_train = self.load_labels(
            str(self.save_dir / self.dataset_name_dict["train_labels"])
        )
        X_test = self.load_images(
            str(self.save_dir / self.dataset_name_dict["test_images"])
        )
        y_test = self.load_labels(
            str(self.save_dir / self.dataset_name_dict["test_labels"])
        )

        # 拼接在一起
        X = np.vstack([X_train, X_test])
        y = np.hstack([y_train, y_test])

        # 筛选指定数字类别
        mask = np.isin(y, digits)
        X, y = X[mask], y[mask]

        # 归一化，然后展平降低维度方便预算
        X = X.astype(np.float32) / 255.0
        X = X.reshape(X.shape[0], -1)

        # 重新划分训练集和测试集
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        self.logger.info("加载 MNIST 数据完毕")

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    mnist_data_processor = DataProcessor(
        r"../datasets/mnist"
    )
    mnist_data_processor.download()
    data = mnist_data_processor.load(digits=(0, 1), test_size=0.5)
    if not data:
        pass
    else:
        X_train, X_test, y_train, y_test = data
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        print(X_train[0], y_train[0], X_test[0], y_test[0])
