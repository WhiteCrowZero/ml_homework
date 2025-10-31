#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Name   : train.py
Author      : wzw
Date Created: 2025/9/28
Description : 训练文件，负责训练模型，mian入口
"""

import numpy as np
from typing import Optional
from data import DataProcessor
from model import LogisticRegression


def load_data(
    digits=(0, 1), test_size=0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mnist_data_processor = DataProcessor(r"../datasets/mnist")
    # # 下载数据集，第一次调用之后可以注释掉
    # mnist_data_processor.download()
    data = mnist_data_processor.load(digits=digits, test_size=test_size)
    if not data:
        raise ValueError("数据加载失败")
    X_train, X_test, y_train, y_test = data
    return X_train, X_test, y_train, y_test


def train(X_train, y_train, model: Optional[LogisticRegression] = None):
    if model is None:
        model = LogisticRegression()
    model.train(X_train, y_train)
    model.save_model()


def test(X_test, y_test, model: Optional[LogisticRegression] = None):
    if model is None:
        model = LogisticRegression()
        model.load_model()
    acc = model.evaluate(X_test, y_test, train_flag=False)
    model.logger.info(f"Test accuracy after loading: {acc:.4f}")


if __name__ == "__main__":
    # 一对一
    # 原始 0/1 二分类
    X_train, X_test, y_train, y_test = load_data(test_size=0.1)
    model = LogisticRegression(learning_rate=0.001, epochs=512)

    # # 一对一（other）
    # # 0/1 二分类映射 其他 单个二分类
    # X_train, X_test, y_train, y_test = load_data(digits=(2, 3), test_size=0.1)
    # model = LogisticRegression(
    #     label_map_dict={0: 2, 1: 3}, learning_rate=0.01, epochs=512
    # )

    # # 一对多
    # # 0/1 二分类映射 其他 组二分类
    # X_train, X_test, y_train, y_test = load_data(digits=(2, 3, 4, 5), test_size=0.1)
    # model = LogisticRegression(
    #     label_map_dict={0: [4, 5], 1: [2, 3]}, learning_rate=0.001, epochs=512
    # )

    # # 多对多
    # X_train, X_test, y_train, y_test = load_data(
    #     digits=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), test_size=0.1
    # )
    # model = LogisticRegression(
    #     label_map_dict={0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9]},
    #     learning_rate=0.01,
    #     epochs=2048,
    # )

    train(X_train, y_train, model)
    model.show_loss_pic("test")
    model.show_acc_pic("test")
    test(X_test, y_test, model)
