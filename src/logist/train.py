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


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mnist_data_processor = DataProcessor(
        r"../datasets/mnist"
    )
    # mnist_data_processor.download()
    data = mnist_data_processor.load(digits=(0, 1), test_size=0.5)
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
    acc = model.evaluate(X_test, y_test)
    model.logger.info(f"Test accuracy after loading: {acc:.4f}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    model = LogisticRegression()
    train(X_train, y_train, model)
    test(X_test, y_test, model)
