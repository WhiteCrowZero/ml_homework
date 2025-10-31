#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Name   : log_utils.py
Date Created: 2025/4/23
Description : 日志记录工具（控制台色彩、文件记录）
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler


class ColoredFormatter(logging.Formatter):
    """为控制台输出添加颜色"""

    COLOR_MAP = {
        logging.DEBUG: "37",  # 白
        logging.INFO: "32",  # 绿
        logging.WARNING: "33",  # 黄
        logging.ERROR: "31",  # 红
        logging.CRITICAL: "35",  # 紫
    }
    CSI = "\033["
    RESET = "\033[0m"

    def format(self, record):
        msg = super().format(record)
        color = self.COLOR_MAP.get(record.levelno)
        return f"{self.CSI}{color}m{msg}{self.RESET}" if color else msg


def setup_logger(
    level=logging.DEBUG,  # 日志级别
    console_fmt="%(asctime)s %(name)s %(levelname)s %(message)s",  # 控制台日志格式
    dateformat="%Y-%m-%d %H:%M:%S",  # 日志时间格式
    log_dir=None,  # 如果不需要文件日志，传 None
    log_filename="temp.log",  # 文件日志名称
    max_bytes=100 * 1024 * 1024,  # 文件日志最大 100MB
    backup_count=3,  # 文件日志最多保留 3 个备份
):
    """
    初始化日志记录器
    可以设置详细参数
    """
    root = logging.getLogger()
    root.setLevel(level)

    # 清空旧 handler
    while root.handlers:
        root.handlers.pop()

    # 1) 控制台彩色输出
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(ColoredFormatter(fmt=console_fmt, datefmt=dateformat))
    root.addHandler(ch)

    # 2) 可选：滚动文件输出（非彩色）
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, log_filename)
        fh = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
        fh.setLevel(level)
        # 文件里就不加色了，用普通 Formatter
        fh.setFormatter(logging.Formatter(fmt=console_fmt, datefmt=dateformat))
        root.addHandler(fh)


def init_logger(name, module_name, log_dir="./logs", level=logging.INFO):
    """
    初始化日志记录器（简化版）
    """
    setup_logger(level=level, log_dir=log_dir, log_filename=f"{name}_{module_name}.log")
    logger = logging.getLogger(module_name)
    return logger


# 使用示例
if __name__ == "__main__":
    setup_logger(level=logging.DEBUG, log_dir="./logs")
    logging.debug("This is a debug message")
