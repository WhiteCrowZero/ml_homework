#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Name   : exceptions.py
Date Created: 2025/10/31
Description : 自定义异常
"""


class InvalidBackboneError(Exception):
    """无效的骨干网络错误"""

    pass


class InvalidDatasetSelection(Exception):
    """无效的数据集选择错误"""

    pass


class InvalidDatasetClassesNum(Exception):
    """无效的数据集选择错误"""

    pass
