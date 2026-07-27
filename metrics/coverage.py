"""基础覆盖率计算：脏污像素 / 有效像素。"""
import numpy as np


def compute_coverage(mask):

    return np.mean(mask > 0)