"""随机种子固定工具。确保训练可复现，固定 random/numpy/torch/cudnn。"""
import random
import numpy as np
import torch


def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False