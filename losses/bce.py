"""BCEWithLogitsLoss 封装。用于 tile 分类损失，内部自动处理 sigmoid。"""
import torch.nn as nn


class BCELoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.loss = nn.BCEWithLogitsLoss()


    def forward(self, pred, target):

        return self.loss(pred, target)