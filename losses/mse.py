"""MSELoss 封装。用于 coverage 回归损失。"""
import torch.nn as nn


class MSELoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.loss = nn.MSELoss()


    def forward(self, pred, target):

        return self.loss(pred, target)