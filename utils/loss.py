import torch
import torch.nn.functional as F
import torch.nn as nn
import math


def mu_law_tonemap(x, mu=1000.0):
    """μ-law tone mapping: 压缩HDR动态范围"""
    x = torch.clamp(x, 0, 1)  # 归一化到[0,1]
    return torch.sign(x) * torch.log(1 + mu * torch.abs(x)) / math.log(1 + mu)

def safnet_loss(pred_hdr, gt_hdr):
    """SAFNet官方Loss实现"""
    # 1. tone mapping到感知均匀域
    pred_tm = mu_law_tonemap(pred_hdr)
    gt_tm = mu_law_tonemap(gt_hdr)
    
    # 2. L1 loss
    loss = F.l1_loss(pred_tm, gt_tm)
    return loss


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
