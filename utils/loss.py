import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torchvision.models import vgg16, VGG16_Weights


# ======================== Tonemapping ========================

def mu_law_tonemap(x, mu=5000.0):
    """μ-law tone mapping: 压缩 HDR 动态范围到感知均匀域"""
    return torch.log(1 + mu * x) / math.log(1 + mu)


# ======================== Base Losses ========================

class MSELoss(nn.Module):
    """L2 loss (gamma domain)"""
    def forward(self, pred, gt):
        return F.mse_loss(pred, gt)


class L1Loss(nn.Module):
    """L1 loss (gamma domain)"""
    def forward(self, pred, gt):
        return F.l1_loss(pred, gt)


class CharbonnierLoss(nn.Module):
    """Charbonnier loss: smooth L1, 在零点附近可导"""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt):
        diff = pred - gt
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


# ======================== Tonemapped Losses ========================

class MuLawL1Loss(nn.Module):
    """mu-law tonemapped L1 (SAFNet 标准 loss)"""
    def __init__(self, mu=5000.0):
        super().__init__()
        self.mu = mu

    def forward(self, pred, gt):
        pred_tm = mu_law_tonemap(pred.clamp(0, 1), self.mu)
        gt_tm = mu_law_tonemap(gt.clamp(0, 1), self.mu)
        return F.l1_loss(pred_tm, gt_tm)


class MuLawCharbonnierLoss(nn.Module):
    """mu-law tonemapped Charbonnier"""
    def __init__(self, mu=5000.0, eps=1e-3):
        super().__init__()
        self.mu = mu
        self.eps = eps

    def forward(self, pred, gt):
        pred_tm = mu_law_tonemap(pred.clamp(0, 1), self.mu)
        gt_tm = mu_law_tonemap(gt.clamp(0, 1), self.mu)
        diff = pred_tm - gt_tm
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


# ======================== Perceptual Loss ========================

class VGGPerceptualLoss(nn.Module):
    """VGG16 perceptual loss (conv3_3 features)"""
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        # 截取到 conv3_3 (index 15): 3 blocks, 浅层纹理特征
        self.features = nn.Sequential(*list(vgg.features[:16]))
        for p in self.features.parameters():
            p.requires_grad = False
        # ImageNet 归一化
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, gt):
        pred_n = (pred - self.mean) / self.std
        gt_n = (gt - self.mean) / self.std
        return F.l1_loss(self.features(pred_n), self.features(gt_n))


class MuLawL1PerceptualLoss(nn.Module):
    """mu-law L1 + perceptual (SAFNet 完整 loss)
    L = L1(T(pred), T(gt)) + w_perceptual * VGG(T(pred), T(gt))
    """
    def __init__(self, mu=5000.0, w_perceptual=0.01):
        super().__init__()
        self.mu = mu
        self.w_perceptual = w_perceptual
        self.vgg = VGGPerceptualLoss()

    def forward(self, pred, gt):
        pred_tm = mu_law_tonemap(pred.clamp(0, 1), self.mu)
        gt_tm = mu_law_tonemap(gt.clamp(0, 1), self.mu)
        l1 = F.l1_loss(pred_tm, gt_tm)
        perceptual = self.vgg(pred_tm, gt_tm)
        return l1 + self.w_perceptual * perceptual


# ======================== FFT Loss ========================

class FFTLoss(nn.Module):
    """频域 L1 loss: 强调高频细节保留"""
    def forward(self, pred, gt):
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        gt_fft = torch.fft.rfft2(gt, norm='ortho')
        return F.l1_loss(torch.abs(pred_fft), torch.abs(gt_fft))


class MuLawL1FFTLoss(nn.Module):
    """mu-law L1 + FFT loss"""
    def __init__(self, mu=5000.0, w_fft=0.1):
        super().__init__()
        self.mu = mu
        self.w_fft = w_fft
        self.fft_loss = FFTLoss()

    def forward(self, pred, gt):
        pred_tm = mu_law_tonemap(pred.clamp(0, 1), self.mu)
        gt_tm = mu_law_tonemap(gt.clamp(0, 1), self.mu)
        l1 = F.l1_loss(pred_tm, gt_tm)
        fft = self.fft_loss(pred_tm, gt_tm)
        return l1 + self.w_fft * fft


# ======================== Loss Registry ========================

LOSS_NAMES = [
    'mse', 'l1', 'charbonnier',
    'mulaw_l1', 'mulaw_charb',
    'mulaw_l1_perceptual', 'mulaw_l1_fft',
]


def build_loss(name):
    """根据名称构建 loss function (nn.Module)"""
    registry = {
        'mse': MSELoss,
        'l1': L1Loss,
        'charbonnier': CharbonnierLoss,
        'mulaw_l1': MuLawL1Loss,
        'mulaw_charb': MuLawCharbonnierLoss,
        'mulaw_l1_perceptual': MuLawL1PerceptualLoss,
        'mulaw_l1_fft': MuLawL1FFTLoss,
    }
    if name not in registry:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(registry.keys())}")
    return registry[name]()
