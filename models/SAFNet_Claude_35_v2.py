"""
SAFNet_Claude_35_v2 — Global SE(2) Motion + Refine-Heavy Reconstruction
========================================================================
Built on the Claude_35 / Claude_51 design line with one major allocation change:
1. Replace dense two-stage optical flow with global SE(2) motion prediction.
2. Expand the reconstruction-heavy path using a Claude_51-style Star RefineNet.
3. Keep the original Claude_35 learned merge, plus a small high-res tail knob.

Target (fused): < 5M params, 95G < FLOPs < 100G for input (1, 36, 384, 768),
which corresponds to full-size RGB output (1, 3, 768, 1536).
"""
import argparse
import copy
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import DeformConv2d as _TorchvisionDeformConv2d
except Exception as e:
    _TorchvisionDeformConv2d = None
    _DEFORM_IMPORT_ERROR = e
else:
    _DEFORM_IMPORT_ERROR = None


if _TorchvisionDeformConv2d is not None:
    DeformConv2d = _TorchvisionDeformConv2d
else:
    class DeformConv2d(nn.Module):
        """Compatibility fallback for environments without torchvision deform op."""
        def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                     padding=0, dilation=1, bias=True):
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, bias=bias
            )
            self._warned = False

        def forward(self, x, offset):
            if not self._warned:
                warnings.warn(
                    "torchvision.ops.DeformConv2d is unavailable; "
                    "falling back to nn.Conv2d for compatibility.",
                    RuntimeWarning,
                )
                if _DEFORM_IMPORT_ERROR is not None:
                    warnings.warn(
                        f"Original import error: {_DEFORM_IMPORT_ERROR}",
                        RuntimeWarning,
                    )
                self._warned = True
            return self.conv(x)


div_size = 16
div_flow = 20.0


def warp(img, flow):
    b, _, h, w = flow.shape
    xx = torch.linspace(-1.0, 1.0, w, device=img.device, dtype=img.dtype)
    yy = torch.linspace(-1.0, 1.0, h, device=img.device, dtype=img.dtype)
    xx = xx.view(1, 1, 1, w).expand(b, -1, h, -1)
    yy = yy.view(1, 1, h, 1).expand(b, -1, -1, w)
    grid = torch.cat([xx, yy], 1)
    flow_ = torch.cat([
        flow[:, 0:1, :, :] / ((w - 1.0) / 2.0),
        flow[:, 1:2, :, :] / ((h - 1.0) / 2.0),
    ], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    return F.grid_sample(
        input=img,
        grid=grid_,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def resize(x, scale_factor):
    return F.interpolate(
        x,
        scale_factor=scale_factor,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=True,
    )


def resize_flow(flow, size_hw):
    """Resize flow to target spatial size with proper magnitude scaling."""
    src_h, src_w = flow.shape[-2:]
    dst_h, dst_w = int(size_hw[0]), int(size_hw[1])
    if src_h == dst_h and src_w == dst_w:
        return flow
    out = F.interpolate(flow, size=(dst_h, dst_w), mode="bilinear", align_corners=False)
    out[:, 0, :, :] *= float(dst_w) / float(src_w)
    out[:, 1, :, :] *= float(dst_h) / float(src_h)
    return out


def convrelu(in_channels, out_channels, kernel_size=3, stride=1,
             padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias=bias
        ),
        nn.PReLU(out_channels),
    )


class DeformConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, bias=True):
        super().__init__()
        offset_channels = 2 * kernel_size * kernel_size
        self.conv_offset = nn.Conv2d(
            in_channels, offset_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=True
        )
        self.deform = DeformConv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias
        )
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        offset = self.conv_offset(x)
        return self.prelu(self.deform(x, offset))


class CondConv2d_1x1(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts=12):
        super().__init__()
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(num_experts, out_channels, in_channels))
        self.bias = nn.Parameter(torch.Tensor(num_experts, out_channels))
        self.routing = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, num_experts, 1),
            nn.Softmax(dim=1),
        )
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        routing_weight = self.routing(x).view(b, self.num_experts)
        weight = torch.matmul(routing_weight, self.weight.view(self.num_experts, -1))
        weight = weight.view(b, self.out_channels, self.in_channels)
        bias = torch.matmul(routing_weight, self.bias)
        out = torch.bmm(weight, x.view(b, c, h * w)) + bias.view(b, self.out_channels, 1)
        return out.view(b, self.out_channels, h, w)


class RepDWConvS(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.channels = channels
        self.dilation = dilation
        self.fused = False
        padding = dilation
        kw = dict(in_channels=channels, out_channels=channels, groups=channels)

        self.conv_3x3 = nn.Conv2d(kernel_size=3, padding=padding, dilation=dilation, bias=True, **kw)
        self.conv_3h = nn.Conv2d(kernel_size=(3, 1), padding=(padding, 0), dilation=(dilation, 1), bias=False, **kw)
        self.conv_3w = nn.Conv2d(kernel_size=(1, 3), padding=(0, padding), dilation=(1, dilation), bias=False, **kw)
        if dilation == 1:
            self.conv_2x2 = nn.Conv2d(kernel_size=2, padding=1, dilation=2, bias=False, **kw)
        else:
            self.conv_1x1 = nn.Conv2d(kernel_size=1, padding=0, bias=False, **kw)

    def forward(self, x):
        if self.fused:
            return self.conv_3x3(x)
        out = self.conv_3x3(x) + self.conv_3h(x) + self.conv_3w(x)
        if self.dilation == 1:
            out = out + self.conv_2x2(x)
        else:
            out = out + self.conv_1x1(x)
        return out

    @torch.no_grad()
    def fuse(self):
        if self.fused:
            return
        weight = self.conv_3x3.weight.data.clone()
        bias = self.conv_3x3.bias.data.clone()

        weight += F.pad(self.conv_3h.weight.data, [1, 1, 0, 0])
        weight += F.pad(self.conv_3w.weight.data, [0, 0, 1, 1])

        if self.dilation == 1:
            weight_2x2 = F.conv_transpose2d(
                self.conv_2x2.weight.data,
                torch.ones(1, 1, 1, 1, device=weight.device),
                stride=2,
            )
            weight += weight_2x2
            del self.conv_2x2
        else:
            weight += F.pad(self.conv_1x1.weight.data, [1, 1, 1, 1])
            del self.conv_1x1

        self.conv_3x3.weight.data.copy_(weight)
        self.conv_3x3.bias.data.copy_(bias)
        del self.conv_3h, self.conv_3w
        self.fused = True


class RepDWConvM(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.channels = channels
        self.dilation = dilation
        self.fused = False
        padding2 = 2 * dilation
        padding1 = dilation
        kw = dict(in_channels=channels, out_channels=channels, groups=channels)

        self.conv_5x5 = nn.Conv2d(kernel_size=5, padding=padding2, dilation=dilation, bias=True, **kw)
        self.conv_5x3 = nn.Conv2d(kernel_size=(5, 3), padding=(padding2, padding1), dilation=dilation, bias=True, **kw)
        self.conv_3x5 = nn.Conv2d(kernel_size=(3, 5), padding=(padding1, padding2), dilation=dilation, bias=True, **kw)
        self.conv_5w = nn.Conv2d(kernel_size=(1, 5), padding=(0, padding2), dilation=(1, dilation), bias=False, **kw)
        self.conv_5h = nn.Conv2d(kernel_size=(5, 1), padding=(padding2, 0), dilation=(dilation, 1), bias=False, **kw)

    def forward(self, x):
        if self.fused:
            return self.conv_5x5(x)
        return (
            self.conv_5x5(x) + self.conv_5x3(x) + self.conv_3x5(x) +
            self.conv_5h(self.conv_5w(x)) + x
        )

    @torch.no_grad()
    def fuse(self):
        if self.fused:
            return
        weight = self.conv_5x5.weight.data.clone()
        bias = self.conv_5x5.bias.data.clone()

        weight += F.pad(self.conv_5x3.weight.data, [1, 1, 0, 0])
        bias += self.conv_5x3.bias.data

        weight += F.pad(self.conv_3x5.weight.data, [0, 0, 1, 1])
        bias += self.conv_3x5.bias.data

        weight_serial = torch.einsum("bcnx,bcyn->bcyx", self.conv_5w.weight.data, self.conv_5h.weight.data)
        weight += weight_serial
        weight[:, 0, 2, 2] += 1.0

        self.conv_5x5.weight.data.copy_(weight)
        self.conv_5x5.bias.data.copy_(bias)
        del self.conv_5x3, self.conv_3x5, self.conv_5w, self.conv_5h
        self.fused = True


class RepDWConvL(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.channels = channels
        self.dilation = dilation
        self.fused = False
        padding3 = 3 * dilation
        padding1 = dilation
        kw = dict(in_channels=channels, out_channels=channels, groups=channels)

        self.conv_7x7 = nn.Conv2d(kernel_size=7, padding=padding3, dilation=dilation, bias=True, **kw)
        self.conv_7x3 = nn.Conv2d(kernel_size=(7, 3), padding=(padding3, padding1), dilation=dilation, bias=True, **kw)
        self.conv_3x7 = nn.Conv2d(kernel_size=(3, 7), padding=(padding1, padding3), dilation=dilation, bias=True, **kw)
        self.conv_7w = nn.Conv2d(kernel_size=(1, 7), padding=(0, padding3), dilation=(1, dilation), bias=False, **kw)
        self.conv_7h = nn.Conv2d(kernel_size=(7, 1), padding=(padding3, 0), dilation=(dilation, 1), bias=False, **kw)

    def forward(self, x):
        if self.fused:
            return self.conv_7x7(x)
        return (
            self.conv_7x7(x) + self.conv_7x3(x) + self.conv_3x7(x) +
            self.conv_7h(self.conv_7w(x)) + x
        )

    @torch.no_grad()
    def fuse(self):
        if self.fused:
            return
        weight = self.conv_7x7.weight.data.clone()
        bias = self.conv_7x7.bias.data.clone()

        weight += F.pad(self.conv_7x3.weight.data, [2, 2, 0, 0])
        bias += self.conv_7x3.bias.data

        weight += F.pad(self.conv_3x7.weight.data, [0, 0, 2, 2])
        bias += self.conv_3x7.bias.data

        weight_serial = torch.einsum("bcnx,bcyn->bcyx", self.conv_7w.weight.data, self.conv_7h.weight.data)
        weight += weight_serial
        weight[:, 0, 3, 3] += 1.0

        self.conv_7x7.weight.data.copy_(weight)
        self.conv_7x7.bias.data.copy_(bias)
        del self.conv_7x3, self.conv_3x7, self.conv_7w, self.conv_7h
        self.fused = True


class ChunkConvV4(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        assert channels % 4 == 0
        group_channels = channels // 4
        self.group_channels = group_channels
        self.rep3 = RepDWConvS(group_channels, dilation=dilation)
        self.rep5 = RepDWConvM(group_channels, dilation=dilation)
        self.rep7 = RepDWConvL(group_channels, dilation=dilation)

    def forward(self, x):
        g = self.group_channels
        x0, x1, x2, x3 = x[:, :g], x[:, g:2 * g], x[:, 2 * g:3 * g], x[:, 3 * g:]
        return torch.cat([x0, self.rep3(x1), self.rep5(x2), self.rep7(x3)], dim=1)


class ChunkConvV3NoL(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        assert channels % 4 == 0
        group_channels = channels // 4
        self.group_channels = group_channels
        self.rep3 = RepDWConvS(group_channels, dilation=dilation)
        self.rep5a = RepDWConvM(group_channels, dilation=dilation)
        self.rep5b = RepDWConvM(group_channels, dilation=dilation)

    def forward(self, x):
        g = self.group_channels
        x0, x1, x2, x3 = x[:, :g], x[:, g:2 * g], x[:, 2 * g:3 * g], x[:, 3 * g:]
        return torch.cat([x0, self.rep3(x1), self.rep5a(x2), self.rep5b(x3)], dim=1)


class RepNeXtBlock(nn.Module):
    def __init__(self, channels, dilation=1, se_reduction=4, expand_ratio=2,
                 cond=False, num_experts=12):
        super().__init__()
        expand_ch = int(np.ceil((channels * float(expand_ratio)) / 4.0) * 4)
        se_ch = max(expand_ch // se_reduction, 8)
        self.norm = nn.GroupNorm(1, channels)
        if cond:
            self.pw1 = CondConv2d_1x1(channels, expand_ch, num_experts)
            self.pw2 = CondConv2d_1x1(expand_ch, channels, num_experts)
        else:
            self.pw1 = nn.Conv2d(channels, expand_ch, 1, bias=True)
            self.pw2 = nn.Conv2d(expand_ch, channels, 1, bias=True)
        self.dw = ChunkConvV4(expand_ch, dilation=dilation)
        self.act = nn.GELU()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expand_ch, se_ch, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(se_ch, expand_ch, 1, bias=True),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)

    def forward(self, x):
        y = self.norm(x)
        y = self.pw1(y)
        y = self.dw(y)
        y = self.act(y)
        y = y * self.se(y)
        y = self.pw2(y)
        return x + y * self.scale


class RepNeXtBlockNoL(nn.Module):
    def __init__(self, channels, dilation=1, se_reduction=4, expand_ratio=2,
                 cond=False, num_experts=12):
        super().__init__()
        expand_ch = int(np.ceil((channels * float(expand_ratio)) / 4.0) * 4)
        se_ch = max(expand_ch // se_reduction, 8)
        self.norm = nn.GroupNorm(1, channels)
        if cond:
            self.pw1 = CondConv2d_1x1(channels, expand_ch, num_experts)
            self.pw2 = CondConv2d_1x1(expand_ch, channels, num_experts)
        else:
            self.pw1 = nn.Conv2d(channels, expand_ch, 1, bias=True)
            self.pw2 = nn.Conv2d(expand_ch, channels, 1, bias=True)
        self.dw = ChunkConvV3NoL(expand_ch, dilation=dilation)
        self.act = nn.GELU()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expand_ch, se_ch, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(se_ch, expand_ch, 1, bias=True),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)

    def forward(self, x):
        y = self.norm(x)
        y = self.pw1(y)
        y = self.dw(y)
        y = self.act(y)
        y = y * self.se(y)
        y = self.pw2(y)
        return x + y * self.scale


class TinyMotionEncoder(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.stage1 = nn.Sequential(
            convrelu(in_channels, 16, 3, 2, 1),
            RepNeXtBlockNoL(16),
        )
        self.stage2 = nn.Sequential(
            convrelu(16, 24, 3, 2, 1),
            RepNeXtBlockNoL(24),
        )
        self.stage3 = nn.Sequential(
            convrelu(24, 32, 3, 2, 1),
            RepNeXtBlockNoL(32),
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x


class GlobalSE2Head(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=True)
        self.dw_act = nn.PReLU(in_channels)
        self.pw = nn.Conv2d(in_channels, 32, 1, 1, 0, bias=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(32, 16)
        self.fc1_act = nn.PReLU(16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, feat_src, feat_ref, image_hw):
        feat = torch.cat([feat_src, feat_ref, feat_ref - feat_src, feat_ref * feat_src], dim=1)
        feat = self.dw_act(self.dw(feat))
        feat = self.pw(feat)
        feat = self.pool(feat).flatten(1)
        feat = self.fc1_act(self.fc1(feat))
        raw = self.fc2(feat)

        height, width = int(image_hw[0]), int(image_hw[1])
        tx = 0.25 * float(width) * torch.tanh(raw[:, 0])
        ty = 0.25 * float(height) * torch.tanh(raw[:, 1])
        theta = (math.pi / 12.0) * torch.tanh(raw[:, 2])
        mask_bias = raw[:, 3]
        return tx, ty, theta, mask_bias


def se2_to_flow(tx, ty, theta, size_hw, device, dtype):
    height, width = int(size_hw[0]), int(size_hw[1])
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)
    cx = (width - 1.0) * 0.5
    cy = (height - 1.0) * 0.5
    x_centered = xx - cx
    y_centered = yy - cy

    cos_theta = torch.cos(theta).view(-1, 1, 1)
    sin_theta = torch.sin(theta).view(-1, 1, 1)
    tx = tx.view(-1, 1, 1)
    ty = ty.view(-1, 1, 1)

    x_warp = cos_theta * x_centered - sin_theta * y_centered + cx + tx
    y_warp = sin_theta * x_centered + cos_theta * y_centered + cy + ty

    flow_x = x_warp - xx
    flow_y = y_warp - yy
    return torch.stack([flow_x, flow_y], dim=1)


class RigidMaskHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(9, 16, 3, 1, 1, bias=True),
            nn.PReLU(16),
            nn.Conv2d(16, 16, 3, 1, 1, bias=True),
            nn.PReLU(16),
            nn.Conv2d(16, 1, 3, 1, 1, bias=True),
        )

    def forward(self, src_warp_rgb, ref_rgb, mask_bias):
        feat = torch.cat([src_warp_rgb, ref_rgb, torch.abs(src_warp_rgb - ref_rgb)], dim=1)
        return self.net(feat) + mask_bias.view(-1, 1, 1, 1)


class LearnedMerge3Frame(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_net = nn.Sequential(
            nn.Conv2d(11, 48, 3, 1, 1),
            RepNeXtBlock(48),
            RepNeXtBlock(48),
        )
        self.attn_head = nn.Conv2d(48, 3, 1, 1, 0)

    def forward(self, img0_w, img4, img8_w, mask0, mask8):
        feat = self.feat_net(torch.cat([img0_w, img4, img8_w, mask0, mask8], dim=1))
        weights = torch.softmax(self.attn_head(feat), dim=1)
        return (
            weights[:, 0:1] * img0_w +
            weights[:, 1:2] * img4 +
            weights[:, 2:3] * img8_w
        )


class StarRefineBlock(nn.Module):
    def __init__(self, channels, dilation=1, expand_ratio=5.5, cond=True, num_experts=10):
        super().__init__()
        expand_ch = int(np.ceil((channels * float(expand_ratio)) / 8.0) * 8)
        self.norm = nn.GroupNorm(1, channels)
        if cond:
            self.pw1 = CondConv2d_1x1(channels, expand_ch, num_experts)
            self.pw2 = CondConv2d_1x1(expand_ch // 2, channels, num_experts)
        else:
            self.pw1 = nn.Conv2d(channels, expand_ch, 1, bias=True)
            self.pw2 = nn.Conv2d(expand_ch // 2, channels, 1, bias=True)
        self.dw = ChunkConvV3NoL(expand_ch // 2, dilation=dilation)
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)

    def forward(self, x):
        identity = x
        y = self.norm(x)
        y = self.pw1(y)
        y1, y2 = y.chunk(2, dim=1)
        y1 = self.dw(y1)
        y = y1 * y2
        y = self.pw2(y)
        return identity + y * self.scale


class RefineNet(nn.Module):
    def __init__(self, img_channels=4):
        super().__init__()
        c0, c1, c2 = 20, 44, 20
        total_c = c0 + c1 + c2

        self.conv0 = nn.Sequential(
            convrelu(img_channels, c0),
            StarRefineBlock(c0, expand_ratio=3.4, cond=False),
        )
        self.conv1 = nn.Sequential(
            DeformConvRelu(img_channels + 2 + 2 + 1 + 1 + 3, c1),
            StarRefineBlock(c1, expand_ratio=3.4, cond=False),
        )
        self.conv2 = nn.Sequential(
            convrelu(img_channels, c2),
            StarRefineBlock(c2, expand_ratio=3.4, cond=False),
        )

        self.blocks = nn.Sequential(
            StarRefineBlock(total_c, dilation=1, cond=True, num_experts=10, expand_ratio=4.0),
            StarRefineBlock(total_c, dilation=2, cond=True, num_experts=10, expand_ratio=4.0),
            StarRefineBlock(total_c, dilation=4, cond=True, num_experts=10, expand_ratio=4.0),
            StarRefineBlock(total_c, dilation=4, cond=True, num_experts=10, expand_ratio=4.0),
            StarRefineBlock(total_c, dilation=2, cond=True, num_experts=10, expand_ratio=4.0),
            StarRefineBlock(total_c, dilation=1, cond=True, num_experts=10, expand_ratio=4.0),
        )

        self.conv3 = nn.Conv2d(total_c, 12, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, img0_c, img4_c, img8_c, flow0, flow8, mask0, mask8, img_hdr_m):
        feat0 = self.conv0(img0_c)
        feat1 = self.conv1(torch.cat([
            img4_c, flow0 / div_flow, flow8 / div_flow, mask0, mask8, img_hdr_m,
        ], 1))
        feat2 = self.conv2(img8_c)

        feat = torch.cat([warp(feat0, flow0), feat1, warp(feat2, flow8)], 1)
        feat = self.blocks(feat)
        res = self.pixel_shuffle(self.conv3(feat))
        img_hdr_m_up = F.interpolate(img_hdr_m, scale_factor=2, mode="bilinear", align_corners=False)
        return torch.clamp(img_hdr_m_up + res, 0, 1)



class SAFNet_Claude_35_v2(nn.Module):
    def __init__(self, tail_hidden_channels=12):
        super().__init__()
        self.tail_hidden_channels = tail_hidden_channels
        self.motion_encoder = TinyMotionEncoder()
        self.motion_head = GlobalSE2Head()
        self.mask_head = RigidMaskHead()
        self.learned_merge = LearnedMerge3Frame()
        self.refinenet = RefineNet()
        self._aux_losses = {}

    def fuse_reparam(self):
        for module in self.modules():
            if isinstance(module, (RepDWConvS, RepDWConvM, RepDWConvL)):
                module.fuse()

    def _predict_pair(self, img_src, img_ref, input_size):
        feat_src = self.motion_encoder(img_src)
        feat_ref = self.motion_encoder(img_ref)
        tx, ty, theta, mask_bias = self.motion_head(feat_src, feat_ref, input_size)

        flow = se2_to_flow(
            tx, ty, theta, input_size,
            device=img_ref.device,
            dtype=img_ref.dtype,
        )
        src_warp = warp(img_src, flow)
        mask_logits = self.mask_head(src_warp[:, :3], img_ref[:, :3], mask_bias)
        return flow, mask_logits

    def forward_flow_mask(self, img0_c, img4_c, img8_c, scale_factor=0.5):
        h, w = img4_c.shape[-2:]
        org_size = (int(h), int(w))
        input_size = (
            int(div_size * math.ceil(float(h) * scale_factor / div_size)),
            int(div_size * math.ceil(float(w) * scale_factor / div_size)),
        )

        if input_size != org_size:
            img0_c = F.interpolate(img0_c, size=input_size, mode="bilinear", align_corners=False)
            img4_c = F.interpolate(img4_c, size=input_size, mode="bilinear", align_corners=False)
            img8_c = F.interpolate(img8_c, size=input_size, mode="bilinear", align_corners=False)

        flow0, mask0_logits = self._predict_pair(img0_c, img4_c, input_size)
        flow8, mask8_logits = self._predict_pair(img8_c, img4_c, input_size)

        if input_size != org_size:
            flow0 = resize_flow(flow0, org_size)
            flow8 = resize_flow(flow8, org_size)
            mask0_logits = F.interpolate(mask0_logits, size=org_size, mode="bilinear", align_corners=False)
            mask8_logits = F.interpolate(mask8_logits, size=org_size, mode="bilinear", align_corners=False)

        return torch.sigmoid(mask0_logits), torch.sigmoid(mask8_logits), flow0, flow8

    def forward(self, x, scale_factor=0.5, refine=True):
        img0_c = x[:, 0:4, :, :]
        img4_c = x[:, 16:20, :, :]
        img8_c = x[:, 32:36, :, :]

        mask0, mask8, flow0, flow8 = self.forward_flow_mask(img0_c, img4_c, img8_c, scale_factor=scale_factor)

        img0_warp = warp(img0_c, flow0)
        img8_warp = warp(img8_c, flow8)
        img_hdr_m = self.learned_merge(
            img0_warp[:, :3], img4_c[:, :3], img8_warp[:, :3], mask0, mask8
        )

        if refine:
            return self.refinenet(img0_c, img4_c, img8_c, flow0, flow8, mask0, mask8, img_hdr_m)
        return F.interpolate(img_hdr_m, scale_factor=2, mode="bilinear", align_corners=False)


def _format_count(num):
    if num >= 1e12:
        return f"{num / 1e12:.3f}T"
    if num >= 1e9:
        return f"{num / 1e9:.3f}G"
    if num >= 1e6:
        return f"{num / 1e6:.3f}M"
    if num >= 1e3:
        return f"{num / 1e3:.3f}K"
    return str(num)


def _print_profile(height=384, width=768, params_min_m=0.0, params_cap_m=5.0,
                   flops_min_g=95.0, flops_cap_g=100.0, tail_hidden_channels=12):
    device = torch.device("cpu")
    dummy = torch.ones(1, 36, height, width, device=device)

    model_before = SAFNet_Claude_35_v2(tail_hidden_channels=tail_hidden_channels).to(device).eval()
    model_after = copy.deepcopy(model_before).to(device).eval()
    model_after.fuse_reparam()

    params_before = sum(p.numel() for p in model_before.parameters())
    params_after = sum(p.numel() for p in model_after.parameters())

    print(f"Input shape: (1, 36, {height}, {width})")
    print(f"Output shape: (1, 3, {height * 2}, {width * 2})")
    print(f"Tail hidden channels : {tail_hidden_channels}")
    print(f"Params before fusion: {params_before:,} ({_format_count(params_before)})")
    print(f"Params after fusion : {params_after:,} ({_format_count(params_after)})")
    print(f"Param delta         : {params_after - params_before:,}")
    params_min_ok = params_after >= int(params_min_m * 1e6)
    params_cap_ok = params_after < int(params_cap_m * 1e6)
    print(f"Params min ({params_min_m:.2f}M): {'PASS' if params_min_ok else 'FAIL'}")
    print(f"Params cap ({params_cap_m:.2f}M): {'PASS' if params_cap_ok else 'FAIL'}")

    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception as e:
        print(f"FLOPs skipped (fvcore unavailable): {e}")
        return

    flops_before = FlopCountAnalysis(model_before, dummy).total()
    flops_after = FlopCountAnalysis(model_after, dummy).total()
    print(f"FLOPs before fusion : {flops_before:.0f} ({_format_count(flops_before)})")
    print(f"FLOPs after fusion  : {flops_after:.0f} ({_format_count(flops_after)})")
    print(f"FLOPs delta         : {flops_after - flops_before:.0f}")
    flops_min_ok = flops_after > float(flops_min_g) * 1e9
    flops_cap_ok = flops_after < float(flops_cap_g) * 1e9
    print(f"FLOPs min ({flops_min_g:.2f}G): {'PASS' if flops_min_ok else 'FAIL'}")
    print(f"FLOPs cap ({flops_cap_g:.2f}G): {'PASS' if flops_cap_ok else 'FAIL'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=384, help="Input height for profiling.")
    parser.add_argument("--width", type=int, default=768, help="Input width for profiling.")
    parser.add_argument("--params_min_m", type=float, default=0.0, help="Param minimum in millions.")
    parser.add_argument("--params_cap_m", type=float, default=5.0, help="Param cap in millions.")
    parser.add_argument("--flops_min_g", type=float, default=95.0, help="FLOPs minimum in billions.")
    parser.add_argument("--flops_cap_g", type=float, default=100.0, help="FLOPs cap in billions.")
    parser.add_argument("--tail_hidden_channels", type=int, default=12, help="High-res tail width knob.")
    args = parser.parse_args()
    _print_profile(
        height=args.height,
        width=args.width,
        params_min_m=args.params_min_m,
        params_cap_m=args.params_cap_m,
        flops_min_g=args.flops_min_g,
        flops_cap_g=args.flops_cap_g,
        tail_hidden_channels=args.tail_hidden_channels,
    )
