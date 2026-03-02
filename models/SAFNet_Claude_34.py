"""
SAFNet_Claude_34 — Large-Kernel Reparameterized SRP + CondConv
==============================================================
Optimized from SAFNet_Claude_33 with stronger representation under deployment budget:
1. Add RepDWConvL (7x7 large-kernel reparameterizable DW branch, UniRepLKNet-style design)
2. Upgrade ChunkConvV3 -> ChunkConvV4 (identity | RepDWConvS | RepDWConvM | RepDWConvL)
3. Increase decoder/refine capacity with larger channels and selective expansion

Target after fusion: params < 5M, MACs < 100G while being larger than SAFNet_Claude_33.
"""
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import copy
import warnings

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

# ======================== Utility Functions ========================
def warp(img, flow):
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], 1).to(img)
    flow_ = torch.cat([
        flow[:, 0:1, :, :] / ((W - 1.0) / 2.0),
        flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)
    ], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    return F.grid_sample(input=img, grid=grid_, mode='bilinear',
                         padding_mode='border', align_corners=True)

def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear",
                         align_corners=False, recompute_scale_factor=True)

def convrelu(in_channels, out_channels, kernel_size=3, stride=1,
             padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                  padding, dilation, groups, bias=bias),
        nn.PReLU(out_channels)
    )

def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                               stride, padding, bias=True)

class DeformConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, bias=True):
        super().__init__()
        offset_channels = 2 * kernel_size * kernel_size
        self.conv_offset = nn.Conv2d(
            in_channels, offset_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=True)
        self.deform = DeformConv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        offset = self.conv_offset(x)
        return self.prelu(self.deform(x, offset))

# ======================== CondConv ========================
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
            nn.Softmax(dim=1)
        )
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        rw = self.routing(x).view(B, self.num_experts)
        w = torch.matmul(rw, self.weight.view(self.num_experts, -1))
        w = w.view(B, self.out_channels, self.in_channels)
        b = torch.matmul(rw, self.bias)
        x = torch.bmm(w, x.view(B, C, H * W)) + b.view(B, self.out_channels, 1)
        return x.view(B, self.out_channels, H, W)

# ======================== Enhanced SRP DW Convolutions ========================
class RepDWConvS(nn.Module):
    """Enhanced 3x3 reparameterizable DW conv (RepNeXt-style).
    Branches: 3x3 + 3x1 + 1x3 + (2x2 dilated or 1x1) → fuse to single 3x3.
    """
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.channels = channels
        self.dilation = dilation
        self.fused = False
        padding = dilation
        kw = dict(in_channels=channels, out_channels=channels, groups=channels)

        self.conv_3x3 = nn.Conv2d(kernel_size=3, padding=padding,
                                   dilation=dilation, bias=True, **kw)
        self.conv_3h = nn.Conv2d(kernel_size=(3, 1), padding=(padding, 0),
                                  dilation=(dilation, 1), bias=False, **kw)
        self.conv_3w = nn.Conv2d(kernel_size=(1, 3), padding=(0, padding),
                                  dilation=(1, dilation), bias=False, **kw)
        if dilation == 1:
            self.conv_2x2 = nn.Conv2d(kernel_size=2, padding=1,
                                       dilation=2, bias=False, **kw)
        else:
            self.conv_1x1 = nn.Conv2d(kernel_size=1, padding=0,
                                       bias=False, **kw)

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
        w = self.conv_3x3.weight.data.clone()
        b = self.conv_3x3.bias.data.clone()

        w += F.pad(self.conv_3h.weight.data, [1, 1, 0, 0])
        w += F.pad(self.conv_3w.weight.data, [0, 0, 1, 1])

        if self.dilation == 1:
            w_2x2 = F.conv_transpose2d(
                self.conv_2x2.weight.data,
                torch.ones(1, 1, 1, 1, device=w.device), stride=2)
            w += w_2x2
            del self.conv_2x2
        else:
            w += F.pad(self.conv_1x1.weight.data, [1, 1, 1, 1])
            del self.conv_1x1

        self.conv_3x3.weight.data.copy_(w)
        self.conv_3x3.bias.data.copy_(b)
        del self.conv_3h, self.conv_3w
        self.fused = True


class RepDWConvM(nn.Module):
    """Enhanced 5x5 reparameterizable DW conv (RepNeXt-style, scaled from 7x7).
    Branches: 5x5 + 5x3 + 3x5 + serial(1x5→5x1) + identity → fuse to single 5x5.
    """
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.channels = channels
        self.dilation = dilation
        self.fused = False
        padding2 = 2 * dilation
        padding1 = dilation
        kw = dict(in_channels=channels, out_channels=channels, groups=channels)

        self.conv_5x5 = nn.Conv2d(kernel_size=5, padding=padding2,
                                   dilation=dilation, bias=True, **kw)
        self.conv_5x3 = nn.Conv2d(kernel_size=(5, 3), padding=(padding2, padding1),
                                   dilation=dilation, bias=True, **kw)
        self.conv_3x5 = nn.Conv2d(kernel_size=(3, 5), padding=(padding1, padding2),
                                   dilation=dilation, bias=True, **kw)
        self.conv_5w = nn.Conv2d(kernel_size=(1, 5), padding=(0, padding2),
                                  dilation=(1, dilation), bias=False, **kw)
        self.conv_5h = nn.Conv2d(kernel_size=(5, 1), padding=(padding2, 0),
                                  dilation=(dilation, 1), bias=False, **kw)

    def forward(self, x):
        if self.fused:
            return self.conv_5x5(x)
        return (self.conv_5x5(x) + self.conv_5x3(x) + self.conv_3x5(x)
                + self.conv_5h(self.conv_5w(x)) + x)

    @torch.no_grad()
    def fuse(self):
        if self.fused:
            return
        w = self.conv_5x5.weight.data.clone()
        b = self.conv_5x5.bias.data.clone()

        w += F.pad(self.conv_5x3.weight.data, [1, 1, 0, 0])
        b += self.conv_5x3.bias.data

        w += F.pad(self.conv_3x5.weight.data, [0, 0, 1, 1])
        b += self.conv_3x5.bias.data

        w_serial = torch.einsum("bcnx,bcyn->bcyx",
                                 self.conv_5w.weight.data,
                                 self.conv_5h.weight.data)
        w += w_serial

        w[:, 0, 2, 2] += 1.0

        self.conv_5x5.weight.data.copy_(w)
        self.conv_5x5.bias.data.copy_(b)
        del self.conv_5x3, self.conv_3x5, self.conv_5w, self.conv_5h
        self.fused = True


class RepDWConvL(nn.Module):
    """Large-kernel 7x7 reparameterizable DW conv.
    Branches: 7x7 + 7x3 + 3x7 + serial(1x7->7x1) + identity -> fuse to single 7x7.
    """
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.channels = channels
        self.dilation = dilation
        self.fused = False
        padding3 = 3 * dilation
        padding1 = dilation
        kw = dict(in_channels=channels, out_channels=channels, groups=channels)

        self.conv_7x7 = nn.Conv2d(kernel_size=7, padding=padding3,
                                   dilation=dilation, bias=True, **kw)
        self.conv_7x3 = nn.Conv2d(kernel_size=(7, 3), padding=(padding3, padding1),
                                   dilation=dilation, bias=True, **kw)
        self.conv_3x7 = nn.Conv2d(kernel_size=(3, 7), padding=(padding1, padding3),
                                   dilation=dilation, bias=True, **kw)
        self.conv_7w = nn.Conv2d(kernel_size=(1, 7), padding=(0, padding3),
                                  dilation=(1, dilation), bias=False, **kw)
        self.conv_7h = nn.Conv2d(kernel_size=(7, 1), padding=(padding3, 0),
                                  dilation=(dilation, 1), bias=False, **kw)

    def forward(self, x):
        if self.fused:
            return self.conv_7x7(x)
        return (self.conv_7x7(x) + self.conv_7x3(x) + self.conv_3x7(x)
                + self.conv_7h(self.conv_7w(x)) + x)

    @torch.no_grad()
    def fuse(self):
        if self.fused:
            return
        w = self.conv_7x7.weight.data.clone()
        b = self.conv_7x7.bias.data.clone()

        w += F.pad(self.conv_7x3.weight.data, [2, 2, 0, 0])
        b += self.conv_7x3.bias.data

        w += F.pad(self.conv_3x7.weight.data, [0, 0, 2, 2])
        b += self.conv_3x7.bias.data

        w_serial = torch.einsum("bcnx,bcyn->bcyx",
                                 self.conv_7w.weight.data,
                                 self.conv_7h.weight.data)
        w += w_serial
        w[:, 0, 3, 3] += 1.0

        self.conv_7x7.weight.data.copy_(w)
        self.conv_7x7.bias.data.copy_(b)
        del self.conv_7x3, self.conv_3x7, self.conv_7w, self.conv_7h
        self.fused = True


# ======================== ChunkConvV4: Multi-Scale DW Conv ========================
class ChunkConvV4(nn.Module):
    """4-group chunked DW conv: identity | RepDWConvS | RepDWConvM | RepDWConvL."""
    def __init__(self, channels, dilation=1):
        super().__init__()
        assert channels % 4 == 0
        g = channels // 4
        self.g = g
        self.rep3 = RepDWConvS(g, dilation=dilation)
        self.rep5 = RepDWConvM(g, dilation=dilation)
        self.rep7 = RepDWConvL(g, dilation=dilation)

    def forward(self, x):
        g = self.g
        x0, x1, x2, x3 = x[:, :g], x[:, g:2*g], x[:, 2*g:3*g], x[:, 3*g:]
        return torch.cat([x0, self.rep3(x1), self.rep5(x2), self.rep7(x3)], dim=1)

# ======================== RepNeXtBlock ========================
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
            nn.Sigmoid()
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

# ======================== Encoder ========================
class Encoder(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.pyramid1 = nn.Sequential(
            convrelu(in_channels, 48, 3, 2, 1),
            RepNeXtBlock(48)
        )
        self.pyramid2 = nn.Sequential(
            convrelu(48, 48, 3, 2, 1),
            RepNeXtBlock(48)
        )
        self.pyramid3 = nn.Sequential(
            convrelu(48, 48, 3, 2, 1),
            RepNeXtBlock(48, expand_ratio=2.25)
        )
        self.pyramid4 = nn.Sequential(
            convrelu(48, 48, 3, 2, 1),
            RepNeXtBlock(48, expand_ratio=2.25)
        )

    def forward(self, img_c):
        f1 = self.pyramid1(img_c)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f1, f2, f3, f4

# ======================== Flow-Guided DCN ========================
def flow_to_dcn_offset(flow, kernel_size=3):
    flow_yx = torch.cat([flow[:, 1:2], flow[:, 0:1]], dim=1)
    return flow_yx.repeat(1, kernel_size * kernel_size, 1, 1)

class FlowGuidedDCN(nn.Module):
    def __init__(self, channels=48, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        offset_channels = 2 * kernel_size * kernel_size
        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels + 2, channels, 3, 1, 1),
            nn.PReLU(channels),
            nn.Conv2d(channels, offset_channels, 3, 1, 1),
        )
        self.dcn = DeformConv2d(channels, channels, kernel_size, 1,
                                kernel_size // 2)

    def forward(self, feat, flow):
        base_offset = flow_to_dcn_offset(flow, self.kernel_size)
        residual = self.offset_conv(torch.cat([feat, flow], dim=1))
        return self.dcn(feat, base_offset + residual)

# ======================== DecoderDCN ========================
class DecoderDCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fgdcn = FlowGuidedDCN(48)
        self.conv1 = DeformConvRelu(150, 100)
        self.blocks = nn.Sequential(
            RepNeXtBlock(100, cond=True, num_experts=24, expand_ratio=2.0),
            RepNeXtBlock(100, cond=True, num_experts=24, expand_ratio=2.0),
        )
        self.conv_out = deconv(100, 6)

    def forward(self, f0, f1, f2, flow0, flow2, mask0, mask2):
        f0_warp = self.fgdcn(f0, flow0)
        f2_warp = self.fgdcn(f2, flow2)
        f_in = torch.cat([f0_warp, f1, f2_warp, flow0, flow2, mask0, mask2], 1)
        f_out = self.conv1(f_in)
        f_out = self.blocks(f_out)
        f_out = self.conv_out(f_out)
        up_flow0 = 2.0 * resize(flow0, scale_factor=2.0) + f_out[:, 0:2]
        up_flow2 = 2.0 * resize(flow2, scale_factor=2.0) + f_out[:, 2:4]
        up_mask0 = resize(mask0, scale_factor=2.0) + f_out[:, 4:5]
        up_mask2 = resize(mask2, scale_factor=2.0) + f_out[:, 5:6]
        return up_flow0, up_flow2, up_mask0, up_mask2

# ======================== Learned Merge (3-Frame) ========================
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
        x = torch.cat([img0_w, img4, img8_w, mask0, mask8], dim=1)
        feat = self.feat_net(x)
        weights = torch.softmax(self.attn_head(feat), dim=1)
        return (weights[:, 0:1] * img0_w +
                weights[:, 1:2] * img4 +
                weights[:, 2:3] * img8_w)

# ======================== RefineNet ========================
class RefineNet(nn.Module):
    def __init__(self, img_channels=4):
        super().__init__()
        c0, c1, c2 = 24, 52, 24
        total_c = c0 + c1 + c2

        self.conv0 = nn.Sequential(convrelu(img_channels, c0), RepNeXtBlock(c0))
        self.conv1 = nn.Sequential(
            DeformConvRelu(img_channels + 2 + 2 + 1 + 1 + 3, c1),
            RepNeXtBlock(c1)
        )
        self.conv2 = nn.Sequential(convrelu(img_channels, c2), RepNeXtBlock(c2))

        self.blocks = nn.Sequential(
            RepNeXtBlock(total_c, dilation=1, cond=True, num_experts=10, expand_ratio=2.0),
            RepNeXtBlock(total_c, dilation=2, cond=True, num_experts=10, expand_ratio=2.0),
            RepNeXtBlock(total_c, dilation=4, cond=True, num_experts=10, expand_ratio=2.0),
            RepNeXtBlock(total_c, dilation=2, cond=True, num_experts=10, expand_ratio=2.0),
            RepNeXtBlock(total_c, dilation=1, cond=True, num_experts=10, expand_ratio=2.0),
        )

        self.conv3 = nn.Conv2d(total_c, 12, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, img0_c, img4_c, img8_c, flow0, flow8, mask0, mask8, img_hdr_m):
        feat0 = self.conv0(img0_c)
        feat1 = self.conv1(torch.cat([
            img4_c, flow0 / div_flow, flow8 / div_flow,
            mask0, mask8, img_hdr_m], 1))
        feat2 = self.conv2(img8_c)

        feat0_warp = warp(feat0, flow0)
        feat2_warp = warp(feat2, flow8)
        feat = torch.cat([feat0_warp, feat1, feat2_warp], 1)

        feat = self.blocks(feat)
        res = self.pixel_shuffle(self.conv3(feat))
        img_hdr_m_up = F.interpolate(img_hdr_m, scale_factor=2,
                                     mode="bilinear", align_corners=False)
        return torch.clamp(img_hdr_m_up + res, 0, 1)

# ======================== SAFNet_Claude_34 ========================
class SAFNet_Claude_34(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = DecoderDCN()
        self.refinenet = RefineNet()
        self.learned_merge = LearnedMerge3Frame()

    def fuse_reparam(self):
        for m in self.modules():
            if isinstance(m, (RepDWConvS, RepDWConvM, RepDWConvL)):
                m.fuse()

    def forward_flow_mask(self, img0_c, img4_c, img8_c, scale_factor=0.5):
        h, w = img4_c.shape[-2:]
        org_size = (int(h), int(w))
        input_size = (
            int(div_size * math.ceil(float(h) * scale_factor / div_size)),
            int(div_size * math.ceil(float(w) * scale_factor / div_size)))

        if input_size != org_size:
            img0_c = F.interpolate(img0_c, size=input_size, mode='bilinear', align_corners=False)
            img4_c = F.interpolate(img4_c, size=input_size, mode='bilinear', align_corners=False)
            img8_c = F.interpolate(img8_c, size=input_size, mode='bilinear', align_corners=False)

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_c)
        f4_1, f4_2, f4_3, f4_4 = self.encoder(img4_c)
        f8_1, f8_2, f8_3, f8_4 = self.encoder(img8_c)

        up_flow0_5 = torch.zeros_like(f4_4[:, 0:2])
        up_flow8_5 = torch.zeros_like(f4_4[:, 0:2])
        up_mask0_5 = torch.zeros_like(f4_4[:, 0:1])
        up_mask8_5 = torch.zeros_like(f4_4[:, 0:1])

        up_flow0_4, up_flow8_4, up_mask0_4, up_mask8_4 = self.decoder(
            f0_4, f4_4, f8_4, up_flow0_5, up_flow8_5, up_mask0_5, up_mask8_5)
        up_flow0_3, up_flow8_3, up_mask0_3, up_mask8_3 = self.decoder(
            f0_3, f4_3, f8_3, up_flow0_4, up_flow8_4, up_mask0_4, up_mask8_4)
        up_flow0_2, up_flow8_2, up_mask0_2, up_mask8_2 = self.decoder(
            f0_2, f4_2, f8_2, up_flow0_3, up_flow8_3, up_mask0_3, up_mask8_3)
        up_flow0_1, up_flow8_1, up_mask0_1, up_mask8_1 = self.decoder(
            f0_1, f4_1, f8_1, up_flow0_2, up_flow8_2, up_mask0_2, up_mask8_2)

        img0_warp_p2 = warp(img0_c, up_flow0_1)
        img8_warp_p2 = warp(img8_c, up_flow8_1)

        f0w_1, f0w_2, f0w_3, f0w_4 = self.encoder(img0_warp_p2)
        f8w_1, f8w_2, f8w_3, f8w_4 = self.encoder(img8_warp_p2)

        up_rflow0_5 = torch.zeros_like(f4_4[:, 0:2])
        up_rflow8_5 = torch.zeros_like(f4_4[:, 0:2])
        up_rmask0_5 = torch.zeros_like(f4_4[:, 0:1])
        up_rmask8_5 = torch.zeros_like(f4_4[:, 0:1])

        up_rflow0_4, up_rflow8_4, up_rmask0_4, up_rmask8_4 = self.decoder(
            f0w_4, f4_4, f8w_4, up_rflow0_5, up_rflow8_5, up_rmask0_5, up_rmask8_5)
        up_rflow0_3, up_rflow8_3, up_rmask0_3, up_rmask8_3 = self.decoder(
            f0w_3, f4_3, f8w_3, up_rflow0_4, up_rflow8_4, up_rmask0_4, up_rmask8_4)
        up_rflow0_2, up_rflow8_2, up_rmask0_2, up_rmask8_2 = self.decoder(
            f0w_2, f4_2, f8w_2, up_rflow0_3, up_rflow8_3, up_rmask0_3, up_rmask8_3)
        up_rflow0_1, up_rflow8_1, up_rmask0_1, up_rmask8_1 = self.decoder(
            f0w_1, f4_1, f8w_1, up_rflow0_2, up_rflow8_2, up_rmask0_2, up_rmask8_2)

        final_flow0 = up_flow0_1 + up_rflow0_1
        final_flow8 = up_flow8_1 + up_rflow8_1

        if input_size != org_size:
            scale_h = org_size[0] / input_size[0]
            scale_w = org_size[1] / input_size[1]
            final_flow0 = F.interpolate(final_flow0, size=org_size, mode='bilinear', align_corners=False)
            final_flow0[:, 0, :, :] *= scale_w
            final_flow0[:, 1, :, :] *= scale_h
            final_flow8 = F.interpolate(final_flow8, size=org_size, mode='bilinear', align_corners=False)
            final_flow8[:, 0, :, :] *= scale_w
            final_flow8[:, 1, :, :] *= scale_h
            up_rmask0_1 = F.interpolate(up_rmask0_1, size=org_size, mode='bilinear', align_corners=False)
            up_rmask8_1 = F.interpolate(up_rmask8_1, size=org_size, mode='bilinear', align_corners=False)

        return torch.sigmoid(up_rmask0_1), torch.sigmoid(up_rmask8_1), final_flow0, final_flow8

    def forward(self, x, scale_factor=0.5, refine=True):
        img0_c = x[:, 0:4, :, :]
        img4_c = x[:, 16:20, :, :]
        img8_c = x[:, 32:36, :, :]

        mask0, mask8, flow0, flow8 = self.forward_flow_mask(
            img0_c, img4_c, img8_c, scale_factor=scale_factor)

        img0_warp = warp(img0_c, flow0)
        img8_warp = warp(img8_c, flow8)

        img_hdr_m = self.learned_merge(
            img0_warp[:, :3], img4_c[:, :3], img8_warp[:, :3], mask0, mask8)

        if refine:
            return self.refinenet(img0_c, img4_c, img8_c,
                                  flow0, flow8, mask0, mask8, img_hdr_m)
        else:
            return F.interpolate(img_hdr_m, scale_factor=2,
                                 mode="bilinear", align_corners=False)

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


def _print_profile(height=384, width=768):
    device = torch.device("cpu")  # macOS compatibility (MPS deform op is often unsupported).
    dummy = torch.ones(1, 36, height, width, device=device)

    model_before = SAFNet_Claude_34().to(device).eval()
    model_after = copy.deepcopy(model_before).to(device).eval()
    model_after.fuse_reparam()

    params_before = sum(p.numel() for p in model_before.parameters())
    params_after = sum(p.numel() for p in model_after.parameters())

    print(f"Input shape: (1, 36, {height}, {width})")
    print(f"Params before fusion: {params_before:,} ({_format_count(params_before)})")
    print(f"Params after fusion : {params_after:,} ({_format_count(params_after)})")
    print(f"Param delta         : {params_after - params_before:,}")

    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception as e:
        print(f"FLOPs skipped (fvcore unavailable): {e}")
        return

    flops_before = FlopCountAnalysis(model_before, dummy).total()
    flops_after = FlopCountAnalysis(model_after, dummy).total()
    print(f"FLOPs before fusion : {flops_before:.0f} ({_format_count(flops_before)})")
    print(f"FLOPs after fusion  : {flops_after:.0f} ({_format_count(flops_after)})")
    print(f"FLOPs delta          : {flops_after - flops_before:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=384, help="Input height for profiling.")
    parser.add_argument("--width", type=int, default=768, help="Input width for profiling.")
    args = parser.parse_args()
    _print_profile(height=args.height, width=args.width)
