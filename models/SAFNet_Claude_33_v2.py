"""
SAFNet_Claude_33 — RepNeXt-Enhanced SRP + CondConv
===================================================
Based on SAFNet_Claude_27_v2 (832K params, 98.59G MACs).
Scales params to ~4.27M while keeping MACs under 100G using:
1. Enhanced Structural Reparameterization (RepDWConvS/M) — zero MACs increase at inference
2. CondConv2d_1x1 in Decoder (24 experts) and RefineNet main blocks (10 experts)

Constraints: ~4.27M params, ~98.6G MACs.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

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


class StripConv(nn.Module):
    def __init__(self, channels, strip_k=7, dilation=1):
        super().__init__()
        self.channels = channels
        self.strip_k = strip_k
        pad_h = (strip_k // 2) * dilation
        self.conv_h = nn.Conv2d(channels, channels, (strip_k, 1), 1,
                                (pad_h, 0), (dilation, 1),
                                groups=channels, bias=True)
        self.conv_v = nn.Conv2d(channels, channels, (1, strip_k), 1,
                                (0, pad_h), (1, dilation),
                                groups=channels, bias=True)

    def forward(self, x):
        return self.conv_h(x) + self.conv_v(x) + x

# ======================== ChunkConvV3: Multi-Scale DW Conv ========================
class ChunkConvV3(nn.Module):
    """4-group chunked DW conv: identity | RepDWConvS | RepDWConvM | StripConv."""
    def __init__(self, channels, dilation=1):
        super().__init__()
        assert channels % 4 == 0
        g = channels // 4
        self.g = g
        self.rep3 = RepDWConvS(g, dilation=dilation)
        self.rep5 = RepDWConvM(g, dilation=dilation)
        self.strip = StripConv(g, strip_k=7, dilation=dilation)

    def forward(self, x):
        g = self.g
        x0, x1, x2, x3 = x[:, :g], x[:, g:2*g], x[:, 2*g:3*g], x[:, 3*g:]
        return torch.cat([x0, self.rep3(x1), self.rep5(x2), self.strip(x3)], dim=1)

# ======================== RepNeXtBlock ========================
class RepNeXtBlock(nn.Module):
    def __init__(self, channels, dilation=1, se_reduction=4, expand_ratio=2,
                 cond=False, num_experts=12):
        super().__init__()
        expand_ch = channels * expand_ratio
        se_ch = max(expand_ch // se_reduction, 8)
        self.norm = nn.GroupNorm(1, channels)
        if cond:
            self.pw1 = CondConv2d_1x1(channels, expand_ch, num_experts)
            self.pw2 = CondConv2d_1x1(expand_ch, channels, num_experts)
        else:
            self.pw1 = nn.Conv2d(channels, expand_ch, 1, bias=True)
            self.pw2 = nn.Conv2d(expand_ch, channels, 1, bias=True)
        self.dw = ChunkConvV3(expand_ch, dilation=dilation)
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
            RepNeXtBlock(48)
        )
        self.pyramid4 = nn.Sequential(
            convrelu(48, 48, 3, 2, 1),
            RepNeXtBlock(48)
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
        self.conv1 = DeformConvRelu(150, 96)
        self.blocks = nn.Sequential(
            RepNeXtBlock(96, cond=True, num_experts=24),
            RepNeXtBlock(96, cond=True, num_experts=24),
        )
        self.conv_out = deconv(96, 6)

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

    def forward(self, img4_w, img0, img8_w, mask4, mask8):
        x = torch.cat([img4_w, img0, img8_w, mask4, mask8], dim=1)
        feat = self.feat_net(x)
        weights = torch.softmax(self.attn_head(feat), dim=1)
        return (weights[:, 0:1] * img4_w +
                weights[:, 1:2] * img0 +
                weights[:, 2:3] * img8_w)

# ======================== RefineNet ========================
class RefineNet(nn.Module):
    def __init__(self, img_channels=4):
        super().__init__()
        c0, c1, c2 = 32, 48, 24
        total_c = c0 + c1 + c2

        self.conv0 = nn.Sequential(convrelu(img_channels, c0), RepNeXtBlock(c0))
        self.conv1 = nn.Sequential(
            DeformConvRelu(img_channels + 2 + 2 + 1 + 1 + 3, c1),
            RepNeXtBlock(c1)
        )
        self.conv2 = nn.Sequential(convrelu(img_channels, c2), RepNeXtBlock(c2))

        self.blocks = nn.Sequential(
            RepNeXtBlock(total_c, dilation=1, cond=True, num_experts=10),
            RepNeXtBlock(total_c, dilation=2, cond=True, num_experts=10),
            RepNeXtBlock(total_c, dilation=4, cond=True, num_experts=10),
            RepNeXtBlock(total_c, dilation=2, cond=True, num_experts=10),
            RepNeXtBlock(total_c, dilation=1, cond=True, num_experts=10),
        )

        self.conv3 = nn.Conv2d(total_c, 12, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, img0_c, img4_c, img8_c, flow4, flow8, mask4, mask8, img_hdr_m):
        feat4 = self.conv0(img4_c)
        feat1 = self.conv1(torch.cat([
            img0_c, flow4 / div_flow, flow8 / div_flow,
            mask4, mask8, img_hdr_m], 1))
        feat2 = self.conv2(img8_c)

        feat4_warp = warp(feat4, flow4)
        feat2_warp = warp(feat2, flow8)
        feat = torch.cat([feat4_warp, feat1, feat2_warp], 1)

        feat = self.blocks(feat)
        res = self.pixel_shuffle(self.conv3(feat))
        img_hdr_m_up = F.interpolate(img_hdr_m, scale_factor=2,
                                     mode="bilinear", align_corners=False)
        return torch.clamp(img_hdr_m_up + res, 0, 1)

# ======================== SAFNet_Claude_33 ========================
class SAFNet_Claude_33(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = DecoderDCN()
        self.refinenet = RefineNet()
        self.learned_merge = LearnedMerge3Frame()

    def fuse_reparam(self):
        for m in self.modules():
            if isinstance(m, (RepDWConvS, RepDWConvM)):
                m.fuse()

    def forward_flow_mask(self, img0_c, img4_c, img8_c, scale_factor=0.5):
        h, w = img0_c.shape[-2:]
        org_size = (int(h), int(w))
        input_size = (
            int(div_size * np.ceil(h * scale_factor / div_size)),
            int(div_size * np.ceil(w * scale_factor / div_size)))

        if input_size != org_size:
            img0_c = F.interpolate(img0_c, size=input_size, mode='bilinear', align_corners=False)
            img4_c = F.interpolate(img4_c, size=input_size, mode='bilinear', align_corners=False)
            img8_c = F.interpolate(img8_c, size=input_size, mode='bilinear', align_corners=False)

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_c)
        f4_1, f4_2, f4_3, f4_4 = self.encoder(img4_c)
        f8_1, f8_2, f8_3, f8_4 = self.encoder(img8_c)

        up_flow4_5 = torch.zeros_like(f0_4[:, 0:2])
        up_flow8_5 = torch.zeros_like(f0_4[:, 0:2])
        up_mask4_5 = torch.zeros_like(f0_4[:, 0:1])
        up_mask8_5 = torch.zeros_like(f0_4[:, 0:1])

        up_flow4_4, up_flow8_4, up_mask4_4, up_mask8_4 = self.decoder(
            f4_4, f0_4, f8_4, up_flow4_5, up_flow8_5, up_mask4_5, up_mask8_5)
        up_flow4_3, up_flow8_3, up_mask4_3, up_mask8_3 = self.decoder(
            f4_3, f0_3, f8_3, up_flow4_4, up_flow8_4, up_mask4_4, up_mask8_4)
        up_flow4_2, up_flow8_2, up_mask4_2, up_mask8_2 = self.decoder(
            f4_2, f0_2, f8_2, up_flow4_3, up_flow8_3, up_mask4_3, up_mask8_3)
        up_flow4_1, up_flow8_1, up_mask4_1, up_mask8_1 = self.decoder(
            f4_1, f0_1, f8_1, up_flow4_2, up_flow8_2, up_mask4_2, up_mask8_2)

        img4_warp_p2 = warp(img4_c, up_flow4_1)
        img8_warp_p2 = warp(img8_c, up_flow8_1)

        f4w_1, f4w_2, f4w_3, f4w_4 = self.encoder(img4_warp_p2)
        f8w_1, f8w_2, f8w_3, f8w_4 = self.encoder(img8_warp_p2)

        up_rflow4_5 = torch.zeros_like(f0_4[:, 0:2])
        up_rflow8_5 = torch.zeros_like(f0_4[:, 0:2])
        up_rmask4_5 = torch.zeros_like(f0_4[:, 0:1])
        up_rmask8_5 = torch.zeros_like(f0_4[:, 0:1])

        up_rflow4_4, up_rflow8_4, up_rmask4_4, up_rmask8_4 = self.decoder(
            f4w_4, f0_4, f8w_4, up_rflow4_5, up_rflow8_5, up_rmask4_5, up_rmask8_5)
        up_rflow4_3, up_rflow8_3, up_rmask4_3, up_rmask8_3 = self.decoder(
            f4w_3, f0_3, f8w_3, up_rflow4_4, up_rflow8_4, up_rmask4_4, up_rmask8_4)
        up_rflow4_2, up_rflow8_2, up_rmask4_2, up_rmask8_2 = self.decoder(
            f4w_2, f0_2, f8w_2, up_rflow4_3, up_rflow8_3, up_rmask4_3, up_rmask8_3)
        up_rflow4_1, up_rflow8_1, up_rmask4_1, up_rmask8_1 = self.decoder(
            f4w_1, f0_1, f8w_1, up_rflow4_2, up_rflow8_2, up_rmask4_2, up_rmask8_2)

        final_flow4 = up_flow4_1 + up_rflow4_1
        final_flow8 = up_flow8_1 + up_rflow8_1

        if input_size != org_size:
            scale_h = org_size[0] / input_size[0]
            scale_w = org_size[1] / input_size[1]
            final_flow4 = F.interpolate(final_flow4, size=org_size, mode='bilinear', align_corners=False)
            final_flow4[:, 0, :, :] *= scale_w
            final_flow4[:, 1, :, :] *= scale_h
            final_flow8 = F.interpolate(final_flow8, size=org_size, mode='bilinear', align_corners=False)
            final_flow8[:, 0, :, :] *= scale_w
            final_flow8[:, 1, :, :] *= scale_h
            up_rmask4_1 = F.interpolate(up_rmask4_1, size=org_size, mode='bilinear', align_corners=False)
            up_rmask8_1 = F.interpolate(up_rmask8_1, size=org_size, mode='bilinear', align_corners=False)

        return torch.sigmoid(up_rmask4_1), torch.sigmoid(up_rmask8_1), final_flow4, final_flow8

    def forward(self, x, scale_factor=0.5, refine=True):
        img0_c = x[:, 0:4, :, :]
        img4_c = x[:, 16:20, :, :]
        img8_c = x[:, 32:36, :, :]

        mask4, mask8, flow4, flow8 = self.forward_flow_mask(
            img0_c, img4_c, img8_c, scale_factor=scale_factor)

        img4_warp = warp(img4_c, flow4)
        img8_warp = warp(img8_c, flow8)

        img_hdr_m = self.learned_merge(
            img4_warp[:, :3], img0_c[:, :3], img8_warp[:, :3], mask4, mask8)

        if refine:
            return self.refinenet(img0_c, img4_c, img8_c,
                                  flow4, flow8, mask4, mask8, img_hdr_m)
        else:
            return F.interpolate(img_hdr_m, scale_factor=2,
                                 mode="bilinear", align_corners=False)


if __name__ == "__main__":
    device = torch.device('cpu')
    model = SAFNet_Claude_33().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,} ({total_params/1e6:.3f}M)")

    from fvcore.nn import FlopCountAnalysis, flop_count_table
    flops = FlopCountAnalysis(model, torch.ones(1, 36, 384, 768).to(device))
    print(f"Total FLOPs of the model : {flops.total() / (1000**4) :.3f}(T)")

    model.fuse_reparam()
    flops = FlopCountAnalysis(model, torch.ones(1, 36, 384, 768).to(device))
    print(f"Total FLOPs of the model after fusion: {flops.total() / (1000**4) :.3f}(T)")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,} ({total_params/1e6:.3f}M)")
