"""
SAFNet_Claude_27 — 3-Frame Input + RepNeXt Block Everywhere
==============================================================
Based on SAFNet_Claude_27. Optimized for NTIRE 2026 HDR Competition.
Input reduced to 3 frames (img0, img4, img8) to save computation.
RepNeXtBlock integrated into Encoder, Decoder, and Merge stages.

Constraints: < 5M params, < 100G FLOPs.
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

def channel_shuffle(x, groups):
    b, c, h, w = x.size()
    channels_per_group = c // groups
    x = x.view(b, groups, channels_per_group, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, -1, h, w)
    return x

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

# ======================== ChunkConv: Multi-Scale DW Conv ========================
class RepDWBranch(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        padding = (kernel_size // 2) * dilation
        self.conv_main = nn.Conv2d(channels, channels, kernel_size, 1, padding,
                                    dilation, groups=channels, bias=True)
        self.conv_1x1 = nn.Conv2d(channels, channels, 1, 1, 0,
                                   groups=channels, bias=True)
        self.fused = False

    def forward(self, x):
        if self.fused:
            return self.conv_main(x)
        return self.conv_main(x) + self.conv_1x1(x) + x

    @torch.no_grad()
    def fuse(self):
        if self.fused:
            return
        k = self.kernel_size
        w_main = self.conv_main.weight.data
        b_main = self.conv_main.bias.data
        p = k // 2
        w_1x1 = F.pad(self.conv_1x1.weight.data, [p, p, p, p])
        b_1x1 = self.conv_1x1.bias.data
        w_id = torch.zeros_like(w_main)
        w_id[:, 0, k // 2, k // 2] = 1.0
        self.conv_main.weight.data = w_main + w_1x1 + w_id
        self.conv_main.bias.data = b_main + b_1x1
        del self.conv_1x1
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

class ChunkConv(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        assert channels % 4 == 0
        g = channels // 4
        self.g = g
        self.rep3 = RepDWBranch(g, kernel_size=3, dilation=dilation)
        self.rep5 = RepDWBranch(g, kernel_size=5, dilation=dilation)
        self.strip = StripConv(g, strip_k=7, dilation=dilation)

    def forward(self, x):
        g = self.g
        x0, x1, x2, x3 = x[:, :g], x[:, g:2*g], x[:, 2*g:3*g], x[:, 3*g:]
        return torch.cat([x0, self.rep3(x1), self.rep5(x2), self.strip(x3)], dim=1)

# ======================== RepNeXtBlock ========================
class RepNeXtBlock(nn.Module):
    def __init__(self, channels, dilation=1, se_reduction=4, expand_ratio=2):
        super().__init__()
        expand_ch = channels * expand_ratio
        se_ch = max(expand_ch // se_reduction, 8)
        self.norm = nn.GroupNorm(1, channels)
        self.pw1 = nn.Conv2d(channels, expand_ch, 1, bias=True)
        self.dw = ChunkConv(expand_ch, dilation=dilation)
        self.act = nn.GELU()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expand_ch, se_ch, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(se_ch, expand_ch, 1, bias=True),
            nn.Sigmoid()
        )
        self.pw2 = nn.Conv2d(expand_ch, channels, 1, bias=True)
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
            RepNeXtBlock(96),
            RepNeXtBlock(96),
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
        c0, c1, c2 = 24, 48, 24
        total_c = c0 + c1 + c2

        self.conv0 = nn.Sequential(convrelu(img_channels, c0), RepNeXtBlock(c0))
        self.conv1 = nn.Sequential(
            DeformConvRelu(img_channels + 2 + 2 + 1 + 1 + 3, c1),
            RepNeXtBlock(c1)
        )
        self.conv2 = nn.Sequential(convrelu(img_channels, c2), RepNeXtBlock(c2))

        self.blocks = nn.Sequential(
            RepNeXtBlock(total_c, dilation=1),
            RepNeXtBlock(total_c, dilation=2),
            RepNeXtBlock(total_c, dilation=4),
            RepNeXtBlock(total_c, dilation=2),
            RepNeXtBlock(total_c, dilation=1),
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

# ======================== SAFNet_Claude_27 ========================
class SAFNet_Claude_27(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = DecoderDCN()
        self.refinenet = RefineNet()
        self.learned_merge = LearnedMerge3Frame()

    def fuse_reparam(self):
        for m in self.modules():
            if isinstance(m, RepDWBranch):
                m.fuse()

    def forward_flow_mask(self, img0_c, img4_c, img8_c, scale_factor=0.5):
        h, w = img4_c.shape[-2:]
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

if __name__ == "__main__":
    device = torch.device('cpu')
    model = SAFNet_Claude_27().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,} ({total_params/1e6:.3f}M)")
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(
        model, (36, 384, 768), verbose=False, print_per_layer_stat=False)
    print(f"MACs: {macs}, Params: {params}")

    from fvcore.nn import FlopCountAnalysis, flop_count_table
    model.fuse_reparam()
    flops = FlopCountAnalysis(model, torch.ones(1, 36, 384, 768).to(device))
    print(f"Total FLOPs of the model after fusion: {flops.total() / (1000**4) :.3f}(T)")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,} ({total_params/1e6:.3f}M)")
