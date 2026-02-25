"""
SAFNet_Claude_18 — 5-Frame Flow + 5-Frame RefineNet
====================================================
Based on Claude_15. Two key changes:
  1. Single-pass 5-frame direct flow (same as Claude_16)
  2. New RefineNet5Frame: uses all 5 frames instead of only 3 anchor frames
     - conv_short (shared): 4->10->10 for img0, img2
     - conv_ref:  19->24->24 for img4 + 4 flows + 4 masks + hdr
     - conv_long (shared):  4->10->10 for img6, img8
     - feat = cat[f0_warp, f2_warp, f4, f6_warp, f8_warp] -> 64ch
     - 8 x HybridBlock(64)
     - conv_out: 64->12, PixelShuffle(2) -> 3ch

All 5 frames get independent optical flow — no 0.67x approximation.
Estimated: ~0.80M params, ~88G MACs
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


# ======================== HybridBlock ========================
class HybridBlock(nn.Module):
    def __init__(self, channels, dilation=1, se_reduction=4, expand_ratio=2):
        super().__init__()
        expand_ch = channels * expand_ratio
        se_ch = max(expand_ch // se_reduction, 8)

        self.norm = nn.GroupNorm(1, channels)
        self.pw1 = nn.Conv2d(channels, expand_ch, 1, bias=True)
        self.dw = nn.Conv2d(expand_ch, expand_ch, 3, 1, dilation, dilation,
                            groups=expand_ch, bias=True)
        self.act = nn.GELU()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expand_ch, se_ch, 1, bias=True),
            nn.ReLU(inplace=True),
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
            convrelu(in_channels, 40, 3, 2, 1),
            convrelu(40, 40, 3, 1, 1))
        self.pyramid2 = nn.Sequential(
            convrelu(40, 40, 3, 2, 1),
            convrelu(40, 40, 3, 1, 1))
        self.pyramid3 = nn.Sequential(
            convrelu(40, 40, 3, 2, 1),
            convrelu(40, 40, 3, 1, 1))
        self.pyramid4 = nn.Sequential(
            convrelu(40, 40, 3, 2, 1),
            convrelu(40, 40, 3, 1, 1))

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
    def __init__(self, channels=40, kernel_size=3):
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
        self.fgdcn = FlowGuidedDCN(40)
        self.conv1 = DeformConvRelu(126, 120)
        self.conv2 = convrelu(120, 120, groups=3)
        self.conv3 = convrelu(120, 120, groups=3)
        self.conv4 = convrelu(120, 120, groups=3)
        self.conv5 = convrelu(120, 120)
        self.conv6 = deconv(120, 6)

    def forward(self, f0, f1, f2, flow0, flow2, mask0, mask2):
        f0_warp = self.fgdcn(f0, flow0)
        f2_warp = self.fgdcn(f2, flow2)
        f_in = torch.cat([f0_warp, f1, f2_warp, flow0, flow2, mask0, mask2], 1)
        f_out = self.conv1(f_in)
        f_out = channel_shuffle(self.conv2(f_out), 3)
        f_out = channel_shuffle(self.conv3(f_out), 3)
        f_out = channel_shuffle(self.conv4(f_out), 3)
        f_out = self.conv5(f_out)
        f_out = self.conv6(f_out)
        up_flow0 = 2.0 * resize(flow0, scale_factor=2.0) + f_out[:, 0:2]
        up_flow2 = 2.0 * resize(flow2, scale_factor=2.0) + f_out[:, 2:4]
        up_mask0 = resize(mask0, scale_factor=2.0) + f_out[:, 4:5]
        up_mask2 = resize(mask2, scale_factor=2.0) + f_out[:, 5:6]
        return up_flow0, up_flow2, up_mask0, up_mask2


# ======================== Learned Merge ========================
class LearnedMerge(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_net = nn.Sequential(
            nn.Conv2d(11, 32, 3, 1, 1),
            nn.PReLU(32),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.PReLU(32),
        )
        self.attn_head = nn.Conv2d(32, 3, 1, 1, 0)

    def forward(self, avg_short, avg_mid, avg_long, mask0, mask2):
        x = torch.cat([avg_short, avg_mid, avg_long, mask0, mask2], dim=1)
        feat = self.feat_net(x)
        weights = torch.softmax(self.attn_head(feat), dim=1)
        return (weights[:, 0:1] * avg_short +
                weights[:, 1:2] * avg_mid +
                weights[:, 2:3] * avg_long)


# ======================== RefineNet5Frame ========================
class RefineNet5Frame(nn.Module):
    """RefineNet that uses all 5 frames instead of only 3 anchor frames.
    Same-exposure frames share conv weights to keep params low.
    """
    def __init__(self, img_channels=4):
        super().__init__()
        # Shared for short-exposure pair (img0, img2)
        self.conv_short = nn.Sequential(
            convrelu(img_channels, 10), convrelu(10, 10))
        # Reference frame (img4) + all flows/masks + hdr
        # 4(img4) + 8(4 flows/div) + 4(4 masks) + 3(hdr) = 19ch
        self.conv_ref = nn.Sequential(
            DeformConvRelu(19, 24), convrelu(24, 24))
        # Shared for long-exposure pair (img6, img8)
        self.conv_long = nn.Sequential(
            convrelu(img_channels, 10), convrelu(10, 10))

        # cat[f0_warp(10), f2_warp(10), f4(24), f6_warp(10), f8_warp(10)] = 64
        total_c = 64
        self.blocks = nn.Sequential(
            HybridBlock(total_c, dilation=1),
            HybridBlock(total_c, dilation=1),
            HybridBlock(total_c, dilation=2),
            HybridBlock(total_c, dilation=4),
            HybridBlock(total_c, dilation=4),
            HybridBlock(total_c, dilation=2),
            HybridBlock(total_c, dilation=1),
            HybridBlock(total_c, dilation=1),
        )

        self.conv_out = nn.Conv2d(total_c, 12, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, img0_c, img2_c, img4_c, img6_c, img8_c,
                flow0, flow2, flow6, flow8,
                mask0, mask2, mask6, mask8, img_hdr_m):
        # Short-exposure features (shared weights)
        feat0 = self.conv_short(img0_c)
        feat2 = self.conv_short(img2_c)
        # Reference features
        feat4 = self.conv_ref(torch.cat([
            img4_c,
            flow0 / div_flow, flow2 / div_flow,
            flow6 / div_flow, flow8 / div_flow,
            mask0, mask2, mask6, mask8,
            img_hdr_m], dim=1))
        # Long-exposure features (shared weights)
        feat6 = self.conv_long(img6_c)
        feat8 = self.conv_long(img8_c)

        # Warp source features to reference
        feat0_warp = warp(feat0, flow0)
        feat2_warp = warp(feat2, flow2)
        feat6_warp = warp(feat6, flow6)
        feat8_warp = warp(feat8, flow8)

        feat = torch.cat([feat0_warp, feat2_warp, feat4,
                          feat6_warp, feat8_warp], dim=1)
        feat = self.blocks(feat)

        res = self.pixel_shuffle(self.conv_out(feat))
        img_hdr_m_up = F.interpolate(img_hdr_m, scale_factor=2,
                                     mode="bilinear", align_corners=False)
        return torch.clamp(img_hdr_m_up + res, 0, 1)


# ======================== SAFNet_Claude_18 ========================
class SAFNet_Claude_18(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = DecoderDCN()
        self.refinenet = RefineNet5Frame()
        self.learned_merge = LearnedMerge()

    def forward_flow_mask(self, img0_c, img2_c, img4_c, img6_c, img8_c,
                          scale_factor=0.5):
        """5-frame direct flow: encode 5, decode Pair A + Pair B (single pass)."""
        h, w = img4_c.shape[-2:]
        org_size = (int(h), int(w))
        input_size = (
            int(div_size * np.ceil(h * scale_factor / div_size)),
            int(div_size * np.ceil(w * scale_factor / div_size)))

        if input_size != org_size:
            img0_c = F.interpolate(img0_c, size=input_size, mode='bilinear', align_corners=False)
            img2_c = F.interpolate(img2_c, size=input_size, mode='bilinear', align_corners=False)
            img4_c = F.interpolate(img4_c, size=input_size, mode='bilinear', align_corners=False)
            img6_c = F.interpolate(img6_c, size=input_size, mode='bilinear', align_corners=False)
            img8_c = F.interpolate(img8_c, size=input_size, mode='bilinear', align_corners=False)

        # Encode all 5 frames (img4 features shared between pairs)
        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_c)
        f2_1, f2_2, f2_3, f2_4 = self.encoder(img2_c)
        f4_1, f4_2, f4_3, f4_4 = self.encoder(img4_c)
        f6_1, f6_2, f6_3, f6_4 = self.encoder(img6_c)
        f8_1, f8_2, f8_3, f8_4 = self.encoder(img8_c)

        zf = torch.zeros_like(f4_4[:, 0:2])
        zm = torch.zeros_like(f4_4[:, 0:1])

        # Pair A: (img0, img4, img8) -> flow_0->4, flow_8->4
        a_f0_4, a_f8_4, a_m0_4, a_m8_4 = self.decoder(
            f0_4, f4_4, f8_4, zf, zf, zm, zm)
        a_f0_3, a_f8_3, a_m0_3, a_m8_3 = self.decoder(
            f0_3, f4_3, f8_3, a_f0_4, a_f8_4, a_m0_4, a_m8_4)
        a_f0_2, a_f8_2, a_m0_2, a_m8_2 = self.decoder(
            f0_2, f4_2, f8_2, a_f0_3, a_f8_3, a_m0_3, a_m8_3)
        a_f0_1, a_f8_1, a_m0_1, a_m8_1 = self.decoder(
            f0_1, f4_1, f8_1, a_f0_2, a_f8_2, a_m0_2, a_m8_2)

        # Pair B: (img2, img4, img6) -> flow_2->4, flow_6->4
        b_f2_4, b_f6_4, b_m2_4, b_m6_4 = self.decoder(
            f2_4, f4_4, f6_4, zf, zf, zm, zm)
        b_f2_3, b_f6_3, b_m2_3, b_m6_3 = self.decoder(
            f2_3, f4_3, f6_3, b_f2_4, b_f6_4, b_m2_4, b_m6_4)
        b_f2_2, b_f6_2, b_m2_2, b_m6_2 = self.decoder(
            f2_2, f4_2, f6_2, b_f2_3, b_f6_3, b_m2_3, b_m6_3)
        b_f2_1, b_f6_1, b_m2_1, b_m6_1 = self.decoder(
            f2_1, f4_1, f6_1, b_f2_2, b_f6_2, b_m2_2, b_m6_2)

        # Rescale flows/masks if resolution was reduced
        if input_size != org_size:
            scale_h = org_size[0] / input_size[0]
            scale_w = org_size[1] / input_size[1]

            a_f0_1 = F.interpolate(a_f0_1, size=org_size, mode='bilinear', align_corners=False)
            a_f0_1[:, 0, :, :] *= scale_w
            a_f0_1[:, 1, :, :] *= scale_h

            b_f2_1 = F.interpolate(b_f2_1, size=org_size, mode='bilinear', align_corners=False)
            b_f2_1[:, 0, :, :] *= scale_w
            b_f2_1[:, 1, :, :] *= scale_h

            b_f6_1 = F.interpolate(b_f6_1, size=org_size, mode='bilinear', align_corners=False)
            b_f6_1[:, 0, :, :] *= scale_w
            b_f6_1[:, 1, :, :] *= scale_h

            a_f8_1 = F.interpolate(a_f8_1, size=org_size, mode='bilinear', align_corners=False)
            a_f8_1[:, 0, :, :] *= scale_w
            a_f8_1[:, 1, :, :] *= scale_h

            a_m0_1 = F.interpolate(a_m0_1, size=org_size, mode='bilinear', align_corners=False)
            b_m2_1 = F.interpolate(b_m2_1, size=org_size, mode='bilinear', align_corners=False)
            b_m6_1 = F.interpolate(b_m6_1, size=org_size, mode='bilinear', align_corners=False)
            a_m8_1 = F.interpolate(a_m8_1, size=org_size, mode='bilinear', align_corners=False)

        return (torch.sigmoid(a_m0_1), torch.sigmoid(b_m2_1),
                torch.sigmoid(b_m6_1), torch.sigmoid(a_m8_1),
                a_f0_1, b_f2_1, b_f6_1, a_f8_1)

    def forward(self, x, scale_factor=0.5, refine=True):
        img0_c = x[:, 0:4, :, :]
        img2_c = x[:, 8:12, :, :]
        img4_c = x[:, 16:20, :, :]
        img6_c = x[:, 24:28, :, :]
        img8_c = x[:, 32:36, :, :]

        mask0, mask2, mask6, mask8, flow0, flow2, flow6, flow8 = \
            self.forward_flow_mask(img0_c, img2_c, img4_c, img6_c, img8_c,
                                   scale_factor=scale_factor)

        # Warp all 4 source frames with independent flows (no 0.67x approx)
        img0_warp = warp(img0_c, flow0)
        img2_warp = warp(img2_c, flow2)
        img6_warp = warp(img6_c, flow6)
        img8_warp = warp(img8_c, flow8)

        # Learned merge: exposure averages then attention blend
        avg_short = (img0_warp[:, :3] + img2_warp[:, :3]) / 2.0
        avg_mid = img4_c[:, :3]
        avg_long = (img6_warp[:, :3] + img8_warp[:, :3]) / 2.0

        img_hdr_m = self.learned_merge(avg_short, avg_mid, avg_long, mask0, mask8)

        if refine:
            return self.refinenet(img0_c, img2_c, img4_c, img6_c, img8_c,
                                  flow0, flow2, flow6, flow8,
                                  mask0, mask2, mask6, mask8, img_hdr_m)
        else:
            return F.interpolate(img_hdr_m, scale_factor=2,
                                 mode="bilinear", align_corners=False)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SAFNet_Claude_18().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,} ({total_params/1e6:.3f}M)")
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(
        model, (36, 384, 768), verbose=False, print_per_layer_stat=True)
    print(f"MACs: {macs}, Params: {params}")
