"""
SAFNet_Claude_21 — 2-Pass Anchor Flow + 5-Frame RefineNet 72ch
================================================================
Key insight from Claude_16~20 experiments:
  - 2-pass iterative flow is essential (don't remove)
  - RefineNet capacity is the quality ceiling (don't shrink)
  - 0.67x flow approximation is NOT the bottleneck

Design: keep Claude_15's proven 2-pass 3-anchor flow EXACTLY,
but replace RefineNet with RefineNet5Frame that takes all 5 frames.
The approximate flows (0.67x) are good enough — RefineNet can compensate.

  - Flow: 2-pass 3-anchor (5enc + 8dec) — identical to Claude_15
  - flow2 = flow0 * 0.67, flow6 = flow8 * 0.67
  - RefineNet5Frame 72ch: conv_short(12) + conv_ref(24) + conv_long(12) = 72
  - All 5 warped frames feed into RefineNet → better temporal denoising

Estimated: ~0.84M params, ~93G MACs
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


# ======================== RefineNet5Frame (72ch) ========================
class RefineNet5Frame(nn.Module):
    """5-frame RefineNet: same-exposure frames share weights.
    total_c = 12 + 12 + 24 + 12 + 12 = 72 (same capacity as Claude_15 RefineNet).
    """
    def __init__(self, img_channels=4):
        super().__init__()
        self.conv_short = nn.Sequential(
            convrelu(img_channels, 12), convrelu(12, 12))
        # 4(img4) + 8(4 flows/div) + 4(4 masks) + 3(hdr) = 19ch
        self.conv_ref = nn.Sequential(
            DeformConvRelu(19, 24), convrelu(24, 24))
        self.conv_long = nn.Sequential(
            convrelu(img_channels, 12), convrelu(12, 12))

        total_c = 72  # 12+12+24+12+12
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
        feat0 = self.conv_short(img0_c)
        feat2 = self.conv_short(img2_c)
        feat4 = self.conv_ref(torch.cat([
            img4_c,
            flow0 / div_flow, flow2 / div_flow,
            flow6 / div_flow, flow8 / div_flow,
            mask0, mask2, mask6, mask8,
            img_hdr_m], dim=1))
        feat6 = self.conv_long(img6_c)
        feat8 = self.conv_long(img8_c)

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


# ======================== SAFNet_Claude_21 ========================
class SAFNet_Claude_21(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = DecoderDCN()
        self.refinenet = RefineNet5Frame()
        self.learned_merge = LearnedMerge()

    def forward_flow_mask(self, img0_c, img1_c, img2_c, scale_factor=0.5):
        """Iterative 2-pass flow estimation — IDENTICAL to Claude_15."""
        h, w = img1_c.shape[-2:]
        org_size = (int(h), int(w))
        input_size = (
            int(div_size * np.ceil(h * scale_factor / div_size)),
            int(div_size * np.ceil(w * scale_factor / div_size)))

        if input_size != org_size:
            img0_c = F.interpolate(img0_c, size=input_size, mode='bilinear', align_corners=False)
            img1_c = F.interpolate(img1_c, size=input_size, mode='bilinear', align_corners=False)
            img2_c = F.interpolate(img2_c, size=input_size, mode='bilinear', align_corners=False)

        # === Pass 1: Encode all 3 frames ===
        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_c)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_c)
        f2_1, f2_2, f2_3, f2_4 = self.encoder(img2_c)

        up_flow0_5 = torch.zeros_like(f1_4[:, 0:2])
        up_flow2_5 = torch.zeros_like(f1_4[:, 0:2])
        up_mask0_5 = torch.zeros_like(f1_4[:, 0:1])
        up_mask2_5 = torch.zeros_like(f1_4[:, 0:1])

        up_flow0_4, up_flow2_4, up_mask0_4, up_mask2_4 = self.decoder(
            f0_4, f1_4, f2_4, up_flow0_5, up_flow2_5, up_mask0_5, up_mask2_5)
        up_flow0_3, up_flow2_3, up_mask0_3, up_mask2_3 = self.decoder(
            f0_3, f1_3, f2_3, up_flow0_4, up_flow2_4, up_mask0_4, up_mask2_4)
        up_flow0_2, up_flow2_2, up_mask0_2, up_mask2_2 = self.decoder(
            f0_2, f1_2, f2_2, up_flow0_3, up_flow2_3, up_mask0_3, up_mask2_3)
        up_flow0_1, up_flow2_1, up_mask0_1, up_mask2_1 = self.decoder(
            f0_1, f1_1, f2_1, up_flow0_2, up_flow2_2, up_mask0_2, up_mask2_2)

        # === Pass 2: Warp sources, re-encode, decode residual ===
        img0_warp_p2 = warp(img0_c, up_flow0_1)
        img2_warp_p2 = warp(img2_c, up_flow2_1)

        f0w_1, f0w_2, f0w_3, f0w_4 = self.encoder(img0_warp_p2)
        f2w_1, f2w_2, f2w_3, f2w_4 = self.encoder(img2_warp_p2)

        up_rflow0_5 = torch.zeros_like(f1_4[:, 0:2])
        up_rflow2_5 = torch.zeros_like(f1_4[:, 0:2])
        up_rmask0_5 = torch.zeros_like(f1_4[:, 0:1])
        up_rmask2_5 = torch.zeros_like(f1_4[:, 0:1])

        up_rflow0_4, up_rflow2_4, up_rmask0_4, up_rmask2_4 = self.decoder(
            f0w_4, f1_4, f2w_4, up_rflow0_5, up_rflow2_5, up_rmask0_5, up_rmask2_5)
        up_rflow0_3, up_rflow2_3, up_rmask0_3, up_rmask2_3 = self.decoder(
            f0w_3, f1_3, f2w_3, up_rflow0_4, up_rflow2_4, up_rmask0_4, up_rmask2_4)
        up_rflow0_2, up_rflow2_2, up_rmask0_2, up_rmask2_2 = self.decoder(
            f0w_2, f1_2, f2w_2, up_rflow0_3, up_rflow2_3, up_rmask0_3, up_rmask2_3)
        up_rflow0_1, up_rflow2_1, up_rmask0_1, up_rmask2_1 = self.decoder(
            f0w_1, f1_1, f2w_1, up_rflow0_2, up_rflow2_2, up_rmask0_2, up_rmask2_2)

        final_flow0 = up_flow0_1 + up_rflow0_1
        final_flow2 = up_flow2_1 + up_rflow2_1

        if input_size != org_size:
            scale_h = org_size[0] / input_size[0]
            scale_w = org_size[1] / input_size[1]
            final_flow0 = F.interpolate(final_flow0, size=org_size, mode='bilinear', align_corners=False)
            final_flow0[:, 0, :, :] *= scale_w
            final_flow0[:, 1, :, :] *= scale_h
            final_flow2 = F.interpolate(final_flow2, size=org_size, mode='bilinear', align_corners=False)
            final_flow2[:, 0, :, :] *= scale_w
            final_flow2[:, 1, :, :] *= scale_h
            up_rmask0_1 = F.interpolate(up_rmask0_1, size=org_size, mode='bilinear', align_corners=False)
            up_rmask2_1 = F.interpolate(up_rmask2_1, size=org_size, mode='bilinear', align_corners=False)

        return torch.sigmoid(up_rmask0_1), torch.sigmoid(up_rmask2_1), final_flow0, final_flow2

    def forward(self, x, scale_factor=0.5, refine=True):
        img0_c = x[:, 0:4, :, :]
        img2_c = x[:, 8:12, :, :]
        img4_c = x[:, 16:20, :, :]
        img6_c = x[:, 24:28, :, :]
        img8_c = x[:, 32:36, :, :]

        # 2-pass 3-anchor flow (identical to Claude_15)
        mask0, mask8, flow0, flow8 = self.forward_flow_mask(
            img0_c, img4_c, img8_c, scale_factor=scale_factor)

        # Approximate flow for middle frames
        flow2 = flow0 * 0.67
        flow6 = flow8 * 0.67
        mask2 = mask0
        mask6 = mask8

        # Warp all 5 frames
        img0_warp = warp(img0_c, flow0)
        img2_warp = warp(img2_c, flow2)
        img6_warp = warp(img6_c, flow6)
        img8_warp = warp(img8_c, flow8)

        # Learned merge (same as Claude_15)
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
    model = SAFNet_Claude_21().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,} ({total_params/1e6:.3f}M)")
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(
        model, (36, 384, 768), verbose=False, print_per_layer_stat=True)
    print(f"MACs: {macs}, Params: {params}")
