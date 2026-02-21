"""
SAFNet_Claude_15 — Joint Innovation (Flow + Warp + Merge)
=========================================================
Based on Claude_5, keeps RefineNet identical.
Combines all three innovations from Claude_12/13/14:
  1. Iterative Flow Refinement (2-pass flow estimation, from Claude_12)
  2. Flow-Guided DCN alignment (replaces bilinear warp, from Claude_13)
  3. Learned Attention HDR Merge (replaces handcrafted merge, from Claude_14)
Ref: RAFT, EDVR PCD, Attention-Guided HDR, RASD (NTIRE 2025)
Estimated: ~0.879M params (+49K), ~95.3G MACs (+18.9G)
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


def weight_3expo_low_tog17(img):
    w = torch.zeros_like(img)
    mask2 = img >= 0.5
    w[mask2] = img[mask2] - 0.5
    w /= 0.5
    return w


def weight_3expo_mid_tog17(img):
    w = torch.zeros_like(img)
    mask1 = img < 0.5
    w[mask1] = img[mask1]
    mask2 = img >= 0.5
    w[mask2] = 1.0 - img[mask2]
    w /= 0.5
    return w


def weight_3expo_high_tog17(img):
    w = torch.zeros_like(img)
    mask1 = img < 0.5
    w[mask1] = 0.5 - img[mask1]
    w /= 0.5
    return w


def merge_hdr_5frame(short_ldr, mid_ldr, long_ldr,
                     short_lin, mid_lin, long_lin, mask0, mask2):
    """Original handcrafted merge (kept as backup, not used in Claude_15)."""
    avg_short_lin = sum(short_lin) / len(short_lin)
    avg_mid_lin = sum(mid_lin) / len(mid_lin)
    avg_long_lin = sum(long_lin) / len(long_lin)
    avg_mid_ldr = sum(mid_ldr) / len(mid_ldr)

    w_low = weight_3expo_low_tog17(avg_mid_ldr) * mask0
    w_mid = (weight_3expo_mid_tog17(avg_mid_ldr)
             + weight_3expo_low_tog17(avg_mid_ldr) * (1.0 - mask0)
             + weight_3expo_high_tog17(avg_mid_ldr) * (1.0 - mask2))
    w_high = weight_3expo_high_tog17(avg_mid_ldr) * mask2
    sum_w = w_low + w_mid + w_high
    sum_img = w_low * avg_short_lin + w_mid * avg_mid_lin + w_high * avg_long_lin
    return sum_img / (sum_w + 1e-9)


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


# ======================== Flow-Guided DCN (from Claude_13) ========================
def flow_to_dcn_offset(flow, kernel_size=3):
    """Convert optical flow to DCN base offset.

    flow: (B, 2, H, W) where flow[:,0]=dx, flow[:,1]=dy
    Returns: (B, 2*K*K, H, W) with interleaved (dy, dx) per kernel position
    """
    flow_yx = torch.cat([flow[:, 1:2], flow[:, 0:1]], dim=1)  # swap to (dy, dx)
    return flow_yx.repeat(1, kernel_size * kernel_size, 1, 1)


class FlowGuidedDCN(nn.Module):
    """Flow-guided deformable alignment: uses flow as base offset + learns residual."""
    def __init__(self, channels=40, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        offset_channels = 2 * kernel_size * kernel_size  # 18 for 3x3
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


# ======================== DecoderDCN (from Claude_13) ========================
class DecoderDCN(nn.Module):
    """Decoder with Flow-Guided DCN replacing bilinear warp."""
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


# ======================== Learned Merge (from Claude_14) ========================
class LearnedMerge(nn.Module):
    """Lightweight attention network for HDR exposure merge.

    Input: avg_short(3ch) + avg_mid(3ch) + avg_long(3ch) + mask0(1ch) + mask2(1ch) = 11ch
    Output: merged HDR image (3ch)
    """
    def __init__(self):
        super().__init__()
        self.feat_net = nn.Sequential(
            nn.Conv2d(11, 32, 3, 1, 1),
            nn.PReLU(32),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.PReLU(32),
        )
        self.attn_head = nn.Conv2d(32, 3, 1, 1, 0)  # 3 exposure weights

    def forward(self, avg_short, avg_mid, avg_long, mask0, mask2):
        x = torch.cat([avg_short, avg_mid, avg_long, mask0, mask2], dim=1)
        feat = self.feat_net(x)
        weights = torch.softmax(self.attn_head(feat), dim=1)  # (B, 3, H, W)
        return (weights[:, 0:1] * avg_short +
                weights[:, 1:2] * avg_mid +
                weights[:, 2:3] * avg_long)


# ======================== RefineNet ========================
class RefineNet(nn.Module):
    def __init__(self, img_channels=4):
        super().__init__()
        c0, c1, c2 = 18, 36, 18
        total_c = c0 + c1 + c2

        self.conv0 = nn.Sequential(convrelu(img_channels, c0), convrelu(c0, c0))
        self.conv1 = nn.Sequential(
            DeformConvRelu(img_channels + 2 + 2 + 1 + 1 + 3, c1),
            convrelu(c1, c1))
        self.conv2 = nn.Sequential(convrelu(img_channels, c2), convrelu(c2, c2))

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

        self.conv3 = nn.Conv2d(total_c, 12, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, img0_c, img1_c, img2_c, flow0, flow2, mask0, mask2, img_hdr_m):
        feat0 = self.conv0(img0_c)
        feat1 = self.conv1(torch.cat([
            img1_c, flow0 / div_flow, flow2 / div_flow,
            mask0, mask2, img_hdr_m], 1))
        feat2 = self.conv2(img2_c)

        feat0_warp = warp(feat0, flow0)
        feat2_warp = warp(feat2, flow2)
        feat = torch.cat([feat0_warp, feat1, feat2_warp], 1)

        feat = self.blocks(feat)

        res = self.pixel_shuffle(self.conv3(feat))
        img_hdr_m_up = F.interpolate(img_hdr_m, scale_factor=2,
                                     mode="bilinear", align_corners=False)
        return torch.clamp(img_hdr_m_up + res, 0, 1)


# ======================== SAFNet_Claude_15 ========================
class SAFNet_Claude_15(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = DecoderDCN()       # DCN warp (Claude_13)
        self.refinenet = RefineNet()
        self.learned_merge = LearnedMerge()  # Attention merge (Claude_14)

    def forward_flow_mask(self, img0_c, img1_c, img2_c, scale_factor=0.5):
        """Iterative 2-pass flow estimation (Claude_12) with DCN decoder (Claude_13)."""
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
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_c)  # cached for pass 2
        f2_1, f2_2, f2_3, f2_4 = self.encoder(img2_c)

        # Pass 1: Decode 4 levels from zero
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

        # === Pass 2: Warp sources with coarse flow, re-encode, decode residual ===
        img0_warp_p2 = warp(img0_c, up_flow0_1)
        img2_warp_p2 = warp(img2_c, up_flow2_1)

        f0w_1, f0w_2, f0w_3, f0w_4 = self.encoder(img0_warp_p2)
        f2w_1, f2w_2, f2w_3, f2w_4 = self.encoder(img2_warp_p2)
        # f1 features reused from pass 1

        # Pass 2: Decode 4 levels from zero (residual flow near zero)
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

        # Combine: final flow = pass1 + residual, mask from pass 2
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

        mask0, mask8, flow0, flow8 = self.forward_flow_mask(
            img0_c, img4_c, img8_c, scale_factor=scale_factor)

        img0_warp = warp(img0_c, flow0)
        img8_warp = warp(img8_c, flow8)
        img2_warp = warp(img2_c, flow0 * 0.67)
        img6_warp = warp(img6_c, flow8 * 0.67)

        # Learned merge: compute exposure averages then blend with attention
        avg_short = (img0_warp[:, :3] + img2_warp[:, :3]) / 2.0
        avg_mid = img4_c[:, :3]
        avg_long = (img8_warp[:, :3] + img6_warp[:, :3]) / 2.0

        img_hdr_m = self.learned_merge(avg_short, avg_mid, avg_long, mask0, mask8)

        if refine:
            return self.refinenet(img0_c, img4_c, img8_c,
                                  flow0, flow8, mask0, mask8, img_hdr_m)
        else:
            return F.interpolate(img_hdr_m, scale_factor=2,
                                 mode="bilinear", align_corners=False)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SAFNet_Claude_15().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,} ({total_params/1e6:.3f}M)")
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(
        model, (36, 384, 768), verbose=False, print_per_layer_stat=True)
    print(f"MACs: {macs}, Params: {params}")
