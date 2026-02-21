"""
SAFNet_Claude_11 — MDTA + DropPath + Dense Skip (Claude_8 × Claude_10)
======================================================================
Combines the two best-performing innovations:
  From Claude_8:  Multi-head Transposed Attention (MDTA) for global context
  From Claude_10: DropPath regularization + Dense skip connections + 80ch width

Architecture:
  - Width: 80ch (20+40+20)
  - Group 1: 3 HybridBlockDP [dilation 1,2,4] → dense fuse (concat + 1x1)
  - Group 2: 2 HybridBlockDP [dilation 2,1] → sequential
  - Tail:    2 MDTABlockDP (transposed attention + DropPath)
  - Total: 5 conv + 2 MDTA = 7 blocks
  - DropPath: linearly increasing 0 → 0.1 across all 7 blocks

Encoder/Decoder: identical to Claude_5 (proven alignment)
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


# ======================== DropPath ========================
def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ======================== HybridBlockDP (from Claude_10) ========================
class HybridBlockDP(nn.Module):
    """DW-Sep inverted bottleneck + SE + DropPath."""
    def __init__(self, channels, dilation=1, se_reduction=4,
                 expand_ratio=2, drop_path_rate=0.):
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
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        y = self.norm(x)
        y = self.pw1(y)
        y = self.dw(y)
        y = self.act(y)
        y = y * self.se(y)
        y = self.pw2(y)
        return x + self.drop_path(y * self.scale)


# ======================== MDTABlockDP (from Claude_8 + DropPath) ========================
class MDTABlockDP(nn.Module):
    """
    Multi-head Transposed Attention + DW-FFN + DropPath.
    Transposed attention: O(C²HW) — efficient for moderate channel counts.
    """
    def __init__(self, channels, num_heads=4, ffn_expand=1, drop_path_rate=0.):
        super().__init__()
        self.num_heads = num_heads

        # Transposed Attention
        self.norm1 = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=True)
        self.proj = nn.Conv2d(channels, channels, 1, bias=True)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # FFN with DW conv
        self.norm2 = nn.GroupNorm(1, channels)
        ffn_ch = channels * ffn_expand
        self.ffn_pw1 = nn.Conv2d(channels, ffn_ch, 1, bias=True)
        self.ffn_dw = nn.Conv2d(ffn_ch, ffn_ch, 3, 1, 1,
                                groups=ffn_ch, bias=True)
        self.ffn_act = nn.GELU()
        self.ffn_pw2 = nn.Conv2d(ffn_ch, channels, 1, bias=True)

        self.attn_scale = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)
        self.ffn_scale = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        # Multi-head Transposed Attention
        y = self.norm1(x)
        qkv = self.qkv(y)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W)
        k = k.reshape(B, self.num_heads, C // self.num_heads, H * W)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(B, C, H, W)
        out = self.proj(out)
        x = x + self.drop_path(out * self.attn_scale)

        # FFN
        y = self.norm2(x)
        y = self.ffn_pw1(y)
        y = self.ffn_dw(y)
        y = self.ffn_act(y)
        y = self.ffn_pw2(y)
        x = x + self.drop_path(y * self.ffn_scale)

        return x


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


# ======================== Decoder ========================
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = DeformConvRelu(126, 120)
        self.conv2 = convrelu(120, 120, groups=3)
        self.conv3 = convrelu(120, 120, groups=3)
        self.conv4 = convrelu(120, 120, groups=3)
        self.conv5 = convrelu(120, 120)
        self.conv6 = deconv(120, 6)

    def forward(self, f0, f1, f2, flow0, flow2, mask0, mask2):
        f0_warp = warp(f0, flow0)
        f2_warp = warp(f2, flow2)
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


# ======================== RefineNet ========================
class RefineNet(nn.Module):
    """
    Combines Claude_8 (MDTA) and Claude_10 (DropPath + Dense Skip) innovations.

    80ch, 7 blocks total:
      Group 1: 3 HybridBlockDP [d=1,2,4] → dense fuse (concat + 1x1 compress)
      Group 2: 2 HybridBlockDP [d=2,1]   → sequential
      Tail:    2 MDTABlockDP              → global context via transposed attention
    DropPath with linearly increasing rate across all 7 blocks.
    """
    def __init__(self, img_channels=4):
        super().__init__()
        c0, c1, c2 = 20, 40, 20
        total_c = c0 + c1 + c2  # 80

        self.conv0 = nn.Sequential(convrelu(img_channels, c0), convrelu(c0, c0))
        self.conv1 = nn.Sequential(
            DeformConvRelu(img_channels + 2 + 2 + 1 + 1 + 3, c1),
            convrelu(c1, c1))
        self.conv2 = nn.Sequential(convrelu(img_channels, c2), convrelu(c2, c2))

        num_blocks = 7
        dpr = [x.item() for x in torch.linspace(0, 0.1, num_blocks)]

        # Group 1: 3 HybridBlockDP with dense skip
        self.group1 = nn.ModuleList([
            HybridBlockDP(total_c, dilation=1, drop_path_rate=dpr[0]),
            HybridBlockDP(total_c, dilation=2, drop_path_rate=dpr[1]),
            HybridBlockDP(total_c, dilation=4, drop_path_rate=dpr[2]),
        ])
        self.fuse1 = nn.Conv2d(total_c * 2, total_c, 1, bias=True)

        # Group 2: 2 HybridBlockDP sequential
        self.group2 = nn.ModuleList([
            HybridBlockDP(total_c, dilation=2, drop_path_rate=dpr[3]),
            HybridBlockDP(total_c, dilation=1, drop_path_rate=dpr[4]),
        ])

        # Tail: 2 MDTABlockDP for global context
        self.attn_blocks = nn.ModuleList([
            MDTABlockDP(total_c, num_heads=4, drop_path_rate=dpr[5]),
            MDTABlockDP(total_c, num_heads=4, drop_path_rate=dpr[6]),
        ])

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

        # Group 1: local feature extraction with dense skip
        group1_in = feat
        for block in self.group1:
            feat = block(feat)
        feat = self.fuse1(torch.cat([group1_in, feat], 1))

        # Group 2: multi-scale refinement
        for block in self.group2:
            feat = block(feat)

        # Tail: global context via transposed attention
        for block in self.attn_blocks:
            feat = block(feat)

        res = self.pixel_shuffle(self.conv3(feat))
        img_hdr_m_up = F.interpolate(img_hdr_m, scale_factor=2,
                                     mode="bilinear", align_corners=False)
        return torch.clamp(img_hdr_m_up + res, 0, 1)


# ======================== SAFNet_Claude_11 ========================
class SAFNet_Claude_11(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.refinenet = RefineNet()

    def forward_flow_mask(self, img0_c, img1_c, img2_c, scale_factor=0.5):
        h, w = img1_c.shape[-2:]
        org_size = (int(h), int(w))
        input_size = (
            int(div_size * np.ceil(h * scale_factor / div_size)),
            int(div_size * np.ceil(w * scale_factor / div_size)))

        if input_size != org_size:
            img0_c = F.interpolate(img0_c, size=input_size, mode='bilinear', align_corners=False)
            img1_c = F.interpolate(img1_c, size=input_size, mode='bilinear', align_corners=False)
            img2_c = F.interpolate(img2_c, size=input_size, mode='bilinear', align_corners=False)

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

        if input_size != org_size:
            scale_h = org_size[0] / input_size[0]
            scale_w = org_size[1] / input_size[1]
            up_flow0_1 = F.interpolate(up_flow0_1, size=org_size, mode='bilinear', align_corners=False)
            up_flow0_1[:, 0, :, :] *= scale_w
            up_flow0_1[:, 1, :, :] *= scale_h
            up_flow2_1 = F.interpolate(up_flow2_1, size=org_size, mode='bilinear', align_corners=False)
            up_flow2_1[:, 0, :, :] *= scale_w
            up_flow2_1[:, 1, :, :] *= scale_h
            up_mask0_1 = F.interpolate(up_mask0_1, size=org_size, mode='bilinear', align_corners=False)
            up_mask2_1 = F.interpolate(up_mask2_1, size=org_size, mode='bilinear', align_corners=False)

        return torch.sigmoid(up_mask0_1), torch.sigmoid(up_mask2_1), up_flow0_1, up_flow2_1

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

        img_hdr_m = merge_hdr_5frame(
            short_ldr=[img0_warp[:, :3], img2_warp[:, :3]],
            mid_ldr=[img4_c[:, :3]],
            long_ldr=[img8_warp[:, :3], img6_warp[:, :3]],
            short_lin=[img0_warp[:, :3], img2_warp[:, :3]],
            mid_lin=[img4_c[:, :3]],
            long_lin=[img8_warp[:, :3], img6_warp[:, :3]],
            mask0=mask0, mask2=mask8
        )

        if refine:
            return self.refinenet(img0_c, img4_c, img8_c,
                                  flow0, flow8, mask0, mask8, img_hdr_m)
        else:
            return F.interpolate(img_hdr_m, scale_factor=2,
                                 mode="bilinear", align_corners=False)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SAFNet_Claude_11().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,} ({total_params/1e6:.3f}M)")
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(
        model, (36, 384, 768), verbose=False, print_per_layer_stat=True)
    print(f"MACs: {macs}, Params: {params}")
