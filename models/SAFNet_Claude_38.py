"""
SAFNet_Claude_38 — Group-Aware Claude_35 Upgrade
================================================
Built on SAFNet_Claude_35 with four PSNR-oriented upgrades:
1. Exposure-group context aggregation compresses 9 raw frames into 3 anchor-centric pseudo-frames.
2. Reliability-aware merge uses warped shallow features, flow magnitude, masks, and exposure priors.
3. Stage-2 refinement further decouples the L2 decoder and high-level low/high adapters.
4. A lightweight high-resolution tail refines the final RGB prediction.

Target (fused): <= 5M params, <= 100G FLOPs for input (1, 36, 384, 768),
which corresponds to full-size RGB output (1, 3, 768, 1536).
"""
import argparse
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.SAFNet_Claude_35 import (
        div_size,
        div_flow,
        warp,
        resize_flow,
        resize,
        convrelu,
        Encoder,
        DecoderDCN,
        DecoderDCNLite,
        FlowFeatureAdapter,
        RepNeXtBlock,
        RefineNet,
        RepDWConvS,
        RepDWConvM,
        RepDWConvL,
        _format_count,
    )
except ModuleNotFoundError:
    from SAFNet_Claude_35 import (  # pragma: no cover - direct script execution fallback
        div_size,
        div_flow,
        warp,
        resize_flow,
        resize,
        convrelu,
        Encoder,
        DecoderDCN,
        DecoderDCNLite,
        FlowFeatureAdapter,
        RepNeXtBlock,
        RefineNet,
        RepDWConvS,
        RepDWConvM,
        RepDWConvL,
        _format_count,
    )


class AnchorGroupAggregator(nn.Module):
    """Use same-exposure context to refine a spatially safe anchor frame."""
    def __init__(self, anchor_index):
        super().__init__()
        self.anchor_index = anchor_index
        self.context = nn.Sequential(
            nn.Conv2d(12, 12, 1, 1, 0, groups=4, bias=True),
            nn.PReLU(12),
            nn.Conv2d(12, 12, 3, 1, 1, groups=4, bias=True),
            nn.PReLU(12),
        )
        self.gate = nn.Conv2d(12, 4, 1, 1, 0, bias=True)
        self.residual = nn.Conv2d(12, 4, 1, 1, 0, bias=True)

    def forward(self, frames):
        # frames: (B, 3, 4, H, W)
        b, _, c, h, w = frames.shape
        anchor = frames[:, self.anchor_index, :, :, :]
        context = self.context(frames.reshape(b, 3 * c, h, w))
        gate = torch.sigmoid(self.gate(context))
        residual = self.residual(context)
        return anchor + gate * residual


class ExposureGroupPreparer(nn.Module):
    """Compress 9 packed raw frames into 3 exposure-aware pseudo-frames."""
    def __init__(self):
        super().__init__()
        self.low = AnchorGroupAggregator(anchor_index=0)
        self.mid = AnchorGroupAggregator(anchor_index=1)
        self.high = AnchorGroupAggregator(anchor_index=2)

    def forward(self, x):
        b, _, h, w = x.shape
        burst = x.reshape(b, 9, 4, h, w)
        img0_c = self.low(burst[:, 0:3, :, :, :])
        img4_c = self.mid(burst[:, 3:6, :, :, :])
        img8_c = self.high(burst[:, 6:9, :, :, :])
        return img0_c, img4_c, img8_c


class MergeFeatureStem(nn.Module):
    def __init__(self, in_channels=4, out_channels=6):
        super().__init__()
        self.stem = nn.Sequential(
            convrelu(in_channels, out_channels, 3, 1, 1),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels, bias=True),
            nn.PReLU(out_channels),
        )

    def forward(self, x):
        return self.stem(x)


class ReliableMerge3Frame(nn.Module):
    def __init__(self, feat_channels=6, hidden_channels=40):
        super().__init__()
        in_channels = 9 + feat_channels * 3 + 2 + 2 + 3
        self.feat_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1, bias=True),
            RepNeXtBlock(hidden_channels, expand_ratio=1.75),
            RepNeXtBlock(hidden_channels, expand_ratio=1.75),
        )
        self.weight_head = nn.Conv2d(hidden_channels, 3, 1, 1, 0, bias=True)
        self.residual_head = nn.Conv2d(hidden_channels, 3, 3, 1, 1, bias=True)

    def forward(self, img0_w, img4, img8_w, mask0, mask8, flow0, flow8,
                feat0_w, feat4, feat8_w):
        flow0_mag = torch.sqrt(torch.clamp(flow0.square().sum(dim=1, keepdim=True), min=1e-8)) / div_flow
        flow8_mag = torch.sqrt(torch.clamp(flow8.square().sum(dim=1, keepdim=True), min=1e-8)) / div_flow
        exp_low = mask0.new_full(mask0.shape, -1.0)
        exp_mid = mask0.new_zeros(mask0.shape)
        exp_high = mask0.new_full(mask0.shape, 1.0)

        x = torch.cat([
            img0_w, img4, img8_w,
            feat0_w, feat4, feat8_w,
            mask0, mask8,
            flow0_mag, flow8_mag,
            exp_low, exp_mid, exp_high,
        ], dim=1)
        feat = self.feat_net(x)
        weights = torch.softmax(self.weight_head(feat), dim=1)
        merged = (
            weights[:, 0:1] * img0_w +
            weights[:, 1:2] * img4 +
            weights[:, 2:3] * img8_w
        )
        residual = 0.10 * torch.tanh(self.residual_head(feat))
        return torch.clamp(merged + residual, 0.0, 1.0)


class HighResTailBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.dw = nn.Conv2d(channels, channels, 5, 1, 2, groups=channels, bias=True)
        self.pw1 = nn.Conv2d(channels, channels * 2, 1, 1, 0, bias=True)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(channels * 2, channels, 1, 1, 0, bias=True)
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)

    def forward(self, x):
        y = self.norm(x)
        y = self.dw(y)
        y = self.pw1(y)
        y = self.act(y)
        y = self.pw2(y)
        return x + y * self.scale


class HighResRefineTail(nn.Module):
    def __init__(self, hidden_channels=12, num_blocks=1):
        super().__init__()
        blocks = [HighResTailBlock(hidden_channels) for _ in range(num_blocks)]
        self.in_proj = nn.Sequential(
            nn.Conv2d(3, hidden_channels, 3, 1, 1, bias=True),
            nn.PReLU(hidden_channels),
        )
        self.blocks = nn.Sequential(*blocks)
        self.out_proj = nn.Conv2d(hidden_channels, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        feat = self.in_proj(x)
        feat = self.blocks(feat)
        return self.out_proj(feat)


class RefineNetPlus(RefineNet):
    def __init__(self, img_channels=4):
        super().__init__(img_channels=img_channels)
        self.tail = HighResRefineTail(hidden_channels=12, num_blocks=1)

    def forward(self, img0_c, img4_c, img8_c, flow0, flow8, mask0, mask8, img_hdr_m):
        out = super().forward(img0_c, img4_c, img8_c, flow0, flow8, mask0, mask8, img_hdr_m)
        out = out + 0.10 * self.tail(out)
        return torch.clamp(out, 0.0, 1.0)


class SAFNet_Claude_38(nn.Module):
    def __init__(self):
        super().__init__()
        self.group_preparer = ExposureGroupPreparer()
        self.encoder = Encoder()
        self.decoder_shared = DecoderDCN(
            mid_channels=96, num_blocks=2, cond=True, num_experts=16, expand_ratio=2.0
        )
        self.decoder_refine_l4 = DecoderDCNLite(mid_channels=72)
        self.decoder_refine_l3 = DecoderDCNLite(mid_channels=72)
        self.decoder_refine_l2 = DecoderDCNLite(mid_channels=72)
        self.adapter_shared_l1 = FlowFeatureAdapter(48)
        self.adapter_low_l2 = FlowFeatureAdapter(48)
        self.adapter_low_l3 = FlowFeatureAdapter(48)
        self.adapter_low_l4 = FlowFeatureAdapter(48)
        self.adapter_high_l2 = FlowFeatureAdapter(48)
        self.adapter_high_l3 = FlowFeatureAdapter(48)
        self.adapter_high_l4 = FlowFeatureAdapter(48)
        self.merge_stem = MergeFeatureStem(4, 6)
        self.learned_merge = ReliableMerge3Frame(feat_channels=6, hidden_channels=40)
        self.refinenet = RefineNetPlus()

    def fuse_reparam(self):
        for m in self.modules():
            if isinstance(m, (RepDWConvS, RepDWConvM, RepDWConvL)):
                m.fuse()

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

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_c)
        f4_1, f4_2, f4_3, f4_4 = self.encoder(img4_c)
        f8_1, f8_2, f8_3, f8_4 = self.encoder(img8_c)

        up_flow0_5 = torch.zeros_like(f4_4[:, 0:2])
        up_flow8_5 = torch.zeros_like(f4_4[:, 0:2])
        up_mask0_5 = torch.zeros_like(f4_4[:, 0:1])
        up_mask8_5 = torch.zeros_like(f4_4[:, 0:1])

        up_flow0_4, up_flow8_4, up_mask0_4, up_mask8_4 = self.decoder_shared(
            f0_4, f4_4, f8_4, up_flow0_5, up_flow8_5, up_mask0_5, up_mask8_5)
        up_flow0_3, up_flow8_3, up_mask0_3, up_mask8_3 = self.decoder_shared(
            f0_3, f4_3, f8_3, up_flow0_4, up_flow8_4, up_mask0_4, up_mask8_4)
        up_flow0_2, up_flow8_2, up_mask0_2, up_mask8_2 = self.decoder_shared(
            f0_2, f4_2, f8_2, up_flow0_3, up_flow8_3, up_mask0_3, up_mask8_3)
        up_flow0_1, up_flow8_1, up_mask0_1, up_mask8_1 = self.decoder_shared(
            f0_1, f4_1, f8_1, up_flow0_2, up_flow8_2, up_mask0_2, up_mask8_2)

        flow0_l1 = resize_flow(up_flow0_1, f0_1.shape[-2:])
        flow0_l2 = resize_flow(up_flow0_1, f0_2.shape[-2:])
        flow0_l3 = resize_flow(up_flow0_1, f0_3.shape[-2:])
        flow0_l4 = resize_flow(up_flow0_1, f0_4.shape[-2:])
        flow8_l1 = resize_flow(up_flow8_1, f8_1.shape[-2:])
        flow8_l2 = resize_flow(up_flow8_1, f8_2.shape[-2:])
        flow8_l3 = resize_flow(up_flow8_1, f8_3.shape[-2:])
        flow8_l4 = resize_flow(up_flow8_1, f8_4.shape[-2:])

        f0w_1 = self.adapter_shared_l1(f0_1, warp(f0_1, flow0_l1))
        f0w_2 = self.adapter_low_l2(f0_2, warp(f0_2, flow0_l2))
        f0w_3 = self.adapter_low_l3(f0_3, warp(f0_3, flow0_l3))
        f0w_4 = self.adapter_low_l4(f0_4, warp(f0_4, flow0_l4))
        f8w_1 = self.adapter_shared_l1(f8_1, warp(f8_1, flow8_l1))
        f8w_2 = self.adapter_high_l2(f8_2, warp(f8_2, flow8_l2))
        f8w_3 = self.adapter_high_l3(f8_3, warp(f8_3, flow8_l3))
        f8w_4 = self.adapter_high_l4(f8_4, warp(f8_4, flow8_l4))

        up_rflow0_5 = torch.zeros_like(f4_4[:, 0:2])
        up_rflow8_5 = torch.zeros_like(f4_4[:, 0:2])
        up_rmask0_5 = torch.zeros_like(f4_4[:, 0:1])
        up_rmask8_5 = torch.zeros_like(f4_4[:, 0:1])

        up_rflow0_4, up_rflow8_4, up_rmask0_4, up_rmask8_4 = self.decoder_refine_l4(
            f0w_4, f4_4, f8w_4, up_rflow0_5, up_rflow8_5, up_rmask0_5, up_rmask8_5)
        up_rflow0_3, up_rflow8_3, up_rmask0_3, up_rmask8_3 = self.decoder_refine_l3(
            f0w_3, f4_3, f8w_3, up_rflow0_4, up_rflow8_4, up_rmask0_4, up_rmask8_4)
        up_rflow0_2, up_rflow8_2, up_rmask0_2, up_rmask8_2 = self.decoder_refine_l2(
            f0w_2, f4_2, f8w_2, up_rflow0_3, up_rflow8_3, up_rmask0_3, up_rmask8_3)
        up_rflow0_1, up_rflow8_1, up_rmask0_1, up_rmask8_1 = self.decoder_shared(
            f0w_1, f4_1, f8w_1, up_rflow0_2, up_rflow8_2, up_rmask0_2, up_rmask8_2)

        final_flow0 = up_flow0_1 + up_rflow0_1
        final_flow8 = up_flow8_1 + up_rflow8_1

        if input_size != org_size:
            scale_h = org_size[0] / input_size[0]
            scale_w = org_size[1] / input_size[1]
            final_flow0 = F.interpolate(final_flow0, size=org_size, mode="bilinear", align_corners=False)
            final_flow0[:, 0, :, :] *= scale_w
            final_flow0[:, 1, :, :] *= scale_h
            final_flow8 = F.interpolate(final_flow8, size=org_size, mode="bilinear", align_corners=False)
            final_flow8[:, 0, :, :] *= scale_w
            final_flow8[:, 1, :, :] *= scale_h
            up_rmask0_1 = F.interpolate(up_rmask0_1, size=org_size, mode="bilinear", align_corners=False)
            up_rmask8_1 = F.interpolate(up_rmask8_1, size=org_size, mode="bilinear", align_corners=False)

        return torch.sigmoid(up_rmask0_1), torch.sigmoid(up_rmask8_1), final_flow0, final_flow8

    def forward(self, x, scale_factor=0.5, refine=True):
        img0_c, img4_c, img8_c = self.group_preparer(x)

        mask0, mask8, flow0, flow8 = self.forward_flow_mask(
            img0_c, img4_c, img8_c, scale_factor=scale_factor)

        img0_warp = warp(img0_c, flow0)
        img8_warp = warp(img8_c, flow8)

        merge_feat0 = warp(self.merge_stem(img0_c), flow0)
        merge_feat4 = self.merge_stem(img4_c)
        merge_feat8 = warp(self.merge_stem(img8_c), flow8)
        img_hdr_m = self.learned_merge(
            img0_warp[:, :3], img4_c[:, :3], img8_warp[:, :3],
            mask0, mask8, flow0, flow8,
            merge_feat0, merge_feat4, merge_feat8)

        if refine:
            return self.refinenet(img0_c, img4_c, img8_c,
                                  flow0, flow8, mask0, mask8, img_hdr_m)
        return F.interpolate(img_hdr_m, scale_factor=2,
                             mode="bilinear", align_corners=False)


def _print_profile(height=384, width=768, params_cap_m=5.0, flops_cap_g=100.0):
    device = torch.device("cpu")
    dummy = torch.ones(1, 36, height, width, device=device)

    model_before = SAFNet_Claude_38().to(device).eval()
    model_after = copy.deepcopy(model_before).to(device).eval()
    model_after.fuse_reparam()

    params_before = sum(p.numel() for p in model_before.parameters())
    params_after = sum(p.numel() for p in model_after.parameters())

    print(f"Input shape: (1, 36, {height}, {width})")
    print(f"Output shape: (1, 3, {height * 2}, {width * 2})")
    print(f"Params before fusion: {params_before:,} ({_format_count(params_before)})")
    print(f"Params after fusion : {params_after:,} ({_format_count(params_after)})")
    print(f"Param delta         : {params_after - params_before:,}")
    params_ok = params_after <= int(params_cap_m * 1e6)
    print(f"Params budget ({params_cap_m:.2f}M): {'PASS' if params_ok else 'FAIL'}")

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
    flops_ok = flops_after <= float(flops_cap_g) * 1e9
    print(f"FLOPs budget ({flops_cap_g:.2f}G): {'PASS' if flops_ok else 'FAIL'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=384, help="Input height for profiling.")
    parser.add_argument("--width", type=int, default=768, help="Input width for profiling.")
    parser.add_argument("--params_cap_m", type=float, default=5.0, help="Param cap in millions.")
    parser.add_argument("--flops_cap_g", type=float, default=100.0, help="FLOPs cap in billions.")
    args = parser.parse_args()
    _print_profile(
        height=args.height,
        width=args.width,
        params_cap_m=args.params_cap_m,
        flops_cap_g=args.flops_cap_g,
    )
