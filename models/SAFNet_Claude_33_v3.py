"""
SAFNet_Claude_33_v3 — 9-Frame Group-Prepared Variant
=====================================================
Built on SAFNet_Claude_33_v2.

Key change:
1. All 9 packed raw frames are consumed.
2. Each exposure group is first aligned to its anchor with the original
   SAFNet flow/mask pipeline, then fused in the packed-raw domain.
3. The main SAFNet_Claude_33_v2 alignment / merge / refine pipeline is reused
   on top of the three aligned pseudo-frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.SAFNet_Claude_33_v2 import SAFNet_Claude_33_v2, RepNeXtBlock, warp
except ImportError:
    from SAFNet_Claude_33_v2 import SAFNet_Claude_33_v2, RepNeXtBlock, warp


class LearnedRawMerge3Frame(nn.Module):
    """Packed-raw counterpart of the original SAFNet learned merge head."""
    def __init__(self, img_channels=4, hidden_channels=12, num_blocks=0):
        super().__init__()
        in_channels = img_channels * 3 + 2
        blocks = [RepNeXtBlock(hidden_channels, expand_ratio=1) for _ in range(num_blocks)]
        self.feat_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1, bias=True),
            nn.PReLU(hidden_channels),
            *blocks,
        )
        self.attn_head = nn.Conv2d(hidden_channels, 3, 1, 1, 0)

    def forward(self, img_a_w, img_anchor, img_b_w, mask_a, mask_b):
        x = torch.cat([img_a_w, img_anchor, img_b_w, mask_a, mask_b], dim=1)
        feat = self.feat_net(x)
        weights = torch.softmax(self.attn_head(feat), dim=1)
        return (weights[:, 0:1] * img_a_w +
                weights[:, 1:2] * img_anchor +
                weights[:, 2:3] * img_b_w)


class SAFNet_Claude_33_v3(SAFNet_Claude_33_v2):
    def __init__(self):
        super().__init__()
        self.group_scale_factor = 0.04
        self.group_raw_merge = LearnedRawMerge3Frame(
            img_channels=4, hidden_channels=12, num_blocks=0)

    def _fuse_group(self, img_anchor, img_a, img_b, scale_factor):
        group_scale = min(scale_factor, self.group_scale_factor)
        mask_a, mask_b, flow_a, flow_b = self.forward_flow_mask(
            img_anchor, img_a, img_b, scale_factor=group_scale)
        img_a_warp = warp(img_a, flow_a)
        img_b_warp = warp(img_b, flow_b)
        return self.group_raw_merge(img_a_warp, img_anchor, img_b_warp, mask_a, mask_b)

    def prepare_group_frames(self, x, scale_factor=0.5):
        b, _, h, w = x.shape
        burst = x.reshape(b, 9, 4, h, w)

        img0_c = self._fuse_group(
            img_anchor=burst[:, 0, :, :, :],
            img_a=burst[:, 1, :, :, :],
            img_b=burst[:, 2, :, :, :],
            scale_factor=scale_factor,
        )
        img4_c = self._fuse_group(
            img_anchor=burst[:, 4, :, :, :],
            img_a=burst[:, 3, :, :, :],
            img_b=burst[:, 5, :, :, :],
            scale_factor=scale_factor,
        )
        img8_c = self._fuse_group(
            img_anchor=burst[:, 8, :, :, :],
            img_a=burst[:, 7, :, :, :],
            img_b=burst[:, 6, :, :, :],
            scale_factor=scale_factor,
        )
        return img0_c, img4_c, img8_c

    def forward(self, x, scale_factor=0.5, refine=True):
        img0_c, img4_c, img8_c = self.prepare_group_frames(x, scale_factor=scale_factor)

        mask4, mask8, flow4, flow8 = self.forward_flow_mask(
            img0_c, img4_c, img8_c, scale_factor=scale_factor)

        img4_warp = warp(img4_c, flow4)
        img8_warp = warp(img8_c, flow8)

        img_hdr_m = self.learned_merge(
            img4_warp[:, :3], img0_c[:, :3], img8_warp[:, :3], mask4, mask8)

        if refine:
            return self.refinenet(img0_c, img4_c, img8_c,
                                  flow4, flow8, mask4, mask8, img_hdr_m)
        return F.interpolate(img_hdr_m, scale_factor=2,
                             mode="bilinear", align_corners=False)


if __name__ == "__main__":
    device = torch.device("cpu")
    model = SAFNet_Claude_33_v3().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,} ({total_params / 1e6:.3f}M)")

    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        print("fvcore is not installed; skip FLOPs analysis.")
    else:
        flops = FlopCountAnalysis(model, torch.ones(1, 36, 384, 768).to(device))
        print(f"Total FLOPs of the model : {flops.total() / (1000**4):.3f}(T)")

        model.fuse_reparam()
        flops = FlopCountAnalysis(model, torch.ones(1, 36, 384, 768).to(device))
        print(f"Total FLOPs of the model after fusion: {flops.total() / (1000**4):.3f}(T)")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params:,} ({total_params / 1e6:.3f}M)")
