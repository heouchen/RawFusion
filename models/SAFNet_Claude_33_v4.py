"""
SAFNet_Claude_33_v4
===================
Minimal staged-training variant built on SAFNet_Claude_33_v3.

Key additions:
1. Frame-wise exposure adapter before group alignment.
2. Auxiliary reconstruction head for stage-1 alignment training.
3. Structured return_aux output for alignment-aware losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.SAFNet_Claude_33_v3 import SAFNet_Claude_33_v3
    from models.SAFNet_Claude_33_v2 import RepNeXtBlock, warp
except ImportError:
    from SAFNet_Claude_33_v3 import SAFNet_Claude_33_v3
    from SAFNet_Claude_33_v2 import RepNeXtBlock, warp


class ExposureAdapter(nn.Module):
    """Shared lightweight residual adapter applied frame-wise."""
    def __init__(self, channels=4, hidden_channels=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, 1, 1, bias=True),
            nn.PReLU(hidden_channels),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                3,
                1,
                1,
                groups=hidden_channels,
                bias=True,
            ),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, channels, 1, 1, 0, bias=True),
        )

    def forward(self, x):
        return x + self.net(x)


class AuxReconstructionHead(nn.Module):
    """Small head that supervises alignment stages without using RefineNet."""
    def __init__(self, hidden_channels=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4 * 3 + 2, hidden_channels, 3, 1, 1, bias=True),
            nn.PReLU(hidden_channels),
            RepNeXtBlock(hidden_channels, expand_ratio=1),
            nn.Conv2d(hidden_channels, 12, 3, 1, 1, bias=True),
            nn.PReLU(12),
            nn.Conv2d(12, 3, 1, 1, 0, bias=True),
        )

    def forward(self, img0_c, img4_warp, img8_warp, mask4, mask8):
        x = torch.cat([img0_c, img4_warp, img8_warp, mask4, mask8], dim=1)
        x = self.net(x)
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class SAFNet_Claude_33_v4(SAFNet_Claude_33_v3):
    def __init__(self):
        super().__init__()
        self.exposure_adapter = ExposureAdapter(channels=4, hidden_channels=16)
        self.aux_head = AuxReconstructionHead(hidden_channels=24)

    def apply_exposure_adapter(self, burst):
        b, n, c, h, w = burst.shape
        burst_flat = burst.reshape(b * n, c, h, w)
        burst_flat = self.exposure_adapter(burst_flat)
        return burst_flat.reshape(b, n, c, h, w)

    def _fuse_group(self, img_anchor, img_a, img_b, scale_factor, return_aux=False):
        group_scale = min(scale_factor, self.group_scale_factor)
        mask_a, mask_b, flow_a, flow_b = self.forward_flow_mask(
            img_anchor, img_a, img_b, scale_factor=group_scale
        )
        img_a_warp = warp(img_a, flow_a)
        img_b_warp = warp(img_b, flow_b)
        fused = self.group_raw_merge(
            img_a_warp, img_anchor, img_b_warp, mask_a, mask_b
        )
        if not return_aux:
            return fused
        return fused, {
            "anchor": img_anchor,
            "warped": (img_a_warp, img_b_warp),
            "flows": (flow_a, flow_b),
            "masks": (mask_a, mask_b),
        }

    def prepare_group_frames(self, burst, scale_factor=0.5, return_aux=False):
        img0_c = self._fuse_group(
            img_anchor=burst[:, 0, :, :, :],
            img_a=burst[:, 1, :, :, :],
            img_b=burst[:, 2, :, :, :],
            scale_factor=scale_factor,
            return_aux=return_aux,
        )
        img4_c = self._fuse_group(
            img_anchor=burst[:, 4, :, :, :],
            img_a=burst[:, 3, :, :, :],
            img_b=burst[:, 5, :, :, :],
            scale_factor=scale_factor,
            return_aux=return_aux,
        )
        img8_c = self._fuse_group(
            img_anchor=burst[:, 8, :, :, :],
            img_a=burst[:, 7, :, :, :],
            img_b=burst[:, 6, :, :, :],
            scale_factor=scale_factor,
            return_aux=return_aux,
        )
        if not return_aux:
            return img0_c, img4_c, img8_c

        group_frames = (img0_c[0], img4_c[0], img8_c[0])
        group_aux = (img0_c[1], img4_c[1], img8_c[1])
        return group_frames[0], group_frames[1], group_frames[2], group_aux

    def forward(self, x, scale_factor=0.5, refine=True, return_aux=False):
        b, _, h, w = x.shape
        burst = x.reshape(b, 9, 4, h, w)
        burst = self.apply_exposure_adapter(burst)

        if return_aux:
            img0_c, img4_c, img8_c, group_aux = self.prepare_group_frames(
                burst, scale_factor=scale_factor, return_aux=True
            )
        else:
            img0_c, img4_c, img8_c = self.prepare_group_frames(
                burst, scale_factor=scale_factor, return_aux=False
            )
            group_aux = None

        mask4, mask8, flow4, flow8 = self.forward_flow_mask(
            img0_c, img4_c, img8_c, scale_factor=scale_factor
        )

        img4_warp = warp(img4_c, flow4)
        img8_warp = warp(img8_c, flow8)
        aux_pred = self.aux_head(img0_c, img4_warp, img8_warp, mask4, mask8)

        img_hdr_m = self.learned_merge(
            img4_warp[:, :3], img0_c[:, :3], img8_warp[:, :3], mask4, mask8
        )

        if refine:
            pred = self.refinenet(
                img0_c, img4_c, img8_c, flow4, flow8, mask4, mask8, img_hdr_m
            )
        else:
            pred = F.interpolate(
                img_hdr_m, scale_factor=2, mode="bilinear", align_corners=False
            )

        if not return_aux:
            return pred

        aux = {
            "aux_pred": aux_pred,
            "group_aux": group_aux,
            "group_frames": (img0_c, img4_c, img8_c),
            "inter_warped": (img4_warp, img8_warp),
            "inter_flows": (flow4, flow8),
            "inter_masks": (mask4, mask8),
        }
        return pred, aux

if __name__ == "__main__":
    device = torch.device("cpu")
    model = SAFNet_Claude_33_v4().to(device)
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
