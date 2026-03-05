#!/usr/bin/env python3
"""
Verify reparameterization consistency for SAFNet_Claude_37.
Fused and unfused outputs should match within numerical tolerance.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from models.SAFNet_Claude_37 import (
    ChunkConvV4,
    RepDWConvL,
    RepDWConvM,
    RepDWConvS,
    SAFNet_Claude_37,
)


def _check_pair(m_ctor, shape, tol_max=1e-4, tol_mean=1e-5, seed=42, **kwargs):
    torch.manual_seed(seed)
    x = torch.randn(*shape)
    m_unfused = m_ctor(**kwargs)
    m_fused = m_ctor(**kwargs)
    m_fused.load_state_dict(m_unfused.state_dict())
    out_unfused = m_unfused(x)
    m_fused.fuse()
    out_fused = m_fused(x)
    diff = (out_unfused - out_fused).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ok = (max_diff < tol_max) and (mean_diff < tol_mean)
    return ok, max_diff, mean_diff


def test_rep_dw_conv_s():
    ok, max_diff, mean_diff = _check_pair(
        RepDWConvS, shape=(2, 8, 16, 16), channels=8, dilation=1
    )
    print(
        f"RepDWConvS d=1: max_diff={max_diff:.2e} mean_diff={mean_diff:.2e} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def test_rep_dw_conv_s_d2():
    ok, max_diff, mean_diff = _check_pair(
        RepDWConvS, shape=(2, 8, 16, 16), channels=8, dilation=2
    )
    print(
        f"RepDWConvS d=2: max_diff={max_diff:.2e} mean_diff={mean_diff:.2e} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def test_rep_dw_conv_m():
    ok, max_diff, mean_diff = _check_pair(
        RepDWConvM, shape=(2, 8, 16, 16), channels=8, dilation=1
    )
    print(
        f"RepDWConvM d=1: max_diff={max_diff:.2e} mean_diff={mean_diff:.2e} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def test_rep_dw_conv_l():
    ok, max_diff, mean_diff = _check_pair(
        RepDWConvL, shape=(2, 8, 16, 16), channels=8, dilation=1
    )
    print(
        f"RepDWConvL d=1: max_diff={max_diff:.2e} mean_diff={mean_diff:.2e} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def test_chunk_conv_v4():
    torch.manual_seed(42)
    x = torch.randn(2, 16, 16, 16)
    m_unfused = ChunkConvV4(16, dilation=1)
    m_fused = ChunkConvV4(16, dilation=1)
    m_fused.load_state_dict(m_unfused.state_dict())
    out_unfused = m_unfused(x)
    for module in m_fused.modules():
        if hasattr(module, "fuse"):
            module.fuse()
    out_fused = m_fused(x)
    diff = (out_unfused - out_fused).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ok = (max_diff < 1e-4) and (mean_diff < 1e-5)
    print(
        f"ChunkConvV4: max_diff={max_diff:.2e} mean_diff={mean_diff:.2e} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def test_full_model():
    torch.manual_seed(42)
    x = torch.randn(1, 36, 64, 64)
    m_unfused = SAFNet_Claude_37().eval()
    m_fused = SAFNet_Claude_37().eval()
    m_fused.load_state_dict(m_unfused.state_dict())
    with torch.no_grad():
        out_unfused = m_unfused(x, scale_factor=0.5, refine=True)
        m_fused.fuse_reparam()
        out_fused = m_fused(x, scale_factor=0.5, refine=True)
    diff = (out_unfused - out_fused).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ok = (max_diff < 1e-3) and (mean_diff < 1e-4)
    print(
        f"SAFNet_Claude_37 full: max_diff={max_diff:.2e} mean_diff={mean_diff:.2e} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def main():
    print("Reparameterization verification for SAFNet_Claude_37\n")
    checks = [
        test_rep_dw_conv_s(),
        test_rep_dw_conv_s_d2(),
        test_rep_dw_conv_m(),
        test_rep_dw_conv_l(),
        test_chunk_conv_v4(),
        test_full_model(),
    ]
    all_ok = all(checks)
    print(f"\nOverall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
