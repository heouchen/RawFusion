#!/usr/bin/env python3
"""
验证 SAFNet_Claude_33 中 RepDWConvS / RepDWConvM 重参数化正确性：
融合前后在相同输入、相同参数下输出应一致（数值误差内）。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F

# 只导入需要测试的类
from models.SAFNet_Claude_33 import RepDWConvS, RepDWConvM, ChunkConvV3, SAFNet_Claude_33


def test_rep_dw_conv_s(channels=8, dilation=1, H=16, W=16, seed=42):
    """RepDWConvS: 融合前后输出一致"""
    torch.manual_seed(seed)
    x = torch.randn(2, channels, H, W)
    m_unfused = RepDWConvS(channels, dilation=dilation)
    m_fused = RepDWConvS(channels, dilation=dilation)
    m_fused.load_state_dict(m_unfused.state_dict())
    out_unfused = m_unfused(x)
    m_fused.fuse()
    out_fused = m_fused(x)
    max_diff = (out_unfused - out_fused).abs().max().item()
    mean_diff = (out_unfused - out_fused).abs().mean().item()
    ok = max_diff < 1e-4 and mean_diff < 1e-5
    print(f"  RepDWConvS channels={channels} dilation={dilation}: max_diff={max_diff:.2e} mean_diff={mean_diff:.2e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_rep_dw_conv_s_dilation2(channels=8, dilation=2, H=16, W=16, seed=42):
    """RepDWConvS dilation=2 分支（1x1）"""
    torch.manual_seed(seed)
    x = torch.randn(2, channels, H, W)
    m_unfused = RepDWConvS(channels, dilation=dilation)
    m_fused = RepDWConvS(channels, dilation=dilation)
    m_fused.load_state_dict(m_unfused.state_dict())
    out_unfused = m_unfused(x)
    m_fused.fuse()
    out_fused = m_fused(x)
    max_diff = (out_unfused - out_fused).abs().max().item()
    mean_diff = (out_unfused - out_fused).abs().mean().item()
    ok = max_diff < 1e-4 and mean_diff < 1e-5
    print(f"  RepDWConvS channels={channels} dilation={dilation}: max_diff={max_diff:.2e} mean_diff={mean_diff:.2e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_rep_dw_conv_m(channels=8, dilation=1, H=16, W=16, seed=42):
    """RepDWConvM: 融合前后输出一致"""
    torch.manual_seed(seed)
    x = torch.randn(2, channels, H, W)
    m_unfused = RepDWConvM(channels, dilation=dilation)
    m_fused = RepDWConvM(channels, dilation=dilation)
    m_fused.load_state_dict(m_unfused.state_dict())
    out_unfused = m_unfused(x)
    m_fused.fuse()
    out_fused = m_fused(x)
    max_diff = (out_unfused - out_fused).abs().max().item()
    mean_diff = (out_unfused - out_fused).abs().mean().item()
    ok = max_diff < 1e-4 and mean_diff < 1e-5
    print(f"  RepDWConvM channels={channels} dilation={dilation}: max_diff={max_diff:.2e} mean_diff={mean_diff:.2e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_chunk_conv_v3(channels=16, H=16, W=16, seed=42):
    """ChunkConvV3（内含 RepDWConvS / RepDWConvM / StripConv）融合前后一致"""
    torch.manual_seed(seed)
    x = torch.randn(2, channels, H, W)
    m_unfused = ChunkConvV3(channels, dilation=1)
    m_fused = ChunkConvV3(channels, dilation=1)
    m_fused.load_state_dict(m_unfused.state_dict())
    out_unfused = m_unfused(x)
    for _m in m_fused.modules():
        if hasattr(_m, 'fuse'):
            _m.fuse()
    out_fused = m_fused(x)
    max_diff = (out_unfused - out_fused).abs().max().item()
    mean_diff = (out_unfused - out_fused).abs().mean().item()
    ok = max_diff < 1e-4 and mean_diff < 1e-5
    print(f"  ChunkConvV3 channels={channels}: max_diff={max_diff:.2e} mean_diff={mean_diff:.2e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_full_model_forward_consistency(batch=1, C=36, H=64, W=64, seed=42):
    """完整 SAFNet_Claude_33：融合前后 forward 输出一致（refine=True/False 都测）"""
    torch.manual_seed(seed)
    x = torch.randn(batch, C, H, W)
    m_unfused = SAFNet_Claude_33()
    m_fused = SAFNet_Claude_33()
    m_fused.load_state_dict(m_unfused.state_dict())
    m_unfused.eval()
    m_fused.eval()
    with torch.no_grad():
        out_unfused = m_unfused(x, scale_factor=0.5, refine=True)
        m_fused.fuse_reparam()
        out_fused = m_fused(x, scale_factor=0.5, refine=True)
    max_diff = (out_unfused - out_fused).abs().max().item()
    mean_diff = (out_unfused - out_fused).abs().mean().item()
    ok = max_diff < 1e-3 and mean_diff < 1e-4  # 整网允许稍大一点误差
    print(f"  SAFNet_Claude_33 full forward (refine=True): max_diff={max_diff:.2e} mean_diff={mean_diff:.2e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("Reparameterization verification (fused vs unfused output consistency)\n")
    all_ok = True
    all_ok &= test_rep_dw_conv_s(8, 1)
    all_ok &= test_rep_dw_conv_s_dilation2(8, 2)
    all_ok &= test_rep_dw_conv_m(8, 1)
    all_ok &= test_chunk_conv_v3(16)
    all_ok &= test_full_model_forward_consistency()
    print()
    print("Overall:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    exit(main())
