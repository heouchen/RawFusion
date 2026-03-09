#!/usr/bin/env python3
"""
Verify reparameterization consistency for SAFNet_Claude_49 to SAFNet_Claude_54.
Fused and unfused outputs should match within numerical tolerance.
"""
import importlib
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


MODEL_IDS = [49, 50, 51, 52, 53, 54]


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


def _test_block(module, cls_name, kwargs, shape, label):
    cls = getattr(module, cls_name, None)
    if cls is None:
        return True
    ok, max_diff, mean_diff = _check_pair(cls, shape=shape, **kwargs)
    print(
        f"{label}: max_diff={max_diff:.2e} mean_diff={mean_diff:.2e} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def _test_chunk_block(module):
    chunk_cls = getattr(module, "ChunkConvV4", None) or getattr(module, "ChunkConvV3NoL", None)
    if chunk_cls is None:
        return True
    torch.manual_seed(42)
    x = torch.randn(2, 16, 16, 16)
    m_unfused = chunk_cls(16, dilation=1)
    m_fused = chunk_cls(16, dilation=1)
    m_fused.load_state_dict(m_unfused.state_dict())
    out_unfused = m_unfused(x)
    for submodule in m_fused.modules():
        if hasattr(submodule, "fuse"):
            submodule.fuse()
    out_fused = m_fused(x)
    diff = (out_unfused - out_fused).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ok = (max_diff < 1e-4) and (mean_diff < 1e-5)
    print(
        f"{chunk_cls.__name__}: max_diff={max_diff:.2e} mean_diff={mean_diff:.2e} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def _test_full_model(module, model_id):
    model_cls = getattr(module, f"SAFNet_Claude_{model_id}")
    torch.manual_seed(42)
    x = torch.randn(1, 36, 64, 64)
    m_unfused = model_cls().eval()
    m_fused = model_cls().eval()
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
        f"SAFNet_Claude_{model_id} full: max_diff={max_diff:.2e} mean_diff={mean_diff:.2e} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def _test_flow_shapes(module, model_id):
    model_cls = getattr(module, f"SAFNet_Claude_{model_id}")
    torch.manual_seed(42)
    x = torch.randn(1, 36, 64, 64)
    model = model_cls().eval()
    with torch.no_grad():
        img0_c, img4_c, img8_c = model.group_preparer(x)
        mask0, mask8, flow0, flow8 = model.forward_flow_mask(img0_c, img4_c, img8_c, scale_factor=0.5)
    assert mask0.shape[-2:] == img4_c.shape[-2:]
    assert mask8.shape[-2:] == img4_c.shape[-2:]
    assert flow0.shape[-2:] == img4_c.shape[-2:]
    assert flow8.shape[-2:] == img4_c.shape[-2:]
    ok = (
        mask0.shape == (1, 1, 64, 64) and
        mask8.shape == (1, 1, 64, 64) and
        flow0.shape == (1, 2, 64, 64) and
        flow8.shape == (1, 2, 64, 64)
    )
    print(f"SAFNet_Claude_{model_id} flow shape -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    all_ok = True
    for model_id in MODEL_IDS:
        print(f"\n=== SAFNet_Claude_{model_id} ===")
        module = importlib.import_module(f"models.SAFNet_Claude_{model_id}")
        all_ok &= _test_block(module, "RepDWConvS", {"channels": 8, "dilation": 1}, (2, 8, 16, 16), "RepDWConvS d=1")
        all_ok &= _test_block(module, "RepDWConvS", {"channels": 8, "dilation": 2}, (2, 8, 16, 16), "RepDWConvS d=2")
        all_ok &= _test_block(module, "RepDWConvM", {"channels": 8, "dilation": 1}, (2, 8, 16, 16), "RepDWConvM d=1")
        all_ok &= _test_block(module, "RepDWConvL", {"channels": 8, "dilation": 1}, (2, 8, 16, 16), "RepDWConvL d=1")
        all_ok &= _test_chunk_block(module)
        all_ok &= _test_full_model(module, model_id)
        all_ok &= _test_flow_shapes(module, model_id)
    print(f"\nOverall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
