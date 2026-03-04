import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import copy

from models import build_model

def _format_count(num):
    if num >= 1e12:
        return f"{num / 1e12:.3f}T"
    if num >= 1e9:
        return f"{num / 1e9:.3f}G"
    if num >= 1e6:
        return f"{num / 1e6:.3f}M"
    if num >= 1e3:
        return f"{num / 1e3:.3f}K"
    return str(num)

def get_model_profile(model, dummy_input):
    params = sum(p.numel() for p in model.parameters())
    flops = 0
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, dummy_input).total()
    except ImportError:
        pass
    return params, flops

def resolve_checkpoint_dir(model_name, exp_name, checkpoint_dir_root='./checkpoint_dir'):
    return os.path.join(checkpoint_dir_root, f'checkpoint_dir_{model_name}_{exp_name}')

def reparam(model_name, exp_name, checkpoint_dir_root='./checkpoint_dir',
            checkpoint=None, output=None, height=384, width=768):
    """Reparameterize a trained model: fuse SRP branches, save fused weights.

    Uses the original model class — no separate _rep.py needed.
    Workflow: model.fuse_reparam() → save → reload with fuse_reparam() + load_state_dict().

    Returns:
        (bool, str): (success, output_path)
    """
    ckpt_dir = resolve_checkpoint_dir(model_name, exp_name, checkpoint_dir_root)
    checkpoint_path = checkpoint or os.path.join(ckpt_dir, 'model_best.pth.tar')
    output_path = output or os.path.join(ckpt_dir, 'model_best_rep.pth.tar')

    device = torch.device("cpu")
    print(f"Model: {model_name}")
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Initialize original model
    model = build_model(model_name).to(device)

    # Load weights
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    # Remove 'module.' prefix if present
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # Create dummy input for profiling and verification
    dummy_input = torch.randn(1, 36, height, width).to(device)

    # Profile before fusion
    params_before, flops_before = get_model_profile(model, dummy_input)

    # Record output before fusion
    with torch.no_grad():
        output_before = model(dummy_input)

    # Perform fusion on a deep copy
    model_fused = copy.deepcopy(model)
    print("Performing reparameterization (fusion in memory)...")
    model_fused.fuse_reparam()
    model_fused.eval()

    # Profile after fusion
    params_after, flops_after = get_model_profile(model_fused, dummy_input)

    # Verify consistency (In-memory)
    with torch.no_grad():
        output_after_fused = model_fused(dummy_input)

    max_diff_fused = (output_before - output_after_fused).abs().max().item()

    # Print profiling results
    print("")
    print("=" * 55)
    print("  Reparameterization Profile")
    print("=" * 55)
    print(f"  Input shape  : (1, 36, {height}, {width})")
    print("-" * 55)
    print(f"  Params before: {params_before:>12,}  ({_format_count(params_before)})")
    print(f"  Params after : {params_after:>12,}  ({_format_count(params_after)})")
    print(f"  Param delta  : {params_after - params_before:>12,}  ({100*(params_after-params_before)/params_before:+.2f}%)")
    print("-" * 55)
    if flops_before > 0:
        print(f"  FLOPs before : {flops_before:>14.0f}  ({_format_count(flops_before)})")
        print(f"  FLOPs after  : {flops_after:>14.0f}  ({_format_count(flops_after)})")
        print(f"  FLOPs delta  : {flops_after - flops_before:>14.0f}  ({100*(flops_after-flops_before)/flops_before:+.2f}%)")
    else:
        print("  FLOPs: skipped (fvcore not installed)")
    print("=" * 55)
    print("")

    # Consistency check 1: in-memory fusion
    print("=" * 55)
    print("  Consistency Verification")
    print("=" * 55)
    print(f"  [1] In-memory fusion vs original")
    print(f"      Max diff: {max_diff_fused:.2e}  (threshold: 5e-4)")

    if max_diff_fused >= 5e-4:
        print(f"      Result:   FAILED")
        print("=" * 55)
        print("  Aborted. Model not saved.")
        return False, output_path

    print(f"      Result:   PASSED")

    # Save fused state dict
    print(f"  Saving fused model to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({'state_dict': model_fused.state_dict()}, output_path)

    # Consistency check 2: reload into fresh model (fuse_reparam + load_state_dict)
    print(f"  [2] Save → reload into fresh model")

    model_reloaded = build_model(model_name).to(device)
    model_reloaded.fuse_reparam()

    try:
        saved_ckpt = torch.load(output_path, map_location='cpu', weights_only=False)
        saved_sd = saved_ckpt['state_dict'] if 'state_dict' in saved_ckpt else saved_ckpt
        msg = model_reloaded.load_state_dict(saved_sd)
        print(f"      Load:     {msg}")
    except Exception as e:
        print(f"      Load:     FAILED ({e})")
        print("=" * 55)
        return False, output_path

    model_reloaded.eval()
    with torch.no_grad():
        output_reloaded = model_reloaded(dummy_input)

    max_diff_reloaded = (output_before - output_reloaded).abs().max().item()
    print(f"      Max diff: {max_diff_reloaded:.2e}  (threshold: 5e-4)")

    if max_diff_reloaded < 5e-4:
        print(f"      Result:   PASSED")
        print("=" * 55)
        print("")
        print(f"  All checks passed. Fused model saved to: {output_path}")
    else:
        print(f"      Result:   FAILED")
        print("=" * 55)
        return False, output_path

    return True, output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reparameterize (fuse) a trained model.")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name, e.g. safnet_claude_33")
    parser.add_argument("--exp_name", type=str, required=True,
                        help="Experiment name (must match training exp_name)")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint_dir",
                        help="Root directory containing checkpoint folders")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override: path to the original trained checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Override: path to save the reparameterized checkpoint")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=768)
    args = parser.parse_args()

    success, out_path = reparam(
        model_name=args.model,
        exp_name=args.exp_name,
        checkpoint_dir_root=args.checkpoint_dir,
        checkpoint=args.checkpoint,
        output=args.output,
        height=args.height,
        width=args.width,
    )
    if not success:
        exit(1)
