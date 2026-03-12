"""
Evaluation script for HDR Burst Reconstruction models.

Features:
  - Accepts model name via CLI, auto-builds the model
  - Auto-finds the best checkpoint (model_best.pth.tar) from checkpoint_dir
  - Saves validation outputs as 96-bit TIF (float32 x 3ch) for NTIRE submission
  - Reports per-scene and average PSNR / SSIM
  - Creates img/ folder under the checkpoint dir for output images
  - Optional TTA (Test-Time Augmentation): 8-fold D4 geometric ensemble

Usage:
  python eval.py --model safnet_claude_5 --exp_name model_cmp_claude5
  python eval.py --model safnet_claude_6 --exp_name model_cmp_claude6 --save_gt
  python eval.py --model safnet_claude_5 --exp_name model_cmp_claude5 --tta
  python eval.py --model safnet_claude_5 --exp_name model_cmp_claude5 --sliding_window
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os
import time
import argparse
import glob
import zipfile

from torchvision.transforms import transforms
from DataLoader.custom_data_class import CustomDataset
from models import build_model, MODEL_NAMES
from utils.utils import calculate_psnr, calculate_ssim
from utils.checkpoint import load_checkpoint, load_model_state_dict


def tta_forward(model, x):
    """8-fold TTA using D4 dihedral group (4 rotations x 2 flips).

    For each of the 8 geometric transforms:
      1. Apply transform to input
      2. Run model inference
      3. Apply inverse transform to output
    Returns the average of all 8 predictions.
    """
    preds = []
    for flip in (False, True):
        for rot_k in range(4):
            x_aug = x
            if flip:
                x_aug = torch.flip(x_aug, dims=[-1])
            if rot_k:
                x_aug = torch.rot90(x_aug, k=rot_k, dims=[-2, -1])

            pred = model(x_aug)

            # Inverse: undo rotation first, then undo flip
            if rot_k:
                pred = torch.rot90(pred, k=(4 - rot_k), dims=[-2, -1])
            if flip:
                pred = torch.flip(pred, dims=[-1])

            preds.append(pred)

    return sum(preds) / len(preds)


def model_forward(model, x, tta=False):
    """Run a single inference pass, optionally with D4 TTA."""
    return tta_forward(model, x) if tta else model(x)


def compute_sliding_starts(length, patch_size, stride):
    """Return patch start offsets that fully cover a dimension."""
    if patch_size <= 0 or stride <= 0:
        raise ValueError('patch_size and stride must be positive integers')

    if length <= patch_size:
        return [0]

    starts = list(range(0, length - patch_size + 1, stride))
    last_start = length - patch_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def build_blend_weight(patch_h, patch_w, device, dtype, min_weight=1e-3):
    """Center-weighted window for overlap blending to suppress seam artifacts."""
    y = torch.linspace(-1.0, 1.0, steps=patch_h, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, steps=patch_w, device=device, dtype=dtype)
    wy = (1.0 - y.abs()).clamp_min(min_weight)
    wx = (1.0 - x.abs()).clamp_min(min_weight)
    return torch.outer(wy, wx).unsqueeze(0).unsqueeze(0)


def sliding_window_forward(model, x, patch_size=(192, 384), stride=(96, 192), tta=False):
    """Run inference on overlapping patches and fuse them back into a full image."""
    patch_h, patch_w = patch_size
    stride_h, stride_w = stride

    if stride_h >= patch_h or stride_w >= patch_w:
        raise ValueError(
            f'stride must be smaller than patch size for overlap, got '
            f'patch_size={patch_size}, stride={stride}'
        )

    b, _, h, w = x.shape
    pad_h = max(0, patch_h - h)
    pad_w = max(0, patch_w - w)
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

    _, _, padded_h, padded_w = x.shape
    y_starts = compute_sliding_starts(padded_h, patch_h, stride_h)
    x_starts = compute_sliding_starts(padded_w, patch_w, stride_w)

    pred_acc = None
    weight_acc = None
    blend_weight = None
    scale_h = None
    scale_w = None
    target_h = None
    target_w = None

    for top in y_starts:
        for left in x_starts:
            patch = x[:, :, top:top + patch_h, left:left + patch_w]
            patch_pred = model_forward(model, patch, tta=tta)
            out_patch_h, out_patch_w = patch_pred.shape[-2:]

            if pred_acc is None:
                if out_patch_h % patch_h != 0 or out_patch_w % patch_w != 0:
                    raise ValueError(
                        'Sliding-window inference requires output spatial size '
                        'to be an integer multiple of input patch size, got '
                        f'input patch={patch_size}, output patch='
                        f'({out_patch_h}, {out_patch_w})'
                    )

                scale_h = out_patch_h // patch_h
                scale_w = out_patch_w // patch_w
                blend_weight = build_blend_weight(
                    out_patch_h,
                    out_patch_w,
                    patch_pred.device,
                    patch_pred.dtype,
                )
                target_h = h * scale_h
                target_w = w * scale_w
                pred_acc = torch.zeros(
                    (b, patch_pred.shape[1], padded_h * scale_h, padded_w * scale_w),
                    device=patch_pred.device,
                    dtype=patch_pred.dtype,
                )
                weight_acc = torch.zeros(
                    (1, 1, padded_h * scale_h, padded_w * scale_w),
                    device=patch_pred.device,
                    dtype=patch_pred.dtype,
                )
            elif patch_pred.shape[-2:] != blend_weight.shape[-2:]:
                raise ValueError(
                    'All sliding-window patch predictions must share the same '
                    f'output size, got {patch_pred.shape[-2:]} vs '
                    f'{blend_weight.shape[-2:]}'
                )

            out_top = top * scale_h
            out_left = left * scale_w
            pred_acc[:, :, out_top:out_top + out_patch_h, out_left:out_left + out_patch_w] += (
                patch_pred * blend_weight
            )
            weight_acc[:, :, out_top:out_top + out_patch_h, out_left:out_left + out_patch_w] += (
                blend_weight
            )

    pred = pred_acc / weight_acc.clamp_min(1e-8)
    return pred[:, :, :target_h, :target_w]


def find_best_checkpoint(checkpoint_dir):
    """Auto-find the best checkpoint in checkpoint_dir.

    Priority:
      1. model_best.pth.tar (saved by training script)
      2. Highest-numbered .pth.tar (latest epoch)
    """
    best_path = os.path.join(checkpoint_dir, 'model_best.pth.tar')
    if os.path.exists(best_path):
        return best_path

    files = sorted(glob.glob(os.path.join(checkpoint_dir, '*.pth.tar')))
    if files:
        return files[-1]

    return None


def eval_model(
    model_name,
    exp_name='default',
    val_root='/home/chen/data/ntire2026/hdr/validation/',
    checkpoint_dir_root='./checkpoint_dir',
    cuda=True,
    mgpu=True,
    save_gt=False,
    save_input=False,
    tta=False,
    rep=False,
    sliding_window=False,
    patch_size=(192, 384),
    stride=None,
):
    """Evaluate a model on the validation set and save output TIF images.

    Args:
        model_name: Model key (e.g. 'safnet_claude_5')
        exp_name: Experiment name used during training
        val_root: Path to validation data
        checkpoint_dir_root: Root dir containing per-experiment checkpoint folders
        cuda: Use GPU
        mgpu: Use DataParallel
        save_gt: Also save GT images alongside predictions
        save_input: Also save per-frame input images
        tta: Enable 8-fold D4 test-time augmentation
    """
    if stride is None:
        stride = (patch_size[0] // 2, patch_size[1] // 2)

    print(f'=== Evaluation: {model_name} (exp: {exp_name}) ===')
    if rep:
        print(f'    Rep: enabled (using reparameterized model)')
    if tta:
        print(f'    TTA: enabled (8-fold D4 geometric ensemble)')
    if sliding_window:
        print(f'    Sliding window: enabled')
        print(f'    Patch size: {patch_size[0]}x{patch_size[1]}')
        print(f'    Stride: {stride[0]}x{stride[1]}')
    print()

    # ---- Resolve checkpoint directory ----
    checkpoint_dir = os.path.join(
        checkpoint_dir_root,
        f'checkpoint_dir_{model_name}_{exp_name}'
    )
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(
            f'Checkpoint directory not found: {checkpoint_dir}\n'
            f'Available dirs: {os.listdir(checkpoint_dir_root)}'
        )

    if rep:
        ckpt_path = os.path.join(checkpoint_dir, 'model_best_rep.pth.tar')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f'Reparameterized checkpoint not found: {ckpt_path}\n'
                f'Run reparam_model.py first to generate it.'
            )
    else:
        ckpt_path = find_best_checkpoint(checkpoint_dir)
        if ckpt_path is None:
            raise FileNotFoundError(
                f'No checkpoint found in {checkpoint_dir}'
            )
    print(f'=> Checkpoint: {ckpt_path}')

    # ---- Output image directory (inside checkpoint dir) ----
    img_dir = os.path.join(checkpoint_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)
    print(f'=> Output images will be saved to: {img_dir}')

    # ---- Dataset ----
    val_set = CustomDataset(
        root_dir=val_root,
        transform=transforms.ToTensor(),
        train=False,
        augment=None,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    scene_ids = val_set.scene_ids
    print(f'=> Validation scenes: {len(val_loader)}')

    # ---- Model ----
    model = build_model(model_name)

    # Rep checkpoints are saved from an already-fused model, so the runtime
    # model must be fused before loading weights.
    if rep:
        model.fuse_reparam()

    # ---- Load weights ----
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    load_model_state_dict(model, checkpoint['state_dict'])

    if cuda:
        model = model.cuda()
    if mgpu:
        model = nn.DataParallel(model)
    epoch = checkpoint.get('epoch', '?')
    best_psnr = checkpoint.get('best_psnr', checkpoint.get('best_loss', '?'))
    print(f'=> Loaded checkpoint (epoch {epoch}, best_psnr {best_psnr})')

    # ---- Param count ----
    num_params = sum(p.numel() for p in model.parameters())
    print(f'=> Model params: {num_params:,} ({num_params / 1e6:.3f}M)')

    # ---- Evaluate ----
    model.eval()
    print('\n--- Evaluation started ---\n')

    psnr_total = 0.0
    ssim_total = 0.0
    time_total = 0.0
    n_metric_scenes = 0  # number of scenes that actually have GT

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # 支持既可以返回 (burst_noise, gt)，也可以只返回 burst_noise 的数据集
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                burst_noise, gt = batch
                has_gt = True
            else:
                # batch 可能是单个 tensor，或长度为 1 的 tuple/list
                burst_noise = batch[0] if isinstance(batch, (list, tuple)) else batch
                gt = None
                has_gt = False

            scene_id = scene_ids[i]
            t0 = time.time()

            if cuda:
                burst_noise = burst_noise.cuda(non_blocking=True)
                if gt is not None:
                    gt = gt.cuda(non_blocking=True)

            # burst_noise: (B, 9, 4, H, W) -> (B, 36, H, W)
            b, f, c, h, w = burst_noise.shape
            burst_noise = burst_noise.view(b, f * c, h, w)

            if sliding_window:
                pred = sliding_window_forward(
                    model,
                    burst_noise,
                    patch_size=patch_size,
                    stride=stride,
                    tta=tta,
                )
            else:
                pred = model_forward(model, burst_noise, tta=tta)
            pred = torch.clamp(pred, 0.0, 1.0)

            t1 = time.time()
            elapsed = t1 - t0
            time_total += elapsed

            # Compute metrics（如果有 GT）
            if gt is not None:
                psnr_val = calculate_psnr(pred.unsqueeze(1), gt.unsqueeze(1))
                ssim_val = calculate_ssim(pred.unsqueeze(1), gt.unsqueeze(1))
                psnr_total += psnr_val
                ssim_total += ssim_val
                n_metric_scenes += 1

                print(f'Scene-{scene_id:03d} | PSNR: {psnr_val:.2f} dB | '
                      f'SSIM: {ssim_val:.4f} | time: {elapsed:.2f}s')
            else:
                # 没有 GT，仅打印时间信息
                print(f'Scene-{scene_id:03d} | time: {elapsed:.2f}s (no GT)')

            # ---- Save output as 96-bit TIF (float32 x 3ch) ----
            # Data pipeline: cv2.imread(BGR) -> ToTensor(BGR CHW) -> model(BGR CHW)
            # So pred is BGR, which is exactly what cv2.imwrite expects.
            pred_np = pred[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
            out_path = os.path.join(img_dir, f'Scene-{scene_id:03d}-out.tif')
            cv2.imwrite(out_path, pred_np)

            if save_gt:
                gt_np = gt[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
                gt_path = os.path.join(img_dir, f'Scene-{scene_id:03d}-gt.tif')
                cv2.imwrite(gt_path, gt_np)

            if save_input:
                burst_cpu = burst_noise[0].cpu()  # (36, H, W)
                for frame_idx in range(9):
                    # Each frame is 4ch packed, save first channel as grayscale
                    frame = burst_cpu[frame_idx * 4]  # (H, W)
                    frame_np = frame.numpy().astype(np.float32)
                    inp_path = os.path.join(
                        img_dir, f'Scene-{scene_id:03d}-in-{frame_idx}.tif'
                    )
                    cv2.imwrite(inp_path, frame_np)

    n_scenes = len(val_loader)
    avg_psnr = psnr_total / n_metric_scenes if n_metric_scenes > 0 else None
    avg_ssim = ssim_total / n_metric_scenes if n_metric_scenes > 0 else None

    print(f'\n--- Results ---')
    if n_metric_scenes > 0:
        print(f'Average PSNR: {avg_psnr:.2f} dB (on {n_metric_scenes} scenes with GT)')
        print(f'Average SSIM: {avg_ssim:.4f} (on {n_metric_scenes} scenes with GT)')
    else:
        print('Average PSNR: N/A (no GT provided)')
        print('Average SSIM: N/A (no GT provided)')

    if n_scenes > 0:
        print(f'Total time:   {time_total:.2f}s ({time_total / n_scenes:.2f}s/scene)')
    else:
        print(f'Total time:   {time_total:.2f}s (no scenes)')
    print(f'Output saved: {img_dir}')

    # ---- Verify output format ----
    # 仅在存在至少一个 scene / 输出文件时才做检查
    if len(scene_ids) > 0:
        sample_out = os.path.join(img_dir, f'Scene-{scene_ids[0]:03d}-out.tif')
        if os.path.exists(sample_out):
            verify = cv2.imread(sample_out, cv2.IMREAD_UNCHANGED)
            if verify is not None:
                print(f'\n=> Output format check: shape={verify.shape}, '
                      f'dtype={verify.dtype}, '
                      f'bits={verify.dtype.itemsize * 8 * verify.shape[2]}bit '
                      f'({verify.dtype.itemsize * 8}bit x {verify.shape[2]}ch)')
                assert verify.dtype == np.float32, \
                    f'ERROR: expected float32, got {verify.dtype}'
                assert len(verify.shape) == 3 and verify.shape[2] == 3, \
                    f'ERROR: expected 3-channel, got shape {verify.shape}'
                print('=> Format OK: 96-bit depth (32bit x 3ch) float32 TIF')

    # ---- Pack all images into result.zip ----
    zip_path = os.path.join(checkpoint_dir, 'result.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
        for fname in sorted(os.listdir(img_dir)):
            fpath = os.path.join(img_dir, fname)
            if os.path.isfile(fpath):
                zf.write(fpath, fname)  # arcname=fname: flat structure in zip
    print(f'=> Packed {len(zf.namelist()) if False else "all"} images into: {zip_path}')
    # Print actual count
    with zipfile.ZipFile(zip_path, 'r') as zf:
        print(f'   {len(zf.namelist())} files in zip, '
              f'size: {os.path.getsize(zip_path) / 1e6:.1f} MB')

    return avg_psnr, avg_ssim


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate HDR Burst Reconstruction model')
    parser.add_argument('--model', type=str, required=True, choices=MODEL_NAMES,
                        help='Model name')
    parser.add_argument('--exp_name', type=str, default='default',
                        help='Experiment name (must match training exp_name)')
    parser.add_argument('--val_root', type=str,
                        default='/home/chen/data/ntire2026/hdr/test/',
                        help='Path to validation data')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='./checkpoint_dir',
                        help='Root directory containing checkpoint folders')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--mgpu', type=int, default=1)
    parser.add_argument('--save_gt', action='store_true',
                        help='Also save GT images')
    parser.add_argument('--save_input', action='store_true',
                        help='Also save input frame images')
    parser.add_argument('--tta', action='store_true',
                        help='Enable 8-fold D4 test-time augmentation (flip+rot90)')
    parser.add_argument('--rep', action='store_true',
                        help='Use reparameterized model (load model_best_rep.pth.tar)')
    parser.add_argument('--sliding_window', action='store_true',
                        help='Run inference on overlapping patches and blend them')
    parser.add_argument('--patch_size', type=int, nargs=2, metavar=('H', 'W'),
                        default=(192, 384),
                        help='Sliding window patch size as H W (default: 192 384)')
    parser.add_argument('--stride', type=int, nargs=2, metavar=('H', 'W'),
                        default=None,
                        help='Sliding window stride as H W (default: patch_size/2)')
    args = parser.parse_args()

    eval_model(
        model_name=args.model,
        exp_name=args.exp_name,
        val_root=args.val_root,
        checkpoint_dir_root=args.checkpoint_dir,
        cuda=bool(args.cuda),
        mgpu=bool(args.mgpu),
        save_gt=args.save_gt,
        save_input=args.save_input,
        tta=args.tta,
        rep=args.rep,
        sliding_window=args.sliding_window,
        patch_size=tuple(args.patch_size),
        stride=tuple(args.stride) if args.stride is not None else None,
    )
