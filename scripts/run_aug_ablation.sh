#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# NTIRE 2026 HDR Burst Reconstruction - Data Augmentation Ablation Study
# ==============================================================================
# 
# This script runs a series of training experiments to find the optimal 
# data augmentation strategy for the SAFNet baseline.
#
# Usage:
#   bash scripts/run_aug_ablation.sh [model_name]
#
# Example:
#   bash scripts/run_aug_ablation.sh safnet
# ==============================================================================

MODEL="${1:-safnet}"

# Data Paths
TRAIN_ROOT="${TRAIN_ROOT:-/home/chen/data/ntire2026/hdr/train/}"
VAL_ROOT="${VAL_ROOT:-/home/chen/data/ntire2026/hdr/validation/}"

# Training Hyperparameters (Ensure these remain constant for all runs)
EPOCHS="${EPOCHS:-300}"         # Recommend starting with 100 for fast ablation
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-2e-4}"
LR_DECAY="${LR_DECAY:-0.95}"
CUDA="${CUDA:-1}"
MGPU="${MGPU:-1}"
RESTART_TRAIN="${RESTART_TRAIN:-1}"

mkdir -p output_log

run_one () {
  local exp_name="$1"; shift
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"

  echo "------------------------------------------------------------"
  echo "[${ts}] STARTING EXPERIMENT: ${exp_name}"
  echo "Model: ${MODEL} | Args: $*"
  echo "------------------------------------------------------------"

  # Standard training call
  python train.py \
    --model "${MODEL}" \
    --exp_name "${exp_name}" \
    --train_root "${TRAIN_ROOT}" \
    --val_root "${VAL_ROOT}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --lr_decay "${LR_DECAY}" \
    --cuda "${CUDA}" \
    --mgpu "${MGPU}" \
    --restart_train "${RESTART_TRAIN}" \
    "$@"
}

# ------------------------------------------------------------------------------
# 1. Baseline: No Augmentation
# ------------------------------------------------------------------------------
# Pure baseline to evaluate the model's performance on the original data distribution.
# run_one "baseline_none" \
#   --aug_enable 0

# ------------------------------------------------------------------------------
# 2. Geometric Diversity (Safe & Expected to help)
# ------------------------------------------------------------------------------
# Flip and 90-degree rotations are standard for image restoration and safe for Bayer.
# run_one "aug_geo_only" \
#   --aug_enable 1 \
#   --aug_crop_enable 0 \
#   --aug_geo_enable 1 \
#   --aug_geo_flip_enable 1 \
#   --aug_geo_rot90_enable 1

# ------------------------------------------------------------------------------
# 3. Physics-Aware Photometric Diversity (Global Intensity)
# ------------------------------------------------------------------------------
# Applying the SAME gain to all frames preserves the burst's exposure ratios.
# run_one "aug_ext_global_exp" \
#   --aug_enable 1 \
#   --aug_crop_enable 0 \
#   --aug_exp_enable 1 \
#   --aug_exp_global 1 \
#   --aug_exp_low_min 0.8 --aug_exp_low_max 1.2 \
#   --aug_exp_mid_min 0.8 --aug_exp_mid_max 1.2 \
#   --aug_exp_high_min 0.8 --aug_exp_high_max 1.2

# ------------------------------------------------------------------------------
# 4. Scale & Context Diversity (Large Scale Crops)
# ------------------------------------------------------------------------------
# Using larger crops to ensure the model sees enough context for optical flow.
# Note: Packed input size is 384x768.
run_one "aug_large_crop" \
  --aug_enable 1 \
  --aug_crop_enable 1 \
  --aug_crop_sizes "192x384,256x512,384x768" \
  --aug_geo_enable 0 \
  --aug_exp_enable 0

# ------------------------------------------------------------------------------
# 5. Combined Strategy (The "Optimized" Config)
# ------------------------------------------------------------------------------
# Combining Geometric, Global Exposure, and Large crops.
# run_one "aug_combined_v1" \
#   --aug_enable 1 \
#   --aug_crop_enable 1 \
#   --aug_crop_sizes "192x384,256x512,384x768" \
#   --aug_geo_enable 1 \
#   --aug_geo_flip_enable 1 \
#   --aug_geo_rot90_enable 1 \
#   --aug_exp_enable 1 \
#   --aug_exp_global 1 \
#   --aug_exp_low_min 0.8 --aug_exp_low_max 1.2 \
#   --aug_exp_mid_min 0.8 --aug_exp_mid_max 1.2 \
#   --aug_exp_high_min 0.8 --aug_exp_high_max 1.2

echo "============================================================"
echo "Ablation Study Finished."
echo "Check output_log/ for PSNR/SSIM curves and logs."
echo "============================================================"
