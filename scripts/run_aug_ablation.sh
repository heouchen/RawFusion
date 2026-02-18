#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# NTIRE 2026 HDR Burst Reconstruction - Data Augmentation Ablation Study
# ==============================================================================
#
# 设计原则：
#   1. 所有实验固定 crop 基线（多尺度裁剪），只改变一个增广因子
#   2. 每个实验仅新增/修改一个变量，确保可归因
#   3. 先测单因子效果，再测最优组合
#
# Usage:
#   bash scripts/run_aug_ablation.sh [model_name]
#
# Example:
#   bash scripts/run_aug_ablation.sh safnet_claude_5_pconv
# ==============================================================================

MODEL="${1:-safnet}"

# Data Paths
TRAIN_ROOT="${TRAIN_ROOT:-/home/chen/data/ntire2026/hdr/train/}"
VAL_ROOT="${VAL_ROOT:-/home/chen/data/ntire2026/hdr/validation/}"

# Training Hyperparameters (所有实验保持一致)
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-2e-4}"
LR_DECAY="${LR_DECAY:-0.95}"
CUDA="${CUDA:-1}"
MGPU="${MGPU:-1}"
RESTART_TRAIN="${RESTART_TRAIN:-1}"

# 固定的 crop 基线（所有实验共享）
CROP_SIZES="96x192,192x384,384x768"

mkdir -p output_log

run_one () {
  local exp_name="$1"; shift
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"

  echo "------------------------------------------------------------"
  echo "[${ts}] STARTING EXPERIMENT: ${exp_name}"
  echo "Model: ${MODEL} | Args: $*"
  echo "------------------------------------------------------------"

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

# ==============================================================================
# 第 1 组：基线
# ==============================================================================

# 1. 纯基线：无任何增广
run_one "abl_01_no_aug" \
  --aug_enable 0

# 2. Crop-only 基线：后续所有实验在此基础上叠加
run_one "abl_02_crop_only" \
  --aug_enable 1 \
  --aug_crop_enable 1 \
  --aug_crop_sizes "${CROP_SIZES}" \
  --aug_geo_enable 0 \
  --aug_exp_enable 0 \
  --aug_wb_enable 0 \
  --aug_noise_enable 0

# ==============================================================================
# 第 2 组：单因子消融（在 crop 基线上 +1 因子）
# ==============================================================================

# 3. + Geometric (flip + rot90)
run_one "abl_03_crop_geo" \
  --aug_enable 1 \
  --aug_crop_enable 1 \
  --aug_crop_sizes "${CROP_SIZES}" \
  --aug_geo_enable 1 \
  --aug_geo_flip_enable 1 \
  --aug_geo_rot90_enable 1 \
  --aug_exp_enable 0 \
  --aug_wb_enable 0 \
  --aug_noise_enable 0

# 4. + Global Exposure (所有帧 + GT 同步缩放，模拟场景亮度变化)
run_one "abl_04_crop_exp_global" \
  --aug_enable 1 \
  --aug_crop_enable 1 \
  --aug_crop_sizes "${CROP_SIZES}" \
  --aug_geo_enable 0 \
  --aug_exp_enable 1 \
  --aug_exp_global 1 \
  --aug_exp_mid_min 0.9 --aug_exp_mid_max 1.1 \
  --aug_wb_enable 0 \
  --aug_noise_enable 0

# 5. + Per-frame Exposure (每帧独立增益，模拟帧间曝光抖动)
run_one "abl_05_crop_exp_perframe" \
  --aug_enable 1 \
  --aug_crop_enable 1 \
  --aug_crop_sizes "${CROP_SIZES}" \
  --aug_geo_enable 0 \
  --aug_exp_enable 1 \
  --aug_exp_global 0 \
  --aug_exp_low_min 0.9 --aug_exp_low_max 1.1 \
  --aug_exp_mid_min 0.9 --aug_exp_mid_max 1.1 \
  --aug_exp_high_min 0.9 --aug_exp_high_max 1.1 \
  --aug_wb_enable 0 \
  --aug_noise_enable 0

# 6. + WB Jitter (白平衡扰动)
run_one "abl_06_crop_wb" \
  --aug_enable 1 \
  --aug_crop_enable 1 \
  --aug_crop_sizes "${CROP_SIZES}" \
  --aug_geo_enable 0 \
  --aug_exp_enable 0 \
  --aug_wb_enable 1 \
  --aug_wb_gain_delta 0.05 \
  --aug_noise_enable 0

# ==============================================================================
# 第 3 组：最优组合（基于单因子结果选择有效因子组合）
# ==============================================================================

# 7. Crop + Geo + Global Exp
run_one "abl_07_crop_geo_exp_global" \
  --aug_enable 1 \
  --aug_crop_enable 1 \
  --aug_crop_sizes "${CROP_SIZES}" \
  --aug_geo_enable 1 \
  --aug_geo_flip_enable 1 \
  --aug_geo_rot90_enable 1 \
  --aug_exp_enable 1 \
  --aug_exp_global 1 \
  --aug_exp_mid_min 0.9 --aug_exp_mid_max 1.1 \
  --aug_wb_enable 0 \
  --aug_noise_enable 0

# 8. Full: Crop + Geo + Global Exp + WB
run_one "abl_08_full" \
  --aug_enable 1 \
  --aug_crop_enable 1 \
  --aug_crop_sizes "${CROP_SIZES}" \
  --aug_geo_enable 1 \
  --aug_geo_flip_enable 1 \
  --aug_geo_rot90_enable 1 \
  --aug_exp_enable 1 \
  --aug_exp_global 1 \
  --aug_exp_mid_min 0.9 --aug_exp_mid_max 1.1 \
  --aug_wb_enable 1 \
  --aug_wb_gain_delta 0.05 \
  --aug_noise_enable 0

echo "============================================================"
echo "Ablation Study Finished. (${MODEL}, 8 experiments)"
echo "Check output_log/ for PSNR/SSIM curves and logs."
echo "============================================================"
