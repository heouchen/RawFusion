#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# NTIRE 2026 HDR Burst Reconstruction - Model Architecture Comparison
# ==============================================================================
#
# 对比 SAFNet_Claude_5 ~ Claude_10 六个模型的性能差异。
# 固定条件：
#   - Loss: MSE
#   - 增广: crop (多尺度) + geometric (flip + rot90)
#   - 其它增广关闭 (exp / wb / noise)
#   - 训练超参统一
#
# Usage:
#   bash scripts/run_model_comparison.sh
#
# 可选环境变量覆盖默认值:
#   EPOCHS=200 BATCH_SIZE=1 CUDA=0 bash scripts/run_model_comparison.sh
# ==============================================================================

# Data Paths
TRAIN_ROOT="${TRAIN_ROOT:-/home/chen/data/ntire2026/hdr/train/}"
VAL_ROOT="${VAL_ROOT:-/home/chen/data/ntire2026/hdr/validation/}"

# Training Hyperparameters (所有实验保持一致)
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-2e-4}"
LR_DECAY="${LR_DECAY:-0.95}"
CUDA="${CUDA:-1}"
MGPU="${MGPU:-1}"
RESTART_TRAIN="${RESTART_TRAIN:-1}"

# 固定增广策略（crop + geo）
# Baseline: 随机多尺度 crop
CROP_SIZES="96x192,192x384,384x768"
PROGRESSIVE_CROP_ENABLE="${PROGRESSIVE_CROP_ENABLE:-0}"
PROGRESSIVE_CROP_SCHEDULE="${PROGRESSIVE_CROP_SCHEDULE:-96x192@0.3,192x384@0.7,384x768@1.0}"
PROGRESSIVE_BATCH_ENABLE="${PROGRESSIVE_BATCH_ENABLE:-0}"
PROGRESSIVE_BATCH_SIZES="${PROGRESSIVE_BATCH_SIZES:-96x192@16,192x384@8,384x768@4}"

# 固定损失函数
LOSS="mse"

mkdir -p output_log
EMA_ENABLE="0"
pretrained=""

# echo "开始执行..."
# sleep 3h
# echo "3小时后继续执行..."

run_one () {
  local model_name="$1"
  local exp_name="$2"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"

  echo "------------------------------------------------------------"
  echo "[${ts}] STARTING: ${exp_name}"
  echo "Model: ${model_name} | Loss: ${LOSS}"
  echo "Crop mode: $([[ \"${PROGRESSIVE_CROP_ENABLE}\" == \"1\" ]] && echo progressive || echo random)"
  echo "Batch mode: $([[ \"${PROGRESSIVE_BATCH_ENABLE}\" == \"1\" ]] && echo dynamic || echo fixed)"
  echo "------------------------------------------------------------"

  python train.py \
    --model "${model_name}" \
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
    --loss "${LOSS}" \
    --aug_enable 1 \
    --aug_crop_enable 1 \
    --aug_crop_sizes "${CROP_SIZES}" \
    --aug_progressive_crop_enable "${PROGRESSIVE_CROP_ENABLE}" \
    --aug_progressive_crop_schedule "${PROGRESSIVE_CROP_SCHEDULE}" \
    --aug_progressive_batch_enable "${PROGRESSIVE_BATCH_ENABLE}" \
    --aug_progressive_batch_sizes "${PROGRESSIVE_BATCH_SIZES}" \
    --aug_geo_enable 1 \
    --aug_geo_flip_enable 1 \
    --aug_geo_rot90_enable 1 \
    --aug_exp_enable 0 \
    --aug_wb_enable 0 \
    --aug_noise_enable 0 \
    --ema "${EMA_ENABLE}" \
    --pretrained "${pretrained}"
}
# Curriculum example:
# PROGRESSIVE_CROP_ENABLE=1 \
# PROGRESSIVE_CROP_SCHEDULE="96x192@0.3,192x384@0.7,384x768@1.0" \
# PROGRESSIVE_BATCH_ENABLE=1 \
# PROGRESSIVE_BATCH_SIZES="96x192@16,192x384@8,384x768@4" \
# bash scripts/run.sh

# run_one "safnet_claude_27_v2" "model_submit_claude27_v2"
# run_one "safnet_claude_29" "model_submit_claude29"
# run_one "safnet_claude_30" "model_submit_claude30"
# run_one "safnet_claude_31" "model_submit_claude31"
# run_one "safnet_claude_32" "model_submit_claude32"
# run_one "safnet_claude_33" "model_submit_claude33"
# run_one "safnet_claude_34" "model_submit_claude34"
# run_one "safnet_claude_35" "model_submit_claude35"
# run_one "safnet_claude_36" "model_submit_claude36"
# run_one "safnet_claude_37" "model_submit_claude37"
# run_one "safnet_claude_38" "model_submit_claude38"
# run_one "safnet_claude_39" "model_submit_claude39"
# run_one "safnet_claude_40" "model_submit_claude40"
# run_one "safnet_claude_41" "model_submit_claude41"

run_one "unet" "model_submit_unet_mse"

EMA_ENABLE='1'
run_one "unet" "model_submit_unet_ema"
EMA_ENABLE='0'

LOSS="l1"
run_one "unet" "model_submit_unet_l1"
LOSS="mse"

PROGRESSIVE_CROP_ENABLE=1 \
PROGRESSIVE_CROP_SCHEDULE="96x192@0.3,192x384@0.7,384x768@1.0" \
run_one "unet" "model_submit_unet_mse_progressive"
