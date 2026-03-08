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
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-2e-4}"
LR_DECAY="${LR_DECAY:-0.95}"
CUDA="${CUDA:-1}"
MGPU="${MGPU:-1}"
RESTART_TRAIN="${RESTART_TRAIN:-1}"

# 固定增广策略（crop + geo）
CROP_SIZES="96x192,192x384,384x768"

# 固定损失函数
LOSS="mse"
EMA_ENABLE="0"
mkdir -p output_log

run_one () {
  local model_name="$1"
  local exp_name="$2"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  echo "------------------------------------------------------------"
  echo "[${ts}] STARTING: ${exp_name}"
  echo "Model: ${model_name} | Loss: ${LOSS}"
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
    --aug_geo_enable 1 \
    --aug_geo_flip_enable 1 \
    --aug_geo_rot90_enable 1 \
    --ema "${EMA_ENABLE}"
}
run_one "safnet_claude_27_v2" "model_cmp_claude27_v2_1"

CROP_SIZES="192x384,384x768"
run_one "safnet_claude_27_v2" "model_cmp_claude27_v2_2"

EMA_ENABLE="1"
CROP_SIZES="96x192,192x384,384x768"
run_one "safnet_claude_27_v2" "model_cmp_claude27_v2_3"

CROP_SIZES="192x384,384x768"
run_one "safnet_claude_27_v2" "model_cmp_claude27_v2_4"

# ==============================================================================
# Claude_29~32: Optimized training recipe for larger models
# - LR halved (1e-4) for larger param count
# - Epochs 500 for longer convergence
# - EMA enabled
# - Crop sizes 192x384,384x768 (skip 96x192 for these models)
# ==============================================================================
EPOCHS="500"
LR="1e-4"
EMA_ENABLE="1"
CROP_SIZES="192x384,384x768"

run_one "safnet_claude_29" "model_cmp_claude29"
run_one "safnet_claude_30" "model_cmp_claude30"
run_one "safnet_claude_31" "model_cmp_claude31"
run_one "safnet_claude_32" "model_cmp_claude32"
run_one "safnet_claude_33" "model_cmp_claude33"
