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
#   - 训练超参统一
#
# Usage:
#   bash scripts/run_model_comparison.sh
#
# 可选环境变量覆盖默认值:
#   EPOCHS=200 BATCH_SIZE=1 CUDA=0 bash scripts/run_model_comparison.sh
#   bash scripts/run.sh --gpu 0 --num_workers 16 --val_every 10 --cudnn_benchmark 1 --compile 1
# ==============================================================================

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -gpu|--gpu)
      export CUDA_VISIBLE_DEVICES="$2"
      export CUDA=1
      # If multiple GPUs are specified (e.g., "0,1"), enable mgpu; otherwise disable it.
      if [[ "$2" == *","* ]]; then
        export MGPU=1
      else
        export MGPU=0
      fi
      echo "Using GPU(s): $2 (CUDA_VISIBLE_DEVICES=$2, CUDA=$CUDA, MGPU=$MGPU)"
      shift 2
      ;;
    --num_workers)
      export NUM_WORKERS="$2"
      shift 2
      ;;
    --val_every)
      export VAL_EVERY="$2"
      shift 2
      ;;
    --cudnn_benchmark)
      export CUDNN_BENCHMARK="$2"
      shift 2
      ;;
    --compile)
      export COMPILE="$2"
      shift 2
      ;;
    *)
      # Ignore other arguments
      shift
      ;;
  esac
done

# Data Paths
TRAIN_ROOT="${TRAIN_ROOT:-/home/chen/data/ntire2026/hdr/train/}"
VAL_ROOT="${VAL_ROOT:-/home/chen/data/ntire2026/hdr/validation/}"

# Training Hyperparameters (所有实验保持一致)
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-5e-5}"
LR_DECAY="${LR_DECAY:-0.95}"
CUDA="${CUDA:-1}"
MGPU="${MGPU:-0}"
RESTART_TRAIN="${RESTART_TRAIN:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_EVERY="${VAL_EVERY:-1}"
CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-0}"
COMPILE="${COMPILE:-0}"

# 固定增广策略（crop + geo）
# Baseline: 随机多尺度 crop
CROP_SIZES="96x192,192x384,384x768"
PROGRESSIVE_CROP_ENABLE="${PROGRESSIVE_CROP_ENABLE:-0}"
PROGRESSIVE_CROP_SCHEDULE="${PROGRESSIVE_CROP_SCHEDULE:-96x192@0.3,192x384@0.7,384x768@1.0}"
PROGRESSIVE_BATCH_ENABLE="${PROGRESSIVE_BATCH_ENABLE:-0}"
PROGRESSIVE_BATCH_SIZES="${PROGRESSIVE_BATCH_SIZES:-96x192@16,192x384@8,384x768@4}"

# 固定损失函数
LOSS="mse"

# 一致性约束
CONSIST_ENABLE="${CONSIST_ENABLE:-0}"
CONSIST_SIZES="${CONSIST_SIZES:-96x192,192x384,384x768}"
CONSIST_WEIGHT="${CONSIST_WEIGHT:-0.1}"

mkdir -p output_log
EMA_ENABLE="0"
pretrained=""


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
  echo "Runtime: workers=${NUM_WORKERS} | val_every=${VAL_EVERY} | cudnn_benchmark=${CUDNN_BENCHMARK} | compile=${COMPILE} | mgpu=${MGPU}"
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
    --num_workers "${NUM_WORKERS}" \
    --cuda "${CUDA}" \
    --mgpu "${MGPU}" \
    --val_every "${VAL_EVERY}" \
    --cudnn_benchmark "${CUDNN_BENCHMARK}" \
    --compile "${COMPILE}" \
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
    --ema "${EMA_ENABLE}" \
    --pretrained "${pretrained}" \
    --consist_enable "${CONSIST_ENABLE}" \
    --consist_sizes "${CONSIST_SIZES}" \
    --consist_weight "${CONSIST_WEIGHT}"
}
# Curriculum example:
# PROGRESSIVE_CROP_ENABLE=1 \
# PROGRESSIVE_CROP_SCHEDULE="96x192@0.3,192x384@0.7,384x768@1.0" \
# PROGRESSIVE_BATCH_ENABLE=1 \
# PROGRESSIVE_BATCH_SIZES="96x192@16,192x384@8,384x768@4" \
# bash scripts/run.sh

# run_one "safnet_claude_27" "model_submit_claude27"
# run_one "safnet_claude_29" "model_submit_claude29"
# run_one "safnet_claude_30" "model_submit_claude30"
# run_one "safnet_claude_31" "model_submit_claude31"
# run_one "safnet_claude_32" "model_submit_claude32"
# run_one "safnet_claude_33" "model_submit_claude33"
pretrained="/home/chen/work/RawFusion/checkpoint_dir/checkpoint_dir_safnet_claude_33_v2_model_submit_claude33_v2/model_best.pth.tar"
run_one "safnet_claude_33_v2" "model_submit_claude33_v2"
# run_one "safnet_claude_34" "model_submit_claude34"
# run_one "safnet_claude_35" "model_submit_claude35"
# run_one "safnet_claude_35_v2" "model_submit_claude35_v2"
# run_one "safnet_claude_36" "model_submit_claude36"
# run_one "safnet_claude_37" "model_submit_claude37"
# run_one "safnet_claude_38" "model_submit_claude38"
# run_one "safnet_claude_39" "model_submit_claude39"
# run_one "safnet_claude_40" "model_submit_claude40"
# run_one "safnet_claude_40_v2" "model_submit_claude40_v2"
# run_one "safnet_claude_41" "model_submit_claude41"
# run_one "safnet_claude_42" "model_submit_claude42"
# run_one "safnet_claude_43" "model_submit_claude43"
# run_one "safnet_claude_44" "model_submit_claude44"
# run_one "safnet_claude_45" "model_submit_claude45"
# run_one "safnet_claude_46" "model_submit_claude46"
# run_one "safnet_claude_47" "model_submit_claude47"
# run_one "safnet_claude_48" "model_submit_claude48"

# run_one "safnet_claude_50" "model_submit_claude50"
# run_one "safnet_claude_50_v2" "model_submit_claude50_v2"
# run_one "safnet_claude_51" "model_submit_claude51"
# run_one "safnet_claude_52" "model_submit_claude52"
# run_one "safnet_claude_53" "model_submit_claude53"
# run_one "safnet_claude_54" "model_submit_claude54"

# CONSIST_ENABLE="${CONSIST_ENABLE:-1}" \
# CROP_SIZES="192x384,384x768" \
# CONSIST_SIZES="192x384" \
# CONSIST_WEIGHT="1" \
# run_one "safnet_claude_40" "model_submit_claude40_consist"

# PROGRESSIVE_CROP_ENABLE=1 \
# PROGRESSIVE_CROP_SCHEDULE="96x192@0.3,192x384@0.7,384x768@1.0" \
# PROGRESSIVE_BATCH_ENABLE=1 \
# PROGRESSIVE_BATCH_SIZES="96x192@16,192x384@8,384x768@4" \
# run_one "unet" "model_submit_unet_mse_progressive"
