#!/usr/bin/env bash
set -euo pipefail

# Main RawFusion training launcher.
#
# Examples:
#   bash scripts/run.sh --gpu 0 --num_workers 4 --val_every 1
#   MODEL=rawnet EXP_NAME=model_submit_rawnet bash scripts/run.sh --gpu 0
#   PRETRAINED=/path/to/model_best.pth.tar EPOCHS=1 LR=1e-6 bash scripts/run.sh

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -gpu|--gpu)
      export CUDA_VISIBLE_DEVICES="$2"
      CUDA=1
      if [[ "$2" == *","* ]]; then
        MGPU=1
      else
        MGPU=0
      fi
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --exp_name)
      EXP_NAME="$2"
      shift 2
      ;;
    --num_workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --val_num_workers)
      VAL_NUM_WORKERS="$2"
      shift 2
      ;;
    --val_every)
      VAL_EVERY="$2"
      shift 2
      ;;
    --cudnn_benchmark)
      CUDNN_BENCHMARK="$2"
      shift 2
      ;;
    --compile)
      COMPILE="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

MODEL="${MODEL:-safnet_claude_33_v3}"
EXP_NAME="${EXP_NAME:-model_submit_claude33_v3}"

TRAIN_ROOT="${TRAIN_ROOT:-/home/chen/data/ntire2026/hdr/train/}"
VAL_ROOT="${VAL_ROOT:-/home/chen/data/ntire2026/hdr/validation/}"

EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-2e-4}"
LR_DECAY="${LR_DECAY:-0.95}"
LOSS="${LOSS:-mse}"
RESTART_TRAIN="${RESTART_TRAIN:-1}"
CUDA="${CUDA:-1}"
MGPU="${MGPU:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-0}"
VAL_EVERY="${VAL_EVERY:-1}"
CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-0}"
COMPILE="${COMPILE:-0}"
PRETRAINED="${PRETRAINED:-}"

CROP_SIZES="${CROP_SIZES:-128x128}"
PROGRESSIVE_CROP_ENABLE="${PROGRESSIVE_CROP_ENABLE:-0}"
PROGRESSIVE_CROP_SCHEDULE="${PROGRESSIVE_CROP_SCHEDULE:-96x192@0.3,192x384@0.7,384x768@1.0}"
PROGRESSIVE_BATCH_ENABLE="${PROGRESSIVE_BATCH_ENABLE:-0}"
PROGRESSIVE_BATCH_SIZES="${PROGRESSIVE_BATCH_SIZES:-96x192@16,192x384@8,384x768@4}"

EMA_ENABLE="${EMA_ENABLE:-0}"
CONSIST_ENABLE="${CONSIST_ENABLE:-0}"
CONSIST_SIZES="${CONSIST_SIZES:-96x192,192x384,384x768}"
CONSIST_WEIGHT="${CONSIST_WEIGHT:-0.1}"

mkdir -p output_log

echo "------------------------------------------------------------"
echo "Model: ${MODEL}"
echo "Experiment: ${EXP_NAME}"
echo "Train root: ${TRAIN_ROOT}"
echo "Val root: ${VAL_ROOT}"
echo "Runtime: cuda=${CUDA} mgpu=${MGPU} workers=${NUM_WORKERS} val_workers=${VAL_NUM_WORKERS}"
echo "Train: epochs=${EPOCHS} batch=${BATCH_SIZE} lr=${LR} loss=${LOSS} crop=${CROP_SIZES}"
if [[ -n "${PRETRAINED}" ]]; then
  echo "Pretrained: ${PRETRAINED}"
fi
echo "------------------------------------------------------------"

CMD=(
  python train.py
  --model "${MODEL}"
  --exp_name "${EXP_NAME}"
  --train_root "${TRAIN_ROOT}"
  --val_root "${VAL_ROOT}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --lr "${LR}"
  --lr_decay "${LR_DECAY}"
  --num_workers "${NUM_WORKERS}"
  --cuda "${CUDA}"
  --mgpu "${MGPU}"
  --val_every "${VAL_EVERY}"
  --cudnn_benchmark "${CUDNN_BENCHMARK}"
  --compile "${COMPILE}"
  --val_num_workers "${VAL_NUM_WORKERS}"
  --restart_train "${RESTART_TRAIN}"
  --loss "${LOSS}"
  --aug_enable 1
  --aug_crop_enable 1
  --aug_crop_sizes "${CROP_SIZES}"
  --aug_progressive_crop_enable "${PROGRESSIVE_CROP_ENABLE}"
  --aug_progressive_crop_schedule "${PROGRESSIVE_CROP_SCHEDULE}"
  --aug_progressive_batch_enable "${PROGRESSIVE_BATCH_ENABLE}"
  --aug_progressive_batch_sizes "${PROGRESSIVE_BATCH_SIZES}"
  --aug_geo_enable 1
  --aug_geo_flip_enable 1
  --aug_geo_rot90_enable 1
  --ema "${EMA_ENABLE}"
  --consist_enable "${CONSIST_ENABLE}"
  --consist_sizes "${CONSIST_SIZES}"
  --consist_weight "${CONSIST_WEIGHT}"
)

if [[ -n "${PRETRAINED}" ]]; then
  CMD+=(--pretrained "${PRETRAINED}")
fi

CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"
