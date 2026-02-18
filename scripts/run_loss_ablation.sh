#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# NTIRE 2026 HDR Burst Reconstruction - Loss Function Ablation Study
# ==============================================================================
#
# 设计原则：
#   1. 固定数据增广（crop+geo）、模型、训练超参，只改变损失函数
#   2. 分 3 组：gamma 域基础 loss → mu-law 域 loss → 复合 loss
#   3. 基于 val PSNR 选 best checkpoint，消除 train loss 偏差
#
# 对比的 7 个损失函数：
#   第 1 组 (gamma 域):    mse, l1, charbonnier
#   第 2 组 (mu-law 域):   mulaw_l1, mulaw_charb
#   第 3 组 (复合 loss):   mulaw_l1_perceptual, mulaw_l1_fft
#
# Usage:
#   bash scripts/run_loss_ablation.sh [model_name]
#
# Example:
#   bash scripts/run_loss_ablation.sh safnet_claude_5_pconv
# ==============================================================================

MODEL="${1:-safnet}"

# Data Paths
TRAIN_ROOT="${TRAIN_ROOT:-/home/chen/data/ntire2026/hdr/train/}"
VAL_ROOT="${VAL_ROOT:-/home/chen/data/ntire2026/hdr/validation/}"

# Training Hyperparameters (所有实验保持一致)
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-2e-4}"
CUDA="${CUDA:-1}"
MGPU="${MGPU:-1}"
RESTART_TRAIN="${RESTART_TRAIN:-1}"

# 固定增广策略（crop + geo，消融实验的最优基线）
AUG_ARGS=(
  --aug_enable 1
  --aug_crop_enable 1
  --aug_crop_sizes "96x192,192x384,384x768"
  --aug_geo_enable 1
  --aug_geo_flip_enable 1
  --aug_geo_rot90_enable 1
  --aug_exp_enable 0
  --aug_wb_enable 0
  --aug_noise_enable 0
)

mkdir -p output_log

run_one () {
  local exp_name="$1"
  local loss_name="$2"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"

  echo "------------------------------------------------------------"
  echo "[${ts}] STARTING EXPERIMENT: ${exp_name}"
  echo "Model: ${MODEL} | Loss: ${loss_name}"
  echo "------------------------------------------------------------"

  python train.py \
    --model "${MODEL}" \
    --exp_name "${exp_name}" \
    --train_root "${TRAIN_ROOT}" \
    --val_root "${VAL_ROOT}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --cuda "${CUDA}" \
    --mgpu "${MGPU}" \
    --restart_train "${RESTART_TRAIN}" \
    --loss "${loss_name}" \
    "${AUG_ARGS[@]}"
}

# ==============================================================================
# 第 1 组：Gamma 域基础损失函数
# ==============================================================================
# 对比不同的基础度量在 gamma 域的效果。
# 预期：MSE < L1 ≈ Charbonnier（MSE 对离群值敏感，过度惩罚亮区）

# 1. MSE (L2) — 原始 baseline
run_one "loss_01_mse" "mse"

# 2. L1 — 更鲁棒的像素级 loss
run_one "loss_02_l1" "l1"

# 3. Charbonnier — smooth L1，零点附近可导，梯度更稳定
run_one "loss_03_charbonnier" "charbonnier"

# ==============================================================================
# 第 2 组：mu-law tonemapped 域损失函数
# ==============================================================================
# mu-law 压缩动态范围后再计算 loss，让暗部和亮部同等重要。
# 预期：mu-law 域 >> gamma 域（参考文献报告 +2~9 dB 提升）

# 4. mu-law L1 — SAFNet 标准 loss (mu=5000)
run_one "loss_04_mulaw_l1" "mulaw_l1"

# 5. mu-law Charbonnier — 在 mu-law 域使用 smooth L1
run_one "loss_05_mulaw_charb" "mulaw_charb"

# ==============================================================================
# 第 3 组：复合损失函数
# ==============================================================================
# 在 mu-law L1 基础上叠加辅助 loss，提升纹理/高频细节。
# 预期：小幅改善（+0.1~0.3 dB），但训练开销增加

# 6. mu-law L1 + VGG Perceptual (w=0.01) — SAFNet 完整 loss
run_one "loss_06_mulaw_l1_perceptual" "mulaw_l1_perceptual"

# 7. mu-law L1 + FFT (w=0.1) — 频域约束，保留高频细节
run_one "loss_07_mulaw_l1_fft" "mulaw_l1_fft"

echo "============================================================"
echo "Loss Ablation Study Finished. (${MODEL}, 7 experiments)"
echo ""
echo "预期结果："
echo "  第 1 组 (gamma 域):   MSE < L1 ≈ Charbonnier"
echo "  第 2 组 (mu-law 域):  mulaw_l1 >> L1, mulaw_charb ≈ mulaw_l1"
echo "  第 3 组 (复合 loss):  mulaw_l1_perceptual ≥ mulaw_l1"
echo ""
echo "Check output_log/ for PSNR/SSIM curves and logs."
echo "============================================================"
