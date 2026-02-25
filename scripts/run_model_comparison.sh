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
    --aug_exp_enable 0 \
    --aug_wb_enable 0 \
    --aug_noise_enable 0
}

# ==============================================================================
# 6 个模型逐一训练
# ==============================================================================

# # 1. Claude_5 — 基线 (DW-Sep + SE, 72ch, 8 blocks)
# run_one "safnet_claude_5" "model_cmp_claude5"

# # 2. Claude_6 — Large Kernel DW 7×7 (ConvNeXt style)
# run_one "safnet_claude_6" "model_cmp_claude6"

# # 3. Claude_7 — SimpleGate + SCA (NAFNet style)
# run_one "safnet_claude_7" "model_cmp_claude7"

# # 4. Claude_8 — Hybrid Conv + Transposed Attention (Restormer style)
# run_one "safnet_claude_8" "model_cmp_claude8"

# # 5. Claude_9 — FFT-Enhanced Block
# run_one "safnet_claude_9" "model_cmp_claude9"

# # 6. Claude_10 — Wider + DropPath + Dense Skip
# run_one "safnet_claude_10" "model_cmp_claude10"

# # 7. Claude_11 — MDTA + DropPath + Dense Skip (Claude_8 × Claude_10)
# run_one "safnet_claude_11" "model_cmp_claude11"

# # 8. Claude_12 — Iterative Flow Refinement (2-pass flow estimation)
# run_one "safnet_claude_12" "model_cmp_claude12"

# # 9. Claude_13 — Flow-Guided DCN Warp (deformable alignment)
# run_one "safnet_claude_13" "model_cmp_claude13"

# # 10. Claude_14 — Learned Attention HDR Merge
# run_one "safnet_claude_14" "model_cmp_claude14"

# # 11. Claude_15 — Joint Innovation (Flow + Warp + Merge)
# run_one "safnet_claude_15" "model_cmp_claude15"

# # 12. Claude_16 — 5-Frame Direct Flow (eliminates 0.67x approx)
# run_one "safnet_claude_16" "model_cmp_claude16"

# # 13. Claude_17 — 5-Frame 2-Pass + RefineNet 56ch
# run_one "safnet_claude_17" "model_cmp_claude17"

# # 14. Claude_18 — 5-Frame Flow + 5-Frame RefineNet
# run_one "safnet_claude_18" "model_cmp_claude18"

# # 15. Claude_19 — 5-Frame 2-Pass + Lite RefineNet 48ch
# run_one "safnet_claude_19" "model_cmp_claude19"

# # 16. Claude_20 — 5-Frame 2-Pass + UltraLite RefineNet 40ch
# run_one "safnet_claude_20" "model_cmp_claude20"

# 17. Claude_21 — 2-Pass Anchor Flow + RefineNet5Frame 72ch
run_one "safnet_claude_21" "model_cmp_claude21"

# 18. Claude_22 — 2-Pass Anchor Flow + Per-Frame LearnedMerge5Frame
run_one "safnet_claude_22" "model_cmp_claude22"

# 19. Claude_23 — 2-Pass Anchor Flow + Frame-Augmented RefineNet
run_one "safnet_claude_23" "model_cmp_claude23"

echo "============================================================"
echo "Model Comparison Finished. (19 models, MSE loss, crop+geo aug)"
echo ""
echo "Models compared:"
echo "  Claude_5:  baseline (DW-Sep + SE, 72ch, 8 blocks)"
echo "  Claude_6:  Large Kernel DW 7x7 (0.876M, 89.95G)"
echo "  Claude_7:  SimpleGate + SCA (0.940M, 98.8G)"
echo "  Claude_8:  Hybrid Conv + MDTA (0.851M, 88.67G)"
echo "  Claude_9:  FFT-Enhanced Block (0.803M, 70.44G)"
echo "  Claude_10: Wider + DropPath + Dense Skip (0.881M, 89.38G)"
echo "  Claude_11: MDTA + DropPath + Dense Skip (0.894M, 100.78G)"
echo "  Claude_12: Iterative Flow Refinement (0.830M, ~87.8G)"
echo "  Claude_13: Flow-Guided DCN Warp (~0.866M, ~78.1G)"
echo "  Claude_14: Learned Attention HDR Merge (~0.843M, ~80.3G)"
echo "  Claude_15: Joint Innovation (~0.879M, ~95.3G)"
echo "  Claude_16: 5-Frame Direct Flow (~0.879M, ~95G)"
echo "  Claude_17: 5-Frame 2-Pass + 56ch RefineNet (~0.76M, ~96G)"
echo "  Claude_18: 5-Frame + 5-Frame RefineNet (~0.80M, ~88G)"
echo "  Claude_19: 5-Frame 2-Pass + Lite RefineNet 48ch (to be measured)"
echo "  Claude_20: 5-Frame 2-Pass + UltraLite RefineNet 40ch (to be measured)"
echo "  Claude_21: 2-Pass Anchor + RefineNet5Frame 72ch (~0.84M, ~93G)"
echo "  Claude_22: 2-Pass Anchor + Per-Frame Merge 5Frame (~0.88M, ~96G)"
echo "  Claude_23: 2-Pass Anchor + Frame-Augmented RefineNet (~0.88M, ~96G)"
echo ""
echo "Check output_log/ for PSNR/SSIM curves and logs."
echo "============================================================"
