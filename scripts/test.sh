#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Full Test Pipeline: Reparam → Eval (rep + TTA + TLC)
# ==============================================================================
#
# Automates the complete inference workflow:
#   1. Reparameterize: fuse SRP branches, save fused weights
#   2. Evaluate: run full-image inference with fused model + TTA + TLC
#
# Usage:
#   bash scripts/test.sh <model_name> <exp_name> [tlc_train_h] [tlc_train_w]
#   bash scripts/test.sh safnet_claude_33_v3 model_submit_claude33_v3
#   bash scripts/test.sh safnet_claude_33_v3 model_submit_claude33_v3 128 128
#
# ==============================================================================

MODEL="${1:-safnet_claude_33_v3}"
EXP="${2:-model_submit_claude33_v3}"
TLC_TRAIN_H="${3:-128}"
TLC_TRAIN_W="${4:-128}"
CKPT_DIR="./checkpoint_dir/checkpoint_dir_${MODEL}_${EXP}"

echo "======================================================"
echo " Model:      ${MODEL}"
echo " Experiment: ${EXP}"
echo " Checkpoint: ${CKPT_DIR}"
echo " TLC train:  ${TLC_TRAIN_H}x${TLC_TRAIN_W}"
echo "======================================================"

# ------------------------------------------------------------------
# Step 1: Reparameterize (fuse SRP branches → model_best_rep.pth.tar)
# ------------------------------------------------------------------
echo ""
echo ">>> Step 1/2: Reparameterization"
echo "------------------------------------------------------"

python reparam_model.py --model "${MODEL}" --exp_name "${EXP}"

# ------------------------------------------------------------------
# Step 2: Evaluate with fused model + TTA + TLC
# ------------------------------------------------------------------
echo ""
echo ">>> Step 2/2: Evaluation (rep + TTA + TLC full-image)"
echo "------------------------------------------------------"

python eval.py \
    --model "${MODEL}" \
    --exp_name "${EXP}" \
    --tta \
    --rep \
    --tlc \
    --tlc_train_size "${TLC_TRAIN_H}" "${TLC_TRAIN_W}"
echo ""
echo "======================================================"
echo " Pipeline complete!"
echo " Outputs:"
echo "   Fused weights: ${CKPT_DIR}/model_best_rep.pth.tar"
echo "   Images:        ${CKPT_DIR}/img/"
echo "   Result zip:    ${CKPT_DIR}/result.zip"
echo "======================================================"
