#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Full Test Pipeline: Reparam → Eval
# ==============================================================================
#
# Automates the complete inference workflow:
#   1. Reparameterize: fuse SRP branches, save fused weights
#   2. Evaluate: run inference with fused model + TTA
#
# Usage:
#   bash scripts/test.sh <model_name> <exp_name>
#   bash scripts/test.sh safnet_claude_33 model_submit_claude33
#   bash scripts/test.sh safnet_claude_34 model_submit_claude34
#
# ==============================================================================

MODEL="${1:-safnet_claude_33}"
EXP="${2:-model_submit_claude33}"
CKPT_DIR="./checkpoint_dir/checkpoint_dir_${MODEL}_${EXP}"

echo "======================================================"
echo " Model:      ${MODEL}"
echo " Experiment: ${EXP}"
echo " Checkpoint: ${CKPT_DIR}"
echo "======================================================"

# ------------------------------------------------------------------
# Step 1: Reparameterize (fuse SRP branches → model_best_rep.pth.tar)
# ------------------------------------------------------------------
echo ""
echo ">>> Step 1/2: Reparameterization"
echo "------------------------------------------------------"

python reparam_model.py --model "${MODEL}" --exp_name "${EXP}"

# ------------------------------------------------------------------
# Step 2: Evaluate with fused model + TTA
# ------------------------------------------------------------------
echo ""
echo ">>> Step 2/2: Evaluation (rep + TTA)"
echo "------------------------------------------------------"

python eval.py \
    --model "${MODEL}" \
    --exp_name "${EXP}" \
    --tta \
    --rep

echo ""
echo "======================================================"
echo " Pipeline complete!"
echo " Outputs:"
echo "   Fused weights: ${CKPT_DIR}/model_best_rep.pth.tar"
echo "   Images:        ${CKPT_DIR}/img/"
echo "   Result zip:    ${CKPT_DIR}/result.zip"
echo "======================================================"
