#!/usr/bin/env bash
# Two-stage mytest pretrain -> finetune pipeline
# Stage 1: pretrain backbone on mytest
# Stage 2: finetune on original data using pretrained backbone

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRETRAIN_OUTPUT="${SCRIPT_DIR}/outputs/mytest_pretrain"
FINETUNE_OUTPUT="${SCRIPT_DIR}/outputs/mytest_pretrain_finetune"

echo "=== Stage 1: mytest pretrain ==="
python "${SCRIPT_DIR}/mytest_pretrain.py" \
    --mytest-root "${SCRIPT_DIR}/../../mytest" \
    --output-dir "${PRETRAIN_OUTPUT}" \
    --backbone convnext_tiny \
    --image-size 288 \
    --label-smoothing 0.1 \
    --cutmix-alpha 0.7 \
    --cutmix-prob 0.3 \
    --max-grad-norm 1.0 \
    --lr-scheduler cosine \
    --seed 42 \
    --epochs 50 \
    --head-only-epochs 5 \
    --batch-size 96 \
    --head-lr 1e-4 \
    --backbone-lr 1e-5 \
    --weight-decay 0.05 \
    --dropout 0.1 \
    --val-split-ratio 0.15 \
    --early-stop 12 \
    --save-every-epoch
echo "Stage 1 done. Checkpoint: ${PRETRAIN_OUTPUT}/best.pt"

echo ""
echo "=== Stage 2: finetune on original data ==="
python "${SCRIPT_DIR}/train_finetune.py" \
    --backbone convnext_tiny \
    --backbone-checkpoint "${PRETRAIN_OUTPUT}/best.pt" \
    --output-dir "${FINETUNE_OUTPUT}" \
    --val-root ../../data/myval \
    --val-mask-split myval \
    --mask-dir ../../mask \
    --image-size 288 \
    --label-smoothing 0.1 \
    --cutmix-alpha 0.7 \
    --cutmix-prob 0.3 \
    --max-grad-norm 1.0 \
    --lr-scheduler cosine \
    --seed 42 \
    --epochs 50 \
    --head-only-epochs 5 \
    --batch-size 64 \
    --head-lr 1e-4 \
    --backbone-lr 1e-5 \
    --weight-decay 0.05 \
    --dropout 0.1 \
    --disable-bayes-correction \
    --early-stop 12 \
    --save-every-epoch
echo "Stage 2 done. Checkpoint: ${FINETUNE_OUTPUT}/best.pt"

echo ""
echo "=== myval evaluation ==="
python "${SCRIPT_DIR}/eval_myval.py" \
    --checkpoint "${FINETUNE_OUTPUT}/best.pt" \
    --image-size 288

echo ""
echo "Pipeline complete."
echo "Pretrain: ${PRETRAIN_OUTPUT}/best.pt"
echo "Finetune: ${FINETUNE_OUTPUT}/best.pt"
