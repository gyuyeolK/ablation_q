#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-runs/smoke_vit_q_ablation}"

python run_q_ablation_vitb16_imagenet100.py \
  --synthetic-data \
  --tiny-model \
  --out-dir "$OUT_DIR" \
  --optimizer muon \
  --sampling rr \
  --q 1 \
  --seed 0 \
  --epochs 1 \
  --batch-size 8 \
  --eval-batch-size 16 \
  --num-workers 0 \
  --image-size 224 \
  --lr 0.001 \
  --side-lr 0.0003 \
  --warmup-steps 1

python run_q_ablation_vitb16_imagenet100.py \
  --synthetic-data \
  --tiny-model \
  --out-dir "$OUT_DIR" \
  --optimizer sgdm \
  --sampling us \
  --seed 0 \
  --epochs 1 \
  --batch-size 8 \
  --eval-batch-size 16 \
  --num-workers 0 \
  --image-size 224 \
  --sgdm-lr 0.01 \
  --side-lr 0.0003 \
  --warmup-steps 1

python aggregate_q_ablation.py \
  --run-root "$OUT_DIR" \
  --out-dir "$OUT_DIR/artifacts"

echo "Smoke test complete: $OUT_DIR"
