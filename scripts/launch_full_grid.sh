#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/path/to/imagenet100}"
OUT_DIR="${OUT_DIR:-runs/vitb16_imagenet100_q_ablation}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
SEEDS="${SEEDS:-0,1}"
Q_VALUES="${Q_VALUES:-1,3,5}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-0.002}"
SGDM_LR="${SGDM_LR:-0.1}"
SIDE_LR="${SIDE_LR:-0.0003}"
MOMENTUM="${MOMENTUM:-0.95}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
AMP_FLAG="${AMP_FLAG:---amp}"

IFS=',' read -ra SEED_ARR <<< "$SEEDS"
IFS=',' read -ra Q_ARR <<< "$Q_VALUES"

for seed in "${SEED_ARR[@]}"; do
  for sampling in rr us; do
    for q in "${Q_ARR[@]}"; do
      python run_q_ablation_vitb16_imagenet100.py \
        --data-root "$DATA_ROOT" \
        --out-dir "$OUT_DIR" \
        --optimizer muon \
        --sampling "$sampling" \
        --q "$q" \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --eval-batch-size "$EVAL_BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --lr "$LR" \
        --side-lr "$SIDE_LR" \
        --momentum "$MOMENTUM" \
        --warmup-steps "$WARMUP_STEPS" \
        $AMP_FLAG
    done

    python run_q_ablation_vitb16_imagenet100.py \
      --data-root "$DATA_ROOT" \
      --out-dir "$OUT_DIR" \
      --optimizer sgdm \
      --sampling "$sampling" \
      --seed "$seed" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --eval-batch-size "$EVAL_BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --sgdm-lr "$SGDM_LR" \
      --side-lr "$SIDE_LR" \
      --momentum "$MOMENTUM" \
      --warmup-steps "$WARMUP_STEPS" \
      $AMP_FLAG
  done
done

python aggregate_q_ablation.py \
  --run-root "$OUT_DIR" \
  --out-dir "$OUT_DIR/artifacts"
