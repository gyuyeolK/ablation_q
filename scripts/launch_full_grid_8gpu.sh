#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/dev/shm/datasets/imagenet100}"
OUT_DIR="${OUT_DIR:-runs/vitb16_imagenet100_q_ablation}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
SEEDS="${SEEDS:-0,1}"
Q_VALUES="${Q_VALUES:-1,3,5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-0.002}"
SGDM_LR="${SGDM_LR:-0.1}"
SIDE_LR="${SIDE_LR:-0.0003}"
MOMENTUM="${MOMENTUM:-0.95}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
AMP_FLAG="${AMP_FLAG:---amp}"
NUM_GPUS="${NUM_GPUS:-8}"

mkdir -p "$OUT_DIR/logs"

commands=()
names=()

IFS=',' read -ra SEED_ARR <<< "$SEEDS"
IFS=',' read -ra Q_ARR <<< "$Q_VALUES"

for seed in "${SEED_ARR[@]}"; do
  for sampling in rr us; do
    for q in "${Q_ARR[@]}"; do
      names+=("muon_${sampling}_q${q}_seed${seed}")
      commands+=("python run_q_ablation_vitb16_imagenet100.py \
        --data-root '$DATA_ROOT' \
        --out-dir '$OUT_DIR' \
        --optimizer muon \
        --sampling '$sampling' \
        --q '$q' \
        --seed '$seed' \
        --epochs '$EPOCHS' \
        --batch-size '$BATCH_SIZE' \
        --eval-batch-size '$EVAL_BATCH_SIZE' \
        --num-workers '$NUM_WORKERS' \
        --lr '$LR' \
        --side-lr '$SIDE_LR' \
        --momentum '$MOMENTUM' \
        --warmup-steps '$WARMUP_STEPS' \
        $AMP_FLAG")
    done

    names+=("sgdm_${sampling}_seed${seed}")
    commands+=("python run_q_ablation_vitb16_imagenet100.py \
      --data-root '$DATA_ROOT' \
      --out-dir '$OUT_DIR' \
      --optimizer sgdm \
      --sampling '$sampling' \
      --seed '$seed' \
      --epochs '$EPOCHS' \
      --batch-size '$BATCH_SIZE' \
      --eval-batch-size '$EVAL_BATCH_SIZE' \
      --num-workers '$NUM_WORKERS' \
      --sgdm-lr '$SGDM_LR' \
      --side-lr '$SIDE_LR' \
      --momentum '$MOMENTUM' \
      --warmup-steps '$WARMUP_STEPS' \
      $AMP_FLAG")
  done
done

num_jobs=${#commands[@]}
echo "Total jobs: $num_jobs"
echo "GPUs: $NUM_GPUS"
echo "Output: $OUT_DIR"

run_job() {
  local idx="$1"
  local gpu="$2"
  local name="${names[$idx]}"
  local cmd="${commands[$idx]}"
  local log="$OUT_DIR/logs/job_${idx}_${name}_gpu${gpu}.log"

  echo "[START] job=$idx gpu=$gpu name=$name"
  echo "$cmd" > "${log}.cmd"

  CUDA_VISIBLE_DEVICES="$gpu" bash -lc "$cmd" > "$log" 2>&1

  echo "[DONE]  job=$idx gpu=$gpu name=$name log=$log"
}

next_job=0

while [ "$next_job" -lt "$num_jobs" ]; do
  for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    if [ "$next_job" -ge "$num_jobs" ]; then
      break
    fi

    run_job "$next_job" "$gpu" &
    next_job=$((next_job + 1))
  done

  wait
  echo "[WAVE DONE] completed up to job $next_job / $num_jobs"
done

echo "[AGGREGATE] Collecting metrics..."
python aggregate_q_ablation.py \
  --run-root "$OUT_DIR" \
  --out-dir "$OUT_DIR/artifacts"

echo "[ALL DONE] Results saved to $OUT_DIR/artifacts"
