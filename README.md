# ViT-Base/16 + ImageNet-100 q-ablation for Muon-NS under RR and US

This package replaces the NanoGPT/FineWeb Newton--Schulz step ablation with a
non-GPT, 100M-scale vision Transformer experiment.

## Experiment

- Model: ViT-Base/16 (`vit_base_patch16_224`) with 100 output classes
- Dataset: ImageNet-100 in ImageFolder format
- Sampling:
  - RR: one fresh without-replacement random permutation per epoch
  - US: IID uniform sampling with replacement, matched to the same number of samples per epoch
- Optimizers:
  - Muon-NS with q in {1, 3, 5}
  - SGDM baseline
- Metrics:
  - train loss
  - validation loss
  - validation top-1 accuracy
  - epoch wall-clock time

## Expected data layout

Prepare ImageNet-100 as:

```text
/path/to/imagenet100/
  train/
    class_000/
      *.JPEG
    class_001/
      *.JPEG
    ...
  val/
    class_000/
      *.JPEG
    class_001/
      *.JPEG
    ...
```

The class folder names do not need to be `class_000`; standard ImageFolder
class folders are accepted. The number of classes should be 100.

## Install

```bash
pip install -r requirements.txt
```

## Full q-ablation grid

```bash
DATA_ROOT=/path/to/imagenet100 \
OUT_DIR=runs/vitb16_imagenet100_q_ablation \
EPOCHS=30 \
BATCH_SIZE=128 \
SEEDS=0,1 \
bash scripts/launch_full_grid.sh
```

This runs:

```text
sampling ∈ {rr, us}
optimizer ∈ {muon, sgdm}
q ∈ {1,3,5} for muon
seeds ∈ {0,1}
```

Total runs: `2 * (3 Muon q-values + 1 SGDM) * number_of_seeds`.

## Smoke test without ImageNet-100

```bash
bash scripts/smoke_test.sh
```

The smoke test uses synthetic images and a tiny ViT config to verify the training,
logging, aggregation, and plotting paths. It is not a scientific experiment.

## Aggregate results

```bash
python aggregate_q_ablation.py \
  --run-root runs/vitb16_imagenet100_q_ablation \
  --out-dir runs/vitb16_imagenet100_q_ablation/artifacts
```

Generated artifacts:

```text
artifacts/all_metrics.csv
artifacts/final_summary.csv
artifacts/q_ablation_final_table.tex
artifacts/q_ablation_curves_epoch.pdf
artifacts/q_ablation_rr_vs_us.pdf
```

## Notes

- Muon is applied only to hidden 2D weight matrices, excluding classifier head
  and normalization/bias/embedding-like parameters.
- Side parameters are updated with AdamW in all cells.
- For SGDM, the same matrix-parameter partition is used for a fair comparison:
  hidden 2D matrices use SGDM, and side parameters use AdamW.
- The implementation uses the Jordan quintic Newton--Schulz polynomial:
  `(a,b,c)=(3.4445,-4.7750,2.0315)`.
