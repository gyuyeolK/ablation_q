#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/dev/shm/datasets/imagenet100}"
KAGGLE_DATASET="${KAGGLE_DATASET:-ambityga/imagenet100}"
ZIP_NAME="imagenet100.zip"

echo "[INFO] Target DATA_ROOT: $DATA_ROOT"
echo "[INFO] Kaggle dataset:   $KAGGLE_DATASET"

# Optional cleanup from earlier failed downloads under /root.
if [ -f /root/datasets/imagenet100/imagenet100.zip ]; then
  echo "[INFO] Removing old partial zip from /root/datasets/imagenet100/imagenet100.zip"
  rm -f /root/datasets/imagenet100/imagenet100.zip
fi

mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

echo "[INFO] Disk space:"
df -h "$DATA_ROOT" || true

if [ -d "$DATA_ROOT/train" ] && [ -d "$DATA_ROOT/val" ]; then
  echo "[INFO] Existing train/ and val/ found. Skipping download."
else
  if [ ! -f "$ZIP_NAME" ]; then
    echo "[INFO] Downloading ImageNet-100 to $DATA_ROOT ..."
    kaggle datasets download -d "$KAGGLE_DATASET" -p "$DATA_ROOT"
  else
    echo "[INFO] Existing $ZIP_NAME found. Reusing it."
  fi

  echo "[INFO] Unzipping $ZIP_NAME ..."
  unzip -q "$ZIP_NAME"

  echo "[INFO] Removing $ZIP_NAME to save /dev/shm space ..."
  rm -f "$ZIP_NAME"

  echo "[INFO] Normalizing ImageFolder structure ..."

  # Merge train.X* shards into train/.
  if compgen -G "train.X*" > /dev/null; then
    mkdir -p train
    for d in train.X*; do
      [ -d "$d" ] || continue
      echo "[INFO] Merging $d -> train/"
      for cls in "$d"/*; do
        [ -d "$cls" ] || continue
        clsname="$(basename "$cls")"
        mkdir -p "train/$clsname"
        find "$cls" -maxdepth 1 -type f -exec mv -t "train/$clsname" {} +
      done
    done
    rmdir train.X* 2>/dev/null || true
  fi

  # If the archive already had a normal train folder, keep it.
  if [ ! -d train ]; then
    echo "[ERROR] Could not find or construct train/."
    echo "[INFO] Current directories:"
    find . -maxdepth 2 -type d | sort | head -100
    exit 1
  fi

  # Merge validation folders into val/.
  # Possible names: val, val.X*, validation, validation.X*
  mkdir -p val

  if compgen -G "val.X*" > /dev/null || compgen -G "validation.X*" > /dev/null; then
    for d in val.X* validation.X*; do
      [ -d "$d" ] || continue
      echo "[INFO] Merging $d -> val/"
      for cls in "$d"/*; do
        [ -d "$cls" ] || continue
        clsname="$(basename "$cls")"
        mkdir -p "val/$clsname"
        find "$cls" -maxdepth 1 -type f -exec mv -t "val/$clsname" {} +
      done
    done
    rmdir val.X* validation.X* 2>/dev/null || true
  fi

  if [ -d validation ]; then
    echo "[INFO] Merging validation/ -> val/"
    for cls in validation/*; do
      [ -d "$cls" ] || continue
      clsname="$(basename "$cls")"
      mkdir -p "val/$clsname"
      find "$cls" -maxdepth 1 -type f -exec mv -t "val/$clsname" {} +
    done
    rmdir validation/* 2>/dev/null || true
    rmdir validation 2>/dev/null || true
  fi
fi

echo "[INFO] Verifying with torchvision.datasets.ImageFolder ..."

python - <<PY
from pathlib import Path
from torchvision.datasets import ImageFolder

root = Path("$DATA_ROOT")
train_dir = root / "train"
val_dir = root / "val"

if not train_dir.exists():
    raise FileNotFoundError(f"Missing train directory: {train_dir}")
if not val_dir.exists():
    raise FileNotFoundError(f"Missing val directory: {val_dir}")

train = ImageFolder(str(train_dir))
val = ImageFolder(str(val_dir))

print("train images:", len(train))
print("val images:", len(val))
print("train classes:", len(train.classes))
print("val classes:", len(val.classes))
print("same classes:", train.classes == val.classes)
print("first classes:", train.classes[:5])

assert len(train.classes) == 100, f"Expected 100 train classes, got {len(train.classes)}"
assert len(val.classes) == 100, f"Expected 100 val classes, got {len(val.classes)}"
assert train.classes == val.classes, "Train/val class folders do not match."
assert len(train) > 0, "No train images found."
assert len(val) > 0, "No val images found."
PY

echo ""
echo "[DONE] ImageNet-100 is ready."
echo "DATA_ROOT=$DATA_ROOT"
echo ""
echo "Example:"
echo "DATA_ROOT=$DATA_ROOT \\"
echo "OUT_DIR=runs/vitb16_imagenet100_q_ablation \\"
echo "EPOCHS=30 \\"
echo "BATCH_SIZE=128 \\"
echo "SEEDS=0,1 \\"
echo "Q_VALUES=1,3,5 \\"
echo "NUM_GPUS=8 \\"
echo "NUM_WORKERS=4 \\"
echo "bash scripts/launch_full_grid_8gpu.sh"
