#!/usr/bin/env python3
"""
ViT-Base/16 + ImageNet-100 q-ablation for Muon-NS under RR and US.

This script trains one cell of the grid:
  - optimizer: muon or sgdm
  - sampling: rr or us
  - q: Newton--Schulz steps for Muon
  - seed: random seed

The expected dataset layout is ImageFolder:
  data_root/train/<class_name>/*.JPEG
  data_root/val/<class_name>/*.JPEG

For smoke testing, pass --synthetic-data.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

try:
    import timm
except ImportError as exc:
    raise RuntimeError("Please install timm: pip install timm") from exc


NS5_A = 3.4445
NS5_B = -4.7750
NS5_C = 2.0315


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SyntheticImageNet100(Dataset):
    """Tiny deterministic synthetic dataset for smoke tests only."""

    def __init__(self, n: int = 512, num_classes: int = 100, image_size: int = 224, seed: int = 0):
        self.n = n
        self.num_classes = num_classes
        self.image_size = image_size
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(self.seed + idx)
        x = torch.randn(3, self.image_size, self.image_size, generator=g)
        y = idx % self.num_classes
        return x, y


def build_transforms(image_size: int = 224):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return train_tf, val_tf


def build_datasets(args):
    if args.synthetic_data:
        train_set = SyntheticImageNet100(n=args.synthetic_train_size, image_size=args.image_size, seed=args.seed)
        val_set = SyntheticImageNet100(n=args.synthetic_val_size, image_size=args.image_size, seed=args.seed + 100000)
        return train_set, val_set, 100

    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Missing val directory: {val_dir}")

    train_tf, val_tf = build_transforms(args.image_size)
    train_set = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_set = datasets.ImageFolder(str(val_dir), transform=val_tf)

    if len(train_set.classes) != args.num_classes:
        raise ValueError(
            f"Expected {args.num_classes} classes, found {len(train_set.classes)} in {train_dir}. "
            "Check that ImageNet-100 is prepared in ImageFolder format."
        )
    if len(val_set.classes) != args.num_classes:
        raise ValueError(
            f"Expected {args.num_classes} classes, found {len(val_set.classes)} in {val_dir}."
        )

    return train_set, val_set, len(train_set.classes)


def make_epoch_loader(dataset: Dataset, args, epoch: int, train: bool = True) -> DataLoader:
    if train:
        g = torch.Generator()
        g.manual_seed(args.seed * 1000003 + epoch)
        if args.sampling == "rr":
            sampler = RandomSampler(dataset, replacement=False, generator=g)
        elif args.sampling == "us":
            sampler = RandomSampler(dataset, replacement=True, num_samples=len(dataset), generator=g)
        else:
            raise ValueError(f"Unknown sampling: {args.sampling}")
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.num_workers > 0,
        )

    return DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )


@torch.no_grad()
def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton--Schulz quintic iteration used by Muon.

    This function accepts a 2D matrix and returns an approximate polar factor.
    It internally works in fp32 for stability and casts back to the original dtype.
    """
    if g.ndim != 2:
        raise ValueError("Muon NS update expects a 2D tensor.")

    orig_dtype = g.dtype
    x = g.float()
    if torch.all(x == 0):
        return torch.zeros_like(g)

    transposed = False
    if x.size(0) > x.size(1):
        x = x.T
        transposed = True

    # Frobenius scaling. The target polar factor is scale-invariant.
    x = x / (x.norm(p="fro") + eps)

    for _ in range(steps):
        a = x @ x.T
        b = NS5_B * a + NS5_C * (a @ a)
        x = NS5_A * x + b @ x

    if transposed:
        x = x.T
    return x.to(dtype=orig_dtype)


class Muon(torch.optim.Optimizer):
    """
    Minimal Muon optimizer for hidden 2D matrices.

    Momentum:
      M_t = beta M_{t-1} + grad_t
      update direction = NS(M_t, q)

    This implementation intentionally keeps non-2D / side parameters outside
    this optimizer; they should be handled by AdamW.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 2e-3,
        beta: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, beta=beta, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise RuntimeError("Muon optimizer received a non-2D parameter.")

                grad = p.grad
                if wd != 0.0:
                    grad = grad.add(p, alpha=wd)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]
                buf.mul_(beta).add_(grad)

                update = zeropower_via_newtonschulz5(buf, steps=ns_steps)

                # A common Muon implementation rescales by sqrt(max(1, rows/cols))
                # to stabilize rectangular matrices. This keeps update magnitudes
                # comparable across q and across layers.
                scale = math.sqrt(max(1.0, p.shape[0] / max(1, p.shape[1])))
                p.add_(update, alpha=-lr * scale)

        return loss


def split_params_for_main_optimizer(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter], dict]:
    """
    Muon/SGDM main optimizer is applied to hidden 2D matrices.
    Side parameters use AdamW in every cell.

    Exclusions:
      - classifier head
      - biases
      - norm parameters
      - embeddings / positional tokens / class token
      - patch projection conv kernels (4D)
    """
    matrix_params: List[nn.Parameter] = []
    side_params: List[nn.Parameter] = []
    counts = {"matrix": 0, "side": 0, "matrix_names": [], "side_names": []}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        lname = name.lower()
        is_hidden_matrix = (
            p.ndim == 2
            and "head" not in lname
            and "norm" not in lname
            and "pos_embed" not in lname
            and "cls_token" not in lname
        )

        if is_hidden_matrix:
            matrix_params.append(p)
            counts["matrix"] += p.numel()
            counts["matrix_names"].append(name)
        else:
            side_params.append(p)
            counts["side"] += p.numel()
            counts["side_names"].append(name)

    return matrix_params, side_params, counts


def build_model(args, num_classes: int) -> nn.Module:
    if args.synthetic_data and args.tiny_model:
        model_name = "vit_tiny_patch16_224"
    else:
        model_name = args.model_name

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        img_size=args.image_size,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
    )
    return model


@dataclass
class RunConfig:
    data_root: str
    out_dir: str
    model_name: str
    optimizer: str
    sampling: str
    q: int
    seed: int
    epochs: int
    batch_size: int
    eval_batch_size: int
    lr: float
    sgdm_lr: float
    side_lr: float
    momentum: float
    weight_decay: float
    side_weight_decay: float
    warmup_steps: int
    image_size: int
    num_classes: int
    amp: bool
    dtype: str
    synthetic_data: bool


def get_lrs(args):
    if args.optimizer == "muon":
        return args.lr
    if args.optimizer == "sgdm":
        return args.sgdm_lr
    raise ValueError(args.optimizer)


def set_train_lr(optimizer, lr: float):
    for group in optimizer.param_groups:
        group["lr"] = lr


def linear_warmup_lr(base_lr: float, global_step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return base_lr
    return base_lr * min(1.0, float(global_step + 1) / float(warmup_steps))


def accuracy_top1(logits: torch.Tensor, target: torch.Tensor) -> int:
    pred = logits.argmax(dim=1)
    return int((pred == target).sum().item())


def train_one_epoch(model, main_opt, side_opt, train_loader, criterion, device, args, epoch: int, global_step: int):
    model.train()
    total_loss = 0.0
    total_seen = 0
    total_correct = 0
    base_lr = get_lrs(args)

    pbar = tqdm(train_loader, desc=f"epoch {epoch:03d} train", dynamic_ncols=True)
    for images, targets in pbar:
        lr_now = linear_warmup_lr(base_lr, global_step, args.warmup_steps)
        set_train_lr(main_opt, lr_now)
        set_train_lr(side_opt, linear_warmup_lr(args.side_lr, global_step, args.warmup_steps))

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        main_opt.zero_grad(set_to_none=True)
        side_opt.zero_grad(set_to_none=True)

        if args.amp and device.type == "cuda":
            amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(images)
                loss = criterion(logits, targets)
        else:
            logits = model(images)
            loss = criterion(logits, targets)

        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        main_opt.step()
        side_opt.step()

        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        total_seen += bs
        total_correct += accuracy_top1(logits.detach(), targets)
        global_step += 1

        pbar.set_postfix(loss=total_loss / max(1, total_seen), acc=total_correct / max(1, total_seen), lr=lr_now)

    return total_loss / total_seen, total_correct / total_seen, global_step


@torch.no_grad()
def evaluate(model, val_loader, criterion, device, args):
    model.eval()
    total_loss = 0.0
    total_seen = 0
    total_correct = 0

    for images, targets in tqdm(val_loader, desc="eval", dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if args.amp and device.type == "cuda":
            amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(images)
                loss = criterion(logits, targets)
        else:
            logits = model(images)
            loss = criterion(logits, targets)

        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        total_seen += bs
        total_correct += accuracy_top1(logits, targets)

    return total_loss / total_seen, total_correct / total_seen


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(payload))


def json_dumps(payload) -> str:
    import json
    return json.dumps(payload, indent=2, sort_keys=True)


def run(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.benchmark = True

    out_dir = Path(args.out_dir)
    run_name = f"model={args.model_name}_opt={args.optimizer}_sampling={args.sampling}_q={args.q}_seed={args.seed}"
    if args.synthetic_data and args.tiny_model:
        run_name = f"synthetic_tiny_opt={args.optimizer}_sampling={args.sampling}_q={args.q}_seed={args.seed}"
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_set, val_set, num_classes = build_datasets(args)
    model = build_model(args, num_classes=num_classes).to(device)

    matrix_params, side_params, counts = split_params_for_main_optimizer(model)

    if args.optimizer == "muon":
        main_opt = Muon(matrix_params, lr=args.lr, beta=args.momentum, ns_steps=args.q, weight_decay=args.weight_decay)
    elif args.optimizer == "sgdm":
        main_opt = SGD(matrix_params, lr=args.sgdm_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError(args.optimizer)

    side_opt = AdamW(side_params, lr=args.side_lr, betas=(0.9, 0.95), weight_decay=args.side_weight_decay)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cfg = RunConfig(
        data_root=args.data_root,
        out_dir=str(run_dir),
        model_name=args.model_name,
        optimizer=args.optimizer,
        sampling=args.sampling,
        q=args.q,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        sgdm_lr=args.sgdm_lr,
        side_lr=args.side_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        side_weight_decay=args.side_weight_decay,
        warmup_steps=args.warmup_steps,
        image_size=args.image_size,
        num_classes=num_classes,
        amp=args.amp,
        dtype=args.dtype,
        synthetic_data=args.synthetic_data,
    )

    metadata = {
        "config": asdict(cfg),
        "total_trainable_params": total_params,
        "matrix_path_params": counts["matrix"],
        "side_params": counts["side"],
        "num_train_samples": len(train_set),
        "num_val_samples": len(val_set),
        "steps_per_epoch": len(train_set) // args.batch_size,
        "matrix_param_names": counts["matrix_names"],
        "side_param_names": counts["side_names"],
    }
    (run_dir / "metadata.json").write_text(json_dumps(metadata))

    print(json_dumps({k: v for k, v in metadata.items() if k not in ["matrix_param_names", "side_param_names"]}))

    criterion = nn.CrossEntropyLoss()
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch", "optimizer", "sampling", "q", "seed",
                "train_loss", "train_acc", "val_loss", "val_acc",
                "epoch_seconds", "global_step",
                "matrix_path_params", "total_params",
            ],
        )
        writer.writeheader()

    global_step = 0
    best_val_acc = -1.0

    val_loader = make_epoch_loader(val_set, args, epoch=0, train=False)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loader = make_epoch_loader(train_set, args, epoch=epoch, train=True)
        train_loss, train_acc, global_step = train_one_epoch(
            model, main_opt, side_opt, train_loader, criterion, device, args, epoch, global_step
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, args)
        epoch_seconds = time.time() - epoch_start
        best_val_acc = max(best_val_acc, val_acc)

        row = {
            "epoch": epoch,
            "optimizer": args.optimizer,
            "sampling": args.sampling,
            "q": args.q,
            "seed": args.seed,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_seconds": epoch_seconds,
            "global_step": global_step,
            "matrix_path_params": counts["matrix"],
            "total_params": total_params,
        }

        with metrics_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

        print(json_dumps(row))

        if args.save_checkpoints and val_acc >= best_val_acc:
            ckpt = {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "args": vars(args),
            }
            torch.save(ckpt, run_dir / "best.pt")

    print(f"Saved metrics to {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="runs/vitb16_imagenet100_q_ablation")
    parser.add_argument("--model-name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--drop", type=float, default=0.0)
    parser.add_argument("--drop-path", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, choices=["muon", "sgdm"], required=True)
    parser.add_argument("--sampling", type=str, choices=["rr", "us"], required=True)
    parser.add_argument("--q", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--lr", type=float, default=2e-3, help="Muon learning rate")
    parser.add_argument("--sgdm-lr", type=float, default=0.1, help="SGDM learning rate")
    parser.add_argument("--side-lr", type=float, default=3e-4)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--side-weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--save-checkpoints", action="store_true")

    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--synthetic-train-size", type=int, default=256)
    parser.add_argument("--synthetic-val-size", type=int, default=128)
    parser.add_argument("--tiny-model", action="store_true", help="Use vit_tiny_patch16_224 for smoke testing")

    args = parser.parse_args()

    if args.optimizer == "sgdm":
        # q has no meaning for SGDM; keep q=0 in filenames/tables.
        args.q = 0

    return args


if __name__ == "__main__":
    run(parse_args())
