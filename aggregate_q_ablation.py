#!/usr/bin/env python3
"""Aggregate and plot ViT-Base/16 ImageNet-100 q-ablation results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_all_metrics(run_root: Path) -> pd.DataFrame:
    paths = sorted(run_root.glob("**/metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"No metrics.csv found under {run_root}")

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["run_dir"] = str(p.parent)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def config_label(row) -> str:
    if row["optimizer"] == "sgdm":
        return "SGDM"
    return f"Muon-NS q={int(row['q'])}"


def summarize_final(df: pd.DataFrame) -> pd.DataFrame:
    last = df.sort_values("epoch").groupby(["optimizer", "sampling", "q", "seed"], as_index=False).tail(1)
    summary = (
        last.groupby(["sampling", "optimizer", "q"], as_index=False)
        .agg(
            train_loss_mean=("train_loss", "mean"),
            train_loss_std=("train_loss", "std"),
            val_loss_mean=("val_loss", "mean"),
            val_loss_std=("val_loss", "std"),
            val_acc_mean=("val_acc", "mean"),
            val_acc_std=("val_acc", "std"),
            epoch_seconds_mean=("epoch_seconds", "mean"),
            n=("seed", "nunique"),
        )
    )
    order = {"rr": 0, "us": 1}
    opt_order = {"muon": 0, "sgdm": 1}
    summary["_sampling_order"] = summary["sampling"].map(order)
    summary["_opt_order"] = summary["optimizer"].map(opt_order)
    summary = summary.sort_values(["_sampling_order", "_opt_order", "q"]).drop(columns=["_sampling_order", "_opt_order"])
    return summary


def write_latex_table(summary: pd.DataFrame, out_path: Path):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{\textbf{Final validation results for the ViT-Base/16 / ImageNet-100 $q$-ablation.} Mean $\pm$ one seed-level standard deviation; lower validation loss and higher top-1 accuracy are better.}")
    lines.append(r"\label{tab:q-ablation-vitb16-imagenet100-final}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Configuration & Train loss & Val. loss & Val. top-1 acc. \\")
    lines.append(r"\midrule")

    for sampling in ["rr", "us"]:
        title = "Random Reshuffling (RR)" if sampling == "rr" else "Uniform with replacement (US)"
        lines.append(rf"\multicolumn{{4}}{{l}}{{\emph{{{title}}}}} \\")
        sub = summary[summary["sampling"] == sampling]
        for _, row in sub.iterrows():
            if row["optimizer"] == "muon":
                name = rf"$\muon$-NS, $q={int(row['q'])}$"
            else:
                name = r"\textsc{SGDM}"
            def pm(mean, std, pct=False):
                if pd.isna(std):
                    std = 0.0
                if pct:
                    return f"{100*mean:.2f} $\\pm$ {100*std:.2f}"
                return f"{mean:.4f} $\\pm$ {std:.4f}"
            lines.append(
                rf"\quad {name} & {pm(row['train_loss_mean'], row['train_loss_std'])} "
                rf"& {pm(row['val_loss_mean'], row['val_loss_std'])} "
                rf"& {pm(row['val_acc_mean'], row['val_acc_std'], pct=True)} \\"
            )
        if sampling == "rr":
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines) + "\n")


def plot_curves(df: pd.DataFrame, out_dir: Path):
    grouped = (
        df.groupby(["sampling", "optimizer", "q", "epoch"], as_index=False)
        .agg(
            train_loss_mean=("train_loss", "mean"),
            train_loss_std=("train_loss", "std"),
            val_loss_mean=("val_loss", "mean"),
            val_loss_std=("val_loss", "std"),
            val_acc_mean=("val_acc", "mean"),
            val_acc_std=("val_acc", "std"),
        )
    )

    # Separate RR/US multi-metric curves.
    for metric, ylabel, fname in [
        ("val_loss", "Validation loss", "q_ablation_curves_epoch.pdf"),
        ("val_acc", "Validation top-1 accuracy", "q_ablation_accuracy_epoch.pdf"),
    ]:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.gca()
        for (sampling, optimizer, q), sub in grouped.groupby(["sampling", "optimizer", "q"]):
            label = f"{sampling.upper()} / " + ("SGDM" if optimizer == "sgdm" else f"Muon q={int(q)}")
            mean = sub[f"{metric}_mean"]
            std = sub[f"{metric}_std"].fillna(0.0)
            x = sub["epoch"]
            ax.plot(x, mean, label=label)
            ax.fill_between(x, mean - std, mean + std, alpha=0.15)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / fname, bbox_inches="tight")
        plt.close(fig)

    # RR vs US for each q using val loss.
    fig = plt.figure(figsize=(8, 5))
    ax = fig.gca()
    muon = grouped[grouped["optimizer"] == "muon"]
    for (q, sampling), sub in muon.groupby(["q", "sampling"]):
        label = f"q={int(q)} / {sampling.upper()}"
        ax.plot(sub["epoch"], sub["val_loss_mean"], label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "q_ablation_rr_vs_us.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir) if args.out_dir else run_root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_metrics(run_root)
    df.to_csv(out_dir / "all_metrics.csv", index=False)

    summary = summarize_final(df)
    summary.to_csv(out_dir / "final_summary.csv", index=False)
    write_latex_table(summary, out_dir / "q_ablation_final_table.tex")
    plot_curves(df, out_dir)

    print(f"Wrote artifacts to {out_dir}")


if __name__ == "__main__":
    main()
