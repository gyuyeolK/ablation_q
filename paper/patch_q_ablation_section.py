#!/usr/bin/env python3
"""
Replace the old q-ablation subsection in a LaTeX file with the ViT-Base/16
ImageNet-100 q-ablation template.

Usage:
  python paper/patch_q_ablation_section.py --input paper.tex --output paper_patched.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path


START = r"\subsection{Newton--Schulz steps (\texorpdfstring{$q$}{q}) ablation under RR and US}"
NEXT_SUBSECTION = r"\subsection{Remark on the theory--practice momentum gap}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replacement", default=str(Path(__file__).with_name("q_ablation_vitb16_imagenet100_replacement.tex")))
    args = parser.parse_args()

    text = Path(args.input).read_text()
    replacement = Path(args.replacement).read_text()

    start = text.find(START)
    if start < 0:
        raise RuntimeError("Could not find q-ablation subsection start.")

    end = text.find(NEXT_SUBSECTION, start)
    if end < 0:
        raise RuntimeError("Could not find next subsection marker.")

    patched = text[:start] + replacement.strip() + "\n\n" + text[end:]
    Path(args.output).write_text(patched)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
