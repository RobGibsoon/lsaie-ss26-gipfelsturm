#!/usr/bin/env python3
"""
Plot throughput results from fp8/results/throughput.csv.

Produces:
  fp8/results/scalability_bf16.png   — tokens/sec/GPU vs nodes, one line per model (BF16)
  fp8/results/scalability_fp8.png    — same for FP8 (best recipe per model)
  fp8/results/fp8_vs_bf16.png        — FP8 speedup over BF16, grouped by model + nodes
  fp8/results/recipe_sweep.png       — tokens/sec/GPU per FP8 recipe (760m, 1 node)

Usage:
    python fp8/scripts/plot_scalability.py [--csv fp8/results/throughput.csv]
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

MODEL_ORDER = ["125m", "350m", "760m", "1.5b", "3b", "8b", "32b", "140b"]
COLORS = plt.cm.tab10.colors


def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def tps(row) -> float:
    return float(row["tps_gpu_median"])


def is_bf16(row) -> bool:
    return row["precision"] == "bf16"


def is_fp8(row) -> bool:
    return row["precision"].startswith("fp8")


def best_per_group(rows, key_fn) -> dict:
    """Return the row with highest tps per group key."""
    groups: dict = {}
    for r in rows:
        k = key_fn(r)
        if k not in groups or tps(r) > tps(groups[k]):
            groups[k] = r
    return groups


def plot_scaling(rows, precision_filter, title, out_path):
    filtered = [r for r in rows if precision_filter(r)]
    by_model = defaultdict(list)
    for r in filtered:
        by_model[r["model"]].append(r)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(MODEL_ORDER):
        pts = sorted(by_model.get(model, []), key=lambda r: int(r["nodes"]))
        if not pts:
            continue
        xs = [int(r["nodes"]) for r in pts]
        ys = [tps(r) for r in pts]
        ax.plot(xs, ys, marker="o", label=model, color=COLORS[i % len(COLORS)])

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}n"))
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xlabel("Nodes (4 GPUs each)")
    ax.set_ylabel("Tokens / sec / GPU")
    ax.set_title(title)
    ax.legend(title="Model size")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_fp8_vs_bf16(rows, out_path):
    bf16_best = best_per_group(
        [r for r in rows if is_bf16(r)],
        key_fn=lambda r: (r["model"], r["nodes"])
    )
    fp8_best = best_per_group(
        [r for r in rows if is_fp8(r)],
        key_fn=lambda r: (r["model"], r["nodes"])
    )

    common_keys = sorted(
        set(bf16_best) & set(fp8_best),
        key=lambda k: (MODEL_ORDER.index(k[0]) if k[0] in MODEL_ORDER else 99, int(k[1]))
    )
    if not common_keys:
        print("[skip] fp8_vs_bf16: no matched (model, nodes) pairs between BF16 and FP8")
        return

    labels = [f"{m}\n{n}n" for m, n in common_keys]
    speedups = [tps(fp8_best[k]) / tps(bf16_best[k]) for k in common_keys]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5))
    bars = ax.bar(labels, speedups, color=COLORS[1])
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="BF16 baseline")
    ax.bar_label(bars, fmt="%.2f×", padding=3, fontsize=8)
    ax.set_ylabel("FP8 / BF16 throughput ratio")
    ax.set_title("FP8 speedup over BF16 (best recipe per cell)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_recipe_sweep(rows, out_path, model="760m", nodes=1):
    candidates = [
        r for r in rows
        if r["model"] == model and int(r["nodes"]) == nodes and is_fp8(r)
    ]
    if not candidates:
        print(f"[skip] recipe_sweep: no FP8 data for {model} {nodes}n")
        return

    candidates.sort(key=tps, reverse=True)
    labels = [r["precision"] for r in candidates]
    vals   = [tps(r) for r in candidates]

    # Add BF16 baseline if present
    bf16 = [r for r in rows if r["model"] == model and int(r["nodes"]) == nodes and is_bf16(r)]
    if bf16:
        best_bf16 = max(bf16, key=tps)
        labels.append("bf16 (baseline)")
        vals.append(tps(best_bf16))

    colors = [COLORS[1]] * (len(labels) - (1 if bf16 else 0)) + ([COLORS[0]] if bf16 else [])
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    bars = ax.barh(labels, vals, color=colors)
    ax.bar_label(bars, fmt="%,.0f", padding=4, fontsize=8)
    ax.set_xlabel("Tokens / sec / GPU")
    ax.set_title(f"FP8 recipe sweep — {model}, {nodes} node")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="fp8/results/throughput.csv")
    ap.add_argument("--out-dir", default="fp8/results")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}  (run parse_logs.py first)")
        return
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(csv_path)
    print(f"Loaded {len(rows)} rows from {csv_path}")

    plot_scaling(
        rows, is_bf16,
        title="Throughput scaling — BF16 (tokens/sec/GPU vs nodes)",
        out_path=out_dir / "scalability_bf16.png",
    )
    plot_scaling(
        rows, is_fp8,
        title="Throughput scaling — FP8 best recipe (tokens/sec/GPU vs nodes)",
        out_path=out_dir / "scalability_fp8.png",
    )
    plot_fp8_vs_bf16(rows, out_dir / "fp8_vs_bf16.png")
    plot_recipe_sweep(rows, out_dir / "recipe_sweep_760m_1n.png", model="760m", nodes=1)
    plot_recipe_sweep(rows, out_dir / "recipe_sweep_8b_1n.png",   model="8b",   nodes=1)


if __name__ == "__main__":
    main()
