"""
Comprehensive Run Comparison Plot.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LABEL_FONT_SIZE = 10
TITLE_FONT_SIZE = 11


def load_run(path: Path) -> dict:
    with path.open("r") as fh:
        return json.load(fh)


def smooth_ema(scalars: list[float], weight: float = 0.95) -> list[float]:
    """Calculates Exponential Moving Average for smoother curves."""
    if not scalars:
        return []
        
    # Bulletproof: Force the very first item to be a float
    last = float(scalars[0])
    smoothed = []
    
    for point in scalars:
        # Bulletproof: Force every point to be a float
        val = float(point)
        smoothed_val = (last * weight) + ((1.0 - weight) * val)
        smoothed.append(smoothed_val)
        last = smoothed_val
        
    return smoothed


def plot_comparison(files: list[Path], title: str, output_file: Path):
    fig = plt.figure(figsize=(9, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.35, wspace=0.25)
    
    ax_curve = fig.add_subplot(gs[0, :])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_violin = fig.add_subplot(gs[1, 1])

    run_names = []
    max_iters = []
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    run_colors = []
    final_losses = []

    for idx, f in enumerate(files):
        data = load_run(f)
        if "lm_loss" not in data or "iteration" not in data:
            continue
            
        iterations = data["iteration"]
        raw_loss = data["lm_loss"]
        
        # Bulletproof list extraction
        lm_loss = []
        for x in raw_loss:
            if isinstance(x, list):
                lm_loss.append(float(x))
            else:
                lm_loss.append(float(x))
                
        run_name = data.get("name", f.stem).upper()
        color = colors[idx % len(colors)]
        
        run_names.append(run_name)
        run_colors.append(color)
        max_iters.append(max(iterations))
        
        # 1. Plot Curve
        smoothed = smooth_ema(lm_loss, weight=0.9)
        ax_curve.plot(iterations, lm_loss, alpha=0.15, color=color, linewidth=1)
        ax_curve.plot(iterations, smoothed, label=run_name, color=color, linewidth=2)

        # 2. Prepare Violin Data
        tail_length = max(1, int(len(lm_loss) * 0.10))
        final_losses.append(lm_loss[-tail_length:])

    # Top Panel
    ax_curve.set_title("Validation Loss Trajectory (Smoothed)", fontweight="bold", fontsize=TITLE_FONT_SIZE)
    ax_curve.set_xlabel("Iteration", fontsize=LABEL_FONT_SIZE)
    ax_curve.set_ylabel("LM Loss", fontsize=LABEL_FONT_SIZE)
    ax_curve.legend(loc="upper right")

    # Bottom Left Panel
    y_pos = np.arange(len(run_names))
    bars = ax_bar.barh(y_pos, max_iters, color=run_colors, alpha=0.8)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(run_names, fontweight="bold")
    ax_bar.set_title("Total Iterations (in 2.5h)", fontweight="bold", fontsize=TITLE_FONT_SIZE)
    ax_bar.set_xlabel("Iterations", fontsize=LABEL_FONT_SIZE)
    
    for bar in bars:
        width = bar.get_width()
        ax_bar.text(width * 0.95, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', ha='right', va='center', color='white', fontweight='bold')

    # Bottom Right Panel
    parts = ax_violin.violinplot(final_losses, showmeans=True, showextrema=True)
    for pc, color in zip(parts['bodies'], run_colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        vp = parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)

    ax_violin.set_xticks(np.arange(1, len(run_names) + 1))
    ax_violin.set_xticklabels(run_names, fontweight="bold")
    ax_violin.set_title("Loss Distribution (Final 10%)", fontweight="bold", fontsize=TITLE_FONT_SIZE)
    ax_violin.set_ylabel("Loss Values", fontsize=LABEL_FONT_SIZE)

    if title:
        fig.suptitle(title, fontsize=TITLE_FONT_SIZE + 2, fontweight="bold", y=0.98)

    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved comprehensive comparison plot to {output_file}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("-t", "--title", type=str, default="")
    ap.add_argument("-o", "--output-file", type=Path, default=Path("comparison_plot.png"))
    args = ap.parse_args()

    files: list[Path] = []
    for f in args.files:
        if f.is_dir():
            files.extend(sorted(f.glob("*.json")))
        elif f.is_file():
            files.append(f)

    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["CMU Sans Serif", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    plot_comparison(files, args.title, args.output_file)
    return 0

if __name__ == "__main__":
    sys.exit(main())