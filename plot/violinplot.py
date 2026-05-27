"""
Compare multiple parsed run JSONs (see logparse.py) with violin plots.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

LABEL_FONT_SIZE = 9
TITLE_FONT_SIZE = 10
COLUMN_WIDTH = 8
WARMUP_ITERS = 5  # skip first iters (compile/warmup spikes)

METRICS = [
    ("throughput", "Throughput (TFLOP/s/GPU)"),
    ("tokens_per_sec_per_gpu", "Tokens per second per GPU"),
]


def load_run(path: Path) -> dict:
    with path.open("r") as fh:
        return json.load(fh)


def get_color_cycle():
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def filenamify(text: str) -> str:
    return text.replace(" ", "_").replace("\n", "_").lower()


def suptitle(fig: plt.Figure, text: str):
    pos = fig.get_axes()[0].get_position()
    fig.suptitle(
        text,
        fontsize=TITLE_FONT_SIZE,
        x=pos.x0,
        y=0.98,
        horizontalalignment="left",
        fontweight="bold",
    )


def left_aligned_xlabel(ax: plt.Axes, text: str):
    ax.set_xlabel(
        text,
        fontsize=LABEL_FONT_SIZE,
        horizontalalignment="left",
    )
    ax.xaxis.set_label_coords(0, -0.12)


def median_ci(
    data, percentile: float = 0.5, confidence: float = 0.99
) -> tuple[float, float]:
    """
    CI for the median via Theorem 2.1 (rank-based normal approx).
    See https://leboudec.github.io/perfeval/
    """
    arr = np.asarray(data, dtype=float)
    n = arr.size
    if n < 2:
        v = float(arr[0]) if n else 0.0
        return v, v
    data_sorted = np.sort(arr)
    z = stats.norm.ppf((1 + confidence) / 2)
    exp_rank = n * percentile
    se_rank = np.sqrt(exp_rank * (1 - percentile))
    j = int(np.floor(exp_rank - z * se_rank) - 1)
    k = int(np.ceil(exp_rank + z * se_rank))
    return float(data_sorted[max(0, j)]), float(data_sorted[min(n - 1, k)])


def custom_violinplot(ax, data, pos, color):
    """
    Horizontal violin with trimmed tails + thick line for median CI.
    """
    arr = np.asarray(data, dtype=float)
    trimmed = stats.trimboth(arr, 0.05)
    parts = ax.violinplot(
        trimmed,
        positions=pos,
        showmedians=True,
        orientation="horizontal",
    )
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(0.5)
    for partname in ("cbars", "cmins", "cmaxes", "cmedians"):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor(color)
            vp.set_linewidth(1)

    ci_min, ci_max = median_ci(arr)
    ax.hlines(pos, ci_min, ci_max, linestyle="-", lw=3, color=color)

def get_name(run: dict, path: Path) -> str:
    if "name" in run:
        return run["name"]
    if "model_size" in run:
        return run["model_size"]
    return path.stem


def compare_runs(run_paths: list[Path], title: str, output_file: Path, metrics: list[tuple[str, str]] = METRICS):
    runs = [load_run(p) for p in run_paths]
    names = [get_name(r, p) for r, p in zip(runs, run_paths)]
    colors = get_color_cycle()

    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        n_metrics, 1, figsize=(COLUMN_WIDTH, n_metrics * len(runs))
    )
    if n_metrics == 1:
        axes = [axes]

    for ax, (key, label) in zip(axes, metrics):
        oom_positions = []
        for i, run in enumerate(runs):
            values = run.get(key, [])
            values = values[WARMUP_ITERS:] if len(values) > WARMUP_ITERS else values
            if not values:
                oom_positions.append(i + 1)
                continue
            pos = [i + 1]
            color = colors[i % len(colors)]
            custom_violinplot(ax, values, pos, color)

        ax.set_yticks(
            [i + 1 for i in range(len(runs))], names, fontsize=LABEL_FONT_SIZE
        )
        for ypos in oom_positions:
            xlim = ax.get_xlim()
            x = xlim[0] + (xlim[1] - xlim[0]) * 0.02
            ax.text(x, ypos, "OOM", va="center", ha="left",
                    fontsize=LABEL_FONT_SIZE, color="crimson", fontweight="bold")
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
        ax.set_title(label, fontsize=TITLE_FONT_SIZE, fontweight="bold", loc="left")

    if title:
        suptitle(fig, title)
    
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"saved {output_file}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-f",
        "--files",
        nargs="+",
        type=Path,
        required=True,
        help="One or more run JSON files (or directories of JSONs) produced by logparse.py",
    )
    ap.add_argument("-t", "--title", type=str, default="", help="Plot title")
    ap.add_argument(
        "-o",
        "--output-file",
        type=Path,
        default=Path("plot.png"),
        help="File to save plot",
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        choices=[k for k, _ in METRICS],
        default=[k for k, _ in METRICS],
        help="Which metric panels to include (default: all). "
             f"Choices: {[k for k, _ in METRICS]}",
    )
    args = ap.parse_args()

    files: list[Path] = []
    for f in args.files:
        if f.is_dir():
            found = sorted(f.glob("*.json"))
            if not found:
                print(f"error: no JSON files in directory: {f}", file=sys.stderr)
                return 1
            files.extend(found)
        elif f.is_file():
            files.append(f)
        else:
            print(f"error: path not found: {f}", file=sys.stderr)
            return 1
    args.files = files

    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["CMU Sans Serif"]
    plt.rcParams["font.weight"] = "medium"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.labelsize"] = LABEL_FONT_SIZE
    plt.rcParams["ytick.labelsize"] = LABEL_FONT_SIZE
    plt.rcParams["legend.fontsize"] = LABEL_FONT_SIZE

    active_metrics = [(k, label) for k, label in METRICS if k in args.metrics]
    compare_runs(args.files, args.title, args.output_file, active_metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
