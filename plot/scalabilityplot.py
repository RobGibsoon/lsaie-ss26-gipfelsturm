"""
Throughput plots from parsed run JSONs.

Two modes:
  --mode size-sweep : X = model size, one line per group (e.g. precision).
                      Use to show FP8 vs BF16 crossover across model scales.
                      Each line should be the BEST config at each size.

  --mode strong     : X = number of GPUs, one line per group (e.g. precision).
                      Use to show how a FIXED model scales with more GPUs.
                      All points must be the SAME model size.

The grouping (legend) comes from each run's "name"/"run_name" field. Examples:
  size-sweep groups: {"bf16-best", "fp8-best"}
  strong groups:     {"bf16", "fp8"}  (all runs for one model size)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LABEL_FONT_SIZE = 9
TITLE_FONT_SIZE = 10
COLUMN_WIDTH = 6
WARMUP_ITERS = 5

SIZE_TO_PARAMS = {
    "125m": 125e6,
    "350m": 350e6,
    "760m": 760e6,
    "1.5b": 1.5e9,
    "3b": 3e9,
    "8b": 8e9,
}


def load_run(path: Path) -> dict:
    with path.open("r") as fh:
        return json.load(fh)


def get_color_cycle():
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


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


def parse_size(s: str) -> float:
    if s in SIZE_TO_PARAMS:
        return SIZE_TO_PARAMS[s]
    s = s.strip().lower()
    if s.endswith("b"):
        return float(s[:-1]) * 1e9
    if s.endswith("m"):
        return float(s[:-1]) * 1e6
    return float(s)


def get_group(run: dict, path: Path) -> str:
    if "name" in run:
        return run["name"]
    if "run_name" in run:
        return run["run_name"]
    return path.stem


def median_metric(run: dict, metric_key: str) -> float:
    """Median of `metric_key` series after discarding warmup iterations."""
    values = run.get(metric_key, [])
    values = values[WARMUP_ITERS:] if len(values) > WARMUP_ITERS else values
    if not values:
        return float("nan")
    return float(np.median(values))


def num_gpus(run: dict) -> int | None:
    """Try common field names for GPU count."""
    for k in ("num_gpus", "n_gpus", "world_size", "gpus", "total_gpus"):
        if k in run:
            return int(run[k])
    # Derive from nodes if available
    nodes = run.get("nodes") or run.get("num_nodes")
    gpus_per_node = run.get("gpus_per_node", 4)  # GH200 default
    if nodes is not None:
        return int(nodes) * int(gpus_per_node)
    return None


def plot_size_sweep(run_paths: list[Path], title: str, output_file: Path, metric: str, ylabel: str):
    runs = [load_run(p) for p in run_paths]

    groups: dict[str, list[tuple[float, float, str]]] = {}
    for run, path in zip(runs, run_paths):
        size_str = run.get("model_size")
        if size_str is None:
            print(f"warn: skip {path}, no model_size", file=sys.stderr)
            continue
        x = parse_size(size_str)
        y = median_metric(run, metric)
        if np.isnan(y):
            print(f"warn: skip {path}, no {metric}", file=sys.stderr)
            continue
        groups.setdefault(get_group(run, path), []).append((x, y, size_str))

    colors = get_color_cycle()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.6))

    all_sizes: dict[float, str] = {}
    for i, (name, points) in enumerate(sorted(groups.items())):
        points.sort(key=lambda t: t[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        for p in points:
            all_sizes[p[0]] = p[2]
        color = colors[i % len(colors)]
        ax.plot(xs, ys, marker="o", linestyle="-", color=color, label=name)

    ax.set_xscale("log")
    ax.set_xlabel("Model size (parameters)", fontsize=LABEL_FONT_SIZE)
    ax.set_title(ylabel, fontsize=TITLE_FONT_SIZE, loc="left")

    if all_sizes:
        ticks = sorted(all_sizes.keys())
        ax.set_xticks(ticks)
        ax.set_xticklabels([all_sizes[t] for t in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())

    ax.grid(True)
    ax.legend(fontsize=LABEL_FONT_SIZE)

    if title:
        suptitle(fig, title)

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"saved {output_file}")


def plot_strong_scaling(run_paths: list[Path], title: str, output_file: Path, metric: str, ylabel: str, show_ideal: bool):
    """X-axis is GPU count; all runs must be the SAME model size."""
    runs = [load_run(p) for p in run_paths]

    sizes_seen = {r.get("model_size") for r in runs if r.get("model_size") is not None}
    if len(sizes_seen) > 1:
        print(
            f"error: strong-scaling plot requires a single model size, got: {sizes_seen}",
            file=sys.stderr,
        )
        sys.exit(1)

    groups: dict[str, list[tuple[int, float]]] = {}
    for run, path in zip(runs, run_paths):
        g = num_gpus(run)
        if g is None:
            print(f"warn: skip {path}, no GPU count field", file=sys.stderr)
            continue
        y = median_metric(run, metric)
        if np.isnan(y):
            print(f"warn: skip {path}, no {metric}", file=sys.stderr)
            continue
        groups.setdefault(get_group(run, path), []).append((g, y))

    colors = get_color_cycle()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.6))

    all_gpu_counts: set[int] = set()
    for i, (name, points) in enumerate(sorted(groups.items())):
        points.sort(key=lambda t: t[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        all_gpu_counts.update(xs)
        color = colors[i % len(colors)]
        ax.plot(xs, ys, marker="o", linestyle="-", color=color, label=name)

        # Optional: dashed "ideal" line at the smallest-GPU-count value
        if show_ideal and len(points) > 1:
            baseline_y = ys[0]
            ax.axhline(
                baseline_y,
                color=color,
                linestyle="--",
                alpha=0.4,
                label=f"{name} ideal (perfect scaling)",
            )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of GPUs", fontsize=LABEL_FONT_SIZE)
    ax.set_title(ylabel, fontsize=TITLE_FONT_SIZE, loc="left")

    if all_gpu_counts:
        ticks = sorted(all_gpu_counts)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())

    ax.grid(True)
    ax.legend(fontsize=LABEL_FONT_SIZE)

    if title:
        suptitle(fig, title)

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"saved {output_file}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "-f",
        "--files",
        nargs="+",
        type=Path,
        required=True,
        help="One or more run JSON files (or directories of JSONs) produced by logparse.py",
    )
    ap.add_argument(
        "--mode",
        choices=["size-sweep", "strong"],
        required=True,
        help="size-sweep: X=model size. strong: X=GPU count (single model size).",
    )
    ap.add_argument(
        "--metric",
        default="throughput",
        help="JSON field to read per-iteration values from (default: throughput). "
             "Use 'tokens_per_sec_per_gpu' if that's what your logparse produces.",
    )
    ap.add_argument(
        "--ylabel",
        default="Throughput (TFLOP/s/GPU)",
        help="Y-axis title text. Match this to whatever --metric actually contains.",
    )
    ap.add_argument(
        "--ideal",
        action="store_true",
        help="In strong-scaling mode, draw a dashed line at the 1-GPU baseline showing perfect per-GPU scaling.",
    )
    ap.add_argument("-t", "--title", type=str, default="", help="Plot title")
    ap.add_argument(
        "-o",
        "--output-file",
        type=Path,
        default=Path("plot.png"),
        help="File to save plot",
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

    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "CMU Serif", "Computer Modern Roman", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.weight"] = "medium"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.labelsize"] = LABEL_FONT_SIZE
    plt.rcParams["ytick.labelsize"] = LABEL_FONT_SIZE
    plt.rcParams["legend.fontsize"] = LABEL_FONT_SIZE

    if args.mode == "size-sweep":
        plot_size_sweep(files, args.title, args.output_file, args.metric, args.ylabel)
    else:
        plot_strong_scaling(files, args.title, args.output_file, args.metric, args.ylabel, args.ideal)
    return 0


if __name__ == "__main__":
    sys.exit(main())