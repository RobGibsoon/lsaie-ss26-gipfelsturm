#!/usr/bin/env python3
"""
Parse Megatron-LM throughput logs into a CSV for plotting.

Usage:
    python fp8/scripts/parse_logs.py [--log-dir logs/logs] [--out fp8/results/throughput.csv]

Reads all gipfel-*.log files, extracts per-step tokens/sec/GPU and TFLOP/s/GPU,
takes the median over the stable window (skipping the first WARMUP_STEPS),
and writes one row per job.
"""

import argparse
import csv
import re
import statistics
from pathlib import Path

WARMUP_STEPS = 10  # skip these many steps before computing median

# Megatron stdout patterns
TOKENS_RE = re.compile(r"tokens/sec/GPU:\s*([\d.]+)")
TFLOPS_RE = re.compile(r"throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+)")
ITER_RE   = re.compile(r"iteration\s+(\d+)/")

# Job-name patterns  (gipfel-fp8-throughput-760m-tp1pp1-fp8-hybrid-delayed-50s-1n-<jobid>.log)
NAME_RE = re.compile(
    r"gipfel(?:-fp8)?-(?:throughput|train|profile)-"
    r"(?P<model>[^-]+(?:-[^-]+)?)-"          # model size (e.g. 1.5b)
    r"(?:tp(?P<tp>\d+)pp(?P<pp>\d+)-)?"      # optional tp/pp tag
    r"(?P<prec>bf16|fp8[^-]*(?:-[^-]+)*)?"   # optional precision tag
    r"-\d+s-(?P<nodes>\d+)n"
)

def parse_log(path: Path) -> dict | None:
    tps_vals, tflops_vals = [], []
    step = 0
    with path.open(errors="ignore") as f:
        for line in f:
            m = ITER_RE.search(line)
            if m:
                step = int(m.group(1))
            if step <= WARMUP_STEPS:
                continue
            m = TOKENS_RE.search(line)
            if m:
                tps_vals.append(float(m.group(1)))
            m = TFLOPS_RE.search(line)
            if m:
                tflops_vals.append(float(m.group(1)))

    if not tps_vals:
        return None

    # Parse job metadata from filename
    stem = path.stem  # strip .log
    nm = NAME_RE.search(stem)
    model  = nm.group("model")  if nm else "unknown"
    tp     = int(nm.group("tp")) if nm and nm.group("tp") else 1
    pp     = int(nm.group("pp")) if nm and nm.group("pp") else 1
    nodes  = int(nm.group("nodes")) if nm else 0
    prec   = nm.group("prec")   if nm and nm.group("prec") else "bf16"

    return {
        "jobid":    stem.rsplit("-", 1)[-1],
        "model":    model,
        "nodes":    nodes,
        "tp":       tp,
        "pp":       pp,
        "precision": prec,
        "tps_gpu_median":   round(statistics.median(tps_vals)),
        "tflops_gpu_median": round(statistics.median(tflops_vals), 1) if tflops_vals else "",
        "tps_gpu_std":      round(statistics.stdev(tps_vals)) if len(tps_vals) > 1 else 0,
        "n_steps":  len(tps_vals),
        "logfile":  path.name,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs/logs", help="Directory with *.log files")
    ap.add_argument("--out", default="fp8/results/throughput.csv")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted(log_dir.glob("gipfel-*.log")):
        row = parse_log(p)
        if row:
            rows.append(row)
        else:
            print(f"[skip] {p.name}: no throughput data (too few steps or warmup only)")

    if not rows:
        print("No data found.")
        return

    fieldnames = ["model", "nodes", "tp", "pp", "precision",
                  "tps_gpu_median", "tflops_gpu_median", "tps_gpu_std",
                  "n_steps", "jobid", "logfile"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    for r in sorted(rows, key=lambda x: (x["model"], x["nodes"], x["precision"])):
        print(f"  {r['model']:6s}  {r['nodes']}n  tp{r['tp']}pp{r['pp']}"
              f"  {r['precision']:25s}  {r['tps_gpu_median']:>8,} tok/s/GPU")


if __name__ == "__main__":
    main()
