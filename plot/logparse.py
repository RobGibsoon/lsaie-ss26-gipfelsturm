"""
Parse Megatron training log into JSON of per-iteration metrics.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ITER_RE = re.compile(
    r"\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]\s+"
    r"iteration\s+(?P<iteration>\d+)/\s*(?P<total_iterations>\d+)\s*\|\s*"
    r"consumed samples:\s*(?P<consumed_samples>\d+)\s*\|\s*"
    r"elapsed time per iteration \(ms\):\s*(?P<elapsed_time_ms>[\d.]+)\s*\|\s*"
    r"throughput per GPU \(TFLOP/s/GPU\):\s*(?P<throughput_tflops_per_gpu>[\d.]+)\s*\|\s*"
    r"tokens/sec/GPU:\s*(?P<tokens_per_sec_per_gpu>\d+)\s*\|\s*"
    r"learning rate:\s*(?P<learning_rate>[\d.E+\-]+)\s*\|\s*"
    r"global batch size:\s*(?P<global_batch_size>\d+)\s*\|\s*"
    r"lm loss:\s*(?P<lm_loss>[\d.E+\-]+)\s*\|\s*"
    r"loss scale:\s*(?P<loss_scale>[\d.E+\-]+)\s*\|\s*"
    r"grad norm:\s*(?P<grad_norm>[\d.E+\-]+)\s*\|\s*"
    r"number of skipped iterations:\s*(?P<skipped_iterations>\d+)\s*\|\s*"
    r"number of nan iterations:\s*(?P<nan_iterations>\d+)"
)

INT_FIELDS = {
    "iteration",
    "total_iterations",
    "consumed_samples",
    "tokens_per_sec_per_gpu",
    "global_batch_size",
    "skipped_iterations",
    "nan_iterations",
}
FLOAT_FIELDS = {
    "elapsed_time_ms",
    "throughput_tflops_per_gpu",
    "learning_rate",
    "lm_loss",
    "loss_scale",
    "grad_norm",
}


def parse(log_path: Path, name: str) -> dict:
    data = {"name": name, "timestamp": None}
    arrays: dict[str, list] = {"timestamp_iter": []}
    for f in INT_FIELDS | FLOAT_FIELDS:
        arrays[f] = []

    with log_path.open("r", errors="replace") as fh:
        for line in fh:
            m = ITER_RE.search(line)
            if not m:
                continue
            gd = m.groupdict()
            if data["timestamp"] is None:
                data["timestamp"] = gd["timestamp"]
            arrays["timestamp_iter"].append(gd["timestamp"])
            for f in INT_FIELDS:
                arrays[f].append(int(gd[f]))
            for f in FLOAT_FIELDS:
                arrays[f].append(float(gd[f]))

    data.update(arrays)
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path, help="Path to log file")
    ap.add_argument("name", help="Legend name for this run")
    ap.add_argument("-o", "--output", type=Path, help="Output JSON path")
    args = ap.parse_args()

    if not args.log.is_file():
        print(f"error: log file not found: {args.log}", file=sys.stderr)
        return 1

    parsed = parse(args.log, args.name)
    payload = json.dumps(parsed, indent=2)
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
