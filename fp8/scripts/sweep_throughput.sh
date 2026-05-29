#!/bin/bash
#
# Submit a throughput sweep across model sizes, parallelism, and precision.
# Reads config.sh for WORKDIR/SBATCH_ACCOUNT. Run from the project root.
#
# Usage:
#   ./fp8/scripts/sweep_throughput.sh [--dry-run]
#
# Output: one submitted SLURM job per cell; sbatch files in logs/.
#
# Edit the arrays below to restrict the sweep.

DRY_RUN=false
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=true
fi

LAUNCH="./fp8/launch.sh"
STEPS=50

submit() {
    local model=$1; local nodes=$2; shift 2; local env_prefix="$*"
    local cmd="$env_prefix $LAUNCH throughput $model $STEPS $nodes"
    if [ "$DRY_RUN" = true ]; then
        echo "[dry-run] $cmd"
    else
        echo "Submitting: $cmd"
        eval "$cmd"
    fi
    sleep 1  # avoid hammering the scheduler
}

################ Phase 1: BF16 baseline ################
echo "=== Phase 1: BF16 baseline ==="
for model in 125m 760m 8b; do
    for nodes in 1 2 4 8; do
        submit "$model" "$nodes" "PRECISION=bf16"
    done
done

# Larger models at fixed node counts (needs enough nodes to fit in memory)
# 32b: 1 node OOMs (DP=1, optimizer states ~136 GB); start from 2 nodes
submit "32b"  2 "PRECISION=bf16 TP=4"
submit "32b"  4 "PRECISION=bf16 TP=4"
# 140b: 4 nodes OOMs (DP=1, ~145 GB); 8 nodes is tight but should fit (DP=2, ~91 GB)
submit "140b" 8 "PRECISION=bf16"

################ Phase 2: FP8 recipe sweep (at 760m, 1 node) ################
# Hybrid is what NVIDIA recommends for Hopper and is what the Megatron test configs use.
# In practice for Hopper: hybrid + delayed is the safe default (what NVIDIA ships in examples).
echo "=== Phase 2: FP8 recipe sweep ==="
for recipe in delayed tensorwise blockwise; do
    for fmt in hybrid e4m3; do
        submit "760m" 1 "PRECISION=fp8 FP8_FORMAT=$fmt FP8_RECIPE=$recipe"
    done
done

# FP8 param-gather (delayed + hybrid only)
submit "760m" 1 "PRECISION=fp8 FP8_FORMAT=hybrid FP8_RECIPE=delayed FP8_PARAM_GATHER=1"

# Also sweep winning recipe candidates at 8b
for recipe in delayed tensorwise; do
    submit "8b" 1 "PRECISION=fp8 FP8_FORMAT=hybrid FP8_RECIPE=$recipe"
done

################ Phase 2b: FP8 at scale (best recipe — update after Phase 2) ################
# Set these after reviewing Phase 2 results:
BEST_FORMAT=${BEST_FP8_FORMAT:-hybrid}
BEST_RECIPE=${BEST_FP8_RECIPE:-delayed}
FP8_ENV="PRECISION=fp8 FP8_FORMAT=$BEST_FORMAT FP8_RECIPE=$BEST_RECIPE"

echo "=== Phase 2b: FP8 at scale (format=$BEST_FORMAT recipe=$BEST_RECIPE) ==="
for model in 125m 760m 8b; do
    for nodes in 1 2 4 8; do
        submit "$model" "$nodes" "$FP8_ENV"
    done
done
submit "32b"  2 "$FP8_ENV TP=4"
submit "32b"  4 "$FP8_ENV TP=4"
submit "140b" 8 "$FP8_ENV"
