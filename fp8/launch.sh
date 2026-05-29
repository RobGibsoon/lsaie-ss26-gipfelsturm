#!/bin/bash
#
# Usage: ./fp8/launch.sh <mode> <model_size> [steps] [nodes]
#
# Modes:     throughput   (50 steps, W&B)
#            train        (N steps, W&B + TensorBoard)
#            profile      (15 steps, NSYS trace, no W&B)
#
# Sizes:     125m, 350m, 760m, 1.5b, 3b, 8b   (TP=1 PP=1 by default)
#            32b                                (TP=4 PP=1 by default, single-node)
#            140b                               (TP=4 PP=4 by default, multi-node)
#
# Env vars (set before calling):
#   PRECISION=bf16|fp8          default: bf16
#   FP8_FORMAT=hybrid|e4m3      default: hybrid   (ignored unless PRECISION=fp8)
#   FP8_RECIPE=delayed|tensorwise|blockwise
#                               default: delayed  (ignored unless PRECISION=fp8)
#   FP8_AMAX_HISTORY_LEN=N      default: 16     (ignored unless PRECISION=fp8)
#   FP8_AMAX_HISTORY_LEN=N      default: 16     (ignored unless PRECISION=fp8)
#   FP8_PARAM_GATHER=0|1        default: 0        (only valid with delayed recipe)
#   TP=N                        override tensor parallel (default per model size)
#   PP=N                        override pipeline parallel (default per model size)
#
# Examples:
#   ./fp8/launch.sh throughput 760m 50 1
#   PRECISION=fp8 ./fp8/launch.sh throughput 760m 50 1
#   PRECISION=fp8 FP8_RECIPE=tensorwise ./fp8/launch.sh throughput 8b 50 4
#   PRECISION=fp8 FP8_PARAM_GATHER=1 ./fp8/launch.sh throughput 760m 50 1
#   ./fp8/launch.sh profile 760m 15 1
#   ./fp8/launch.sh throughput 32b 50 1

set -euo pipefail

source "$(dirname "$0")/../config.sh"

MODE=${1:?Usage: ./fp8/launch.sh <mode> <model_size> [steps] [nodes]}
MODEL_SIZE=${2:?Usage: ./fp8/launch.sh <mode> <model_size> [steps] [nodes]}

PRECISION=${PRECISION:-bf16}
FP8_FORMAT=${FP8_FORMAT:-hybrid}
FP8_RECIPE=${FP8_RECIPE:-delayed}
FP8_AMAX_HISTORY_LEN=${FP8_AMAX_HISTORY_LEN:-16}
FP8_PARAM_GATHER=${FP8_PARAM_GATHER:-0}

if [ "$FP8_PARAM_GATHER" = "1" ] && [ "$FP8_RECIPE" != "delayed" ]; then
    echo "Error: FP8_PARAM_GATHER=1 requires FP8_RECIPE=delayed (got '$FP8_RECIPE')"
    exit 1
fi

################ Mode config ################
case $MODE in
    throughput)
        TRAINING_STEPS=${3:-50}
        NODES=${4:-4}
        TIME=00:20:00
        EVAL_INTERVAL=$TRAINING_STEPS
        EVAL_ITERS=0
        LR_WARMUP_ITERS=10
        LOGGING_EXTRA="--log-timers-to-tensorboard"
        WANDB=true
        PROFILE_ARGS=""
        NSYS_PREFIX=""
        ;;
    train)
        TRAINING_STEPS=${3:?Usage: ./fp8/launch.sh train <model_size> <steps> [nodes]}
        NODES=${4:-4}
        TIME=02:30:00
        EVAL_INTERVAL=1000
        EVAL_ITERS=10
        LR_WARMUP_ITERS=200
        LOGGING_EXTRA="
    --tensorboard-dir \$TENSORBOARD_DIR
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard"
        WANDB=true
        PROFILE_ARGS=""
        NSYS_PREFIX=""
        ;;
    profile)
        TRAINING_STEPS=${3:-15}
        NODES=${4:-1}
        TIME=00:30:00
        EVAL_INTERVAL=$TRAINING_STEPS
        EVAL_ITERS=0
        LR_WARMUP_ITERS=5
        LOGGING_EXTRA=""
        WANDB=false
        PROFILE_ARGS="--profile --profile-step-start 10 --profile-step-end 12 --profile-ranks 0"
        NSYS_PREFIX="nsys profile -s none -w true \
            --trace='nvtx,cudnn,cublas,cuda' \
            --output=\$LOG_DIR/trace-\$SLURM_JOB_ID.nsys-rep \
            --force-overwrite true \
            --capture-range=cudaProfilerApi \
            --capture-range-end=stop -x true"
        ;;
    *)
        echo "Unknown mode: $MODE. Choose: throughput, train, profile"
        exit 1
        ;;
esac

################ Model config ################
case $MODEL_SIZE in
    125m)
        NUM_LAYERS=12;  HIDDEN=768;   FFN=2048;  HEADS=12; KV_HEADS=4;  MBS=16
        TP_DEFAULT=1; PP_DEFAULT=1
        ;;
    350m)
        NUM_LAYERS=24;  HIDDEN=1024;  FFN=2816;  HEADS=16; KV_HEADS=4;  MBS=8
        TP_DEFAULT=1; PP_DEFAULT=1
        ;;
    760m)
        NUM_LAYERS=24;  HIDDEN=1536;  FFN=4096;  HEADS=16; KV_HEADS=4;  MBS=4
        TP_DEFAULT=1; PP_DEFAULT=1
        ;;
    1.5b)
        NUM_LAYERS=48;  HIDDEN=1600;  FFN=4352;  HEADS=20; KV_HEADS=4;  MBS=4
        TP_DEFAULT=1; PP_DEFAULT=1
        ;;
    3b)
        NUM_LAYERS=32;  HIDDEN=3072;  FFN=8192;  HEADS=24; KV_HEADS=8;  MBS=4
        TP_DEFAULT=1; PP_DEFAULT=1
        ;;
    8b)
        NUM_LAYERS=32;  HIDDEN=4096;  FFN=14336; HEADS=32; KV_HEADS=8;  MBS=1
        TP_DEFAULT=1; PP_DEFAULT=1
        ;;
    32b)
        # ~32B scale (Llama-2 34B-ish), TP=4
        # KV_HEADS=4 so HEADS(52) % KV_HEADS(4) == 0 (required by TransformerEngine)
        # Needs >=2 nodes: 1 node (DP=1) OOMs on optimizer states (~136 GB); 2 nodes (DP=2) is tight (~85 GB + activations)
        NUM_LAYERS=60;  HIDDEN=6656;  FFN=17920; HEADS=52; KV_HEADS=4;  MBS=1
        TP_DEFAULT=4; PP_DEFAULT=1
        NODES=${4:-2}
        ;;
    140b)
        # GPT-3-style ~145B architecture (~multi-node scale, TP=4 PP=4)
        # Needs 8 nodes + activation recompute: 4 nodes (DP=1) OOMs on optimizer states;
        # 8 nodes (DP=2) is tight (~91 GB before activations) so we enable selective recompute
        NUM_LAYERS=80;  HIDDEN=12288; FFN=32768; HEADS=96; KV_HEADS=8;  MBS=1
        TP_DEFAULT=4; PP_DEFAULT=4
        NODES=${4:-8}
        #ACTIVATION_RECOMPUTE="--recompute-granularity selective" perhaps uncomment if still OOMs with 8 nodes
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE. Choose: 125m, 350m, 760m, 1.5b, 3b, 8b, 32b, 140b"
        exit 1
        ;;
esac

TP=${TP:-$TP_DEFAULT}
PP=${PP:-$PP_DEFAULT}
ACTIVATION_RECOMPUTE=${ACTIVATION_RECOMPUTE:-}

GBS=256
SEQ_LEN=4096

# Clamp MBS so that MBS * DP <= GBS (required: GBS divisible by MBS * DP)
DP_SIZE=$(( NODES * 4 / (TP * PP) ))
while [ $((MBS * DP_SIZE)) -gt $GBS ]; do
    MBS=$((MBS / 2))
    if [ "$MBS" -lt 1 ]; then
        echo "Error: cannot fit MBS>=1 with DP=$DP_SIZE and GBS=$GBS"
        exit 1
    fi
done
if [ $((GBS % (MBS * DP_SIZE))) -ne 0 ]; then
    echo "Error: GBS=$GBS not divisible by MBS=$MBS * DP=$DP_SIZE"
    exit 1
fi

# Build a descriptive job name that includes precision
PREC_TAG="${PRECISION}"
if [ "$PRECISION" = "fp8" ]; then
    PREC_TAG="fp8-${FP8_FORMAT}-${FP8_RECIPE}"
fi
JOB_NAME="gipfel-fp8-${MODE}-${MODEL_SIZE}-tp${TP}pp${PP}-${PREC_TAG}-${TRAINING_STEPS}s-${NODES}n"

################ Precision flags ################
MIXED_PRECISION_FLAGS="    --bf16"
if [ "$PRECISION" = "fp8" ]; then
    MIXED_PRECISION_FLAGS="${MIXED_PRECISION_FLAGS}
    --fp8-format ${FP8_FORMAT}
    --fp8-recipe ${FP8_RECIPE}
    --fp8-amax-history-len ${FP8_AMAX_HISTORY_LEN}
    --fp8-amax-compute-algo max"
    if [ "$FP8_PARAM_GATHER" = "1" ]; then
        MIXED_PRECISION_FLAGS="${MIXED_PRECISION_FLAGS}
    --fp8-param-gather"
    fi
fi

################ Parallelism flags ################
DISTRIBUTED_FLAGS="    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --use-distributed-optimizer
    --overlap-grad-reduce"
# --overlap-param-gather is incompatible with pipeline parallelism
if [ "${PP}" -eq 1 ]; then
    DISTRIBUTED_FLAGS="${DISTRIBUTED_FLAGS}
    --overlap-param-gather"
fi

################ W&B block ################
if [ "$WANDB" = true ]; then
    WANDB_BLOCK='
# WANDB
if [ -n "$WANDB_API_KEY" ]; then
    echo "[$(date)] WANDB enabled."
    TRAINING_CMD="$TRAINING_CMD \
        --wandb-save-dir $LOG_DIR \
        --wandb-project $PROJECT_NAME \
        --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
    export WANDB_MODE=disabled
    echo "[$(date)] WANDB disabled."
fi'
else
    WANDB_BLOCK='export WANDB_MODE=disabled'
fi

################ Generate script ################
mkdir -p "$WORKDIR/logs"

SCRIPT="$WORKDIR/logs/${JOB_NAME}.sbatch"

cat > "$SCRIPT" << 'HEADER'
#!/bin/bash
HEADER

cat >> "$SCRIPT" << SBATCH_DIRECTIVES
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --time=${TIME}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --no-requeue
SBATCH_DIRECTIVES

cat >> "$SCRIPT" << 'BODY_HEAD'

echo "START TIME: $(date)"

################ Configs ################
BODY_HEAD

cat >> "$SCRIPT" << BODY_WORKDIR
WORKDIR=${WORKDIR}
MEGATRON_LM_DIR=\$WORKDIR/Megatron-LM
DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/cache
BODY_WORKDIR

cat >> "$SCRIPT" << CONFIGS

# Training config
MBS=${MBS}
GBS=${GBS}
SEQ_LEN=${SEQ_LEN}
TRAINING_STEPS=${TRAINING_STEPS}

# Logging
PROJECT_NAME=gipfelsturm
EXP_NAME=${MODE}-${MODEL_SIZE}-tp${TP}pp${PP}-${PREC_TAG}-\${SLURM_NNODES}n
LOG_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/\$PROJECT_NAME/\$EXP_NAME
TENSORBOARD_DIR=\$LOG_DIR/tensorboard
CONFIGS

cat >> "$SCRIPT" << 'SETUP'

#########################################

mkdir -p logs $LOG_DIR $TENSORBOARD_DIR $DATASET_CACHE_DIR

cd $MEGATRON_LM_DIR
flock $MEGATRON_LM_DIR/.git-lock bash -c "cd $MEGATRON_LM_DIR && git checkout -- . && git apply $WORKDIR/patches/*.patch"
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.inductor_cache
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
MASTER_ADDR=$(hostname)
MASTER_PORT=25678

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
)

SETUP

cat >> "$SCRIPT" << MODEL
NETWORK_SIZE_ARGS=(
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN}
    --ffn-hidden-size ${FFN}
    --num-attention-heads ${HEADS}
    --group-query-attention
    --num-query-groups ${KV_HEADS}
    --max-position-embeddings \$SEQ_LEN
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --seq-length \$SEQ_LEN
)
MODEL

cat >> "$SCRIPT" << TRAINING

TRAINING_ARGS=(
    --micro-batch-size \$MBS
    --global-batch-size \$GBS
    --train-iters \$TRAINING_STEPS
    --log-interval 1
    --eval-interval ${EVAL_INTERVAL}
    --eval-iters ${EVAL_ITERS}
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --optimizer adam
    --dataloader-type single
    --no-check-for-nan-in-loss-and-grad
    --manual-gc
    --manual-gc-interval 50
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

LEARNING_RATE_ARGS=(
    --lr 3e-4
    --lr-decay-style constant
    --lr-warmup-iters ${LR_WARMUP_ITERS}
)
TRAINING

cat >> "$SCRIPT" << 'REST'

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)

REST

cat >> "$SCRIPT" << MIXED_PRECISION

MIXED_PRECISION_ARGS=(
${MIXED_PRECISION_FLAGS}
)

MIXED_PRECISION

cat >> "$SCRIPT" << DISTRIBUTED

DISTRIBUTED_ARGS=(
${DISTRIBUTED_FLAGS}
)

DISTRIBUTED

cat >> "$SCRIPT" << 'LOGGING_START'

LOGGING_ARGS=(
    --log-throughput
    --log-progress
LOGGING_START

cat >> "$SCRIPT" << LOGGING_EXTRA
${LOGGING_EXTRA}
$([ -n "$PROFILE_ARGS" ] && echo "    $PROFILE_ARGS")
$([ -n "$ACTIVATION_RECOMPUTE" ] && echo "    $ACTIVATION_RECOMPUTE")
)
LOGGING_EXTRA

cat >> "$SCRIPT" << 'TOKENIZER'

TOKENIZER_ARGS=(
    --tokenizer-type GPT2BPETokenizer
    --vocab-file $WORKDIR/data/gpt2-vocab.json
    --merge-file $WORKDIR/data/gpt2-merges.txt
)

DATA_ARGS=(
    --data-path $DATA_PREFIX
    --data-cache-path $DATASET_CACHE_DIR
    --split 99,1,0
    --num-workers 1
)

TORCHRUN_ARGS=(
    --nproc-per-node $SLURM_GPUS_PER_NODE
    --nnodes $SLURM_NNODES
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
    --rdzv_backend c10d
    --max_restarts 0
    --tee 3
)

TRAINING_CMD="torchrun ${TORCHRUN_ARGS[@]} $MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]}"

TOKENIZER

cat >> "$SCRIPT" << 'WANDB_PLACEHOLDER'
WANDB_PLACEHOLDER

sed -i '/^WANDB_PLACEHOLDER$/d' "$SCRIPT"
cat >> "$SCRIPT" << WANDB_INSERT
${WANDB_BLOCK}
WANDB_INSERT

cat >> "$SCRIPT" << FOOTER

echo "CMD: \$TRAINING_CMD"
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 --cpus-per-task \$SLURM_CPUS_PER_TASK --wait 60 bash -c "numactl --membind=0-3 ${NSYS_PREFIX} \$TRAINING_CMD"

echo "END TIME: \$(date)"
FOOTER

chmod +x "$SCRIPT"

echo "Generated: $SCRIPT"
sbatch "$SCRIPT"
