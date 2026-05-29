#!/bin/bash
#SBATCH --job-name=yarn-v2-1k-code-math
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=32
#SBATCH --mem=480G
#SBATCH --partition=standard-g
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --account=project_465002530
#SBATCH -o /scratch/project_462000963/bmoell/yarn-v2-1k-code-math-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/yarn-v2-1k-code-math-%j.err

# YaRN continued pre-training v2 — multilingual + code + math mix.
#
# Identical to yarn_multilingual_v2_1k.sbatch except the data mix includes:
#   - StarCoder 10%-sample (oellm-v1-256k tokenizer): 10% of training tokens
#   - MegaMath-full subsets (oellm-v1-256k tokenizer): 10% of training tokens
#     (proportionally weighted by file size across non-empty subsets)
#   - Multilingual 31-language tier mix: remaining 80% of training tokens
#
# All other settings (model, parallelism, optimizer, iterations) are the same
# as the baseline yarn_multilingual_v2_1k.sbatch for a clean A/B comparison.
#
# Data paths:
#   Multilingual: $HF_DATA_DIR/merged  (same as baseline)
#   MegaMath:     /scratch/project_462000963/preprocessed/oellm-v1-256k/LLM360/MegaMath-full/
#   StarCoder:    /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/starcoder-10p-sample/
#
# Submit: sbatch lumi/slurm/yarn_multilingual_v2_1k_code_math.sbatch

set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p /scratch/project_462000963/bmoell/logs-yarn-multilingual-v2-code-math
    sbatch "$0" "$@"
    exit
fi

ln -sf "/scratch/project_462000963/bmoell/yarn-v2-1k-code-math-${SLURM_JOBID}.out" \
       "/scratch/project_462000963/bmoell/yarn-v2-1k-code-math-latest.out" 2>/dev/null || true

if [[ -v SLURM_RESTART_COUNT ]]; then
    echo "RUN RESTARTED (count: $SLURM_RESTART_COUNT)"
fi

module purge

# ── Environment ──────────────────────────────────────────────────────────────
export CONTAINER=/scratch/project_462000963/containers/staging/rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0.sif
export BIND_DIRS=/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd

export ROCM_PATH=/opt/rocm-6.4.4
export LD_LIBRARY_PATH="$ROCM_PATH/lib:/usr/lib:${LD_LIBRARY_PATH:-}"
BIND_DIRS="${BIND_DIRS},/opt/rocm/lib/librccl.so:/opt/venv/lib/python3.12/site-packages/torch/lib/librccl.so"

export PYTHONUSERBASE=""
export CC=gcc
export CXX=g++

c="fe"
BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9998
export WORLD_SIZE=$SLURM_NTASKS
export OMP_NUM_THREADS=2
export HSA_ENABLE_SDMA=0

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
export HSA_FORCE_FINE_GRAIN_PCIE=1
export FI_HMEM=rocr
export FI_HMEM_ROCR_USE_DMABUF=0
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072
export PYTHONWARNINGS=ignore
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0

# Load W&B key from ~/.wandb_credentials (never commit the key itself)
if [[ -f "$HOME/.wandb_credentials" ]]; then
    source "$HOME/.wandb_credentials"
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKDIR=/scratch/project_462000963/bmoell
MEGATRON_DIR=/flash/project_462000963/jouni/test/NVIDIA-Megatron-LM
LAUNCH_SCRIPT=/flash/project_462000963/jouni/oellm/launch.sh

OUTPUT_DIR=/flash/project_462000963/bmoell/yarn-multilingual-v2-1k-code-math
CHECKPOINT_PATH=$OUTPUT_DIR/checkpoints
TENSORBOARD_DIR=$OUTPUT_DIR/tensorboard/${SLURM_JOB_NAME}-${SLURM_JOBID}
DATA_CACHE_PATH=$OUTPUT_DIR/cache_9B_multilingual_v2_code_math
mkdir -p "$CHECKPOINT_PATH" "$TENSORBOARD_DIR" "$DATA_CACHE_PATH"

export HF_HOME="$WORKDIR/hf"

# ── Data — multilingual (80%) + MegaMath (10%) + StarCoder (10%) ─────────────
HF_DATA_DIR=${HF_DATA_DIR:-/flash/project_462000963/bmoell/data_tokenized_hf_multilingual}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-/flash/project_462000963/jouni/checkpoints/converted-checkpoints/oellm-datamix-9b-80-20}

MEGAMATH_DIR=/scratch/project_462000963/preprocessed/oellm-v1-256k/LLM360/MegaMath-full
STARCODER_PREFIX=/scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/starcoder-10p-sample/starcoder-10p-sample

echo "=== Building data_path.args: multilingual 80% + MegaMath 10% + StarCoder 10% ==="
singularity exec -B "$BIND_DIRS" "$CONTAINER" python3 -u - \
    "$HF_DATA_DIR/merged" "$HF_DATA_DIR/data_path_v2_code_math.args" \
    "$MEGAMATH_DIR" "$STARCODER_PREFIX" << 'PYEOF'
import sys, re, pathlib

merged_dir      = pathlib.Path(sys.argv[1])
out_file        = pathlib.Path(sys.argv[2])
megamath_dir    = pathlib.Path(sys.argv[3])
starcoder_prefix = pathlib.Path(sys.argv[4])

MULTILINGUAL_WEIGHT = 0.80
MATH_WEIGHT         = 0.10
CODE_WEIGHT         = 0.10

# ── Multilingual tier files ───────────────────────────────────────────────────
tier_weights = {"16k_plus": 0.75, "4_16k": 0.15, "under4k": 0.10}
_pat = re.compile(r'^([a-z]+)_(16k_plus|4_16k|under4k)_text_document\.bin$')
langs = sorted(set(
    m.group(1)
    for p in merged_dir.glob('*.bin')
    for m in [_pat.match(p.name)] if m
))
if not langs:
    print("ERROR: no language bin files found in merged dir — check path")
    sys.exit(1)
print(f"Discovered {len(langs)} languages: {langs}")

entries = []
for lang in langs:
    for tier, tw in tier_weights.items():
        prefix = merged_dir / f"{lang}_{tier}_text_document"
        if not (prefix.parent / (prefix.name + ".bin")).exists():
            print(f"  WARNING: {prefix}.bin not found — omitting")
            continue
        entries.append((MULTILINGUAL_WEIGHT * tw / len(langs), str(prefix)))

# ── MegaMath subsets (proportional by .bin file size within 10% budget) ───────
megamath_subsets = [
    "megamath-web",
    "megamath-text-code-block",
    "megamath-web-pro",
    "megamath-qa",
    "megamath-translated-code",
]
math_sizes = {}
for subset in megamath_subsets:
    bin_path = megamath_dir / subset / f"{subset}_text_document.bin"
    if bin_path.exists() and bin_path.stat().st_size > 1024:
        math_sizes[subset] = bin_path.stat().st_size
    else:
        print(f"  WARNING: {bin_path} missing or empty — omitting")

total_math_bytes = sum(math_sizes.values())
for subset, sz in math_sizes.items():
    prefix = str(megamath_dir / subset / f"{subset}_text_document")
    w = MATH_WEIGHT * (sz / total_math_bytes)
    entries.append((w, prefix))
    print(f"  MegaMath {subset}: {sz/1e9:.1f} GB  weight={w:.4f}")

# ── StarCoder (single entry, full 10% budget) ─────────────────────────────────
starcoder_bin = pathlib.Path(str(starcoder_prefix) + ".bin")
if not starcoder_bin.exists():
    print(f"ERROR: StarCoder bin not found at {starcoder_bin}")
    sys.exit(1)
entries.append((CODE_WEIGHT, str(starcoder_prefix)))
print(f"  StarCoder: {starcoder_bin.stat().st_size/1e9:.1f} GB  weight={CODE_WEIGHT:.4f}")

# ── Normalize and write ───────────────────────────────────────────────────────
total_w = sum(w for w, _ in entries)
entries = [(w / total_w, p) for w, p in entries]
args_line = " ".join(f"{w:.6f} {p}" for w, p in entries)
out_file.write_text(args_line + "\n")

print(f"\nWritten {len(entries)} entries → {out_file}")
print(f"  Multilingual ({len(langs)} langs × 3 tiers): {MULTILINGUAL_WEIGHT:.0%}")
print(f"  MegaMath ({len(math_sizes)} subsets): {MATH_WEIGHT:.0%}")
print(f"  StarCoder 10%-sample: {CODE_WEIGHT:.0%}")
PYEOF

if [ -z "${DATA_PATH:-}" ] && [ -f "${HF_DATA_DIR}/data_path_v2_code_math.args" ]; then
    DATA_PATH="$(cat "${HF_DATA_DIR}/data_path_v2_code_math.args")"
elif [ -z "${DATA_PATH:-}" ]; then
    echo "ERROR: data_path_v2_code_math.args not written — check merged dir" >&2
    exit 1
fi

# ── Model (9B: 32 layers, hidden 4096, 32 heads) ──────────────────────────────
NUM_LAYERS=32
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=14336
NUM_ATTENTION_HEADS=32
NUM_QUERY_GROUPS=32
TIE_WORD_EMBEDDINGS=1
INIT_METHOD_STD=0.02
SEQ_LENGTH=32768
ROTARY_BASE=10000

# Start from the same base checkpoint as the baseline run.
LOAD_CKPT_PATH=/flash/project_462000963/jouni/checkpoints/oellm-9b-80-20-TP-2-PP-4
SAVE_CKPT_PATH=$CHECKPOINT_PATH
mkdir -p "$SAVE_CKPT_PATH"

# ── Parallelism (256 GPUs: TP=2, PP=4, CP=4) ─────────────────────────────────
PIPELINE_MODEL_PARALLEL_SIZE=4
TENSOR_MODEL_PARALLEL_SIZE=2
CONTEXT_PARALLEL_SIZE=4
NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE=1
PROFILE=0

# ── Optimizer ─────────────────────────────────────────────────────────────────
ADAM_BETA1=0.9
ADAM_BETA2=0.95
ADAM_EPS=1e-8
LR=1e-5
MIN_LR=1e-7
WARMUP_FRACTION=1/10
COOLDOWN_FRACTION=1/5
CLIP_GRAD=1.0
WEIGHT_DECAY=0.05

# ── Training — 1000 iters (same as baseline for A/B comparison) ───────────────
FSDP=0
GLOBAL_BATCH_SIZE=128
MICRO_BATCH_SIZE=1
RECOMPUTATION=1
TRAIN_TOKENS=4_194_304_000   # 1000 iters at GBS=128, SEQ=32768

LOG_INTERVAL=1
SAVE_INTERVAL=100
EVAL_INTERVAL=5000
EVAL_ITERS=100

######################################################################
# DERIVED SETTINGS (do not edit — computed from above)
######################################################################

confirm_unset() {
    local varname="$1"
    if [ -n "${!varname+x}" ]; then
        echo "Error: variable '$varname' should not be set." >&2
        exit 1
    fi
}
confirm_unset "TRAIN_ITERS"
confirm_unset "LR_DECAY_ITERS"
confirm_unset "LR_WSD_DECAY_ITERS"

divide_rounding_up() { echo $((($1+$2-1)/$2)); }

TRAIN_TOKENS=${TRAIN_TOKENS//_}
ITER_TOKENS=$((SEQ_LENGTH * GLOBAL_BATCH_SIZE))
TRAIN_ITERS=$(divide_rounding_up $TRAIN_TOKENS $ITER_TOKENS)

WARMUP_CALC=$((TRAIN_ITERS * ${WARMUP_FRACTION}))
if [ $WARMUP_CALC -gt 25000 ]; then
    LR_WARMUP_ITERS=25000
else
    LR_WARMUP_ITERS=$WARMUP_CALC
fi

LR_WSD_DECAY_ITERS=$((TRAIN_ITERS*${COOLDOWN_FRACTION}))
LR_DECAY_ITERS=$TRAIN_ITERS

######################################################################
# BUILD ARGS
######################################################################

DATA_ARGS=(
    --data-path "$DATA_PATH"
    --data-cache-path "$DATA_CACHE_PATH"
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_MODEL"
    --make-vocab-size-divisible-by 128
    --dataloader-type single
    --num-workers 1
)

MODEL_ARGS=(
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-attention-heads $NUM_ATTENTION_HEADS
)

if [ "$NUM_QUERY_GROUPS" != "$NUM_ATTENTION_HEADS" ]; then
    MODEL_ARGS+=(
        --group-query-attention
        --num-query-groups $NUM_QUERY_GROUPS
    )
fi

if [ "$TIE_WORD_EMBEDDINGS" = "0" ]; then
    MODEL_ARGS+=(--untie-embeddings-and-output-weights)
fi

if [ "$FSDP" = "1" ]; then
    PARALLEL_ARGS=(--use-torch-fsdp2)
else
    PARALLEL_ARGS=(
        --tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE
        --pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE
        --context-parallel-size $CONTEXT_PARALLEL_SIZE
        --sequence-parallel
        --use-distributed-optimizer
    )
fi

if [ "$PROFILE" = "1" ]; then
    PROFILE_ARGS=(
        --use-pytorch-profiler
        --profile-ranks 0
        --profile-step-start 5
        --profile-step-end 7
    )
else
    PROFILE_ARGS=()
fi

MODEL_ARGS+=(
    --use-flash-attn
    --attention-softmax-in-fp32
    --max-position-embeddings $SEQ_LENGTH
    --seq-length $SEQ_LENGTH
    --position-embedding-type yarn
    --yarn-original-max-position-embeddings 2048
    --yarn-scaling-factor 16.0
    --yarn-mscale 1.277    # 0.1 * ln(16.0) + 1.0 — fixes attention decay at 32K
    --rotary-base $ROTARY_BASE
    --disable-bias-linear
    --init-method-std $INIT_METHOD_STD
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters $TRAIN_ITERS
    --bf16
    --swiglu
    --no-async-tensor-model-parallel-allreduce
    --no-masked-softmax-fusion
    --no-gradient-accumulation-fusion
    --no-bias-dropout-fusion
    --no-rope-fusion
    --distributed-timeout-minutes 30
    --overlap-grad-reduce
    --empty-unused-memory-level 2
    --no-load-optim
    --no-load-rng
)

if [ "$RECOMPUTATION" = "1" ]; then
    MODEL_ARGS+=(--recompute-activations)
fi

OPTIMIZER_ARGS=(
    --optimizer adam
    --adam-beta1 $ADAM_BETA1
    --adam-beta2 $ADAM_BETA2
    --adam-eps $ADAM_EPS
    --lr $LR
    --min-lr $MIN_LR
    --lr-decay-style WSD
    --lr-wsd-decay-style linear
    --lr-warmup-iters $LR_WARMUP_ITERS
    --lr-decay-iters $LR_DECAY_ITERS
    --lr-wsd-decay-iters $LR_WSD_DECAY_ITERS
    --clip-grad $CLIP_GRAD
    --weight-decay $WEIGHT_DECAY
)

OUTPUT_ARGS=(
    --eval-interval $EVAL_INTERVAL
    --eval-iters $EVAL_ITERS
    --tensorboard-dir "$TENSORBOARD_DIR"
    --tensorboard-queue-size 5
    --log-throughput
    --log-progress
    --log-interval $LOG_INTERVAL
    --wandb-project openeuro-longctx
    --wandb-exp-name "yarn-v2-1k-code-math-${SLURM_JOBID}"
    --wandb-entity birgermoell
)

if [ $PIPELINE_MODEL_PARALLEL_SIZE -gt 1 ] && [ $NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE -gt 1 ]; then
    PARALLEL_ARGS+=(--num-layers-per-virtual-pipeline-stage $NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE)
fi

CHECKPOINT_ARGS=(
    --ckpt-format torch
    --load "$LOAD_CKPT_PATH"
    --save "$SAVE_CKPT_PATH"
    --save-interval $SAVE_INTERVAL
)

COMMAND="
    $MEGATRON_DIR/pretrain_gpt.py
    ${MODEL_ARGS[@]}
    ${OPTIMIZER_ARGS[@]}
    ${PARALLEL_ARGS[@]}
    ${OUTPUT_ARGS[@]}
    ${CHECKPOINT_ARGS[@]}
    ${DATA_ARGS[@]}
    ${PROFILE_ARGS[@]}
"

echo '============= COMMAND: ============='
echo "$COMMAND"
echo '===================================='
echo "START $SLURM_JOBID: $(date)"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"

srun \
    --label \
    --cpu-bind=mask_cpu:$BIND_MASK \
    singularity exec \
    -B "$BIND_DIRS" \
    "$CONTAINER" \
    "$LAUNCH_SCRIPT" \
    $COMMAND

echo "END $SLURM_JOBID: $(date)"
