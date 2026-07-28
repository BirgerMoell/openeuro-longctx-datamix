#!/bin/bash
#SBATCH --job-name=dsa-sparse8k-500
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/project_465002530/users/bmoell/longctx-extend/logs/%j.out
#SBATCH --error=/scratch/project_465002530/users/bmoell/longctx-extend/logs/%j.err

# Sustained sparse-adaptation gate: 500 updates after the completed iteration-300
# frozen-indexer warm checkpoint. All 36 layers use DSA with selected-set KL.
set -euo pipefail
module purge

DSADIR=${DSADIR:-/scratch/project_465002530/users/bmoell/dsa_exp/repo_10a949e}
MEG=${MEGATRON_ROOT:-/scratch/project_465002530/users/luomajou/oellm-test/NVIDIA-Megatron-LM}
EXT=${LONGCTX_ROOT:-/scratch/project_465002530/users/bmoell/longctx-extend}
WARM=${DSA_WARM_CHECKPOINT:-$EXT/output_dsa_warm_all_s_8k}
OUT=${DSA_OUT:-$EXT/output_dsa_sparse_all_s_8k}
TOK=${TOKENIZER_PATH:-/scratch/project_465002530/users/pyysalos/tokenizers/openeurollm/tokenizer-256k}
CONTAINER=${CONTAINER:-/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif}
BIND_DIRS=${BIND_DIRS:-/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd}
PRETRAIN_GPT=${PRETRAIN_GPT:-$MEG/pretrain_gpt.py}
DATA_BLEND_FILE=${DATA_BLEND_FILE:-$EXT/jouni_blend.txt}
TARGET_ITER=${TARGET_ITER:-800}
SAVE_INTERVAL=${SAVE_INTERVAL:-50}

for required in "$DSADIR/gpt_builders.py" "$DSADIR/gpt_builders_dsa.py" \
                "$DSADIR/MEGATRON_REVISION" "$PRETRAIN_GPT" "$DATA_BLEND_FILE" \
                "$WARM/latest_checkpointed_iteration.txt"; do
  if [ ! -e "$required" ]; then
    echo "FATAL: required path is missing: $required"
    exit 1
  fi
done

EXPECTED_MEGATRON_REVISION=$(<"$DSADIR/MEGATRON_REVISION")
MEG_REAL=$(readlink -f "$MEG")
ACTUAL_MEGATRON_REVISION=$(git -c safe.directory="$MEG_REAL" -C "$MEG" rev-parse HEAD)
if [ "$ACTUAL_MEGATRON_REVISION" != "$EXPECTED_MEGATRON_REVISION" ]; then
  echo "FATAL: Megatron revision mismatch"
  echo "expected: $EXPECTED_MEGATRON_REVISION"
  echo "actual:   $ACTUAL_MEGATRON_REVISION"
  exit 1
fi
if [ "$(<"$WARM/latest_checkpointed_iteration.txt")" != "300" ]; then
  echo "FATAL: expected completed warm-up iteration 300"
  exit 1
fi
if ! find "$WARM/iter_0000300" -maxdepth 1 -name '*.distcp' -type f -print -quit | grep -q .; then
  echo "FATAL: iteration 300 has no distributed-checkpoint shards"
  exit 1
fi

mkdir -p "$OUT"
export DSA_RESUME=1 DSA_LOAD_BASE=""
LOAD_ARGS="--load $WARM --no-load-optim --no-load-rng"
START_ITER=300
if [ -f "$OUT/latest_checkpointed_iteration.txt" ]; then
  START_ITER=$(<"$OUT/latest_checkpointed_iteration.txt")
  if ! [[ "$START_ITER" =~ ^[0-9]+$ ]]; then
    echo "FATAL: invalid latest checkpoint marker: $START_ITER"
    exit 1
  fi
  CKDIR=$(printf "%s/iter_%07d" "$OUT" "$START_ITER")
  if [ ! -d "$CKDIR" ] || ! find "$CKDIR" -maxdepth 1 -name '*.distcp' -type f -print -quit | grep -q .; then
    echo "FATAL: latest marker points to an incomplete sparse checkpoint: $CKDIR"
    exit 1
  fi
  if [ "$START_ITER" -ge "$TARGET_ITER" ]; then
    echo "Sparse adaptation already reached iteration $START_ITER (target $TARGET_ITER)."
    exit 0
  fi
  LOAD_ARGS="--load $OUT"
  echo "##### RESUME sparse adaptation from iteration $START_ITER"
elif find "$OUT" -maxdepth 1 -name 'iter_*' -print -quit | grep -q .; then
  echo "FATAL: sparse iter_* directories exist without a valid latest marker in $OUT"
  exit 1
fi

export PYTHONUSERBASE="" CUDA_DEVICE_MAX_CONNECTIONS=1
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR MASTER_PORT=${MASTER_PORT:-9989}
export WORLD_SIZE=$SLURM_NTASKS OMP_NUM_THREADS=2
export HSA_ENABLE_SDMA=0 HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB
export PYTHONWARNINGS=ignore HF_HOME=${HF_HOME:-/scratch/project_465002530/users/bmoell/hf_home}
export MIOPEN_FIND_MODE=FAST
export MIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH:-/flash/project_465002530/users/bmoell/miopen_cache}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p "$MIOPEN_USER_DB_PATH"

export DSA_PATTERN=SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
export DSA_TOPK=2048 DSA_N_HEADS=16 DSA_HEAD_DIM=128 DSA_LOSS_COEFF=0.1
export DSA_SPARSE_RUN=1 DSA_SPARSE=1 DSA_FREEZE_MODEL=0 DSA_ALLOW_DENSE_LAYERS=0
export DSA_GRAD_PROBE=1
export DSA_RECALL_LOG=1 DSA_RECALL_EVERY=36 DSA_RECALL_KS=512,1024,2048
export DSA_INDEX_BLOCK=8192 DSA_INDEX_Q_BLOCK=8192 DSA_KL_Q_BLOCK=128
export DSA_REQUIRE_NON_INTERLEAVED_ROPE=1

DATA=$(<"$DATA_BLEND_FILE")
if [ -z "$DATA" ]; then
  echo "FATAL: empty data blend: $DATA_BLEND_FILE"
  exit 1
fi

c=fe
BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

CMD=" $PRETRAIN_GPT \
  --num-layers 36 --hidden-size 4096 --ffn-hidden-size 12288 \
  --num-attention-heads 32 --group-query-attention --num-query-groups 8 \
  --kv-channels 128 --qk-layernorm --normalization RMSNorm --swiglu \
  --disable-bias-linear --untie-embeddings-and-output-weights \
  --position-embedding-type rope --rotary-base 64000000 \
  --max-position-embeddings 262144 --seq-length 8192 --no-create-attention-mask-in-dataloader \
  --tensor-model-parallel-size 8 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --sequence-parallel --use-distributed-optimizer \
  --recompute-activations --recompute-granularity selective \
  --micro-batch-size 1 --global-batch-size 1 --train-iters $TARGET_ITER --bf16 \
  --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
  --lr 7.3e-6 --min-lr 7.3e-6 --lr-decay-style constant --lr-warmup-iters 0 \
  --clip-grad 1.0 --weight-decay 0.1 --transformer-impl transformer_engine \
  --attention-backend unfused --no-async-tensor-model-parallel-allreduce \
  --no-masked-softmax-fusion --no-gradient-accumulation-fusion \
  --no-bias-dropout-fusion --no-rope-fusion --overlap-grad-reduce \
  --distributed-timeout-minutes 10 \
  --data-path $DATA --data-cache-path $EXT/cache_dsa_sparse8k --split 100,0,0 \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model $TOK \
  --make-vocab-size-divisible-by 128 --dataloader-type cyclic --num-workers 1 \
  --ckpt-format torch_dist $LOAD_ARGS \
  --save $OUT --save-interval $SAVE_INTERVAL \
  --eval-interval 100000 --eval-iters 0 --log-interval 1 --log-throughput "

echo "##### DSA SPARSE 8K SUSTAINED START $(date)"
echo "start=$START_ITER target=$TARGET_ITER save_interval=$SAVE_INTERVAL out=$OUT"
srun --label --cpu-bind=mask_cpu:$BIND_MASK \
  singularity exec -B "$DSADIR" -B "$EXT" -B "$BIND_DIRS" \
  "$CONTAINER" bash -lc \
  "export PYTHONPATH=$DSADIR:$MEG:\$PYTHONPATH RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID; python3 -u $CMD"
echo "##### DSA SPARSE 8K SUSTAINED OK $(date)"
