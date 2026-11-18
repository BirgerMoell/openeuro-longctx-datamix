#!/bin/bash
#SBATCH --job-name=dsa-warm
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output logs/%j.out
#SBATCH --error logs/%j.err

# Fail-closed DSA indexer warm-up. Cold starts require a complete, compatible base
# checkpoint; resumes require a complete DSA checkpoint. The base model is frozen.
# Frozen warm-up defaults to selective activation recomputation. The model builder makes
# the frozen embedding output require activation gradients so checkpointed DSA cores are
# recomputed during backward and their auxiliary indexer loss is retained.
if [ -z ${SLURM_JOB_ID+x} ]; then
  mkdir -p logs
  sbatch "$0" "$@"
  exit
fi

set -euo pipefail
module purge

CONTAINER=/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND_DIRS=/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd
MEG=/scratch/project_465002530/users/luomajou/oellm-test/NVIDIA-Megatron-LM
DSADIR=/scratch/project_465002530/users/bmoell/dsa_exp
EXT=/scratch/project_465002530/users/bmoell/longctx-extend
BASE_CKPT=$EXT/output_256k_v2/ckpt_262144/iter_0000059
TOK=/scratch/project_465002530/users/pyysalos/tokenizers/openeurollm/tokenizer-256k

export PYTHONUSERBASE="" CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-9995}
export WORLD_SIZE=$SLURM_NTASKS OMP_NUM_THREADS=2 HSA_ENABLE_SDMA=0 HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB PYTHONWARNINGS=ignore
export HF_HOME=/scratch/project_465002530/users/bmoell/hf_home MIOPEN_FIND_MODE=FAST
export MIOPEN_USER_DB_PATH=/flash/project_465002530/users/bmoell/miopen_cache
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p "$MIOPEN_USER_DB_PATH" "$EXT/_quarantine"

export DSA_PATTERN=${DSA_PATTERN:-SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS}
export DSA_TOPK=${DSA_TOPK:-8192}
export DSA_N_HEADS=${DSA_N_HEADS:-16}
export DSA_HEAD_DIM=${DSA_HEAD_DIM:-128}
export DSA_LOSS_COEFF=${DSA_LOSS_COEFF:-0.1}
export DSA_SPARSE=0 DSA_FREEZE_MODEL=1
export DSA_RECALL_LOG=${DSA_RECALL_LOG:-1} DSA_RECALL_EVERY=${DSA_RECALL_EVERY:-90}

SEQ=${DSA_SEQ_LENGTH:-8192}
if [ "${SMOKE:-0}" = "1" ]; then
  ITERS=${ITERS:-2}
  SAVE=${SAVE:-1}
  WARM=${WARM:-1}
  OUT=${DSA_OUT:-$EXT/output_dsa_smoke_grad}
else
  ITERS=${ITERS:-150}
  SAVE=${SAVE:-50}
  WARM=${WARM:-$((ITERS / 10 + 1))}
  OUT=${DSA_OUT:-$EXT/output_dsa_warm}
fi
mkdir -p "$OUT"

DATA=$(<"$EXT/jouni_blend.txt")
if [ -z "$DATA" ]; then
  echo "FATAL: empty data blend"
  exit 1
fi

RECOMPUTE_ARGS=""
if [ "${DSA_RECOMPUTE:-1}" = "1" ]; then
  RECOMPUTE_ARGS="--recompute-activations --recompute-granularity selective"
fi

c=fe
BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

CMD=" $DSADIR/pretrain_gpt.py \
  --num-layers 36 --hidden-size 4096 --ffn-hidden-size 12288 \
  --num-attention-heads 32 --group-query-attention --num-query-groups 8 \
  --kv-channels 128 --qk-layernorm --normalization RMSNorm --swiglu \
  --disable-bias-linear --untie-embeddings-and-output-weights \
  --position-embedding-type rope --rotary-base 64000000 \
  --max-position-embeddings 262144 --seq-length $SEQ --no-create-attention-mask-in-dataloader \
  --tensor-model-parallel-size 8 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --sequence-parallel --use-distributed-optimizer $RECOMPUTE_ARGS \
  --micro-batch-size 1 --global-batch-size 8 --train-iters $ITERS --bf16 \
  --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
  --lr 1e-3 --min-lr 1e-4 --lr-decay-style cosine --lr-warmup-iters $WARM --lr-decay-iters $ITERS \
  --clip-grad 1.0 --weight-decay 0.0 --transformer-impl transformer_engine --attention-backend unfused \
  --no-async-tensor-model-parallel-allreduce --no-masked-softmax-fusion \
  --no-gradient-accumulation-fusion --no-bias-dropout-fusion --no-rope-fusion \
  --overlap-grad-reduce --distributed-timeout-minutes 20 \
  --data-path $DATA --data-cache-path $EXT/cache_dsa_warm --split 100,0,0 \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model $TOK \
  --make-vocab-size-divisible-by 128 --dataloader-type cyclic --num-workers 1 \
  --ckpt-format torch_dist --no-ckpt-fully-parallel-save \
  --save $OUT --no-save-optim --no-save-rng --save-interval $SAVE \
  --eval-interval 100000 --eval-iters 0 --log-interval 1 --log-throughput "

export DSA_RESUME=0
RESUME=""
if [ -f "$OUT/latest_checkpointed_iteration.txt" ]; then
  IT=$(<"$OUT/latest_checkpointed_iteration.txt")
  CKDIR=$(printf "%s/iter_%07d" "$OUT" "${IT:-0}")
  if [ -d "$CKDIR" ] && find "$CKDIR" -maxdepth 1 -name '*.distcp' -type f -print -quit | grep -q .; then
    RESUME="--load $OUT --no-load-optim --no-load-rng"
    export DSA_RESUME=1 DSA_LOAD_BASE=""
    echo "##### RESUME from complete DSA checkpoint iteration $IT"
  else
    echo "FATAL: latest checkpoint marker points to an incomplete checkpoint: $CKDIR"
    exit 1
  fi
fi

if [ "$DSA_RESUME" = "0" ]; then
  if find "$OUT" -maxdepth 1 -name 'iter_*' -print -quit | grep -q .; then
    echo "FATAL: partial iter_* directories exist without a valid latest marker in $OUT"
    exit 1
  fi
  export DSA_LOAD_BASE=$BASE_CKPT
  echo "##### COLD START from base $BASE_CKPT"
fi

echo "##### DSA WARMUP $(date) seq=$SEQ iters=$ITERS save=$SAVE recompute=${DSA_RECOMPUTE:-1}"
srun --label --cpu-bind=mask_cpu:$BIND_MASK \
  singularity exec -B "$EXT" -B "$DSADIR" -B "$BIND_DIRS" "$CONTAINER" \
  bash -lc "export PYTHONPATH=$DSADIR:$MEG:\$PYTHONPATH RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID; python3 -u $CMD $RESUME"

echo "##### DSA WARMUP OK $(date)"
