#!/bin/bash
#SBATCH --job-name=prelude-128k-2x
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=16
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=1-00:00:00
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
#
# DIAGNOSTIC: re-run prelude's 128K stage from the SAME 64K ckpt with 2x tokens (4B vs 2B),
# everything else IDENTICAL to the original (theta=32M, CP=8, gbs64, lr1e-5, selective recompute,
# same multilingual Jouni blend). Tests base-stickiness: does more 128K budget flatten the
# residual U-shape (depth-0.5 88% -> toward 100%)? Anti-hang: 20-min fast-fail + 5x retry + requeue.
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -uo pipefail; module purge
CONTAINER=/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND_DIRS=/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd
export PYTHONUSERBASE="" CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST"|head -n1) MASTER_PORT=9959
export WORLD_SIZE=$SLURM_NTASKS OMP_NUM_THREADS=2 HSA_ENABLE_SDMA=0 HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB
export PYTHONWARNINGS=ignore NVTE_DEBUG=0 NVTE_DEBUG_LEVEL=0 HF_HOME=/scratch/project_465002530/users/bmoell/hf_home
export MIOPEN_FIND_MODE=FAST
export MIOPEN_USER_DB_PATH=/flash/project_465002530/users/bmoell/miopen_cache
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH; mkdir -p $MIOPEN_USER_DB_PATH
c="fe"; BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

MEG=/scratch/project_465002530/users/luomajou/oellm-test/NVIDIA-Megatron-LM
BASE_DIR="$SLURM_SUBMIT_DIR"; LAUNCH=$BASE_DIR/launch.sh
OUT=$BASE_DIR/output_prelude_128k_2x; mkdir -p "$OUT"
TOK=/scratch/project_465002530/users/pyysalos/tokenizers/openeurollm/tokenizer-256k
LOAD=$BASE_DIR/output_prelude_2/ckpt_65536
DATA=$(cat "$BASE_DIR/jouni_blend.txt")
if [ -z "$DATA" ]; then echo "FATAL: empty blend"; exit 1; fi
echo "128K-2x blend: $(echo $DATA|wc -w|awk '{print $1/2}') pairs"

run_stage () {  # seq theta cp tokens gbs load save
  local seq=$1 theta=$2 cp=$3 toks=$4 gbs=$5 load=$6 save=$7
  local iters=$(( toks / (seq * gbs) )); [ $iters -lt 20 ] && iters=20
  local CMD=" $MEG/pretrain_gpt.py \
    --num-layers 36 --hidden-size 4096 --ffn-hidden-size 12288 \
    --num-attention-heads 32 --group-query-attention --num-query-groups 8 \
    --kv-channels 128 --qk-layernorm --normalization RMSNorm --swiglu \
    --disable-bias-linear --untie-embeddings-and-output-weights \
    --position-embedding-type rope --rotary-base $theta \
    --max-position-embeddings $seq --seq-length $seq --use-flash-attn --no-create-attention-mask-in-dataloader \
    --tensor-model-parallel-size 8 --pipeline-model-parallel-size 1 \
    --context-parallel-size $cp --sequence-parallel --use-distributed-optimizer \
    --recompute-activations --recompute-granularity selective \
    --micro-batch-size 1 --global-batch-size $gbs --train-iters $iters --bf16 \
    --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
    --lr 1e-5 --min-lr 1e-6 --lr-decay-style cosine --lr-warmup-iters $(( iters/20 + 1 )) \
    --lr-decay-iters $iters --clip-grad 1.0 --weight-decay 0.1 \
    --no-async-tensor-model-parallel-allreduce --no-masked-softmax-fusion \
    --no-gradient-accumulation-fusion --no-bias-dropout-fusion --no-rope-fusion \
    --overlap-grad-reduce --distributed-timeout-minutes 20 \
    --data-path $DATA --data-cache-path $BASE_DIR/cache_128k_2x --split 100,0,0 \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model $TOK \
    --make-vocab-size-divisible-by 128 --dataloader-type cyclic --num-workers 1 \
    --ckpt-format torch_dist --load $load --finetune --save $save --no-save-optim --no-save-rng \
    --save-interval 100 --eval-interval 100000000 --eval-iters 0 --log-interval 5 --log-throughput "
  local a
  for a in 1 2 3 4 5; do
    echo "##### 128K-2x STAGE seq=$seq theta=$theta cp=$cp iters=$iters ATTEMPT=$a  $(date)"
    if srun --label --cpu-bind=mask_cpu:$BIND_MASK \
        singularity exec -B "$BASE_DIR" -B "$BIND_DIRS" "$CONTAINER" "$LAUNCH" $CMD; then
      echo "##### 128K-2x stage OK (attempt $a)"; return 0
    fi
    echo "##### 128K-2x attempt $a failed/hung — retry (15s)"; sleep 15
  done
  echo "##### 128K-2x FAILED after 5 attempts"; return 1
}

# 4B tokens = 2x the original 2B 128K budget (476 iters vs 238)
run_stage 131072 32000000 8 4000000000 64 "$LOAD" "$OUT/ckpt_131072" || exit 1
echo "==== PRELUDE 128K-2x done -> $OUT/ckpt_131072 ===="
echo "P128K_2X_DONE $(date)"
