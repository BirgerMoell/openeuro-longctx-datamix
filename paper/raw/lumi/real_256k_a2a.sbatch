#!/bin/bash
#SBATCH --job-name=real-256k-a2a
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=16
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=18:00:00
#SBATCH --exclusive
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
#
# Extend the WORKING 128K v3 model -> 256K. theta=64M (law), CP=16 + cp-comm-type a2a (the fix
# for the p2p first-step hang), dense, genuine-256K HF data + 128K mixed-context, no valid split,
# intermediate saves. Target: first real 256K checkpoint.
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
CONTAINER=/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND_DIRS=/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd
export PYTHONUSERBASE="" CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST"|head -n1) MASTER_PORT=9977
export WORLD_SIZE=$SLURM_NTASKS OMP_NUM_THREADS=2 HSA_ENABLE_SDMA=0 HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB
export PYTHONWARNINGS=ignore NVTE_DEBUG=0 NVTE_DEBUG_LEVEL=0 HF_HOME=/scratch/project_465002530/users/bmoell/hf_home
export MIOPEN_FIND_MODE=FAST
export MIOPEN_USER_DB_PATH=/flash/project_465002530/users/bmoell/miopen_cache
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH; mkdir -p $MIOPEN_USER_DB_PATH
c="fe"; BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

MEG=/scratch/project_465002530/users/luomajou/oellm-test/NVIDIA-Megatron-LM
BASE_DIR="$SLURM_SUBMIT_DIR"; LAUNCH=$BASE_DIR/launch.sh
OUT=$BASE_DIR/output_256k_a2a; mkdir -p "$OUT"
TOK=/scratch/project_465002530/users/pyysalos/tokenizers/openeurollm/tokenizer-256k
D256=/scratch/project_465002530/users/bmoell/longctx256_data
LOAD=$BASE_DIR/output_real_v3/ckpt_131072_v3_th32m

# 70% genuine-256K + 30% 128K mixed-context
DATA=$(python3 - "$D256" <<'PY'
import sys, glob, os
D=sys.argv[1]
b256=[f[:-4] for f in sorted(glob.glob(D+"/*/bin/*256k_text_document.bin"))]
b128=[f[:-4] for f in sorted(glob.glob(D+"/*/bin/*128k_text_document.bin"))]
parts=[f"{0.70/max(len(b256),1):.6f} {p}" for p in b256]+[f"{0.30/max(len(b128),1):.6f} {p}" for p in b128]
print(" ".join(parts))
PY
)

seq=262144; theta=64000000; cp=8; toks=500000000; gbs=32
iters=$(( toks / (seq * gbs) )); [ $iters -lt 20 ] && iters=20
echo "##### 256K seq=$seq theta=$theta cp=$cp a2a gbs=$gbs iters=$iters  $(date)"
CMD=" $MEG/pretrain_gpt.py \
  --num-layers 36 --hidden-size 4096 --ffn-hidden-size 12288 \
  --num-attention-heads 32 --group-query-attention --num-query-groups 8 \
  --kv-channels 128 --qk-layernorm --normalization RMSNorm --swiglu \
  --disable-bias-linear --untie-embeddings-and-output-weights \
  --position-embedding-type rope --rotary-base $theta \
  --max-position-embeddings $seq --seq-length $seq --use-flash-attn --no-create-attention-mask-in-dataloader \
  --tensor-model-parallel-size 8 --pipeline-model-parallel-size 1 \
  --context-parallel-size $cp --cp-comm-type a2a --sequence-parallel --use-distributed-optimizer \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 36 \
  --micro-batch-size 1 --global-batch-size $gbs --train-iters $iters --bf16 \
  --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
  --lr 8e-6 --min-lr 8e-7 --lr-decay-style cosine --lr-warmup-iters $(( iters/20 + 1 )) \
  --lr-decay-iters $iters --clip-grad 1.0 --weight-decay 0.1 \
  --no-async-tensor-model-parallel-allreduce --no-masked-softmax-fusion \
  --no-gradient-accumulation-fusion --no-bias-dropout-fusion --no-rope-fusion \
  --overlap-grad-reduce --distributed-timeout-minutes 180 \
  --data-path $DATA --data-cache-path $BASE_DIR/cache_256k_a2a --split 100,0,0 \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model $TOK \
  --make-vocab-size-divisible-by 128 --dataloader-type cyclic --num-workers 1 \
  --ckpt-format torch_dist --load $LOAD --finetune --save $OUT/ckpt_262144 --no-save-optim --no-save-rng \
  --save-interval 50 --eval-interval 100000000 --eval-iters 0 --log-interval 2 --log-throughput "
srun --label --cpu-bind=mask_cpu:$BIND_MASK \
  singularity exec -B "$BASE_DIR" -B "$BIND_DIRS" "$CONTAINER" "$LAUNCH" $CMD
echo "==== 256K done -> $OUT/ckpt_262144 ===="
echo "S256K_A2A_DONE $(date)"
