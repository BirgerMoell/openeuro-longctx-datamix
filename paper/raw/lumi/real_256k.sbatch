#!/bin/bash
#SBATCH --job-name=real-256k-th32m
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=16
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
#
# First 256K attempt: extend the 32M 128K ckpt -> 256K at theta=32M, CP=16, on Birger's HF
# genuine-long 256K data (arxiv/books/code/RFC/docsite, concat to 128K+256K). Intermediate saves.
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
CONTAINER=/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND_DIRS=/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd
export PYTHONUSERBASE="" CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST"|head -n1) MASTER_PORT=9985
export WORLD_SIZE=$SLURM_NTASKS OMP_NUM_THREADS=2 HSA_ENABLE_SDMA=0 HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB
export PYTHONWARNINGS=ignore NVTE_DEBUG=0 NVTE_DEBUG_LEVEL=0 HF_HOME=/scratch/project_465002530/users/bmoell/hf_home
export MIOPEN_FIND_MODE=FAST
export MIOPEN_USER_DB_PATH=/flash/project_465002530/users/bmoell/miopen_cache
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH; mkdir -p $MIOPEN_USER_DB_PATH
c="fe"; BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

MEG=/scratch/project_465002530/users/luomajou/oellm-test/NVIDIA-Megatron-LM
BASE_DIR="$SLURM_SUBMIT_DIR"; LAUNCH=$BASE_DIR/launch.sh
OUT=$BASE_DIR/output_256k; mkdir -p "$OUT"
TOK=/scratch/project_465002530/users/pyysalos/tokenizers/openeurollm/tokenizer-256k
D256=/scratch/project_465002530/users/bmoell/longctx256_data
LOAD128=$BASE_DIR/output_theta_sweep/ckpt_131072_th32000000

# DATA: 70% genuine-256K + 30% 128K (mixed-context), equal across the long sources
DATA=$(python3 - "$D256" <<'PY'
import sys, glob, os
D=sys.argv[1]
b256=sorted(glob.glob(D+"/*/bin/*256k_text_document.bin"))
b128=sorted(glob.glob(D+"/*/bin/*128k_text_document.bin"))
b256=[f[:-4] for f in b256]; b128=[f[:-4] for f in b128]
parts=[]
for p in b256: parts.append(f"{0.70/max(len(b256),1):.6f} {p}")
for p in b128: parts.append(f"{0.30/max(len(b128),1):.6f} {p}")
print(" ".join(parts))
PY
)

run_stage () {  # seq theta cp tokens gbs load save
  local seq=$1 theta=$2 cp=$3 toks=$4 gbs=$5 load=$6 save=$7
  local iters=$(( toks / (seq * gbs) )); [ $iters -lt 20 ] && iters=20
  echo "##### 256K STAGE seq=$seq theta=$theta cp=$cp tokens=$toks gbs=$gbs iters=$iters  $(date)"
  local CMD=" $MEG/pretrain_gpt.py \
    --num-layers 36 --hidden-size 4096 --ffn-hidden-size 12288 \
    --num-attention-heads 32 --group-query-attention --num-query-groups 8 \
    --kv-channels 128 --qk-layernorm --normalization RMSNorm --swiglu \
    --disable-bias-linear --untie-embeddings-and-output-weights \
    --position-embedding-type rope --rotary-base $theta \
    --max-position-embeddings $seq --seq-length $seq --use-flash-attn --no-create-attention-mask-in-dataloader \
    --tensor-model-parallel-size 8 --pipeline-model-parallel-size 1 \
    --context-parallel-size $cp --sequence-parallel --use-distributed-optimizer \
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 36 \
    --micro-batch-size 1 --global-batch-size $gbs --train-iters $iters --bf16 \
    --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
    --lr 8e-6 --min-lr 8e-7 --lr-decay-style cosine --lr-warmup-iters $(( iters/20 + 1 )) \
    --lr-decay-iters $iters --clip-grad 1.0 --weight-decay 0.1 \
    --no-async-tensor-model-parallel-allreduce --no-masked-softmax-fusion \
    --no-gradient-accumulation-fusion --no-bias-dropout-fusion --no-rope-fusion \
    --overlap-grad-reduce --distributed-timeout-minutes 180 \
    --data-path $DATA --data-cache-path $BASE_DIR/cache_256k --split 100,0,0 \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model $TOK \
    --make-vocab-size-divisible-by 128 --dataloader-type cyclic --num-workers 1 \
    --ckpt-format torch_dist --load $load --finetune --save $save --no-save-optim --no-save-rng \
    --save-interval 50 --eval-interval 100000000 --eval-iters 0 --log-interval 2 --log-throughput "
  srun --label --cpu-bind=mask_cpu:$BIND_MASK \
    singularity exec -B "$BASE_DIR" -B "$BIND_DIRS" "$CONTAINER" "$LAUNCH" $CMD
}

run_stage 262144 32000000 8 400000000 16 "$LOAD128" "$OUT/ckpt_262144_th32m"
echo "==== 256K @theta32M done -> $OUT/ckpt_262144_th32m ===="
echo "S256K_DONE $(date)"
