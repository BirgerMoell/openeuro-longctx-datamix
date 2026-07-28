#!/bin/bash
#SBATCH --job-name=real-v3-128k-th32m
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=16
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=1-00:00:00
#SBATCH --exclusive
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
#
# v3 REAL run: 128K from the v2 64K ckpt at theta=16M (best-guess fix for depth-0, per theta sweep).
# Jouni length-biased data. Intermediate saves every 100 iters. Resumable.
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
CONTAINER=/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND_DIRS=/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd
export PYTHONUSERBASE="" CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST"|head -n1) MASTER_PORT=9987
export WORLD_SIZE=$SLURM_NTASKS OMP_NUM_THREADS=2 HSA_ENABLE_SDMA=0 HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB
export PYTHONWARNINGS=ignore NVTE_DEBUG=0 NVTE_DEBUG_LEVEL=0 HF_HOME=/scratch/project_465002530/users/bmoell/hf_home
export MIOPEN_FIND_MODE=FAST
export MIOPEN_USER_DB_PATH=/flash/project_465002530/users/bmoell/miopen_cache
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH; mkdir -p $MIOPEN_USER_DB_PATH
c="fe"; BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

MEG=/scratch/project_465002530/users/luomajou/oellm-test/NVIDIA-Megatron-LM
BASE_DIR="$SLURM_SUBMIT_DIR"; LAUNCH=$BASE_DIR/launch.sh
OUT=$BASE_DIR/output_real_v3; mkdir -p "$OUT"
TOK=/scratch/project_465002530/users/pyysalos/tokenizers/openeurollm/tokenizer-256k
SAMPLE=/flash/project_465002530/preprocessed/oellm-v1-256k/long-ctx-sample
LOAD64=$BASE_DIR/output_real_v2_16n/ckpt_65536

DATA=$(python3 - "$SAMPLE" <<'PY'
import sys, glob, struct, numpy as np, os
S=sys.argv[1]; parts=[]
for idx in sorted(glob.glob(S+"/*.idx")):
    name=os.path.basename(idx)[:-4]
    if name.startswith("test_"): continue
    f=open(idx,"rb"); f.read(9); struct.unpack("<Q",f.read(8)); struct.unpack("<B",f.read(1))
    nseq=struct.unpack("<Q",f.read(8))[0]; struct.unpack("<Q",f.read(8))
    tot=int(np.frombuffer(f.read(nseq*4),dtype=np.int32).sum())
    if tot<=0: continue
    parts.append((tot, S+"/"+name))
T=sum(t for t,_ in parts)
print(" ".join(f"{t/T:.6f} {p}" for t,p in parts))
PY
)

run_stage () {  # seq theta cp tokens load save
  local seq=$1 theta=$2 cp=$3 toks=$4 load=$5 save=$6
  local iters=$(( toks / (seq * 64) ))
  echo "##### v3 STAGE seq=$seq theta=$theta cp=$cp tokens=$toks iters=$iters  $(date)"
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
    --micro-batch-size 1 --global-batch-size 64 --train-iters $iters --bf16 \
    --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
    --lr 1e-5 --min-lr 1e-6 --lr-decay-style cosine --lr-warmup-iters $(( iters/20 + 1 )) \
    --lr-decay-iters $iters --clip-grad 1.0 --weight-decay 0.1 \
    --no-async-tensor-model-parallel-allreduce --no-masked-softmax-fusion \
    --no-gradient-accumulation-fusion --no-bias-dropout-fusion --no-rope-fusion \
    --overlap-grad-reduce --distributed-timeout-minutes 120 \
    --data-path $DATA --data-cache-path $BASE_DIR/cache_v2 --split 99,1,0 \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model $TOK \
    --make-vocab-size-divisible-by 128 --dataloader-type cyclic --num-workers 1 \
    --ckpt-format torch_dist --load $load --finetune --save $save --no-save-optim --no-save-rng \
    --save-interval 100 --eval-interval 100000000 --eval-iters 1 --log-interval 5 --log-throughput "
  srun --label --cpu-bind=mask_cpu:$BIND_MASK \
    singularity exec -B "$BASE_DIR" -B "$BIND_DIRS" "$CONTAINER" "$LAUNCH" $CMD
}

run_stage 131072 32000000 8 2000000000 "$LOAD64" "$OUT/ckpt_131072_v3_th32m"
echo "==== v3 128K @theta16M done -> $OUT/ckpt_131072_v3_th32m ===="
echo "V3_DONE $(date)"
