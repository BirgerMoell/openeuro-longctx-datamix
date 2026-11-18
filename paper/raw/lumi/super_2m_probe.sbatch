#!/bin/bash
#SBATCH --job-name=super-2m-probe
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
# 2M FEASIBILITY PROBE on the baby model: extend the 128K theta=32M ckpt to 2,097,152 tokens
# with sliding-window attention (window 8192, 1 full layer / 8) + theta=256M (scaling law) +
# CP=16 to shard the 2M sequence, full recompute. Tiny budget (~6 iters). Goal: does 2M run at
# all (memory / first-step / throughput)? Learns the constraints for the super-long track.
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
CONTAINER=/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND_DIRS=/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd
export PYTHONUSERBASE="" CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST"|head -n1) MASTER_PORT=9979
export WORLD_SIZE=$SLURM_NTASKS OMP_NUM_THREADS=2 HSA_ENABLE_SDMA=0 HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB
export PYTHONWARNINGS=ignore NVTE_DEBUG=0 NVTE_DEBUG_LEVEL=0 HF_HOME=/scratch/project_465002530/users/bmoell/hf_home
c="fe"; BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

MEG=/scratch/project_465002530/users/luomajou/oellm-test/NVIDIA-Megatron-LM
BASE_DIR="$SLURM_SUBMIT_DIR"; LAUNCH=$BASE_DIR/launch_256k.sh   # persistent /flash MIOpen + long heartbeat
TOK=/scratch/project_465002530/users/pyysalos/tokenizers/openeurollm/tokenizer-256k
D2M=/scratch/project_465002530/users/bmoell/superlong_data/bin
LOAD=$BASE_DIR/output_theta_sweep/ckpt_131072_th32000000

# DATA: 2M sources; synthetic_recall weighted heavily (the long-range skill)
DATA=$(python3 - "$D2M" <<'PY'
import sys, glob, os
D=sys.argv[1]
bins=[f[:-4] for f in sorted(glob.glob(D+"/*2048k_text_document.bin"))]
rec=[p for p in bins if 'synthetic_recall' in p]
oth=[p for p in bins if 'synthetic_recall' not in p]
parts=[]
for p in rec: parts.append(f"{0.40/max(len(rec),1):.6f} {p}")
for p in oth: parts.append(f"{0.60/max(len(oth),1):.6f} {p}")
print(" ".join(parts))
PY
)

seq=2097152; theta=256000000; cp=16; gbs=8; iters=6
echo "##### 2M PROBE seq=$seq theta=$theta cp=16 window=8192,0 skip-freq=8 gbs=$gbs iters=$iters  $(date)"
CMD=" $MEG/pretrain_gpt.py \
  --num-layers 36 --hidden-size 4096 --ffn-hidden-size 12288 \
  --num-attention-heads 32 --group-query-attention --num-query-groups 8 \
  --kv-channels 128 --qk-layernorm --normalization RMSNorm --swiglu \
  --disable-bias-linear --untie-embeddings-and-output-weights \
  --position-embedding-type rope --rotary-base $theta \
  --max-position-embeddings $seq --seq-length $seq --use-flash-attn --no-create-attention-mask-in-dataloader \
  --window-size 8192,0 --window-attn-skip-freq 8 \
  --tensor-model-parallel-size 8 --pipeline-model-parallel-size 1 \
  --context-parallel-size $cp --cp-comm-type a2a --sequence-parallel --use-distributed-optimizer \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 36 \
  --micro-batch-size 1 --global-batch-size $gbs --train-iters $iters --bf16 \
  --optimizer adam --lr 5e-6 --min-lr 5e-7 --lr-decay-style cosine --lr-warmup-iters 1 --lr-decay-iters $iters \
  --clip-grad 1.0 --weight-decay 0.1 \
  --no-async-tensor-model-parallel-allreduce --no-masked-softmax-fusion \
  --no-gradient-accumulation-fusion --no-bias-dropout-fusion --no-rope-fusion \
  --overlap-grad-reduce --distributed-timeout-minutes 540 \
  --data-path $DATA --data-cache-path $BASE_DIR/cache_2m --split 100,0,0 \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model $TOK \
  --make-vocab-size-divisible-by 128 --dataloader-type cyclic --num-workers 1 \
  --ckpt-format torch_dist --load $LOAD --finetune --no-load-optim --no-load-rng \
  --save $BASE_DIR/output_2m/ckpt_2m --save-interval 3 --no-save-optim --no-save-rng \
  --eval-interval 100000000 --eval-iters 0 --log-interval 1 --log-throughput "
srun --label --cpu-bind=mask_cpu:$BIND_MASK \
  singularity exec -B "$BASE_DIR" -B "$BIND_DIRS" "$CONTAINER" "$LAUNCH" $CMD
echo "==== 2M PROBE done ===="
echo "PROBE_2M_DONE $(date)"
