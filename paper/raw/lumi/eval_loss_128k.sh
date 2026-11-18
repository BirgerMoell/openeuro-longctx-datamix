#!/bin/bash
#SBATCH --job-name=eval-loss-128k
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=0-01:00:00
#SBATCH --exclusive
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
#
# Standalone 128K test: load the saved 64K checkpoint (ckpt_65536), extend to 131072 with
# CP=4, train a few iters. Single small data source so the dataset index build is fast (the
# thing that timed out the pipeline). Answers: does CP=4 @ 128K fit + train on MI250X?
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
base="${SLURM_JOB_NAME}-${SLURM_JOBID}"; ln -sf "${base}.out" logs/latest.out; ln -sf "${base}.err" logs/latest.err

CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND_DIRS=/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd
export PYTHONUSERBASE="" CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST"|head -n1) MASTER_PORT=9998
export WORLD_SIZE=$SLURM_NTASKS
export OMP_NUM_THREADS=2 HSA_ENABLE_SDMA=0 HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB
export PYTHONWARNINGS=ignore NVTE_DEBUG=0 NVTE_DEBUG_LEVEL=0 HF_HOME=/scratch/project_462000963/cache/
export MIOPEN_FIND_MODE=FAST                      # pick first working kernel; skip 30-60min search (else NCCL/comms TIMEOUT at long seq). [Jouni]
export MIOPEN_USER_DB_PATH=/flash/project_465002530/users/bmoell/miopen_cache   # persistent perf-db (prepopulates across stages/runs)
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH
c="fe"; BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

MEGATRON_DIR=/scratch/project_465002530/users/luomajou/oellm-test/NVIDIA-Megatron-LM
BASE_DIR="$SLURM_SUBMIT_DIR"; LAUNCH_SCRIPT="$BASE_DIR/launch.sh"
LOAD=$BASE_DIR/output_real/ckpt_131072           # the saved 64K checkpoint
SAVE=$BASE_DIR/output_real/ckpt_131072
TOKENIZER=/scratch/project_465002530/users/pyysalos/tokenizers/openeurollm/tokenizer-256k
DATA=/scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/eng_Latn

CMD=" $MEGATRON_DIR/pretrain_gpt.py \
  --num-layers 36 --hidden-size 4096 --ffn-hidden-size 12288 \
  --num-attention-heads 32 --group-query-attention --num-query-groups 8 \
  --kv-channels 128 --qk-layernorm --normalization RMSNorm --swiglu \
  --disable-bias-linear --untie-embeddings-and-output-weights \
  --position-embedding-type rope --rotary-base 5000000 \
  --max-position-embeddings 131072 --seq-length 131072 \
  --use-flash-attn --no-create-attention-mask-in-dataloader --attention-dropout 0.0 --hidden-dropout 0.0 \
  --tensor-model-parallel-size 8 --pipeline-model-parallel-size 1 \
  --context-parallel-size 8 --sequence-parallel --use-distributed-optimizer \
  --recompute-activations --recompute-granularity selective \
  --micro-batch-size 1 --global-batch-size 64 --train-iters 1 --bf16 \
  --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
  --lr 3e-6 --min-lr 3e-6 --lr-decay-style constant --lr-warmup-iters 1 \
  --clip-grad 1.0 --weight-decay 0.1 \
  --no-async-tensor-model-parallel-allreduce --no-masked-softmax-fusion \
  --no-gradient-accumulation-fusion --no-bias-dropout-fusion --no-rope-fusion \
  --overlap-grad-reduce --distributed-timeout-minutes 120 \
  --data-path $DATA --data-cache-path $BASE_DIR/cache --split 0,100,0 \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model $TOKENIZER \
  --make-vocab-size-divisible-by 128 --dataloader-type cyclic --num-workers 0 \
  --ckpt-format torch_dist --load $LOAD --skip-train  --no-save-optim --no-save-rng \
   --eval-interval 1000 --eval-iters 1 --log-interval 1 --log-throughput "

echo "128K test: load=$LOAD  CP=4  $(date)"
srun --label --cpu-bind=mask_cpu:$BIND_MASK \
  singularity exec -B "$BASE_DIR" -B "$BIND_DIRS" "$CONTAINER" "$LAUNCH_SCRIPT" $CMD
echo "128K test done $(date)"
