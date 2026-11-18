#!/bin/bash
#SBATCH --job-name=extend-pipeline
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=4
#SBATCH --mem=480G
#SBATCH --partition=dev-g
#SBATCH --time=0-00:30:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --account=project_465002530
#SBATCH --output logs/%j.out
#SBATCH --error logs/%j.err
#
# TINY full long-context pipeline on the baby model: 16k -> 32k -> 64k -> 128k.
# Each stage loads the previous stage's checkpoint (--finetune), raises rotary-base
# (ABF), trains a few iters, and saves a checkpoint the next stage consumes.
# Context Parallelism (CP=1/1/2/4) keeps each stage in memory. 4 nodes = 32 GPUs.
# Goal: validate the WHOLE staged path cheaply and find where (if anywhere) 128k breaks.

if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
base="${SLURM_JOB_NAME}-${SLURM_JOBID}-${timestamp}"
mv -f "logs/${SLURM_JOBID}.out" "logs/${base}.out" 2>/dev/null || true
mv -f "logs/${SLURM_JOBID}.err" "logs/${base}.err" 2>/dev/null || true
ln -sf "${base}.out" logs/latest.out; ln -sf "${base}.err" logs/latest.err
module purge

CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
export BIND_DIRS=/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd
export PYTHONUSERBASE=""
MEGATRON_DIR="/scratch/project_465002530/users/luomajou/oellm-test/NVIDIA-Megatron-LM"
BASE_DIR="$SLURM_SUBMIT_DIR"
OUTPUT_DIR="$BASE_DIR/output_pipeline"
LAUNCH_SCRIPT="$BASE_DIR/launch.sh"
mkdir -p "$OUTPUT_DIR"

c="fe"; BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS
export OMP_NUM_THREADS=2
export HSA_ENABLE_SDMA=0
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTHONWARNINGS=ignore
export NVTE_DEBUG=0; export NVTE_DEBUG_LEVEL=0
export HF_HOME="/scratch/project_462000963/cache/"

DATA_PATH="\
0.400000 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/eng_Latn \
0.1000 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/nemotron-cc/1.0/10p-sample/nemotron-ha-10p-sample \
0.1000 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/nemotron-cc/1.0/10p-sample/nemotron-ma-10p-sample \
0.2 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/nemotron-cc/1.0/10p-sample/nemotron-mha-10p-sample \
0.1000 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceTB/finemath-10p-sample/finemath4plus-10p-sample \
0.1000 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/starcoder-10p-sample/starcoder-10p-sample
"
DATA_CACHE_PATH="$BASE_DIR/cache"
TOKENIZER_MODEL="/scratch/project_465002530/users/pyysalos/tokenizers/openeurollm/tokenizer-256k"
BABY=/flash/project_462000963/training/qwen3_9b_hf_baby_ckpts/megatron

# ── Stage table: seq, rotary-base (ABF), context-parallel, iters ──────────────
SEQS=(16384 32768 65536 131072)
THETA=(500000 1000000 2000000 5000000)
CP=(1 1 2 4)
ITERS=(8 8 8 8)

run_stage () {
  local seq=$1 theta=$2 cp=$3 iters=$4 load=$5 save=$6
  echo "############ STAGE seq=$seq theta=$theta CP=$cp iters=$iters ############"
  echo "load=$load  save=$save  $(date)"
  local CMD=" $MEGATRON_DIR/pretrain_gpt.py \
    --num-layers 36 --hidden-size 4096 --ffn-hidden-size 12288 \
    --num-attention-heads 32 --group-query-attention --num-query-groups 8 \
    --kv-channels 128 --qk-layernorm --normalization RMSNorm --swiglu \
    --disable-bias-linear --untie-embeddings-and-output-weights \
    --position-embedding-type rope --rotary-base $theta \
    --max-position-embeddings $seq --seq-length $seq \
    --use-flash-attn --attention-dropout 0.0 --hidden-dropout 0.0 \
    --tensor-model-parallel-size 8 --pipeline-model-parallel-size 1 \
    --context-parallel-size $cp --sequence-parallel --use-distributed-optimizer \
    --recompute-activations --recompute-granularity selective \
    --micro-batch-size 1 --global-batch-size 8 --train-iters $iters --bf16 \
    --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
    --lr 3e-6 --min-lr 3e-6 --lr-decay-style constant --lr-warmup-iters 1 \
    --clip-grad 1.0 --weight-decay 0.1 \
    --no-async-tensor-model-parallel-allreduce --no-masked-softmax-fusion \
    --no-gradient-accumulation-fusion --no-bias-dropout-fusion --no-rope-fusion \
    --overlap-grad-reduce --distributed-timeout-minutes 15 \
    --data-path $DATA_PATH --data-cache-path $DATA_CACHE_PATH --split 99,1,0 \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model $TOKENIZER_MODEL \
    --make-vocab-size-divisible-by 128 --dataloader-type cyclic --num-workers 2 \
    --ckpt-format torch_dist --load $load --finetune --save $save --no-save-optim --no-save-rng \
    --save-interval $iters --eval-interval 1000 --eval-iters 1 --log-interval 1 --log-throughput "
  srun --label --cpu-bind=mask_cpu:$BIND_MASK \
    singularity exec -B "$BASE_DIR" -B "$BIND_DIRS" "$CONTAINER" \
    "$LAUNCH_SCRIPT" $CMD
}

LOAD=$BABY
for i in 0 1 2 3; do
  seq=${SEQS[$i]}; save=$OUTPUT_DIR/ckpt_${seq}
  if run_stage ${SEQS[$i]} ${THETA[$i]} ${CP[$i]} ${ITERS[$i]} "$LOAD" "$save"; then
    echo "==== STAGE ${seq} OK -> $save ===="
    LOAD=$save
  else
    echo "==== STAGE ${seq} FAILED (rc=$?). Stopping; earlier checkpoints kept. ===="
    exit 1
  fi
done
echo "PIPELINE_DONE $(date)"
