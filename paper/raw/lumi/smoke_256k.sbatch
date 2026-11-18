#!/bin/bash
#SBATCH --job-name=smoke-256k
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=3:00:00
#SBATCH --exclusive
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
#
# 256K SMOKE TEST: does 256K run at all? Load v3 128K -> seq 262144, theta=64M, CP=8 + a2a +
# full recompute, ~5 iters, 1 data source, NO save. Pass = produces training iterations.
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
CONTAINER=/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND_DIRS=/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd
export PYTHONUSERBASE="" CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST"|head -n1) MASTER_PORT=9967
export WORLD_SIZE=$SLURM_NTASKS OMP_NUM_THREADS=2 HSA_ENABLE_SDMA=0 HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB
export PYTHONWARNINGS=ignore NVTE_DEBUG=0 NVTE_DEBUG_LEVEL=0 HF_HOME=/scratch/project_465002530/users/bmoell/hf_home
c="fe"; BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

MEG=/scratch/project_465002530/users/luomajou/oellm-test/NVIDIA-Megatron-LM
BASE_DIR="$SLURM_SUBMIT_DIR"; LAUNCH=$BASE_DIR/launch_256k.sh   # persistent /flash MIOpen + long heartbeat
TOK=/scratch/project_465002530/users/pyysalos/tokenizers/openeurollm/tokenizer-256k
LOAD=$BASE_DIR/output_real_v3/ckpt_131072_v3_th32m
DATA="1.0 /scratch/project_465002530/users/bmoell/longctx256_data/natural/bin/natural_arxiv_full_concat_256k_text_document"

echo "##### 256K SMOKE seq=262144 theta=64M cp=8 a2a fullrecompute iters=5  $(date)"
CMD=" $MEG/pretrain_gpt.py \
  --num-layers 36 --hidden-size 4096 --ffn-hidden-size 12288 \
  --num-attention-heads 32 --group-query-attention --num-query-groups 8 \
  --kv-channels 128 --qk-layernorm --normalization RMSNorm --swiglu \
  --disable-bias-linear --untie-embeddings-and-output-weights \
  --position-embedding-type rope --rotary-base 64000000 \
  --max-position-embeddings 262144 --seq-length 262144 --use-flash-attn --no-create-attention-mask-in-dataloader \
  --tensor-model-parallel-size 8 --pipeline-model-parallel-size 1 \
  --context-parallel-size 8 --sequence-parallel --use-distributed-optimizer \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 36 \
  --micro-batch-size 1 --global-batch-size 8 --train-iters 5 --bf16 \
  --optimizer adam --lr 1e-6 --min-lr 1e-6 --lr-decay-style constant --lr-warmup-iters 0 --lr-decay-iters 5 \
  --clip-grad 1.0 --weight-decay 0.0 \
  --no-async-tensor-model-parallel-allreduce --no-masked-softmax-fusion \
  --no-gradient-accumulation-fusion --no-bias-dropout-fusion --no-rope-fusion \
  --overlap-grad-reduce --distributed-timeout-minutes 170 \
  --data-path $DATA --data-cache-path $BASE_DIR/cache_256k_smoke --split 100,0,0 \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model $TOK \
  --make-vocab-size-divisible-by 128 --dataloader-type cyclic --num-workers 1 \
  --ckpt-format torch_dist --load $LOAD --finetune --no-load-optim --no-load-rng \
  --eval-interval 100000000 --eval-iters 0 --log-interval 1 --log-throughput "
srun --label --cpu-bind=mask_cpu:$BIND_MASK \
  singularity exec -B "$BASE_DIR" -B "$BIND_DIRS" "$CONTAINER" "$LAUNCH" $CMD
echo "SMOKE_256K_DONE $(date)"
