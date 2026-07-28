#!/bin/bash
#SBATCH --job-name=sparse-512k
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=16
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
#
# DSA DENSE-WARMUP on the real 256K ckpt: add the lightning indexer (new params), train it via KL
# to mimic attention (DSA_SPARSE=0), model NOT frozen (autoscaler balances model+indexer). Hybrid
# pattern from dsa_layer_search (18/36 sparse). Short 16K context = cheap+fast (indexer learns
# "which tokens matter", length-generalizes). 1 node (TP=8, CP=1). Watch: indexer loss ↓, grad>0.
#
# ARCHIVED / FAIL-CLOSED:
# This launcher predates the validated selected-set-KL sparse path. Its body combines
# DSA_SPARSE_RUN=1 with DSA_SPARSE=0, uses context parallelism that the current sparse
# path cannot implement correctly, and would leave global dense layers at 512K.
# Keep the historical recipe below for auditability, but never submit it.
echo "FATAL: sparse_512k.sbatch is archived and intentionally disabled."
echo "Run dsa_sparse_8k_correctness.sbatch, then implement/validate CP top-k before 512K."
exit 1

if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -uo pipefail; module purge
CONTAINER=/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND_DIRS=/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd
MEG=/scratch/project_465002530/users/luomajou/oellm-test/NVIDIA-Megatron-LM
DSADIR=/scratch/project_465002530/users/bmoell/dsa_exp
EXT=/scratch/project_465002530/users/bmoell/longctx-extend
export PYTHONUSERBASE="" CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST"|head -n1) MASTER_PORT=9975
export WORLD_SIZE=$SLURM_NTASKS OMP_NUM_THREADS=2 HSA_ENABLE_SDMA=0 HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB PYTHONWARNINGS=ignore
export HF_HOME=/scratch/project_465002530/users/bmoell/hf_home MIOPEN_FIND_MODE=FAST
export MIOPEN_USER_DB_PATH=/flash/project_465002530/users/bmoell/miopen_cache
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH; mkdir -p $MIOPEN_USER_DB_PATH
# DSA config (dense-warmup): hybrid pattern, k=2048, indexer KL on, model NOT frozen
export DSA_PATTERN=FFFFSSFFFFFFFFSSFFFSSSSSSSSSSSSSSFFF
export DSA_TOPK=2048 DSA_SPARSE_RUN=1 DSA_N_HEADS=4 DSA_HEAD_DIM=128 DSA_LOSS_COEFF=0.0 DSA_SPARSE=0 DSA_FREEZE_MODEL=0
export DSA_LOAD_BASE=/scratch/project_465002530/users/bmoell/longctx-extend/output_256k_v2/ckpt_262144/iter_0000059
c="fe"; BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

OUT=$EXT/output_sparse_512k; mkdir -p "$OUT"
TOK=/scratch/project_465002530/users/pyysalos/tokenizers/openeurollm/tokenizer-256k
LOAD=$EXT/output_256k_v2/ckpt_262144
DATA=$(cat "$EXT/blend_512k.txt"); [ -z "$DATA" ] && { echo FATAL empty blend; exit 1; }

CMD=" $DSADIR/pretrain_gpt.py \
  --num-layers 36 --hidden-size 4096 --ffn-hidden-size 12288 \
  --num-attention-heads 32 --group-query-attention --num-query-groups 8 \
  --kv-channels 128 --qk-layernorm --normalization RMSNorm --swiglu \
  --disable-bias-linear --untie-embeddings-and-output-weights \
  --position-embedding-type rope --rotary-base 128000000 \
  --max-position-embeddings 262144 --seq-length 524288 --no-create-attention-mask-in-dataloader \
  --tensor-model-parallel-size 8 --pipeline-model-parallel-size 1 --context-parallel-size 16 \
  --sequence-parallel --use-distributed-optimizer \
  --recompute-activations --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --micro-batch-size 1 --global-batch-size 64 --train-iters 6 --bf16 \
  --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
  --lr 1e-4 --min-lr 1e-5 --lr-decay-style cosine --lr-warmup-iters 1 --lr-decay-iters 6 \
  --clip-grad 1.0 --weight-decay 0.1 --transformer-impl transformer_engine --attention-backend unfused \
  --no-async-tensor-model-parallel-allreduce --no-masked-softmax-fusion \
  --no-gradient-accumulation-fusion --no-bias-dropout-fusion --no-rope-fusion \
  --overlap-grad-reduce --distributed-timeout-minutes 20 \
  --data-path $DATA --data-cache-path $EXT/cache_sparse_512k --split 100,0,0 \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model $TOK \
  --make-vocab-size-divisible-by 128 --dataloader-type cyclic --num-workers 1 \
  --ckpt-format torch_dist \
  --save $OUT --no-save-optim --no-save-rng --save-interval 100 \
  --eval-interval 100000 --eval-iters 0 --log-interval 2 --log-throughput "

for a in 1 2 3 4 5; do
  echo "##### DSA WARMUP ATTEMPT=$a  $(date)"
  if srun --label --cpu-bind=mask_cpu:$BIND_MASK singularity exec -B "$EXT" -B "$DSADIR" -B "$BIND_DIRS" "$CONTAINER" \
      bash -lc "export PYTHONPATH=$MEG:$DSADIR:\$PYTHONPATH RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID; python3 -u $CMD"; then
    echo "##### DSA WARMUP OK (attempt $a)"; echo "DSA_WARMUP_DONE $(date)"; exit 0
  fi
  echo "##### warmup attempt $a failed/hung — retry"; sleep 15
done
echo "##### DSA WARMUP FAILED after 5 attempts"; exit 1
