#!/bin/bash
#SBATCH --job-name=extend-validate
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=8                     # 64 GCDs; raise for faster wall-clock (cost is ~node-invariant)
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=0-03:00:00
#SBATCH --exclusive
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
#
# REAL staged long-context extension of the baby: 4K(base) -> 16K -> 32K -> 64K -> 128K,
# with real token budgets and the real long-context data mix. Each stage --finetune-loads the
# previous stage's checkpoint, bumps rotary-base (ABF), trains TOKENS[i], saves a checkpoint.
# Resumable: skips stages whose checkpoint already exists.
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
base="${SLURM_JOB_NAME}-${SLURM_JOBID}"; ln -sf "${base}.out" logs/latest.out; ln -sf "${base}.err" logs/latest.err

CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND_DIRS=/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd
export PYTHONUSERBASE="" CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST"|head -n1) MASTER_PORT=9997
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
OUT=$BASE_DIR/output_validate; mkdir -p "$OUT"
TOK=/scratch/project_465002530/users/pyysalos/tokenizers/openeurollm/tokenizer-256k
BABY=/flash/project_462000963/training/qwen3_9b_hf_baby_ckpts/megatron
LCS=/scratch/project_465002530/preprocessed/oellm-v1-256k/long-ctx-sample

# Real long-context data mix (length-biased sample). TODO: finalize weights to match the
# pretraining domain mix (web/code/math/pdf) per data/SOURCES.md; add more finepdfs langs.
# All 37 OELLM finepdfs languages (eng boosted) + edu + code + math + web. Verified bin+idx.
DATA="1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/als_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/bos_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/bul_Cyrl 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/cat_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/ces_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/dan_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/deu_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/ekk_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/ell_Grek 20 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/eng_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/eus_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/fin_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/fra_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/gle_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/glg_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/hrv_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/hun_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/isl_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/ita_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/kat_Geor 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/lit_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/lvs_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/mkd_Cyrl 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/mlt_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/nld_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/nno_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/nob_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/pol_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/por_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/ron_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/slk_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/slv_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/spa_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/srp_Cyrl 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/swe_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/tur_Latn 1 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/ukr_Cyrl 5 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-edu-10p-sample/eng_Latn 15 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/starcoder-10p-sample/starcoder-10p-sample 6 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceTB/finemath-10p-sample/finemath4plus-10p-sample 5 /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/nemotron-cc/1.0/10p-sample/nemotron-mha-10p-sample"

# Stage table: seq, rotary-base (ABF), context-parallel, TOKENS (real budget)
SEQS=(16384 32768 65536 131072)
THETA=(500000 1000000 2000000 5000000)
CP=(1 1 2 8)        # 128K -> CP=8 for memory headroom (CP=4 hit 94%)
TOKENS=(1048576 2097152 4194304 8388608)   # ~8 iters/stage (cheap validation, GBS=8)

# Packed-doc boundary reset. KEEP OFF for extension that relies on packing short docs:
# reset would isolate each sub-doc (positions restart at 0, no cross-boundary attention) ->
# NO long-range training signal, defeating the extension. Turn ON *only* for stages whose
# data is genuinely >= seq-length (single long docs) — then it's cleaner (no cross-doc bleed).
RESET_ARGS=""   # set to "--reset-attention-mask --reset-position-ids" only with truly-long docs

run_stage () {
  local seq=$1 theta=$2 cp=$3 toks=$4 load=$5 save=$6
  local iters=$(( toks / (seq * 8) ))           # GBS=64
  echo "##### STAGE seq=$seq theta=$theta CP=$cp tokens=$toks iters=$iters  $(date)"
  local CMD=" $MEGATRON_DIR/pretrain_gpt.py \
    --num-layers 36 --hidden-size 4096 --ffn-hidden-size 12288 \
    --num-attention-heads 32 --group-query-attention --num-query-groups 8 \
    --kv-channels 128 --qk-layernorm --normalization RMSNorm --swiglu \
    --disable-bias-linear --untie-embeddings-and-output-weights \
    --position-embedding-type rope --rotary-base $theta \
    --max-position-embeddings $seq --seq-length $seq --use-flash-attn \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --tensor-model-parallel-size 8 --pipeline-model-parallel-size 1 \
    --context-parallel-size $cp --sequence-parallel --use-distributed-optimizer \
    --recompute-activations --recompute-granularity selective \
    --micro-batch-size 1 --global-batch-size 8 --train-iters $iters --bf16 \
    --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
    --lr 1e-5 --min-lr 1e-6 --lr-decay-style cosine --lr-warmup-iters $(( iters/20 + 1 )) \
    --lr-decay-iters $iters --clip-grad 1.0 --weight-decay 0.1 \
    --no-async-tensor-model-parallel-allreduce --no-masked-softmax-fusion \
    --no-gradient-accumulation-fusion --no-bias-dropout-fusion --no-rope-fusion \
    --overlap-grad-reduce --distributed-timeout-minutes 30 $RESET_ARGS \
    --data-path $DATA --data-cache-path $BASE_DIR/cache_real --split 99,1,0 \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model $TOK \
    --make-vocab-size-divisible-by 128 --dataloader-type cyclic --num-workers 2 \
    --ckpt-format torch_dist --load $load --finetune --save $save \
    --save-interval $iters --eval-interval $(( iters/2 + 1 )) --eval-iters 5 \
    --log-interval 5 --log-throughput "
  srun --label --cpu-bind=mask_cpu:$BIND_MASK \
    singularity exec -B "$BASE_DIR" -B "$BIND_DIRS" "$CONTAINER" "$LAUNCH_SCRIPT" $CMD
}

LOAD=$BABY
for i in 0 1 2 3; do
  seq=${SEQS[$i]}; save=$OUT/ckpt_${seq}
  if [ -f "$save/latest_checkpointed_iteration.txt" ]; then echo "skip $seq (exists)"; LOAD=$save; continue; fi
  run_stage ${SEQS[$i]} ${THETA[$i]} ${CP[$i]} ${TOKENS[$i]} "$LOAD" "$save"
  echo "==== STAGE ${seq} DONE -> $save ===="
  # RULER retrieval eval at this length (loss != working retrieval). Submitted as a SEPARATE
  # job so the training allocation isn't blocked on generation. Needs vLLM-ROCm (see ruler_eval.sh).
  sbatch ruler_eval.sh "$save" "${seq}" 2>/dev/null && echo "queued RULER eval for ${seq}" || echo "RULER eval not queued (wire ruler_eval.sh)"
  LOAD=$save
done
echo "REAL_PIPELINE_DONE $(date)"
