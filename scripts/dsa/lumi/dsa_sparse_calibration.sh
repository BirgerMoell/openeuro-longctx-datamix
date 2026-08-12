#!/bin/bash
# Fail-closed 512K calibration: two 36-update phases and a final reload update.
set -euo pipefail
module purge

: "${DSADIR:?}" "${SEQ_LENGTH:?}" "${CP_SIZE:?}" "${ROTARY_BASE:?}" "${OUT:?}"
MEG=${MEGATRON_ROOT:-/scratch/project_465002530/users/luomajou/oellm-test/NVIDIA-Megatron-LM}
EXT=${LONGCTX_ROOT:-/scratch/project_465002530/users/bmoell/longctx-extend}
WARM=${DSA_WARM_CHECKPOINT:-$EXT/output_dsa_warm_all_s_8k}
TOK=${TOKENIZER_PATH:-/scratch/project_465002530/users/pyysalos/tokenizers/openeurollm/tokenizer-256k}
CONTAINER=${CONTAINER:-/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif}
BIND_DIRS=${BIND_DIRS:-/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd}
DATA_BLEND_FILE=${DATA_BLEND_FILE:?}
DATA_CACHE_PATH=${DATA_CACHE_PATH:-$EXT/cache_dsa_sparse_${SEQ_LENGTH}}
MID_LEVEL_DATASET_SURPLUS=${MID_LEVEL_DATASET_SURPLUS:-0.5}
INNER=$DSADIR/lumi/dsa_sparse_train_inner.sh

for required in "$DSADIR/gpt_builders.py" "$DSADIR/MEGATRON_REVISION" \
                "$DSADIR/lumi/preflight_sparse.py" "$INNER" "$MEG/pretrain_gpt.py" \
                "$DATA_BLEND_FILE" "$WARM/latest_checkpointed_iteration.txt" "$CONTAINER"; do
  if [ ! -e "$required" ]; then
    echo "FATAL: missing required path: $required"
    exit 1
  fi
done
if [ "$SLURM_NTASKS" -ne $((8 * CP_SIZE)) ]; then
  echo "FATAL: expected $((8 * CP_SIZE)) ranks for TP=8 CP=$CP_SIZE, got $SLURM_NTASKS"
  exit 1
fi

export PYTHONUSERBASE="" PYTHONDONTWRITEBYTECODE=1 CUDA_DEVICE_MAX_CONNECTIONS=1
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR MASTER_PORT=${MASTER_PORT:-9993}
export WORLD_SIZE=$SLURM_NTASKS OMP_NUM_THREADS=2
export HSA_ENABLE_SDMA=0 HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB
export PYTHONWARNINGS=ignore HF_HOME=${HF_HOME:-/scratch/project_465002530/users/bmoell/hf_home}
export MIOPEN_FIND_MODE=FAST
export MIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH:-/flash/project_465002530/users/bmoell/miopen_cache}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p "$MIOPEN_USER_DB_PATH" "$OUT"

export DSA_PATTERN=${DSA_PATTERN:-SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS}
export DSA_TOPK=${DSA_TOPK:-2048} DSA_N_HEADS=${DSA_N_HEADS:-16}
export DSA_HEAD_DIM=${DSA_HEAD_DIM:-128} DSA_LOSS_COEFF=${DSA_LOSS_COEFF:-0.1}
export DSA_SPARSE_RUN=1 DSA_SPARSE=1 DSA_FREEZE_MODEL=0 DSA_ALLOW_DENSE_LAYERS=0
export DSA_ROUTER=block_cp DSA_BLOCK_SIZE=${DSA_BLOCK_SIZE:-128}
export DSA_ROUTED_BLOCKS=${DSA_ROUTED_BLOCKS:-15}
export DSA_CP_MAX_SEQ=524288 DSA_ALLOW_LONGER_CP=0
export DSA_GRAD_PROBE=${DSA_GRAD_PROBE:-1} DSA_RECALL_LOG=${DSA_RECALL_LOG:-1}
export DSA_RECALL_EVERY=${DSA_RECALL_EVERY:-36}
export DSA_RECALL_KS=${DSA_RECALL_KS:-512,1024,2048}
export DSA_KL_Q_BLOCK=${DSA_KL_Q_BLOCK:-64}
export DSA_REQUIRE_NON_INTERLEAVED_ROPE=1 DSA_RESUME=1 DSA_LOAD_BASE=""
export MEG EXT OUT DATA_BLEND_FILE DATA_CACHE_PATH SEQ_LENGTH CP_SIZE ROTARY_BASE
export MID_LEVEL_DATASET_SURPLUS
export TOKENIZER_PATH=$TOK

singularity exec -B "$DSADIR" -B "$EXT" -B "$BIND_DIRS" "$CONTAINER" bash -lc \
  "export PYTHONPATH=$DSADIR:$MEG:\${PYTHONPATH:-}; python3 $DSADIR/lumi/preflight_sparse.py \
    --dsa-dir $DSADIR --megatron-root $MEG --warm-checkpoint $WARM \
    --data-blend $DATA_BLEND_FILE --seq-length $SEQ_LENGTH --cp-size $CP_SIZE \
    --block-size $DSA_BLOCK_SIZE --routed-blocks $DSA_ROUTED_BLOCKS --topk $DSA_TOPK \
    --mid-level-dataset-surplus $MID_LEVEL_DATASET_SURPLUS"

checkpoint_complete () {
  local root=$1 expected=$2 marker dir
  marker=$root/latest_checkpointed_iteration.txt
  [ -f "$marker" ] || return 1
  [ "$(<"$marker")" = "$expected" ] || return 1
  dir=$(printf "%s/iter_%07d" "$root" "$expected")
  [ -f "$dir/.metadata" ] && [ -f "$dir/common.pt" ] && \
    find "$dir" -maxdepth 1 -name '*.distcp' -type f -print -quit | grep -q .
}

if checkpoint_complete "$OUT" 373; then
  echo "##### calibration already complete at iteration 373: $OUT"
  exit 0
fi

current=300
load_dir=$WARM
no_load_state=1
if [ -f "$OUT/latest_checkpointed_iteration.txt" ]; then
  current=$(<"$OUT/latest_checkpointed_iteration.txt")
  case "$current" in
    336|372)
      if ! checkpoint_complete "$OUT" "$current"; then
        echo "FATAL: output marker points to incomplete iteration $current"
        exit 1
      fi
      load_dir=$OUT
      no_load_state=0
      ;;
    *)
      echo "FATAL: unexpected calibration checkpoint marker $current"
      exit 1
      ;;
  esac
elif find "$OUT" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "FATAL: output contains files but has no complete checkpoint marker: $OUT"
  exit 1
fi

c=fe
BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

run_train () {
  local from=$1 target=$2 no_state=$3 save_every=$4
  export LOAD_DIR=$from TARGET_ITER=$target NO_LOAD_STATE=$no_state EXIT_INTERVAL=""
  export TRAIN_ITERS=$target SAVE_INTERVAL=$save_every
  echo "##### calibration process: load=$from target=$target seq=$SEQ_LENGTH cp=$CP_SIZE topk=$DSA_TOPK"
  srun --label --cpu-bind=mask_cpu:$BIND_MASK \
    singularity exec -B "$DSADIR" -B "$EXT" -B "$BIND_DIRS" \
    "$CONTAINER" bash "$INNER"
}

if [ "$current" -lt 336 ]; then
  run_train "$load_dir" 336 "$no_load_state" 336
  if ! checkpoint_complete "$OUT" 336; then
    echo "FATAL: calibration process did not produce complete iteration 336"
    exit 1
  fi
  current=336
  load_dir=$OUT
  no_load_state=0
fi

if [ "$current" -lt 372 ]; then
  run_train "$load_dir" 372 "$no_load_state" 372
  if ! checkpoint_complete "$OUT" 372; then
    echo "FATAL: calibration process did not produce complete iteration 372"
    exit 1
  fi
fi

# Third Python/srun process: prove that optimizer, scheduler, and RNG reload
# after all 72 adaptation updates before accepting the calibration artifact.
run_train "$OUT" 373 0 373
if ! checkpoint_complete "$OUT" 373; then
  echo "FATAL: reload process did not produce complete iteration 373"
  exit 1
fi
echo "##### DSA 512K k2048 calibration PASS: updates=73 seq=$SEQ_LENGTH cp=$CP_SIZE out=$OUT"
