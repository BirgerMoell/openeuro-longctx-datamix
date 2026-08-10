#!/bin/bash
# Common fail-closed driver: one sparse update, full checkpoint, fresh-process reload, second update.
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

export DSA_PATTERN=SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
export DSA_TOPK=512 DSA_N_HEADS=16 DSA_HEAD_DIM=128 DSA_LOSS_COEFF=0.1
export DSA_SPARSE_RUN=1 DSA_SPARSE=1 DSA_FREEZE_MODEL=0 DSA_ALLOW_DENSE_LAYERS=0
export DSA_ROUTER=block_cp DSA_BLOCK_SIZE=256 DSA_ROUTED_BLOCKS=1
export DSA_CP_MAX_SEQ=524288 DSA_ALLOW_LONGER_CP=0
export DSA_GRAD_PROBE=1 DSA_RECALL_LOG=0 DSA_KL_Q_BLOCK=128
export DSA_REQUIRE_NON_INTERLEAVED_ROPE=1 DSA_RESUME=1 DSA_LOAD_BASE=""
export MEG EXT OUT DATA_BLEND_FILE DATA_CACHE_PATH SEQ_LENGTH CP_SIZE ROTARY_BASE
export TOKENIZER_PATH=$TOK

singularity exec -B "$DSADIR" -B "$EXT" -B "$BIND_DIRS" "$CONTAINER" bash -lc \
  "export PYTHONPATH=$DSADIR:$MEG:\${PYTHONPATH:-}; python3 $DSADIR/lumi/preflight_sparse.py \
    --dsa-dir $DSADIR --megatron-root $MEG --warm-checkpoint $WARM \
    --data-blend $DATA_BLEND_FILE --seq-length $SEQ_LENGTH --cp-size $CP_SIZE \
    --block-size $DSA_BLOCK_SIZE --routed-blocks $DSA_ROUTED_BLOCKS --topk $DSA_TOPK"

if [ "${RUN_GPU_TESTS:-0}" = "1" ]; then
  echo "##### GPU dense-reference and 2-rank RCCL gates"
  srun --nodes=1 --ntasks=1 --gpus-per-task=1 singularity exec \
    -B "$DSADIR" -B "$BIND_DIRS" "$CONTAINER" bash -lc \
    "export PYTHONPATH=$DSADIR:$MEG:\${PYTHONPATH:-}; python3 $DSADIR/test_dsa_correctness.py"
  TEST_PORT=$((MASTER_PORT + 1))
  srun --nodes=1 --ntasks=2 --ntasks-per-node=2 --gpus-per-task=1 singularity exec \
    -B "$DSADIR" -B "$BIND_DIRS" "$CONTAINER" bash -lc \
    "export PYTHONPATH=$DSADIR:$MEG:\${PYTHONPATH:-} MASTER_ADDR=$MASTER_ADDR \
      MASTER_PORT=$TEST_PORT WORLD_SIZE=2 RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID; \
      python3 $DSADIR/test_cp_distributed.py --backend nccl"
fi

checkpoint_complete () {
  local root=$1 expected=$2 marker dir
  marker=$root/latest_checkpointed_iteration.txt
  [ -f "$marker" ] || return 1
  [ "$(<"$marker")" = "$expected" ] || return 1
  dir=$(printf "%s/iter_%07d" "$root" "$expected")
  [ -f "$dir/.metadata" ] && [ -f "$dir/common.pt" ] && \
    find "$dir" -maxdepth 1 -name '*.distcp' -type f -print -quit | grep -q .
}

if checkpoint_complete "$OUT" 302; then
  echo "##### round trip already complete at iteration 302: $OUT"
  exit 0
fi
if [ -f "$OUT/latest_checkpointed_iteration.txt" ] && ! checkpoint_complete "$OUT" 301; then
  echo "FATAL: output marker is neither a complete iteration 301 nor 302 checkpoint"
  exit 1
fi
if [ ! -f "$OUT/latest_checkpointed_iteration.txt" ] && \
   find "$OUT" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "FATAL: output contains files but has no valid checkpoint marker: $OUT"
  exit 1
fi

c=fe
BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

run_train () {
  local load_dir=$1 target=$2 no_load_state=$3 exit_interval=$4
  export LOAD_DIR=$load_dir TARGET_ITER=$target NO_LOAD_STATE=$no_load_state EXIT_INTERVAL=$exit_interval
  echo "##### sparse train process: load=$load_dir target=$target seq=$SEQ_LENGTH cp=$CP_SIZE"
  srun --label --cpu-bind=mask_cpu:$BIND_MASK \
    singularity exec -B "$DSADIR" -B "$EXT" -B "$BIND_DIRS" \
    "$CONTAINER" bash "$INNER"
}

if ! checkpoint_complete "$OUT" 301; then
  run_train "$WARM" 301 1 301
  if ! checkpoint_complete "$OUT" 301; then
    echo "FATAL: first process did not produce a complete iteration 301 checkpoint"
    exit 1
  fi
fi

# This is deliberately a new Python/srun process. It must load model, optimizer,
# RNG, and scheduler state written by the first process before update 302.
run_train "$OUT" 302 0 ""
if ! checkpoint_complete "$OUT" 302; then
  echo "FATAL: reload process did not produce a complete iteration 302 checkpoint"
  exit 1
fi
echo "##### DSA sparse round trip PASS: seq=$SEQ_LENGTH cp=$CP_SIZE out=$OUT"
