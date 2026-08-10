#!/bin/bash
# Executed once per Slurm rank inside the pinned LUMI container.
set -euo pipefail

: "${DSADIR:?}" "${MEG:?}" "${EXT:?}" "${LOAD_DIR:?}" "${OUT:?}"
: "${DATA_BLEND_FILE:?}" "${DATA_CACHE_PATH:?}" "${SEQ_LENGTH:?}" "${CP_SIZE:?}"
: "${ROTARY_BASE:?}" "${TARGET_ITER:?}" "${TOKENIZER_PATH:?}"

export RANK=${RANK:-${SLURM_PROCID:?}} LOCAL_RANK=${LOCAL_RANK:-${SLURM_LOCALID:?}}
export PYTHONPATH="$DSADIR:$MEG:${PYTHONPATH:-}"
# Importing Megatron emits informational lines on stdout. The resolved module
# path is deliberately printed last; retain only that line for the comparison.
MODULE_PATH=$(python3 -c 'import pathlib, gpt_builders; print(pathlib.Path(gpt_builders.__file__).resolve())' | tail -n 1)
EXPECTED_PATH=$(readlink -f "$DSADIR/gpt_builders.py")
if [ "$MODULE_PATH" != "$EXPECTED_PATH" ]; then
  echo "FATAL: wrong gpt_builders import on rank ${RANK:-?}"
  echo "expected: $EXPECTED_PATH"
  echo "actual:   $MODULE_PATH"
  exit 1
fi

# Read the entire whitespace-delimited blend. `read` returns failure when a
# regular file ends without a newline (as the production blend currently
# does), so append an explicit NUL delimiter and consume through that instead.
IFS=$' \t\r\n' read -r -d '' -a DATA_ARGS < <(cat "$DATA_BLEND_FILE"; printf '\0')
if [ "${#DATA_ARGS[@]}" -eq 0 ] || [ $(( ${#DATA_ARGS[@]} % 2 )) -ne 0 ]; then
  echo "FATAL: invalid weight/prefix data blend: $DATA_BLEND_FILE"
  exit 1
fi

LOAD_ARGS=(--load "$LOAD_DIR")
if [ "${NO_LOAD_STATE:-0}" = "1" ]; then
  LOAD_ARGS+=(--no-load-optim --no-load-rng)
fi
EXIT_ARGS=()
if [ -n "${EXIT_INTERVAL:-}" ]; then
  EXIT_ARGS+=(--exit-interval "$EXIT_INTERVAL")
fi

# Module mode is mandatory. Executing "$MEG/pretrain_gpt.py" directly puts the
# Megatron directory at sys.path[0] and silently shadows this overlay's
# gpt_builders.py even when PYTHONPATH is otherwise correct.
exec python3 -u -m pretrain_gpt \
  --num-layers 36 --hidden-size 4096 --ffn-hidden-size 12288 \
  --num-attention-heads 32 --group-query-attention --num-query-groups 8 \
  --kv-channels 128 --qk-layernorm --normalization RMSNorm --swiglu \
  --disable-bias-linear --untie-embeddings-and-output-weights \
  --position-embedding-type rope --rotary-base "$ROTARY_BASE" \
  --max-position-embeddings "$SEQ_LENGTH" --seq-length "$SEQ_LENGTH" \
  --no-create-attention-mask-in-dataloader \
  --tensor-model-parallel-size 8 --pipeline-model-parallel-size 1 \
  --context-parallel-size "$CP_SIZE" --sequence-parallel --use-distributed-optimizer \
  --recompute-activations --recompute-granularity full --recompute-method uniform \
  --recompute-num-layers 1 \
  --micro-batch-size 1 --global-batch-size 1 --train-iters 302 --bf16 \
  --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
  --lr 1e-6 --min-lr 1e-6 --lr-decay-style constant --lr-warmup-iters 0 \
  --override-opt-param-scheduler --clip-grad 1.0 --weight-decay 0.1 \
  --transformer-impl transformer_engine --attention-backend unfused \
  --no-async-tensor-model-parallel-allreduce --no-masked-softmax-fusion \
  --no-gradient-accumulation-fusion --no-bias-dropout-fusion --no-rope-fusion \
  --overlap-grad-reduce --distributed-timeout-minutes 30 \
  --data-path "${DATA_ARGS[@]}" --data-cache-path "$DATA_CACHE_PATH" --split 100,0,0 \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$TOKENIZER_PATH" \
  --make-vocab-size-divisible-by 128 --dataloader-type cyclic --num-workers 1 \
  --ckpt-format torch_dist "${LOAD_ARGS[@]}" --save "$OUT" --save-interval "$TARGET_ITER" \
  "${EXIT_ARGS[@]}" \
  --eval-interval 100000000 --eval-iters 0 --log-interval 1 --log-throughput
