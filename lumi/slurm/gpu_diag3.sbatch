#!/bin/bash
#SBATCH -p dev-g
#SBATCH -t 00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH -J longctx-diag3
#SBATCH -A project_462000963
#SBATCH -o /scratch/project_462000963/bmoell/longctx-diag3-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/longctx-diag3-%j.err

set -euo pipefail

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
WORKDIR=/scratch/project_462000963/bmoell
MEGATRON_DIR=$WORKDIR/NVIDIA-Megatron-LM

if ! type module &>/dev/null; then
  source /etc/profile
fi
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export ROCR_VISIBLE_DEVICES=0
export MIOPEN_USER_DB_PATH=$WORKDIR/miopen_cache
export MIOPEN_CUSTOM_CACHE_DIR=$WORKDIR/miopen_cache
export MIOPEN_FIND_ENFORCE=NONE
export HSA_ENABLE_SDMA=0
mkdir -p $WORKDIR/miopen_cache

# Write the timeout-with-stacktrace wrapper
cat > /tmp/run_megatron_timed.py << 'PYEOF'
#!/usr/bin/env python3
"""Run pretrain_gpt.py with a stack-trace dump if it hangs."""
import signal, sys, os, traceback

TIMEOUT_SECS = 120  # 2-minute timeout

def dump_and_die(sig, frame):
    print(f"\n=== HANG DETECTED after {TIMEOUT_SECS}s — stack traces ===", flush=True)
    for tid, fr in sys._current_frames().items():
        print(f"\n--- Thread {tid} ---", flush=True)
        for line in traceback.format_stack(fr):
            print(line, end='', flush=True)
    print("\n=== END TRACES ===", flush=True)
    sys.stdout.flush()
    os.kill(os.getpid(), signal.SIGKILL)

signal.signal(signal.SIGALRM, dump_and_die)
signal.alarm(TIMEOUT_SECS)

# Import everything needed for pretrain_gpt
import importlib.util, runpy
spec = importlib.util.spec_from_file_location("__main__", sys.argv[1])
code = open(sys.argv[1]).read()
# Remove the if __name__ == '__main__' guard so it runs inline
exec(compile(code, sys.argv[1], 'exec'), {'__name__': '__main__', '__file__': sys.argv[1]})
PYEOF

echo "=== Megatron pretrain_gpt.py with stack-trace timeout (2 min) ==="
singularity exec "$SIF" bash -c "
  set -euo pipefail
  export PYTHONPATH=$MEGATRON_DIR:\${PYTHONPATH:-}
  export MIOPEN_USER_DB_PATH=$WORKDIR/miopen_cache
  export MIOPEN_CUSTOM_CACHE_DIR=$WORKDIR/miopen_cache
  export MIOPEN_FIND_ENFORCE=NONE
  export HSA_ENABLE_SDMA=0
  export ROCR_VISIBLE_DEVICES=0
  export MASTER_ADDR=localhost
  export MASTER_PORT=29500
  export WORLD_SIZE=1
  export RANK=0
  export LOCAL_RANK=0

  # Run with stack-trace timeout wrapper
  python3 -u /tmp/run_megatron_timed.py $MEGATRON_DIR/pretrain_gpt.py \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 2 \
    --hidden-size 64 \
    --num-attention-heads 2 \
    --seq-length 128 \
    --max-position-embeddings 128 \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --train-iters 3 \
    --eval-iters 2 \
    --eval-interval 5 \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size 262144 \
    --split 900,50,50 \
    --lr 0.0001 \
    --min-lr 0.00001 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --no-save-optim \
    --no-save-rng \
    --distributed-backend gloo \
    --transformer-impl local \
    --no-gradient-accumulation-fusion \
    --no-bias-dropout-fusion \
    --no-masked-softmax-fusion \
    --num-workers 0 \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0
"
echo "Training run complete"
