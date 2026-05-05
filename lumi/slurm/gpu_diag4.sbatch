#!/bin/bash
#SBATCH -p dev-g
#SBATCH -t 00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH -J longctx-diag4
#SBATCH -A project_462000963
#SBATCH -o /scratch/project_462000963/bmoell/longctx-diag4-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/longctx-diag4-%j.err

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
export HSA_ENABLE_SDMA=0

echo "=== Stage A: gloo dist.barrier() with WORLD_SIZE=1 ==="
singularity exec "$SIF" python3 -u -c "
import torch.distributed as dist
import os, time

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'

print('init_process_group...', flush=True)
dist.init_process_group(backend='gloo')
print('init done', flush=True)

print('barrier call 1...', flush=True)
t0 = time.time()
dist.barrier()
print(f'barrier 1 done in {(time.time()-t0)*1000:.1f}ms', flush=True)

print('barrier call 2...', flush=True)
t0 = time.time()
dist.barrier()
print(f'barrier 2 done in {(time.time()-t0)*1000:.1f}ms', flush=True)

dist.destroy_process_group()
print('Stage A PASSED', flush=True)
"
echo ""

echo "=== Stage B: Megatron timer with barrier=True ==="
singularity exec "$SIF" python3 -u -c "
import sys, os
sys.path.insert(0, '$MEGATRON_DIR')

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29501'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'

import torch.distributed as dist
dist.init_process_group(backend='gloo')

import megatron.core.parallel_state as ps
ps.initialize_model_parallel(1, 1)

from megatron.training.utils import print_rank_0
from megatron.core.timers import Timers
timers = Timers(['interval-time'], 0)
print('Starting timer with barrier=True...', flush=True)
import time
t0 = time.time()
timers('interval-time').start(barrier=True)
print(f'Timer started in {(time.time()-t0)*1000:.1f}ms', flush=True)
timers('interval-time').stop()
print('Stage B PASSED', flush=True)
"
echo ""

echo "=== Stage C: Megatron pretrain_gpt.py patched to skip first barrier ==="
# Patch training.py to print before and after barrier, and after each sub-step
singularity exec "$SIF" python3 -u -c "
import sys, os, time
sys.path.insert(0, '$MEGATRON_DIR')

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29502'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['MIOPEN_USER_DB_PATH'] = '$WORKDIR/miopen_cache'
os.environ['MIOPEN_CUSTOM_CACHE_DIR'] = '$WORKDIR/miopen_cache'
os.environ['MIOPEN_FIND_ENFORCE'] = 'NONE'
os.environ['ROCR_VISIBLE_DEVICES'] = '0'
os.environ['HSA_ENABLE_SDMA'] = '0'

# Monkeypatch torch.distributed.barrier to print timing
import torch.distributed as dist
_orig_barrier = dist.barrier
def _patched_barrier(*args, **kwargs):
    print(f'[DIAG] dist.barrier() called from:', flush=True)
    import traceback
    for line in traceback.format_stack()[:-1][-5:]:
        print(line.rstrip(), flush=True)
    t0 = time.time()
    result = _orig_barrier(*args, **kwargs)
    print(f'[DIAG] dist.barrier() returned in {(time.time()-t0)*1000:.1f}ms', flush=True)
    return result
dist.barrier = _patched_barrier

# Monkeypatch forward_step to print
import megatron.training.training as training_module
_orig_train_step = training_module.train_step
def _patched_train_step(*args, **kwargs):
    print('[DIAG] entering train_step', flush=True)
    t0 = time.time()
    result = _orig_train_step(*args, **kwargs)
    print(f'[DIAG] train_step done in {(time.time()-t0)*1000:.1f}ms', flush=True)
    return result
training_module.train_step = _patched_train_step

# Now run pretrain_gpt with the patches active
sys.argv = [
    'pretrain_gpt.py',
    '--tensor-model-parallel-size', '1',
    '--pipeline-model-parallel-size', '1',
    '--num-layers', '2',
    '--hidden-size', '64',
    '--num-attention-heads', '2',
    '--seq-length', '128',
    '--max-position-embeddings', '128',
    '--micro-batch-size', '1',
    '--global-batch-size', '1',
    '--train-iters', '3',
    '--eval-iters', '2',
    '--eval-interval', '5',
    '--mock-data',
    '--tokenizer-type', 'NullTokenizer',
    '--vocab-size', '262144',
    '--split', '900,50,50',
    '--lr', '0.0001',
    '--min-lr', '0.00001',
    '--lr-decay-style', 'cosine',
    '--log-interval', '1',
    '--no-save-optim',
    '--no-save-rng',
    '--distributed-backend', 'gloo',
    '--transformer-impl', 'local',
    '--no-gradient-accumulation-fusion',
    '--no-bias-dropout-fusion',
    '--no-masked-softmax-fusion',
    '--num-workers', '0',
    '--disable-bias-linear',
    '--attention-dropout', '0.0',
    '--hidden-dropout', '0.0',
]
exec(open('$MEGATRON_DIR/pretrain_gpt.py').read())
print('Stage C PASSED')
"
echo ""
echo "=== Done ==="
