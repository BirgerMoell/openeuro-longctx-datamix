#!/bin/bash
#SBATCH -p dev-g
#SBATCH -t 00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH -J longctx-diag5
#SBATCH -A project_462000963
#SBATCH -o /scratch/project_462000963/bmoell/longctx-diag5-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/longctx-diag5-%j.err

set -euo pipefail

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
WORKDIR=/scratch/project_462000963/bmoell
MEGATRON_DIR=$WORKDIR/NVIDIA-Megatron-LM

if ! type module &>/dev/null; then source /etc/profile; fi
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export ROCR_VISIBLE_DEVICES=0
export HSA_ENABLE_SDMA=0
export MIOPEN_FIND_ENFORCE=NONE
export MIOPEN_USER_DB_PATH=$WORKDIR/miopen_cache
export MIOPEN_CUSTOM_CACHE_DIR=$WORKDIR/miopen_cache

echo "=== A: apex FusedLayerNorm on GPU ==="
singularity exec "$SIF" python3 -u -c "
import torch, time
device = torch.device('cuda')
print('Testing apex FusedLayerNorm...', flush=True)
try:
    from apex.normalization import FusedLayerNorm
    norm = FusedLayerNorm(256).to(device)
    x = torch.randn(512, 1, 256, device=device)
    t0 = time.time()
    out = norm(x)
    torch.cuda.synchronize()
    print(f'FusedLayerNorm OK in {(time.time()-t0)*1000:.1f}ms shape={out.shape}', flush=True)
except Exception as e:
    print(f'FusedLayerNorm error: {e}', flush=True)
print('A DONE', flush=True)
"
echo ""

echo "=== B: apex FusedRMSNorm on GPU ==="
singularity exec "$SIF" python3 -u -c "
import torch, time
device = torch.device('cuda')
print('Testing apex FusedRMSNorm...', flush=True)
try:
    from apex.normalization import FusedRMSNorm
    norm = FusedRMSNorm(256).to(device)
    x = torch.randn(512, 1, 256, device=device)
    t0 = time.time()
    out = norm(x)
    torch.cuda.synchronize()
    print(f'FusedRMSNorm OK in {(time.time()-t0)*1000:.1f}ms shape={out.shape}', flush=True)
except Exception as e:
    print(f'FusedRMSNorm error: {e}', flush=True)
print('B DONE', flush=True)
"
echo ""

echo "=== C: apex apply_rotary_pos_emb kernel ==="
singularity exec "$SIF" python3 -u -c "
import torch, time
device = torch.device('cuda')
print('Testing apex apply_rotary_pos_emb...', flush=True)
try:
    from apex.transformer.functional import apply_rotary_pos_emb
    seq, batch, nhead, head_dim = 512, 1, 4, 64
    t = torch.randn(seq, batch, nhead, head_dim, device=device)
    # Create a simple RoPE embedding
    freqs = torch.randn(seq, 1, 1, head_dim//2, device=device)
    t0 = time.time()
    out = apply_rotary_pos_emb(t, freqs)
    torch.cuda.synchronize()
    print(f'apply_rotary_pos_emb OK in {(time.time()-t0)*1000:.1f}ms', flush=True)
except Exception as e:
    print(f'apply_rotary_pos_emb error: {e}', flush=True)
print('C DONE', flush=True)
"
echo ""

echo "=== D: torch.cuda RNG get/set state ==="
singularity exec "$SIF" python3 -u -c "
import torch, time
device = torch.device('cuda')
print('Testing CUDA RNG state operations...', flush=True)
t0 = time.time()
state = torch.cuda.get_rng_state()
print(f'get_rng_state: {(time.time()-t0)*1000:.1f}ms, state type={type(state)}, len={len(state)}', flush=True)
t0 = time.time()
torch.cuda.set_rng_state(state)
print(f'set_rng_state: {(time.time()-t0)*1000:.1f}ms', flush=True)
t0 = time.time()
torch.cuda.manual_seed(42)
print(f'manual_seed: {(time.time()-t0)*1000:.1f}ms', flush=True)
print('D DONE', flush=True)
"
echo ""

echo "=== E: Megatron CudaRNGStatesTracker.fork() ==="
singularity exec "$SIF" python3 -u -c "
import torch, time, sys, os
sys.path.insert(0, '$MEGATRON_DIR')
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29504'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
import torch.distributed as dist
dist.init_process_group(backend='gloo')
import megatron.core.parallel_state as ps
ps.initialize_model_parallel(1, 1)
from megatron.core import tensor_parallel
# Seed the tracker
tensor_parallel.model_parallel_cuda_manual_seed(1234)
print('Testing get_cuda_rng_tracker().fork()...', flush=True)
t0 = time.time()
with tensor_parallel.get_cuda_rng_tracker().fork():
    print(f'  inside fork() in {(time.time()-t0)*1000:.1f}ms', flush=True)
    x = torch.randn(4, 4, device='cuda')
print(f'fork() completed in {(time.time()-t0)*1000:.1f}ms', flush=True)
print('E DONE', flush=True)
"
echo ""

echo "=== F: Megatron ColumnParallelLinear (QKV-style) forward ==="
singularity exec "$SIF" python3 -u -c "
import torch, time, sys, os
sys.path.insert(0, '$MEGATRON_DIR')
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29505'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
import torch.distributed as dist
dist.init_process_group(backend='gloo')
import megatron.core.parallel_state as ps
ps.initialize_model_parallel(1, 1)
from megatron.core import tensor_parallel
tensor_parallel.model_parallel_cuda_manual_seed(1234)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.transformer_config import TransformerConfig
config = TransformerConfig(num_layers=1, hidden_size=256, num_attention_heads=4)
# QKV projection: hidden -> 3 * hidden
lin = ColumnParallelLinear(256, 3*256, config=config, init_method=torch.nn.init.xavier_uniform_, bias=False).cuda()
x = torch.randn(512, 1, 256, device='cuda')
print('Testing ColumnParallelLinear forward...', flush=True)
t0 = time.time()
out, _ = lin(x)
torch.cuda.synchronize()
print(f'ColumnParallelLinear OK in {(time.time()-t0)*1000:.1f}ms shape={out.shape}', flush=True)
print('F DONE', flush=True)
"
echo ""

echo "=== All stages done ==="
