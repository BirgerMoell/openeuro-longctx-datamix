#!/bin/bash
#SBATCH -p dev-g
#SBATCH -t 00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH -J longctx-diag2
#SBATCH -A project_462000963
#SBATCH -o /scratch/project_462000963/bmoell/longctx-diag2-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/longctx-diag2-%j.err

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
# Recommended LUMI AMD GPU flags
export HSA_ENABLE_SDMA=0
mkdir -p $WORKDIR/miopen_cache

echo "=== Stage 2.5: scaled_dot_product_attention on GPU ==="
singularity exec "$SIF" python3 -u -c "
import torch
import torch.nn.functional as F
import time
device = torch.device('cuda')
print('SDPA backends:', torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(), flush=True)

# Disable flash/efficient SDPA — use math backend
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
print('Disabled flash/mem-efficient SDPA, using math backend', flush=True)

B, n_head, T, head_dim = 1, 2, 128, 32
q = torch.randn(B, n_head, T, head_dim, device=device)
k = torch.randn(B, n_head, T, head_dim, device=device)
v = torch.randn(B, n_head, T, head_dim, device=device)
print('Starting SDPA...', flush=True)
t0 = time.time()
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
torch.cuda.synchronize()
print(f'SDPA OK in {(time.time()-t0)*1000:.1f}ms: {out.shape}', flush=True)
print('Stage 2.5 PASSED', flush=True)
"
echo ""

echo "=== Stage 2.6: Megatron DotProductAttention layer alone ==="
singularity exec "$SIF" python3 -u -c "
import sys
sys.path.insert(0, '$MEGATRON_DIR')
import torch
import os

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29501'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'

import torch.distributed as dist
dist.init_process_group(backend='gloo')

import megatron.core.parallel_state as ps
ps.initialize_model_parallel(1, 1)

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.dot_product_attention import DotProductAttention

config = TransformerConfig(
    num_layers=1,
    hidden_size=64,
    num_attention_heads=2,
    attention_dropout=0.0,
    hidden_dropout=0.0,
)

print('Building DotProductAttention...', flush=True)
attn = DotProductAttention(config=config, layer_number=1).cuda()
print('DotProductAttention built, running forward...', flush=True)

import time
B, T, n_head, head_dim = 1, 128, 2, 32
q = torch.randn(T, B, n_head, head_dim, device='cuda')
k = torch.randn(T, B, n_head, head_dim, device='cuda')
v = torch.randn(T, B, n_head, head_dim, device='cuda')
mask = None
t0 = time.time()
out = attn(q, k, v, mask)
torch.cuda.synchronize()
print(f'DotProductAttention forward OK in {(time.time()-t0)*1000:.1f}ms: {out.shape}', flush=True)
print('Stage 2.6 PASSED', flush=True)
"
echo ""

echo "=== Stage 2.7: Megatron TransformerLayer forward ==="
singularity exec "$SIF" python3 -u -c "
import sys
sys.path.insert(0, '$MEGATRON_DIR')
import torch, os, time

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29502'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'

import torch.distributed as dist
dist.init_process_group(backend='gloo')
import megatron.core.parallel_state as ps
ps.initialize_model_parallel(1, 1)

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer

config = TransformerConfig(
    num_layers=1,
    hidden_size=64,
    num_attention_heads=2,
    attention_dropout=0.0,
    hidden_dropout=0.0,
    add_bias_linear=False,
)

print('Building TransformerLayer...', flush=True)
layer = TransformerLayer(config=config, layer_number=1).cuda()
print('Running TransformerLayer forward...', flush=True)

B, T, H = 1, 128, 64
x = torch.randn(T, B, H, device='cuda')
mask = None
t0 = time.time()
out, _ = layer(x, attention_mask=mask)
torch.cuda.synchronize()
print(f'TransformerLayer forward OK in {(time.time()-t0)*1000:.1f}ms: {out.shape}', flush=True)
print('Stage 2.7 PASSED', flush=True)
"
echo ""

echo "=== Stage 2.8: backward through TransformerLayer ==="
singularity exec "$SIF" python3 -u -c "
import sys
sys.path.insert(0, '$MEGATRON_DIR')
import torch, os, time

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29503'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'

import torch.distributed as dist
dist.init_process_group(backend='gloo')
import megatron.core.parallel_state as ps
ps.initialize_model_parallel(1, 1)

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer

config = TransformerConfig(
    num_layers=1,
    hidden_size=64,
    num_attention_heads=2,
    attention_dropout=0.0,
    hidden_dropout=0.0,
    add_bias_linear=False,
)

layer = TransformerLayer(config=config, layer_number=1).cuda()
opt = torch.optim.SGD(layer.parameters(), lr=0.01)

B, T, H = 1, 128, 64
print('Running forward+backward...', flush=True)
for i in range(3):
    t0 = time.time()
    x = torch.randn(T, B, H, device='cuda', requires_grad=True)
    out, _ = layer(x, attention_mask=None)
    loss = out.sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    print(f'  iter {i+1}: loss={loss.item():.4f}  ({(time.time()-t0)*1000:.0f}ms)', flush=True)
print('Stage 2.8 PASSED', flush=True)
"
echo ""

echo "=== All stages passed ==="
