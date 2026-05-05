#!/bin/bash
#SBATCH -p dev-g
#SBATCH -t 00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH -J longctx-gpudiag
#SBATCH -A project_462000963
#SBATCH -o /scratch/project_462000963/bmoell/longctx-gpudiag-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/longctx-gpudiag-%j.err

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
mkdir -p $WORKDIR/miopen_cache

echo "=== Stage 1: GPU matmul timing (should complete in <30s) ==="
singularity exec "$SIF" python3 -u -c "
import torch, time
print('torch:', torch.__version__, flush=True)
print('ROCm:', torch.cuda.is_available(), flush=True)
device = torch.device('cuda')
# Warm-up
x = torch.randn(1024, 1024, device=device)
y = torch.randn(1024, 1024, device=device)
torch.cuda.synchronize()
print('Starting matmul iterations...', flush=True)
for i in range(5):
    t0 = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    print(f'  iter {i+1}: {(time.time()-t0)*1000:.1f}ms, sum={z.sum().item():.2f}', flush=True)
print('Stage 1 PASSED', flush=True)
"
echo ""

echo "=== Stage 2: Tiny PyTorch transformer (no Megatron) ==="
singularity exec "$SIF" python3 -u -c "
import torch, torch.nn as nn, time
print('Building tiny model...', flush=True)
device = torch.device('cuda')
vocab_size = 262144
hidden = 64
seq_len = 128

model = nn.Sequential(
    nn.Embedding(vocab_size, hidden),
).to(device)

# Test embedding first
x = torch.randint(0, vocab_size, (1, seq_len), device=device)
out = model(x)
torch.cuda.synchronize()
print(f'Embedding OK: {out.shape}', flush=True)

# Now add a simple linear + loss
proj = nn.Linear(hidden, vocab_size, bias=False).to(device)
opt = torch.optim.SGD(list(model.parameters()) + list(proj.parameters()), lr=0.01)
crit = nn.CrossEntropyLoss()

print('Starting 5 training iterations...', flush=True)
for i in range(5):
    t0 = time.time()
    x = torch.randint(0, vocab_size, (1, seq_len), device=device)
    tgt = torch.randint(0, vocab_size, (1, seq_len), device=device)
    opt.zero_grad()
    emb = model(x)
    logits = proj(emb)
    loss = crit(logits.view(-1, vocab_size), tgt.view(-1))
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    print(f'  iter {i+1}: loss={loss.item():.4f}  ({(time.time()-t0)*1000:.0f}ms)', flush=True)
print('Stage 2 PASSED', flush=True)
"
echo ""

echo "=== Stage 3: Minimal Megatron pretrain_gpt.py (2 layers, hidden=64) ==="
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

singularity exec "$SIF" bash -c "
  set -euo pipefail
  export PYTHONPATH=$MEGATRON_DIR:\${PYTHONPATH:-}
  export MIOPEN_USER_DB_PATH=$WORKDIR/miopen_cache
  export MIOPEN_CUSTOM_CACHE_DIR=$WORKDIR/miopen_cache
  export MIOPEN_FIND_ENFORCE=NONE
  export ROCR_VISIBLE_DEVICES=0
  export MASTER_ADDR=localhost
  export MASTER_PORT=29500
  export WORLD_SIZE=1
  export RANK=0
  export LOCAL_RANK=0
  python3 -u $MEGATRON_DIR/pretrain_gpt.py \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 2 \
    --hidden-size 64 \
    --num-attention-heads 2 \
    --seq-length 128 \
    --max-position-embeddings 128 \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --train-iters 5 \
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
echo "Stage 3 PASSED"

echo ""
echo "=== All stages passed ==="
