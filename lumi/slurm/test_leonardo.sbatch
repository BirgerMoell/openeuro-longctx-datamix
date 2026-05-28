#!/bin/bash
#SBATCH --job-name=test-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=boost_usr_prod
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --account=OELLM_prod2026
#SBATCH -o %j_test.out
#SBATCH -e %j_test.err

set -euo pipefail

echo "=== Leonardo GPU test — $(date) ==="
echo ""

echo "--- Node info ---"
hostname
echo ""

echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

echo "--- Modules ---"
module purge
module load python/3.11.7
module load cuda/12.1
python3 --version
echo ""

echo "--- Venv setup ---"
VENV=$WORK/ruler_venv
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install --quiet --upgrade pip
    pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121
    pip install --quiet transformers accelerate
    echo "Venv created"
else
    source "$VENV/bin/activate"
    echo "Venv reused from $WORK"
fi
echo ""

echo "--- Python imports ---"
python3 -c "
import torch
import transformers
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
    print('GPU memory: %.1f GB' % (torch.cuda.get_device_properties(0).total_memory / 1e9))
print('transformers:', transformers.__version__)
"
echo ""

echo "--- Quick model load (tokenizer only) ---"
MODEL_LOCAL=/leonardo_work/OELLM_prod2026/models/oellm-9b-yarn-v2-32k
python3 -c "
from transformers import AutoTokenizer
import os
model_path = os.environ.get('MODEL_LOCAL', '/leonardo_work/OELLM_prod2026/models/oellm-9b-yarn-v2-32k')
print('Loading tokenizer from', model_path)
tok = AutoTokenizer.from_pretrained(model_path)
test = tok('Hello, world! Bonjour le monde!')
print('Tokenizer OK — %d tokens' % len(test['input_ids']))
"
echo ""

echo "=== All checks passed — $(date) ==="
