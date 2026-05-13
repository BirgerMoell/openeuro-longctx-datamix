#!/bin/bash
#SBATCH --job-name=upload-to-hf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=dev-g
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --account=project_462000963
#SBATCH -o /scratch/project_462000963/bmoell/upload-to-hf-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/upload-to-hf-%j.err

# Upload YaRN multilingual checkpoint to HuggingFace Hub.
#
# Uploads all files from the converted HF checkpoint directory to:
#   birgermoell/oellm-9b-yarn-multilingual-32k
#
# Model: OpenEuroLLM 9B continued pre-trained with YaRN (factor=16.0)
#        on 8 European languages at 32K context (job 18536300)
#
# Submit: sbatch lumi/slurm/upload_to_hf.sbatch

set -euo pipefail

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
WORKDIR=/scratch/project_462000963/bmoell

MODEL_DIR=/flash/project_462000963/bmoell/yarn-multilingual/converted/checkpoint_0001000
HF_REPO=birgermoell/oellm-9b-yarn-multilingual-32k

if ! type module &>/dev/null; then source /etc/profile; fi
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export HF_HOME="$WORKDIR/hf"

echo "=== Upload to HuggingFace — $(date) ==="
echo "Source: $MODEL_DIR"
echo "Target: https://huggingface.co/$HF_REPO"
echo ""

singularity exec \
    --bind /pfs,/scratch,/projappl,/project,/flash,/appl \
    "$SIF" \
    python3 -u - << 'PYEOF'
import os, sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

hf_home  = os.environ["HF_HOME"]
token    = (Path(hf_home) / "token").read_text().strip()
api      = HfApi(token=token)
repo_id  = "birgermoell/oellm-9b-yarn-multilingual-32k"
model_dir = "/flash/project_462000963/bmoell/yarn-multilingual/converted/checkpoint_0001000"

# Create repo if it doesn't exist
try:
    create_repo(repo_id, repo_type="model", private=False, token=token, exist_ok=True)
    print(f"Repo ready: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"create_repo: {e}")

# Upload all files
files = sorted(Path(model_dir).iterdir())
print(f"\nUploading {len(files)} files:")
for f in files:
    size_mb = f.stat().st_size / 1e6
    print(f"  {f.name} ({size_mb:.0f} MB) ...", flush=True)
    api.upload_file(
        path_or_fileobj=str(f),
        path_in_repo=f.name,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    print(f"  {f.name} done", flush=True)

print(f"\n=== Upload complete ===")
print(f"Model: https://huggingface.co/{repo_id}")
PYEOF

echo ""
echo "=== Done — $(date) ==="
