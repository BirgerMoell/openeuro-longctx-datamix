#!/bin/bash
#SBATCH --job-name=upload-yarn-v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=dev-g
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --account=project_465002530
#SBATCH -o /scratch/project_462000963/bmoell/upload-yarn-v2-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/upload-yarn-v2-%j.err

# Upload YaRN v2 multilingual checkpoint + model card to HuggingFace Hub.
# Source: /flash/project_462000963/bmoell/yarn-multilingual-v2-1k/hf
# Target: birgermoell/oellm-9b-yarn-multilingual-v2-32k
#
# v2 vs v1: fixes mscale (0.1*ln(16)+1.0=1.277) which caused depth=0%
#           failures at 32K in v1.  Also extends from 8→35 European languages.
#
# Submit: sbatch lumi/slurm/upload_yarn_v2_1k.sbatch

set -euo pipefail

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
WORKDIR=/scratch/project_462000963/bmoell

MODEL_DIR=/flash/project_462000963/bmoell/yarn-multilingual-v2-1k/hf
HF_REPO=birgermoell/oellm-9b-yarn-multilingual-v2-32k
REPO_DIR=/scratch/project_462000963/bmoell/openeuro-longctx-datamix

if ! type module &>/dev/null; then source /etc/profile; fi
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export HF_HOME="$WORKDIR/hf"

echo "=== Upload YaRN v2 to HuggingFace — $(date) ==="
echo "Source: $MODEL_DIR"
echo "Target: https://huggingface.co/$HF_REPO"
echo ""

singularity exec \
    --bind /pfs,/scratch,/projappl,/project,/flash,/appl \
    "$SIF" \
    python3 -u - << 'PYEOF'
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

hf_home   = os.environ["HF_HOME"]
token     = (Path(hf_home) / "token").read_text().strip()
api       = HfApi(token=token)
repo_id   = "birgermoell/oellm-9b-yarn-multilingual-v2-32k"
model_dir = Path("/flash/project_462000963/bmoell/yarn-multilingual-v2-1k/hf")
card_path = Path("/scratch/project_462000963/bmoell/openeuro-longctx-datamix/lumi/slurm/model_card_yarn_v2.md")

# Create repo
try:
    create_repo(repo_id, repo_type="model", private=False, token=token, exist_ok=True)
    print(f"Repo ready: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"create_repo: {e}")

# Upload model card first
if card_path.exists():
    print(f"\nUploading model card ...", flush=True)
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    print("  README.md done", flush=True)

# Upload all model files
files = sorted(model_dir.iterdir())
print(f"\nUploading {len(files)} model files:")
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
