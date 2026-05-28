# Leonardo (CINECA) — Access & Best Practices

## Access

Leonardo uses certificate-based SSH via `smallstep`, not SSH keys.
Certificates expire after 12 hours — re-run this before each session:

```bash
step ssh login 'birger.moell@ai.se' --provisioner cineca-hpc
ssh pmoell00@login.leonardo.cineca.it
```

If you get "REMOTE HOST IDENTIFICATION HAS CHANGED" (normal — multiple login nodes):
```bash
ssh-keygen -R login.leonardo.cineca.it
ssh -o StrictHostKeyChecking=no pmoell00@login.leonardo.cineca.it
```

From Claude Code sessions, always add these SSH flags to avoid rotating host key errors:
```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null pmoell00@login.leonardo.cineca.it
```

---

## Storage layout

| Variable   | Path example                              | Limit         | Use for |
|------------|-------------------------------------------|---------------|---------|
| `$HOME`    | `/leonardo/home/userexternal/pmoell00`    | **50 GB hard** | dotfiles, ssh config only |
| `$WORK`    | `/leonardo_work/OELLM_prod2026/`          | 1 TB          | venvs, model weights, code |
| `$SCRATCH` | `/leonardo_scratch/...`                   | No limit      | job outputs, temporary data |

**Rules:**
- Venvs and model checkpoints → `$WORK`
- Job output files (`.out`, `.err`, results) → `$SCRATCH`
- `$SCRATCH` files older than 40 days are auto-deleted — copy anything you want to keep to `$WORK` or download it
- Never fill `$HOME` — 50 GB is a hard wall, jobs will fail

---

## Compute nodes have no internet access

Compute nodes are network-isolated. Do all downloads on the **login node** before submitting jobs:

```bash
# Download a HuggingFace model to $WORK
source $WORK/ruler_venv/bin/activate
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('birgermoell/oellm-9b-yarn-multilingual-v2-32k',
                  local_dir='$WORK/models/oellm-9b-yarn-v2-32k')
"

# Install packages (also login node only)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

In sbatch files always use local paths — never `from_pretrained('org/repo')`.

---

## Venv setup (one-time)

```bash
bash lumi/setup_venv_leonardo.sh
```

This creates `$WORK/ruler_venv` with torch 2.5.1+cu121, transformers, accelerate, lm_eval.

---

## Submitting jobs

```bash
sbatch lumi/slurm/test_leonardo.sbatch        # GPU smoke test
sbatch lumi/slurm/eval_ruler_leonardo.sbatch  # RULER eval
```

Account: `OELLM_prod2026`  
Partition: `boost_usr_prod` (GPU, A100 64GB)  
Check budget: `saldo -b`  
Check queue: `squeue -u $USER`

---

## Performance gotchas

- **Never run `ls -l`, `ls -R`, or `df` on directories with many files** — it hammers the Lustre filesystem and slows the whole cluster
- **Never put tens of thousands of small files in one directory** — tar them up or use HDF5/Parquet instead
- For I/O-heavy training, stream data from `$SCRATCH` not `$WORK`
- Use `$WORK` for things that must persist; use `$SCRATCH` for anything ephemeral

---

## Project info

- Account: `OELLM_prod2026`
- Compute budget: ~14M GPU hours, ~33% used as of May 2026
- Model weights: `$WORK/models/oellm-9b-yarn-v2-32k`
- Venv: `$WORK/ruler_venv`
- Support: superc@cineca.it
