#!/bin/bash
#SBATCH --job-name=ruler-eval
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=0-00:30:00
#SBATCH --output logs/ruler-%j.out
#SBATCH --error  logs/ruler-%j.err
#
# RULER / OneRuler retrieval eval for a long-context checkpoint at a given length.
# Usage:  sbatch ruler_eval.sh <megatron_ckpt_dir> <seq_len>
# This is the REAL "did the extension work?" check — retrieval, not LM loss.
# SKELETON: needs the generation stack (vLLM-ROCm). Wire to BirgerMoell/OneRuler-OELLM.
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
CKPT="${1:?megatron ckpt dir}"; SEQLEN="${2:?seq length, e.g. 131072}"
CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND=/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
TOK=/scratch/project_465002530/users/pyysalos/tokenizers/openeurollm/tokenizer-256k
OUT="$SLURM_SUBMIT_DIR/eval_ruler/$(basename "$CKPT")_${SEQLEN}"; mkdir -p "$OUT"
echo "RULER eval: ckpt=$CKPT seqlen=$SEQLEN -> $OUT"

srun singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  set -e; cd $SLURM_SUBMIT_DIR
  # 1) Megatron torch_dist -> HF (for a generation backend). TODO: use the project converter
  #    (BirgerMoell/megatron-hf-converter) to write '$OUT/hf'.
  # 2) Serve with vLLM (ROCm) at max-model-len=$SEQLEN.   <-- validate vLLM on MI250X first.
  # BASE-MODEL MODE (model is NOT SFT'd): score by FORCED-COMPLETION / log-likelihood, NOT free
  # generation. Run RETRIEVAL tasks only — NIAH (single/multi-key/value/query), Variable Tracking,
  # Common-Word-Extraction. SKIP QA / instruction-following tasks (unreliable on a base LM).
  # NIAH-as-completion (model next-token-completes the needle) directly measures long-range retrieval.
  # 3) Run OneRuler-OELLM (BirgerMoell/OneRuler-OELLM) tasks (NIAH multi/VT/CWE) at length $SEQLEN
  #    + shorter lengths to confirm no short-context regression. Multilingual via OneRuler.
  # 4) Write scores to '$OUT/metrics.json' (keys: ruler@$SEQLEN, and per shorter length).
  echo 'TODO: wire converter + vLLM-ROCm + OneRuler-OELLM. Gate target: RULER must not drop >2 pts vs shorter length.'
"
echo "RULER eval done $(date)"
