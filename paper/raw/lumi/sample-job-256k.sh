#!/bin/bash
#SBATCH --job-name=sample
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --partition=small
#SBATCH --time=0-04:00:00
#SBATCH --account=project_465002530
#SBATCH --output=/scratch/project_465002530/users/luomajou/idx-analysis/logs/sample-%j.out
#SBATCH --error=/scratch/project_465002530/users/luomajou/idx-analysis/logs/sample-%j.err

# Usage: sbatch sample-job.sh <input_prefix> <output_name> <target_docs>
#   input_prefix : full path without .bin/.idx extension
#   output_name  : output filename (written under OUTDIR)
#   target_docs  : number of documents to sample

module use /appl/local/csc/modulefiles/
module load pytorch

INPUT="$1"
OUTPUT_NAME="$2"
TARGET_DOCS="$3"

SCRIPT=/scratch/project_465002530/users/luomajou/idx-analysis/sample_idx.py
OUTDIR=/scratch/project_465002530/users/bmoell/longctx256_data/multilingual_sampled

echo "Job $SLURM_JOBID: sampling $OUTPUT_NAME"
echo "  input:  $INPUT"
echo "  output: $OUTDIR/$OUTPUT_NAME"
echo "  target: $TARGET_DOCS docs"

python3 "$SCRIPT" \
    --input  "$INPUT" \
    --output "$OUTDIR/$OUTPUT_NAME" \
    --target-docs "$TARGET_DOCS" \
    --medium-threshold 65536 \
    --long-threshold 262144 \
    --short-frac 0.60 \
    --medium-frac 0.25 \
    --long-frac 0.15

echo "DONE: $OUTPUT_NAME"
