#!/bin/bash
#SBATCH --job-name=conv-yarn-v2
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --partition=dev-g
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=1
#SBATCH --account=project_462000963
#SBATCH -o /scratch/project_462000963/bmoell/conv-yarn-v2-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/conv-yarn-v2-%j.err

# Convert YaRN v2 Megatron checkpoint (iter 1000, TP=2 PP=4) to HuggingFace format.
# Uses vitiugin/megatron-workdir-lumi Megatron which has weights_only=False already
# and uses the older loader_mcore / saver_llama_mistral API (no loader_base issues).
# Output: /flash/project_462000963/bmoell/yarn-multilingual-v2-1k/hf/

set -euo pipefail

CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif
export SINGULARITY_BIND=/pfs,/scratch,/projappl,/project,/flash,/appl,/usr/lib64/libjansson.so.4,/usr/lib64/libcxi.so.1,/opt/cray,/var/spool/slurmd

MEGATRON_DIR=/scratch/project_462000963/users/vitiugin/conversion/megatron-workdir-lumi/Megatron-LM
CKPT_DIR=/flash/project_462000963/bmoell/yarn-multilingual-v2-1k/checkpoints/iter_0001000
TOKENIZER_DIR=/flash/project_462000963/jouni/checkpoints/converted-checkpoints/oellm-datamix-9b-80-20
OUTPUT_DIR=/flash/project_462000963/bmoell/yarn-multilingual-v2-1k/hf

# Build a temp symlink dir so Megatron convert.py finds the right iteration
ITER_LABEL="iter_0001000"
UUID=$(cat /proc/sys/kernel/random/uuid)
TMP_DIR="/tmp/megatron_conv_${UUID}"
mkdir -p "$TMP_DIR"
ln -s "$(readlink -f "$CKPT_DIR")" "$TMP_DIR/$ITER_LABEL"
echo "1000" > "$TMP_DIR/latest_checkpointed_iteration.txt"

cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

mkdir -p "$OUTPUT_DIR"

echo "=== Converting iter 1000 → HF ==="
echo "  Source : $CKPT_DIR"
echo "  Output : $OUTPUT_DIR"
echo "  START  : $(date)"

singularity exec "$CONTAINER" \
  python3 "$MEGATRON_DIR/tools/checkpoint/convert.py" \
    --model-type GPT \
    --loader mcore \
    --saver llama_mistral \
    --load-dir "$TMP_DIR" \
    --save-dir "$OUTPUT_DIR" \
    --tokenizer-dir "$TOKENIZER_DIR"

echo "=== Conversion done: $(date) ==="
echo "Output: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
