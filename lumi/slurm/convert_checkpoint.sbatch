#!/bin/bash
#SBATCH --job-name=convert-checkpoint
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --partition=dev-g
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=1
#SBATCH --account=project_462000963
#SBATCH -o /scratch/project_462000963/bmoell/convert-checkpoint-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/convert-checkpoint-%j.err

# Convert a Megatron-LM checkpoint to HuggingFace format (LlamaForCausalLM).
#
# Uses poro2-scripts-dev convert.py with --loader mcore --saver llama_mistral,
# adapted from Jouni Luoma's convert_oellm_9b_meg_to_hf.sh.
#
# Input:  CKPT_ITER_DIR — path to a specific iter_XXXXXXXX directory
# Output: OUT_DIR/checkpoint_XXXXXXXX/  — HuggingFace model files
#
# After conversion, manually add to config.json:
#   "rope_scaling": {"factor": 16.0, "original_max_position_embeddings": 2048, "type": "yarn"},
#   "rope_theta": 10000
#
# Submit: sbatch lumi/slurm/convert_checkpoint.sbatch
# Or with custom paths:
#   sbatch --export=ALL,CKPT_ITER_DIR=<path>,OUT_DIR=<path> convert_checkpoint.sbatch

set -euo pipefail

CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif
export SINGULARITY_BIND=/pfs,/scratch,/projappl,/project,/flash,/appl,/usr/lib64/libjansson.so.4,/usr/lib64/libcxi.so.1,/opt/cray,/var/spool/slurmd

PORO2_MEGATRON=/scratch/project_462000963/users/jouni/CPT_from_finetuned/poro2-scripts-dev/Megatron-LM
TOKENIZER_DIR=/flash/project_462000963/jouni/checkpoints/converted-checkpoints/oellm-datamix-9b-80-20

CKPT_ITER_DIR=${CKPT_ITER_DIR:-/flash/project_462000963/bmoell/yarn-multilingual/checkpoints/iter_0001000}
OUT_DIR=${OUT_DIR:-/flash/project_462000963/bmoell/yarn-multilingual/converted}

echo "=== Megatron → HuggingFace conversion — $(date) ==="
echo "Input:     $CKPT_ITER_DIR"
echo "Output:    $OUT_DIR"
echo "Tokenizer: $TOKENIZER_DIR"

# Extract iteration number from the dir name
iter_segment=$(basename "$CKPT_ITER_DIR")
if [[ $iter_segment =~ iter_([0-9]+) ]]; then
    raw_iteration="${BASH_REMATCH[1]}"
    iteration=$((10#$raw_iteration))
else
    echo "ERROR: could not parse iteration from $iter_segment" >&2
    exit 1
fi

OUTPUT_DIR="${OUT_DIR}/checkpoint_${raw_iteration}"
echo "Output dir: $OUTPUT_DIR"

# Megatron's convert.py uses latest_checkpointed_iteration.txt to find the
# checkpoint, so we create a temporary directory with a symlink and the right .txt.
uuid=$(uuidgen)
tmp_dir="/tmp/tmp_ckpt_${uuid}"
mkdir -p "$tmp_dir"
trap "rm -rf '$tmp_dir'" EXIT

ln -s "$(readlink -f "$CKPT_ITER_DIR")" "$tmp_dir/$iter_segment"
echo "$iteration" > "$tmp_dir/latest_checkpointed_iteration.txt"

singularity exec "$CONTAINER" /bin/bash -c "
    pip install --quiet transformers==4.48.2
    python3 $PORO2_MEGATRON/tools/checkpoint/convert.py \
        --model-type GPT \
        --loader mcore \
        --saver llama_mistral \
        --load-dir '$tmp_dir' \
        --save-dir '$OUTPUT_DIR' \
        --tokenizer-dir '$TOKENIZER_DIR'
"

echo ""
echo "=== Conversion complete — $(date) ==="
echo "Output: $OUTPUT_DIR"
echo ""
echo "IMPORTANT: manually add rope_scaling to $OUTPUT_DIR/config.json:"
echo '  "rope_scaling": {"factor": 16.0, "original_max_position_embeddings": 2048, "type": "yarn"},'
echo '  "rope_theta": 10000'
