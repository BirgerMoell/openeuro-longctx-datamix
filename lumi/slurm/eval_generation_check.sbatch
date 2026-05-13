#!/bin/bash
#SBATCH --job-name=eval-gen-check
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=dev-g
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=1
#SBATCH --account=project_462000963
#SBATCH -o /scratch/project_462000963/bmoell/eval-gen-check-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/eval-gen-check-%j.err

# Generation sanity check for the YaRN multilingual HF checkpoint.
#
# Feeds a short prompt in each of the 8 training languages and prints
# 128 tokens of continuation. Pass if the model generates fluent,
# in-language text — no instruction following required.
#
# Languages: bg cs da et fi fr hr nl
# Model:     iter_0001000 HF checkpoint (YaRN factor=16.0, ctx=32K)
#
# Submit: sbatch lumi/slurm/eval_generation_check.sbatch

set -euo pipefail

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
WORKDIR=/scratch/project_462000963/bmoell
MODEL_PATH=/flash/project_462000963/bmoell/yarn-multilingual/converted/checkpoint_0001000

if ! type module &>/dev/null; then source /etc/profile; fi
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export HF_HOME="$WORKDIR/hf"

echo "=== Generation sanity check — $(date) ==="
echo "Model: $MODEL_PATH"
echo ""

singularity exec \
    --bind /pfs,/scratch,/projappl,/project,/flash,/appl \
    "$SIF" \
    python3 -u - << 'PYEOF'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/flash/project_462000963/bmoell/yarn-multilingual/converted/checkpoint_0001000"

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Loading model (bfloat16)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
print(f"Model loaded. Device map: {model.hf_device_map}\n", flush=True)

# Short natural-language prompts — one per training language
# Each is a sentence beginning that the model should continue fluently
prompts = [
    ("bg", "Българската литература е богата и разнообразна. Тя включва"),
    ("cs", "Česká republika je středoevropský stát s bohatou historií. Hlavní město Praha"),
    ("da", "Danmark er et nordisk land med en lang kystlinje. Landet er kendt for"),
    ("et", "Eesti on väike riik Põhja-Euroopas. Pealinn Tallinn on"),
    ("fi", "Suomi on Pohjois-Euroopan maa, joka tunnetaan tuhansista järvistään. Suomen kieli"),
    ("fr", "La France est un pays situé en Europe occidentale. Sa capitale, Paris,"),
    ("hr", "Hrvatska je mediteranska zemlja s dugom obalom Jadranskog mora. Zagreb je"),
    ("nl", "Nederland is een land in Noordwest-Europa bekend om zijn windmolens en tulpen. De hoofdstad Amsterdam"),
]

SEP = "─" * 72

for lang, prompt in prompts:
    print(SEP)
    print(f"[{lang.upper()}] PROMPT: {prompt}")
    print()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,          # greedy — deterministic, easiest to inspect
            temperature=1.0,
            repetition_penalty=1.1,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    continuation = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"CONTINUATION: {continuation}")
    print(flush=True)

print(SEP)
print("\n=== Done ===")
PYEOF

echo ""
echo "=== Complete — $(date) ==="
