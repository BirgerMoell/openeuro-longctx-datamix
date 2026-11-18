#!/bin/bash
#SBATCH --job-name=extract-longctx
#SBATCH --account=project_465002530
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=0-08:00:00
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
# Build length-biased long-context data: extract long docs from the 37 finepdfs languages
# into a 64K mix (docs in [64K,512K]) and a 128K mix (docs in [128K,1M]).
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -uo pipefail; module purge
C=/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
CAT=/scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue
SAMP=/scratch/project_465002530/users/bmoell/build_longctx_mix.py
OUT64=/scratch/project_465002530/users/bmoell/longctx-extend/longctx64_mix
OUT128=/scratch/project_465002530/users/bmoell/longctx-extend/longctx128_mix
mkdir -p "$OUT64" "$OUT128"

LANGS="als_Latn bos_Latn bul_Cyrl cat_Latn ces_Latn dan_Latn deu_Latn ekk_Latn ell_Grek eng_Latn eus_Latn fin_Latn fra_Latn gle_Latn glg_Latn hrv_Latn hun_Latn isl_Latn ita_Latn kat_Geor lit_Latn lvs_Latn mkd_Cyrl mlt_Latn nld_Latn nno_Latn nob_Latn pol_Latn por_Latn ron_Latn slk_Latn slv_Latn spa_Latn srp_Cyrl swe_Latn tur_Latn ukr_Cyrl"

run() {  # run <srcprefix> <outprefix> <budget> <minlen> <maxlen>
  srun singularity exec -B /pfs -B /scratch -B /flash "$C" \
    python3 "$SAMP" --src "$1" --out "$2" --budget-tokens "$3" --min-len "$4" --max-len "$5" \
    2>&1 | grep -aviE "addto|lmod|add_item|torch.from_numpy|Warning"
}

for lg in $LANGS; do
  src=$CAT/HuggingFaceFW/finepdfs-10p-sample/$lg
  [ -f "$src.idx" ] || { echo "skip $lg (missing)"; continue; }
  if [ "$lg" = "eng_Latn" ]; then b64=400000000; b128=400000000; else b64=60000000; b128=60000000; fi
  echo "### $lg"
  run "$src" "$OUT64/lc64_$lg"  "$b64"  65536  524288
  run "$src" "$OUT128/lc128_$lg" "$b128" 131072 1048576
done
# English edu adds more long English
run $CAT/HuggingFaceFW/finepdfs-edu-10p-sample/eng_Latn "$OUT64/lc64_edu_eng" 200000000 65536 524288
run $CAT/HuggingFaceFW/finepdfs-edu-10p-sample/eng_Latn "$OUT128/lc128_edu_eng" 200000000 131072 1048576

echo "=== EXTRACTION DONE $(date) ==="
echo "64K mix: $(ls $OUT64/*.bin 2>/dev/null | wc -l) files"; du -sh "$OUT64" 2>/dev/null
echo "128K mix: $(ls $OUT128/*.bin 2>/dev/null | wc -l) files"; du -sh "$OUT128" 2>/dev/null
