#!/bin/bash
#SBATCH --job-name=extend-v2-64-128
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=16
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH --exclusive
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
#
# v2 rerun: improved 64K then 128K, length-biased data (65% long + 35% original blend).
# Curriculum: ckpt_32768 -> ckpt_65536_v2 -> ckpt_131072_v2. 16K/32K reused (already 100%).
# Resumable: skips a stage whose checkpoint exists.
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge

CONTAINER=/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND_DIRS=/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd
export PYTHONUSERBASE="" CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST"|head -n1) MASTER_PORT=9995
export WORLD_SIZE=$SLURM_NTASKS OMP_NUM_THREADS=2 HSA_ENABLE_SDMA=0 HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB
export PYTHONWARNINGS=ignore NVTE_DEBUG=0 NVTE_DEBUG_LEVEL=0 HF_HOME=/scratch/project_462000963/cache/
export MIOPEN_FIND_MODE=FAST
export MIOPEN_USER_DB_PATH=/flash/project_465002530/users/bmoell/miopen_cache
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH; mkdir -p $MIOPEN_USER_DB_PATH
c="fe"; BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

MEG=/scratch/project_465002530/users/luomajou/oellm-test/NVIDIA-Megatron-LM
BASE_DIR="$SLURM_SUBMIT_DIR"; LAUNCH=$BASE_DIR/launch.sh
OUT=$BASE_DIR/output_real_v2; mkdir -p "$OUT"
TOK=/scratch/project_465002530/users/pyysalos/tokenizers/openeurollm/tokenizer-256k
CAT=/scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue

# ---- build blended DATA strings (65% length-biased long docs + 35% original mix) ----
build_data () {  # build_data <lcmix_dir> <lc_glob>
  python3 - "$1" "$2" <<'PY'
import sys, glob, os
lcdir, lcglob = sys.argv[1], sys.argv[2]
CAT="/scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue"
LANGS="als bos bul cat ces dan deu ekk ell eng eus fin fra gle glg hrv hun isl ita kat lit lvs mkd mlt nld nno nob pol por ron slk slv spa srp swe tur ukr".split()
# 65% length-biased: weight per lc file; English boosted x6
lc=sorted(glob.glob(os.path.join(lcdir, lcglob)))
lc=[f[:-4] for f in lc if f.endswith('.bin')]
parts=[]
nb=len(lc) or 1
for p in lc:
    w = 0.65*0.30 if 'eng' in os.path.basename(p) else 0.65*0.70/max(nb-1,1)
    parts.append(f"{w:.6f} {p}")
# 35% original catalogue mix (finepdfs all langs eng-boosted + code/math/web)
fp=f"{CAT}/HuggingFaceFW/finepdfs-10p-sample"
orig=[]
import re
def lang3(l): return {"eng":"eng_Latn"}.get(l)  # placeholder
# map short->full code dir names
full={ "als":"als_Latn","bos":"bos_Latn","bul":"bul_Cyrl","cat":"cat_Latn","ces":"ces_Latn","dan":"dan_Latn","deu":"deu_Latn","ekk":"ekk_Latn","ell":"ell_Grek","eng":"eng_Latn","eus":"eus_Latn","fin":"fin_Latn","fra":"fra_Latn","gle":"gle_Latn","glg":"glg_Latn","hrv":"hrv_Latn","hun":"hun_Latn","isl":"isl_Latn","ita":"ita_Latn","kat":"kat_Geor","lit":"lit_Latn","lvs":"lvs_Latn","mkd":"mkd_Cyrl","mlt":"mlt_Latn","nld":"nld_Latn","nno":"nno_Latn","nob":"nob_Latn","pol":"pol_Latn","por":"por_Latn","ron":"ron_Latn","slk":"slk_Latn","slv":"slv_Latn","spa":"spa_Latn","srp":"srp_Cyrl","swe":"swe_Latn","tur":"tur_Latn","ukr":"ukr_Cyrl"}
tot_orig=0.35
# eng weight 20x vs others (mirror original mix)
units=20 + (len(LANGS)-1)
for l in LANGS:
    pth=f"{fp}/{full[l]}"
    if not os.path.exists(pth+".idx"): continue
    u = 20 if l=="eng" else 1
    orig.append(f"{tot_orig*0.75*u/units:.6f} {pth}")
orig.append(f"{tot_orig*0.15:.6f} {CAT}/starcoder-10p-sample/starcoder-10p-sample")
orig.append(f"{tot_orig*0.07:.6f} {CAT}/HuggingFaceTB/finemath-10p-sample/finemath4plus-10p-sample")
orig.append(f"{tot_orig*0.03:.6f} {CAT}/nemotron-cc/1.0/10p-sample/nemotron-mha-10p-sample")
print(" ".join(parts+orig))
PY
}

DATA64=$(build_data "$BASE_DIR/longctx64_mix" "lc64_*.bin")
DATA128=$(build_data "$BASE_DIR/longctx128_mix" "lc128_*.bin")

run_stage () {  # run_stage <seq> <theta> <cp> <tokens> <load> <save> <data>
  local seq=$1 theta=$2 cp=$3 toks=$4 load=$5 save=$6 data=$7
  local iters=$(( toks / (seq * 64) ))
  echo "##### v2 STAGE seq=$seq theta=$theta cp=$cp tokens=$toks iters=$iters  $(date)"
  local CMD=" $MEG/pretrain_gpt.py \
    --num-layers 36 --hidden-size 4096 --ffn-hidden-size 12288 \
    --num-attention-heads 32 --group-query-attention --num-query-groups 8 \
    --kv-channels 128 --qk-layernorm --normalization RMSNorm --swiglu \
    --disable-bias-linear --untie-embeddings-and-output-weights \
    --position-embedding-type rope --rotary-base $theta \
    --max-position-embeddings $seq --seq-length $seq --use-flash-attn --no-create-attention-mask-in-dataloader \
    --tensor-model-parallel-size 8 --pipeline-model-parallel-size 1 \
    --context-parallel-size $cp --sequence-parallel --use-distributed-optimizer \
    --recompute-activations --recompute-granularity selective \
    --micro-batch-size 1 --global-batch-size 64 --train-iters $iters --bf16 \
    --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
    --lr 1e-5 --min-lr 1e-6 --lr-decay-style cosine --lr-warmup-iters $(( iters/20 + 1 )) \
    --lr-decay-iters $iters --clip-grad 1.0 --weight-decay 0.1 \
    --no-async-tensor-model-parallel-allreduce --no-masked-softmax-fusion \
    --no-gradient-accumulation-fusion --no-bias-dropout-fusion --no-rope-fusion \
    --overlap-grad-reduce --distributed-timeout-minutes 120 \
    --data-path $data --data-cache-path $BASE_DIR/cache_v2 --split 99,1,0 \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model $TOK \
    --make-vocab-size-divisible-by 128 --dataloader-type cyclic --num-workers 1 \
    --ckpt-format torch_dist --load $load --finetune --save $save --no-save-optim --no-save-rng \
    --save-interval 100 --eval-interval 100000000 --eval-iters 1 --log-interval 5 --log-throughput "
  srun --label --cpu-bind=mask_cpu:$BIND_MASK \
    singularity exec -B "$BASE_DIR" -B "$BIND_DIRS" "$CONTAINER" "$LAUNCH" $CMD
}

# Stage 64K (from the good 32K ckpt) -> v2
if [ ! -f "$OUT/ckpt_65536/latest_checkpointed_iteration.txt" ]; then
  run_stage 65536 2000000 2 3000000000 "$BASE_DIR/output_real/ckpt_32768" "$OUT/ckpt_65536" "$DATA64"
fi
echo "==== 64K v2 done -> $OUT/ckpt_65536 ===="
# Stage 128K (from the new 64K) -> v2
run_stage 131072 5000000 8 2000000000 "$OUT/ckpt_65536" "$OUT/ckpt_131072" "$DATA128"
echo "==== 128K v2 done -> $OUT/ckpt_131072 ===="
echo "V2_PIPELINE_DONE $(date)"
