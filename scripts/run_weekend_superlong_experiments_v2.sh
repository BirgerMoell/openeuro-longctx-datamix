#!/usr/bin/env bash
set -u

GPU_ID="${1:?usage: run_weekend_superlong_experiments_v2.sh GPU_ID}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/birger/megatron_hf_conversion/.venv/bin/python}"
TRAINER="${TRAINER:-/home/ubuntu/birger/train_hf_superlong_smoke.py}"
RUN_ROOT="${RUN_ROOT:-/home/ubuntu/birger/runs/hf_superlong/weekend_20260613_v2}"
MODEL="${MODEL:-openeurollm/datamix-2b-80-20}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_exp() {
  local name="$1"
  shift
  local out_dir="${RUN_ROOT}/${name}"
  mkdir -p "${out_dir}"

  if [[ -f "${out_dir}/final_eval.json" ]]; then
    echo "[$(date -Is)] SKIP ${name}: final_eval.json exists"
    return 0
  fi

  echo "[$(date -Is)] START ${name} on GPU ${GPU_ID}" | tee -a "${RUN_ROOT}/launcher_gpu${GPU_ID}.log"
  {
    echo "name=${name}"
    echo "gpu=${GPU_ID}"
    echo "host=$(hostname)"
    echo "started_at=$(date -Is)"
    echo "python=${PYTHON_BIN}"
    echo "trainer=${TRAINER}"
    echo "run_root=${RUN_ROOT}"
    echo "pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF}"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || true
  } > "${out_dir}/launcher.log" 2>&1

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${TRAINER}" \
    --model "${MODEL}" \
    --local-files-only \
    --output-dir "${out_dir}" \
    "$@" > "${out_dir}/run.log" 2>&1
  local code=$?

  echo "${code}" > "${out_dir}/exit_code.txt"
  date -Is > "${out_dir}/finished_at.txt"
  {
    echo "finished_at=$(cat "${out_dir}/finished_at.txt")"
    echo "exit_code=${code}"
    tail -n 40 "${out_dir}/run.log" || true
  } >> "${out_dir}/launcher.log" 2>&1
  echo "[$(date -Is)] DONE ${name} exit=${code}" | tee -a "${RUN_ROOT}/launcher_gpu${GPU_ID}.log"
  return 0
}

common=(
  --max-position 2097152
  --train-start-position 32768
  --position-strategy pose_bridge
  --eval-position-strategy pose_bridge
  --rope-scaling yarn
  --dtype bfloat16
  --eval-before
)

case "${GPU_ID}" in
  0)
    run_exp g0_fixedeval_yarn_status_4k_wrong2_seed2026061321 \
      "${common[@]}" \
      --seq-len 4096 \
      --train-steps 2400 \
      --last-n-layers 8 \
      --no-train-lm-head \
      --generative-weight 0 \
      --contrastive-weight 1.0 \
      --contrastive-wrong-candidates 2 \
      --contrastive-temperature 0.5 \
      --eval-every 600 \
      --eval-lengths 4096,8192 \
      --eval-depths 0.9,0.5,0.05 \
      --train-depth-schedule 0.9,0.5,0.05 \
      --eval-trials 20 \
      --distractors 24 \
      --wrong-candidates 2 \
      --value-kind status \
      --seed 2026061321 \
      --eval-seed 2026061391

    run_exp g0_fixedeval_yarn_status_4k_wrong2_temp1_seed2026061322 \
      "${common[@]}" \
      --seq-len 4096 \
      --train-steps 2400 \
      --last-n-layers 8 \
      --no-train-lm-head \
      --generative-weight 0 \
      --contrastive-weight 1.0 \
      --contrastive-wrong-candidates 2 \
      --contrastive-temperature 1.0 \
      --eval-every 600 \
      --eval-lengths 4096,8192 \
      --eval-depths 0.9,0.5,0.05 \
      --train-depth-schedule 0.9,0.5,0.05 \
      --eval-trials 20 \
      --distractors 24 \
      --wrong-candidates 2 \
      --value-kind status \
      --seed 2026061322 \
      --eval-seed 2026061391

    run_exp g0_fixedeval_yarn_status_4k_wrong2_depth09first_seed2026061323 \
      "${common[@]}" \
      --seq-len 4096 \
      --train-steps 2400 \
      --last-n-layers 8 \
      --no-train-lm-head \
      --generative-weight 0 \
      --contrastive-weight 1.0 \
      --contrastive-wrong-candidates 2 \
      --contrastive-temperature 0.5 \
      --eval-every 600 \
      --eval-lengths 4096,8192 \
      --eval-depths 0.9,0.5,0.05 \
      --train-depth-schedule 0.9,0.9,0.5,0.05 \
      --eval-trials 20 \
      --distractors 24 \
      --wrong-candidates 2 \
      --value-kind status \
      --seed 2026061323 \
      --eval-seed 2026061391
    ;;

  1)
    run_exp g1_fixedeval_yarn_status_2k_wrong3_seed2026061324 \
      "${common[@]}" \
      --seq-len 2048 \
      --train-steps 3000 \
      --last-n-layers 8 \
      --no-train-lm-head \
      --generative-weight 0 \
      --contrastive-weight 1.0 \
      --contrastive-wrong-candidates 3 \
      --contrastive-temperature 0.5 \
      --eval-every 750 \
      --eval-lengths 2048,4096,8192 \
      --eval-depths 0.9,0.5,0.05 \
      --train-depth-schedule 0.9,0.5,0.05 \
      --eval-trials 16 \
      --distractors 24 \
      --wrong-candidates 3 \
      --value-kind status \
      --seed 2026061324 \
      --eval-seed 2026061392

    run_exp g1_fixedeval_yarn_status_2k_wrong3_last12_seed2026061325 \
      "${common[@]}" \
      --seq-len 2048 \
      --train-steps 3000 \
      --last-n-layers 12 \
      --no-train-lm-head \
      --generative-weight 0 \
      --contrastive-weight 1.0 \
      --contrastive-wrong-candidates 3 \
      --contrastive-temperature 0.5 \
      --eval-every 750 \
      --eval-lengths 2048,4096,8192 \
      --eval-depths 0.9,0.5,0.05 \
      --train-depth-schedule 0.9,0.5,0.05 \
      --eval-trials 16 \
      --distractors 24 \
      --wrong-candidates 3 \
      --value-kind status \
      --seed 2026061325 \
      --eval-seed 2026061392

    run_exp g1_fixedeval_yarn_code_2k_wrong3_last12_seed2026061326 \
      "${common[@]}" \
      --seq-len 2048 \
      --train-steps 3000 \
      --last-n-layers 12 \
      --no-train-lm-head \
      --generative-weight 0 \
      --contrastive-weight 1.0 \
      --contrastive-wrong-candidates 3 \
      --contrastive-temperature 0.5 \
      --eval-every 750 \
      --eval-lengths 2048,4096 \
      --eval-depths 0.9,0.5 \
      --train-depth-schedule 0.9,0.5 \
      --eval-trials 16 \
      --distractors 24 \
      --wrong-candidates 3 \
      --value-kind code \
      --seed 2026061326 \
      --eval-seed 2026061393
    ;;

  *)
    echo "Unknown GPU_ID ${GPU_ID}; expected 0 or 1" >&2
    exit 2
    ;;
esac

echo "[$(date -Is)] GPU ${GPU_ID} v2 queue complete" | tee -a "${RUN_ROOT}/launcher_gpu${GPU_ID}.log"
