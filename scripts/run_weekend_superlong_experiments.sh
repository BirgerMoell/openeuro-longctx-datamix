#!/usr/bin/env bash
set -u

GPU_ID="${1:?usage: run_weekend_superlong_experiments.sh GPU_ID}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/birger/megatron_hf_conversion/.venv/bin/python}"
TRAINER="${TRAINER:-/home/ubuntu/birger/train_hf_superlong_smoke.py}"
RUN_ROOT="${RUN_ROOT:-/home/ubuntu/birger/runs/hf_superlong/weekend_20260613}"
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

common_2m=(
  --max-position 2097152
  --train-start-position 32768
  --position-strategy pose_bridge
  --eval-position-strategy pose_bridge
  --dtype bfloat16
)

case "${GPU_ID}" in
  0)
    run_exp g0_yarn_status_depthsched_contrastive_only_2k_last8_seed2026061301 \
      "${common_2m[@]}" \
      --seq-len 2048 \
      --rope-scaling yarn \
      --train-steps 3000 \
      --last-n-layers 8 \
      --no-train-lm-head \
      --generative-weight 0 \
      --contrastive-weight 1.0 \
      --contrastive-wrong-candidates 3 \
      --contrastive-temperature 0.5 \
      --eval-before \
      --eval-every 750 \
      --eval-lengths 2048,4096,8192 \
      --eval-depths 0.9,0.5,0.05 \
      --train-depth-schedule 0.9,0.5,0.05 \
      --eval-trials 12 \
      --distractors 24 \
      --wrong-candidates 3 \
      --value-kind status \
      --seed 2026061301

    run_exp g0_yarn_status_depthsched_mixed_temp025_2k_last8_seed2026061302 \
      "${common_2m[@]}" \
      --seq-len 2048 \
      --rope-scaling yarn \
      --train-steps 2400 \
      --last-n-layers 8 \
      --no-train-lm-head \
      --generative-weight 0.2 \
      --contrastive-weight 1.0 \
      --contrastive-wrong-candidates 3 \
      --contrastive-temperature 0.25 \
      --eval-before \
      --eval-every 600 \
      --eval-lengths 2048,4096,8192 \
      --eval-depths 0.9,0.5,0.05 \
      --train-depth-schedule 0.9,0.5,0.05 \
      --eval-trials 12 \
      --distractors 24 \
      --wrong-candidates 3 \
      --value-kind status \
      --seed 2026061302

    run_exp g0_yarn_code_depth09_contrastive_only_2k_last8_seed2026061303 \
      "${common_2m[@]}" \
      --seq-len 2048 \
      --rope-scaling yarn \
      --train-steps 2400 \
      --last-n-layers 8 \
      --no-train-lm-head \
      --generative-weight 0 \
      --contrastive-weight 1.0 \
      --contrastive-wrong-candidates 3 \
      --contrastive-temperature 0.5 \
      --eval-before \
      --eval-every 600 \
      --eval-lengths 2048,4096 \
      --eval-depths 0.9 \
      --train-depths 0.9 \
      --eval-trials 16 \
      --distractors 24 \
      --wrong-candidates 3 \
      --value-kind code \
      --seed 2026061303

    run_exp g0_linear_status_depthsched_contrastive_only_2k_last8_seed2026061304 \
      "${common_2m[@]}" \
      --seq-len 2048 \
      --rope-scaling linear \
      --train-steps 2400 \
      --last-n-layers 8 \
      --no-train-lm-head \
      --generative-weight 0 \
      --contrastive-weight 1.0 \
      --contrastive-wrong-candidates 3 \
      --contrastive-temperature 0.5 \
      --eval-before \
      --eval-every 600 \
      --eval-lengths 2048,4096,8192 \
      --eval-depths 0.9,0.5,0.05 \
      --train-depth-schedule 0.9,0.5,0.05 \
      --eval-trials 12 \
      --distractors 24 \
      --wrong-candidates 3 \
      --value-kind status \
      --seed 2026061304

    run_exp g0_yarn_status_depthsched_contrastive_only_2k_last8_seed2026061311 \
      "${common_2m[@]}" \
      --seq-len 2048 \
      --rope-scaling yarn \
      --train-steps 3000 \
      --last-n-layers 8 \
      --no-train-lm-head \
      --generative-weight 0 \
      --contrastive-weight 1.0 \
      --contrastive-wrong-candidates 3 \
      --contrastive-temperature 0.5 \
      --eval-before \
      --eval-every 750 \
      --eval-lengths 2048,4096,8192 \
      --eval-depths 0.9,0.5,0.05 \
      --train-depth-schedule 0.9,0.5,0.05 \
      --eval-trials 12 \
      --distractors 24 \
      --wrong-candidates 3 \
      --value-kind status \
      --seed 2026061311
    ;;

  1)
    run_exp g1_yarn_status_depthsched_contrastive_only_4k_last8_wrong3_seed2026061305 \
      "${common_2m[@]}" \
      --seq-len 4096 \
      --rope-scaling yarn \
      --train-steps 2000 \
      --last-n-layers 8 \
      --no-train-lm-head \
      --generative-weight 0 \
      --contrastive-weight 1.0 \
      --contrastive-wrong-candidates 3 \
      --contrastive-temperature 0.5 \
      --eval-before \
      --eval-every 500 \
      --eval-lengths 4096,8192 \
      --eval-depths 0.9,0.5,0.05 \
      --train-depth-schedule 0.9,0.5,0.05 \
      --eval-trials 8 \
      --distractors 24 \
      --wrong-candidates 3 \
      --value-kind status \
      --seed 2026061305

    run_exp g1_yarn_status_depthsched_contrastive_only_2k_last12_seed2026061306 \
      "${common_2m[@]}" \
      --seq-len 2048 \
      --rope-scaling yarn \
      --train-steps 3000 \
      --last-n-layers 12 \
      --no-train-lm-head \
      --generative-weight 0 \
      --contrastive-weight 1.0 \
      --contrastive-wrong-candidates 3 \
      --contrastive-temperature 0.5 \
      --eval-before \
      --eval-every 750 \
      --eval-lengths 2048,4096,8192 \
      --eval-depths 0.9,0.5,0.05 \
      --train-depth-schedule 0.9,0.5,0.05 \
      --eval-trials 12 \
      --distractors 24 \
      --wrong-candidates 3 \
      --value-kind status \
      --seed 2026061306

    run_exp g1_dynamic_status_depthsched_contrastive_only_2k_last8_seed2026061307 \
      "${common_2m[@]}" \
      --seq-len 2048 \
      --rope-scaling dynamic \
      --train-steps 2400 \
      --last-n-layers 8 \
      --no-train-lm-head \
      --generative-weight 0 \
      --contrastive-weight 1.0 \
      --contrastive-wrong-candidates 3 \
      --contrastive-temperature 0.5 \
      --eval-before \
      --eval-every 600 \
      --eval-lengths 2048,4096,8192 \
      --eval-depths 0.9,0.5,0.05 \
      --train-depth-schedule 0.9,0.5,0.05 \
      --eval-trials 12 \
      --distractors 24 \
      --wrong-candidates 3 \
      --value-kind status \
      --seed 2026061307

    run_exp g1_yarn_code_depthsched_contrastive_only_2k_last12_seed2026061308 \
      "${common_2m[@]}" \
      --seq-len 2048 \
      --rope-scaling yarn \
      --train-steps 2400 \
      --last-n-layers 12 \
      --no-train-lm-head \
      --generative-weight 0 \
      --contrastive-weight 1.0 \
      --contrastive-wrong-candidates 3 \
      --contrastive-temperature 0.5 \
      --eval-before \
      --eval-every 600 \
      --eval-lengths 2048,4096 \
      --eval-depths 0.9,0.5 \
      --train-depth-schedule 0.9,0.5 \
      --eval-trials 12 \
      --distractors 24 \
      --wrong-candidates 3 \
      --value-kind code \
      --seed 2026061308

    run_exp g1_yarn_status_depthsched_contrastive_only_4k_last8_wrong2_seed2026061312 \
      "${common_2m[@]}" \
      --seq-len 4096 \
      --rope-scaling yarn \
      --train-steps 2400 \
      --last-n-layers 8 \
      --no-train-lm-head \
      --generative-weight 0 \
      --contrastive-weight 1.0 \
      --contrastive-wrong-candidates 2 \
      --contrastive-temperature 0.5 \
      --eval-before \
      --eval-every 600 \
      --eval-lengths 4096,8192 \
      --eval-depths 0.9,0.5,0.05 \
      --train-depth-schedule 0.9,0.5,0.05 \
      --eval-trials 10 \
      --distractors 24 \
      --wrong-candidates 2 \
      --value-kind status \
      --seed 2026061312
    ;;

  *)
    echo "Unknown GPU_ID ${GPU_ID}; expected 0 or 1" >&2
    exit 2
    ;;
esac

echo "[$(date -Is)] GPU ${GPU_ID} queue complete" | tee -a "${RUN_ROOT}/launcher_gpu${GPU_ID}.log"
