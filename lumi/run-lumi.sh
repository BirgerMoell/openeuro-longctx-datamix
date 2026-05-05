#!/usr/bin/env bash
set -euo pipefail

LUMI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COMMAND="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

PROJECT="${LONGCTX_LUMI_PROJECT:-}"
WORKDIR="${LONGCTX_LUMI_WORKDIR:-$LUMI_DIR/work}"
JOB_ID_FILE="$LUMI_DIR/.last-job-id"

usage() {
  cat <<'EOF'
openeuro-longctx-datamix LUMI runner

Usage:
  bash run-lumi.sh hello  --project project_462000963
  bash run-lumi.sh setup  --project project_462000963
  bash run-lumi.sh status
  bash run-lumi.sh logs

Commands:
  hello   Submit a hello-world data-pipeline job (no GPU needed)
  setup   Install the package and verify the environment (interactive)
  status  Show status of the last submitted job
  logs    Tail the most recent log file

Options:
  --project ID    Slurm account, e.g. project_462000963
  --workdir PATH  Work/cache directory (default: ./work)
  --repo    PATH  Local clone of openeuro-longctx-datamix (cloned if absent)

Environment overrides:
  LONGCTX_LUMI_PROJECT
  LONGCTX_LUMI_WORKDIR
  LONGCTX_REPO_DIR
EOF
}

REPO_DIR="${LONGCTX_REPO_DIR:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)
      PROJECT="${2:?missing project id after --project}"
      shift 2
      ;;
    --workdir)
      WORKDIR="${2:?missing path after --workdir}"
      shift 2
      ;;
    --repo)
      REPO_DIR="${2:?missing path after --repo}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

infer_project() {
  local path="${PWD}"
  if [[ "$path" =~ /(scratch|project|flash)/(project_[0-9]+)(/|$) ]]; then
    printf '%s\n' "${BASH_REMATCH[2]}"
  fi
}

need_project() {
  if [[ -z "$PROJECT" ]]; then
    PROJECT="$(infer_project || true)"
  fi
  if [[ -z "$PROJECT" ]]; then
    echo "Could not infer Slurm project. Pass --project project_XXXXXXXXX." >&2
    exit 2
  fi
}

ensure_dirs() {
  mkdir -p "$WORKDIR" "$LUMI_DIR/logs"
}

submit_job() {
  local name="$1"
  local sbatch_file="$2"
  need_project
  ensure_dirs
  local output
  output="$(sbatch \
    -A "$PROJECT" \
    --export=ALL,LONGCTX_LUMI_PROJECT="$PROJECT",LONGCTX_LUMI_WORKDIR="$WORKDIR",LONGCTX_REPO_DIR="${REPO_DIR:-$WORKDIR/openeuro-longctx-datamix}",LONGCTX_LUMI_SCRIPT="$LUMI_DIR/run-lumi.sh" \
    "$sbatch_file")"
  local job_id
  job_id="$(awk '{print $4}' <<<"$output")"
  printf '%s\n' "$job_id" > "$JOB_ID_FILE"
  echo "$name submitted as job $job_id"
  echo "Logs: $LUMI_DIR/logs/"
}

status() {
  if [[ -f "$JOB_ID_FILE" ]]; then
    local job_id
    job_id="$(cat "$JOB_ID_FILE")"
    squeue -j "$job_id" || true
  else
    squeue -u "$USER" || true
  fi
}

logs() {
  local latest
  latest="$(ls -t "$LUMI_DIR"/logs/*.out 2>/dev/null | head -n 1 || true)"
  if [[ -z "$latest" ]]; then
    echo "No log files found in $LUMI_DIR/logs yet." >&2
    exit 1
  fi
  echo "Tailing $latest"
  tail -n 120 -f "$latest"
}

case "$COMMAND" in
  hello)
    submit_job "longctx hello-world data pipeline" "$LUMI_DIR/slurm/hello.sbatch"
    echo "Rerun anytime with: bash run-lumi.sh hello --project project_462000963"
    ;;
  setup)
    ensure_dirs
    export LONGCTX_LUMI_WORKDIR="$WORKDIR"
    export LONGCTX_REPO_DIR="${REPO_DIR:-$WORKDIR/openeuro-longctx-datamix}"
    bash "$LUMI_DIR/slurm/run-inside-lumi.sh" setup
    ;;
  status)
    status
    ;;
  logs)
    logs
    ;;
  _inside)
    export LONGCTX_LUMI_WORKDIR="$WORKDIR"
    export LONGCTX_REPO_DIR="${REPO_DIR:-$WORKDIR/openeuro-longctx-datamix}"
    bash "$LUMI_DIR/slurm/run-inside-lumi.sh" "${1:?missing internal mode}"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $COMMAND" >&2
    usage
    exit 2
    ;;
esac
