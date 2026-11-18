#!/bin/bash
#SBATCH --job-name=convert-prelude-hf2meg
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=2:00:00
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
CONTAINER=/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND_DIRS=/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd
HF_MODEL=/scratch/project_465002530/users/bmoell/prelude_hf/iter_0124800
MEG_OUT=/scratch/project_465002530/users/bmoell/longctx-extend/prelude_meg
BRIDGE=/flash/project_465002530/tools/Megatron-Bridge-LUMI
mkdir -p "$MEG_OUT"
srun singularity exec -B "$BIND_DIRS" "$CONTAINER" bash -lc "
  export PYTHONPATH=$BRIDGE/python-packages:$BRIDGE/3rdparty/Megatron-LM:$BRIDGE/src
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  echo '### prelude HF -> Megatron '\$(date)
  python $BRIDGE/examples/conversion/convert_checkpoints.py import --hf-model $HF_MODEL --megatron-path $MEG_OUT 2>&1
"
echo '### import done'; ls -la "$MEG_OUT" | head
echo PRELUDE_CONVERT_DONE $(date)
