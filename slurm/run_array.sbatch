#!/bin/bash
# A sweep: line N of LIST (= $SLURM_ARRAY_TASK_ID) is "<config.yaml> [key=value ...]". Max 2 GPUs per user:
#   sbatch --array=1-$(grep -cv '^\s*#' slurm/lists/phase3.txt)%2 slurm/run_array.sbatch slurm/lists/phase3.txt
# Blank and '#' lines are ignored when counting. SCRIPT works as in run_experiment.sbatch.
#SBATCH --job-name=facecloak-arr
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G                 # SD pipeline + VAE need < 10 GB host RAM; 32G often blocks on a busy node
#SBATCH --time=4:00:00
#SBATCH --output=runs/slurm/%x-%A_%a.out
set -euo pipefail
source slurm/env.sh
LIST=${1:?usage: sbatch --array=1-N%2 slurm/run_array.sbatch <list.txt>}
LINE=$(grep -v '^\s*#' "$LIST" | grep -v '^\s*$' | sed -n "${SLURM_ARRAY_TASK_ID}p")
[ -n "$LINE" ] || { echo "no line $SLURM_ARRAY_TASK_ID in $LIST"; exit 1; }
echo "task $SLURM_ARRAY_TASK_ID: $LINE"
read -r -a ARGS <<< "$LINE"
python "${SCRIPT:-scripts/run_experiment.py}" --config "${ARGS[0]}" "${ARGS[@]:1}"
