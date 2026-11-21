#!/bin/bash
# One experiment on one A5000.  Submit from the repo root:
#   sbatch slurm/run_experiment.sbatch configs/experiments/E0_baseline.yaml [key=value ...]
#   SCRIPT=scripts/calibrate_tau.py sbatch --export=ALL slurm/run_experiment.sbatch configs/base.yaml
# SCRIPT defaults to scripts/run_experiment.py; every stage resumes from disk, so a requeue just re-runs this.
#SBATCH --job-name=facecloak
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G                 # SD pipeline + VAE need < 10 GB host RAM; 32G often blocks on a busy node
#SBATCH --time=4:00:00
#SBATCH --output=runs/slurm/%x-%j.out
set -euo pipefail
source slurm/env.sh
CONFIG=${1:?usage: sbatch slurm/run_experiment.sbatch <config.yaml> [key=value ...]}
shift
python "${SCRIPT:-scripts/run_experiment.py}" --config "$CONFIG" "$@"
