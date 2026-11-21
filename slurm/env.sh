# Sourced by the sbatch scripts: environment, caches on /mnt, repo as working directory.
export FC_ENV=${FC_ENV:-/mnt/users/$USER/envs/facecloak}
export HF_HOME=${HF_HOME:-/mnt/users/$USER/models/hf}
export KAGGLEHUB_CACHE=${KAGGLEHUB_CACHE:-/mnt/users/$USER/facecloak/kagglehub}
export TORCH_HOME=${TORCH_HOME:-/mnt/users/$USER/models/torch}
export PYTHONUNBUFFERED=1
source ~/miniconda3/bin/activate "$FC_ENV"
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
echo "[$(date '+%F %T')] $(hostname) job=${SLURM_JOB_ID:-local} gpu=${CUDA_VISIBLE_DEVICES:-none} python=$(which python)"
