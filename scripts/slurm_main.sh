#!/usr/bin/env bash
#SBATCH --job-name=unlearn-main
#SBATCH --partition=htc
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=outputs/slurm_logs/main-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python scripts/run_experiment.py --config "${MAIN_CONFIG}"
