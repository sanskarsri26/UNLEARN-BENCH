#!/usr/bin/env bash
#SBATCH --job-name=unlearn-training-calibration
#SBATCH --partition=lightwork
#SBATCH --gres=gpu:a100.20gb:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=outputs/slurm_logs/training_calibration-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/calibrate_training.py --config "${CALIBRATION_CONFIG}"
