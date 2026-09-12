#!/usr/bin/env bash
#SBATCH --job-name=unlearn-track-b
#SBATCH --partition=lightwork
#SBATCH --gres=gpu:a100.20gb:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=outputs/slurm_logs/track-b-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ "${TRACK_B_CALIBRATION:-0}" == "1" ]]; then
  python scripts/evaluate_stereoset_track_b.py \
    --model "${TRACK_B_MODEL}" \
    --methods trained_full \
    --seeds 11 \
    --limit 64 \
    --output results/track_b_calibration \
    --batch-size 32
else
  python scripts/evaluate_stereoset_track_b.py \
    --model "${TRACK_B_MODEL}" \
    --output results/track_b \
    --batch-size 32
fi
