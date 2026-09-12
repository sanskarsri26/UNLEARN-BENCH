#!/usr/bin/env bash
#SBATCH --job-name=unlearn-device-smoke
#SBATCH --partition=htc
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=outputs/slurm_logs/device_smoke-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python - <<'PY'
import json
import torch
from unlearn_bench.utils.device import available_devices, select_device

context = select_device("cuda", "auto")
print(json.dumps({
    "available": available_devices(),
    "selected": context.manifest(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
}, indent=2, sort_keys=True))
PY
python scripts/run_experiment.py --config configs/experiments/smoke.yaml --device cuda --precision auto
python scripts/run_experiment.py --config configs/experiments/smoke.yaml --device cuda --precision fp32
python scripts/validate_results.py
