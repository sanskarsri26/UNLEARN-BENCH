# Reproducibility

Use Python 3.10 or newer. The core fixture requires NumPy, PyYAML, and PyTorch; exact minimums and
optional Hugging Face/figure/dev dependencies are in `pyproject.toml`.

```bash
pip install -e '.[dev,hf,figures]'
python scripts/prepare_data.py
python scripts/run_experiment.py --config configs/experiments/smoke.yaml --device cpu --precision fp32
python scripts/run_experiment.py --config configs/experiments/smoke.yaml --method pcgu --seed 11 --device cpu --precision fp32
python scripts/evaluate.py --run RUN_ID
python scripts/aggregate_results.py
python scripts/make_figures.py
python scripts/analyze_track_b.py --validate-only
python scripts/validate_results.py
```

Every run stores its resolved configuration, seed, git commit, UTC timestamp, dataset and partition
hashes, model/tokenizer revisions, software versions, hardware, runtime, peak VRAM, trainable
parameter count, checkpoint size, checkpoint path, raw prediction path, and metric path in JSON.
Raw predictions contain per-example losses, probabilities, log probabilities, margins, and oracle
divergence. Summary tables and figures read these artifacts rather than hard-coded values.

The main and ablation configurations are guarded by a calibration-review marker. The historical
confirmatory marker is intentionally bound to its frozen source hashes and no longer authorizes a
rerun after analysis/provenance changes. A replication must create a new preregistration and marker.

Device selection is explicit and auditable; see [device portability](device_portability.md). An
unavailable explicit device is an error. MPS CPU fallback is disabled unless both the environment
and resolved experiment configuration explicitly enable it.

## Confirmatory and report regeneration

The original Slurm entry points were:

```bash
MAIN_CONFIG=configs/experiments/main.yaml sbatch scripts/slurm_main.sh
MAIN_CONFIG=configs/experiments/main_mamba.yaml sbatch scripts/slurm_main.sh
```

`python scripts/analyze_confirmatory.py` strictly requires all 54 retained checkpoints and verifies
their SHA-256 hashes before regenerating confirmatory tables and figures. Those checkpoints total
31,481,561,142 bytes and are deliberately excluded from Git; their expected hashes and sizes remain
in committed manifests. `python scripts/validate_results.py` supports a clean checkout by validating
that external-retention metadata when the weights are absent. All Track B predictions are small
enough to be tracked, so `python scripts/analyze_track_b.py` can fully hash, recompute, and regenerate
that report in a clean checkout.

## Verified execution environments

- Confirmatory Track A: Python 3.12.9, PyTorch 2.6.0+cu124, Transformers 4.51.3, FP32, NVIDIA
  A100-SXM4-80GB exposed through 20 GiB MIG partitions. Mamba used the logged sequential eager path.
- Track B: the same software/model revisions, FP32 CUDA, A100 1g.20gb and 2g.20gb MIG partitions.
- MacBook Air M3: not available in this environment. The code has explicit MPS selection and
  fail-closed fallback tests, but macOS version, unified memory, runtime, and thermal behavior are
  intentionally unreported until measured on real hardware.

Configuration reproducibility is supported; bitwise equality across CUDA, MPS, software versions,
or GPU partition types is not promised.
