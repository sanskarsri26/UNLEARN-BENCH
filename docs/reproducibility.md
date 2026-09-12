# Reproducibility

Use Python 3.10 or newer. The core fixture requires NumPy, PyYAML, and PyTorch; exact minimums and
optional Hugging Face/figure/dev dependencies are in `pyproject.toml`.

```bash
pip install -e '.[dev,figures]'
python scripts/prepare_data.py
python scripts/run_experiment.py --config configs/experiments/smoke.yaml --device cpu --precision fp32
python scripts/evaluate.py --run RUN_ID
python scripts/aggregate_results.py
python scripts/make_figures.py
```

Every run stores its resolved configuration, seed, git commit, UTC timestamp, dataset and partition
hashes, model/tokenizer revisions, software versions, hardware, runtime, peak VRAM, trainable
parameter count, checkpoint size, checkpoint path, raw prediction path, and metric path in JSON.
Raw predictions contain per-example losses, probabilities, log probabilities, margins, and oracle
divergence. Summary tables and figures read these artifacts rather than hard-coded values.

The main and ablation configurations are guarded by a calibration-review marker. Do not create that
marker until the measured compute budget has been reviewed.

Device selection is explicit and auditable; see [device portability](device_portability.md). An
unavailable explicit device is an error. MPS CPU fallback is disabled unless both the environment
and resolved experiment configuration explicitly enable it.
