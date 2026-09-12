# Frozen confirmatory compute budget

The confirmatory small-tier matrix is 2 models × 9 methods × 3 seeds = 54 terminal cells. It uses
FP32 on an NVIDIA A100 and is frozen in `docs/preregistration_main.md`. The prior 410M/370M candidate
tier is deferred.

## Measured basis

| Model | Exploratory nine-method job | Peak allocation | Checkpoint bytes | Write time |
| --- | ---: | ---: | ---: | ---: |
| Pythia-160M | 30 s | 7.32 GiB | 649,350,482 | 0.596 s |
| Mamba-130M-HF | 58 s | 10.57 GiB | 516,633,264 | 0.445 s |

The method-calibration jobs included 20-step full and exact training and the final intervention
budgets. They completed all 18 cells without OOM or non-finite parameters. Mamba ran on an A100 20
GiB MIG device using the sequential eager implementation. The observed 88 A100-seconds for one seed
of both models scale to approximately 264 seconds for three seeds before additional checkpoint I/O.

## Frozen authorization ceiling

- Expected compute: under 0.10 A100 GPU-hours including measured checkpoint writes.
- Authorized ceiling: 1.0 A100 GPU-hour, allowing scheduler and filesystem variance.
- Stop threshold from the project protocol: 24 GPU-hours; the frozen ceiling is well below it.
- Peak-memory request: 20 GiB A100 MIG or larger; calibrated maximum is 10.57 GiB.
- Checkpoint storage: 31,481,561,142 bytes (31.48 GB, 29.32 GiB) for 54 checkpoint copies.

Wall time is measured but is not the compute-matching variable. Approximate interventions are
matched on 48 loss-contributing examples processed. Token counts, runtime, throughput, and peak
memory remain reported compute outcomes.

M3/MPS throughput and thermal behavior remain unmeasured because the available environment is
Linux/x86_64. The confirmatory protocol therefore authorizes A100 only and makes no fabricated M3
estimate. A valid `CALIBRATION_REVIEWED` marker tied to the preregistration commit is still required
before execution.
