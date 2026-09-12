# Compute budget (awaiting GPU calibration)

The canonical CPU fixture passed all nine configured methods on 2026-09-12 from commit `901cafb`.
The attributed per-run times sum to 1.965 seconds; the full command, including creation of shared
full-training and exact-retrain checkpoints, took 6.37 seconds. It evaluated 108 method/example
pairs (a coarse 55.0 pairs/second over attributed run time), allocated 0 GPU bytes, and wrote nine
14,697-byte checkpoints (132,273 bytes total). This validates plumbing only and is not a substitute
for a Pythia/Mamba GPU estimate.

Provisional main matrix (frozen for costing, not authorized for execution):

| Tier | Models | Methods | Seeds | Runs |
| --- | ---: | ---: | ---: | ---: |
| Smoke GPU | 2 (Pythia-160M, Mamba-130M) | 9 | 1 | 18 |
| Main | 2 (Pythia-410M, Mamba-370M) | 9 | 3 | 54 |

Before creating `results/manifests/CALIBRATION_REVIEWED`, record examples/s, steps/s, peak VRAM,
wall time, checkpoint size, estimated GPU-hours per run and for the full matrix, storage, hardware,
software versions, and reviewer/date. Main runs remain configuration-gated until then.
