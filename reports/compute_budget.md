# Compute budget (awaiting GPU calibration)

The CPU fixture passed all nine configured methods on 2026-09-12. The method-level manifest times
sum to 1.725 seconds; the full command, including creation of shared full-training and exact-retrain
checkpoints, took 11.05 seconds. It evaluated 108 method/example pairs (a coarse 62.6 pairs/second
over method-level time), allocated 0 GPU bytes, and wrote nine 14,697-byte checkpoints (132,273
bytes total). This validates plumbing only and is not a substitute for a Pythia/Mamba GPU estimate.

Provisional main matrix (frozen for costing, not authorized for execution):

| Tier | Models | Methods | Seeds | Runs |
| --- | ---: | ---: | ---: | ---: |
| Smoke GPU | 2 (Pythia-160M, Mamba-130M) | 9 | 1 | 18 |
| Main | 2 (Pythia-410M, Mamba-370M) | 9 | 3 | 54 |

Before creating `results/manifests/CALIBRATION_REVIEWED`, record examples/s, steps/s, peak VRAM,
wall time, checkpoint size, estimated GPU-hours per run and for the full matrix, storage, hardware,
software versions, and reviewer/date. Main runs remain configuration-gated until then.
