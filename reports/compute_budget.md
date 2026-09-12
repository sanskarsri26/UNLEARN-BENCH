# Compute budget (A100 calibrated; M3 pending)

The canonical CPU fixture passed all nine configured methods on 2026-09-12 from commit `901cafb`.
The attributed per-run times sum to 1.965 seconds; the full command, including creation of shared
full-training and exact-retrain checkpoints, took 6.37 seconds. It evaluated 108 method/example
pairs (a coarse 55.0 pairs/second over attributed run time), allocated 0 GPU bytes, and wrote nine
14,697-byte checkpoints (132,273 bytes total). This validates plumbing only and is not a substitute
for a Pythia/Mamba GPU estimate.

Real-model A100 calibration was completed from commit `81beb6d` with FP32 Pythia-160M and
Mamba-130M. The 18 model/method cells completed in 31 and 60 seconds of scheduler wall time,
respectively (91 A100-seconds total). Peak PyTorch allocation was 7.32 GiB for Pythia and 10.24 GiB
for Mamba. Mamba used an A100 20 GiB MIG device and the sequential eager implementation because its
optional fused kernels were unavailable. See `reports/device_calibration.md` for per-method data.

Provisional confirmatory matrix (costing only, not authorized for execution):

| Tier | Models | Methods | Seeds | Runs |
| --- | ---: | ---: | ---: | ---: |
| Calibrated exploratory | 2 (Pythia-160M, Mamba-130M) | 9 | 1 | 18 |
| Candidate main small tier | 2 (Pythia-160M, Mamba-130M) | 9 | 3 | 54 |
| Deferred large tier | 2 (Pythia-410M, Mamba-370M) | 9 | 3 | 54 |

The small-tier candidate is comfortably below the 24 GPU-hour stop threshold based on the measured
short-run throughput, but the estimate is not yet frozen: full-length optimizer behavior,
real-checkpoint write time/storage, and a utility-preserving Pythia setup still need calibration.
M3 MPS/CPU performance and thermal behavior are also unmeasured because the active environment is
Linux/x86_64. Do not create `results/manifests/CALIBRATION_REVIEWED` yet. Main runs remain gated.
