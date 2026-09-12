# Device portability

## Selection and fallback policy

Experiments accept `device: auto|cuda|mps|cpu` and `precision: auto|fp32|fp16|bf16` in YAML or
through `scripts/run_experiment.py`. `auto` selects CUDA, then MPS, then CPU. An explicit unavailable
device raises an error; it never falls back to another device. CUDA auto precision selects BF16 only
when PyTorch reports support and a forward/backward probe succeeds. MPS and CPU auto precision use
FP32. Explicit precision requests must pass the same probe; CPU FP16 is rejected by policy. The
real-model calibration and main configurations explicitly use FP32 because ordinary AdamW does not
maintain FP32 master parameters for BF16 model weights; mixed-precision training remains unverified.

`PYTORCH_ENABLE_MPS_FALLBACK=1` is rejected unless `allow_mps_fallback: true` is also set. When it is
allowed, the manifest records that the environment fallback is enabled and states that PyTorch does
not expose per-operation fallback telemetry. Such runs require profiler evidence before a method is
classified as MPS-supported-with-fallback.

Schema-v2 run manifests record requested and actual device policy in the resolved configuration,
actual device, dtype, backend, fallback state, CUDA and MPS availability, software versions,
determinism settings, and device-memory measurements. Schema-v1 historical artifacts remain valid
and are not rewritten.

## Current verification matrix

Statuses below describe evidence collected in this repository, not expected framework support.

| Model/backend | CPU | MPS | CUDA | Current experiment status |
| --- | --- | --- | --- | --- |
| tiny-association fixture | Supported, all nine methods | Not yet hardware-tested | Not yet hardware-tested | CPU supported |
| Pythia-160M / 410M | Loader configured, experiment adapter pending | Not yet hardware-tested | Not yet hardware-tested | Unsupported end to end |
| Mamba-130M / 370M | Loader configured, experiment adapter pending | Not yet hardware-tested | Not yet hardware-tested | Unsupported end to end |

The device-policy tests mock MPS and CUDA availability to verify ordering, explicit overrides,
unsupported-device errors, dtype policy, manifest validation, and no-silent-fallback behavior. They
do not establish that a real model/method runs correctly on hardware that was unavailable to the
test host.

## Reproducibility scope

Python, NumPy, PyTorch, all visible CUDA devices, and DataLoader workers have deterministic seeding
utilities. Deterministic algorithms are requested with warnings enabled and cuDNN benchmarking is
disabled. This provides configuration reproducibility. Bitwise reproducibility is claimed only
within an unchanged software and device stack, never across CUDA, MPS, and CPU.

## Required hardware validation

Before Phase 3 can be declared complete, run every intended model/method combination on the target
M3 and A100 (or explicitly classify it as unsupported), inspect MPS fallback behavior, and record
precision and memory results. Mamba must retain its configured architecture; no substitute is
permitted. Phase 4 calibration must then measure sustained throughput, including a long enough M3
window to reveal obvious thermal slowdown.
