# Device calibration

**Status: EXPLORATORY COMPUTE CALIBRATION — NOT A CONFIRMATORY RESULT.**

This report is regenerated from schema-v2 raw run artifacts by
`python scripts/summarize_calibration.py`. Calibration commits: 81beb6d.

## Measured A100 results

| Model | Method | Hardware / dtype | Load s | Optimize s | Eval s | Ex/s | Tok/s | Peak GPU GiB | Peak process GiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mamba-130m-hf | continued_retain | NVIDIA A100-SXM4-80GB MIG 1g.20gb / fp32 | 12.17 | 0.919 | 1.471 | 26.11 | 29.38 | 6.32 | 1.75 |
| mamba-130m-hf | counterfactual | NVIDIA A100-SXM4-80GB MIG 1g.20gb / fp32 | 12.17 | 1.320 | 1.473 | 27.28 | 38.64 | 8.02 | 1.75 |
| mamba-130m-hf | exact_retrain | NVIDIA A100-SXM4-80GB MIG 1g.20gb / fp32 | 12.17 | 5.326 | 1.461 | 30.04 | 33.80 | 3.92 | 1.75 |
| mamba-130m-hf | gradient_ascent | NVIDIA A100-SXM4-80GB MIG 1g.20gb / fp32 | 12.17 | 1.838 | 1.473 | 19.58 | 24.48 | 8.91 | 1.75 |
| mamba-130m-hf | npo | NVIDIA A100-SXM4-80GB MIG 1g.20gb / fp32 | 12.17 | 1.856 | 1.477 | 19.39 | 24.24 | 9.87 | 1.75 |
| mamba-130m-hf | pcgu | NVIDIA A100-SXM4-80GB MIG 1g.20gb / fp32 | 12.17 | 0.759 | 1.475 | 10.54 | 18.44 | 10.24 | 1.75 |
| mamba-130m-hf | sham | NVIDIA A100-SXM4-80GB MIG 1g.20gb / fp32 | 12.17 | 0.666 | 1.466 | 0.00 | 0.00 | 5.64 | 1.75 |
| mamba-130m-hf | trained_full | NVIDIA A100-SXM4-80GB MIG 1g.20gb / fp32 | 12.17 | 8.621 | 1.464 | 27.84 | 34.80 | 3.44 | 1.75 |
| mamba-130m-hf | untouched | NVIDIA A100-SXM4-80GB MIG 1g.20gb / fp32 | 12.17 | 0.018 | 1.476 | 0.00 | 0.00 | 2.96 | 1.75 |
| pythia-160m | continued_retain | NVIDIA A100-SXM4-80GB / fp32 | 13.48 | 0.206 | 0.373 | 116.42 | 130.97 | 6.71 | 1.83 |
| pythia-160m | counterfactual | NVIDIA A100-SXM4-80GB / fp32 | 13.48 | 0.185 | 0.365 | 194.25 | 275.19 | 6.71 | 1.83 |
| pythia-160m | exact_retrain | NVIDIA A100-SXM4-80GB / fp32 | 13.48 | 0.941 | 0.368 | 170.01 | 191.26 | 4.30 | 1.83 |
| pythia-160m | gradient_ascent | NVIDIA A100-SXM4-80GB / fp32 | 13.48 | 0.318 | 0.369 | 113.32 | 141.65 | 7.32 | 1.83 |
| pythia-160m | npo | NVIDIA A100-SXM4-80GB / fp32 | 13.48 | 0.325 | 0.371 | 110.86 | 138.57 | 7.32 | 1.83 |
| pythia-160m | pcgu | NVIDIA A100-SXM4-80GB / fp32 | 13.48 | 0.177 | 0.369 | 45.29 | 79.25 | 6.69 | 1.83 |
| pythia-160m | sham | NVIDIA A100-SXM4-80GB / fp32 | 13.48 | 0.865 | 0.369 | 0.00 | 0.00 | 5.19 | 1.83 |
| pythia-160m | trained_full | NVIDIA A100-SXM4-80GB / fp32 | 13.48 | 2.672 | 0.369 | 89.82 | 112.27 | 4.30 | 1.83 |
| pythia-160m | untouched | NVIDIA A100-SXM4-80GB / fp32 | 13.48 | 0.012 | 0.400 | 0.00 | 0.00 | 3.70 | 1.83 |

## PCGU memory

| Model | Two gradient sets GiB | Ranking mask GiB |
| --- | ---: | ---: |
| mamba-130m-hf | 0.962 | 0.120 |
| pythia-160m | 1.209 | 0.151 |

## Compatibility and feasibility

| Model | M3 MPS | M3 CPU | A100 | Recommended device |
| --- | --- | --- | --- | --- |
| Pythia-160M | NOT MEASURED | NOT MEASURED | SUPPORTED (FP32) | A100 until M3 calibration |
| Mamba-130M | NOT MEASURED | NOT MEASURED | SUPPORTED WITH SEQUENTIAL EAGER KERNEL FALLBACK (FP32) | A100 |
| Pythia-410M | NOT MEASURED | NOT MEASURED | NOT MEASURED | Undetermined |
| Mamba-370M | NOT MEASURED | NOT MEASURED | NOT MEASURED | Undetermined |

The Mamba job log explicitly reported that optional fused selective-scan and causal-convolution kernels were unavailable and that Transformers used its sequential implementation. This is a kernel implementation fallback on CUDA, not an MPS-to-CPU device fallback. No architecture was substituted.

The current session is Linux/x86_64, not the target MacBook Air M3. MPS compatibility, CPU fallback behavior, sustained thermal throughput, and M3 memory therefore remain unmeasured. No local-feasibility classification is inferred from A100 data.

## Scientific interpretation

The calibration endpoints were inspected only to detect gross failures. Pythia's short calibration fit the controlled associations while catastrophically damaging held-out utility; Mamba did not show the same gross collapse under this exploratory setup. These observations motivate pre-confirmatory optimization calibration but are not method comparisons, architecture claims, or headline results. No calibration endpoint may be promoted into the confirmatory report.

## Remaining measurements

Checkpoint write time for a real saved checkpoint, sustained M3 throughput, MPS allocation, CPU fallback profiling, and large-tier scaling remain outstanding. The confirmatory matrix must not be authorized until those omissions are resolved or explicitly excluded before preregistration.
