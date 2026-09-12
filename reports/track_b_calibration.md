# Track B compute calibration

The first 64 dataset-order StereoSet intrasentence items were scored for the seed-11 full-trained
checkpoint only. Bias metrics were not used for matrix selection.

| Model | Scheduler job | Scoring time | Peak allocation | Status |
| --- | --- | ---: | ---: | --- |
| Pythia-160M | `63045661` | 0.327 s | 1.00 GiB | completed |
| Mamba-130M-HF | `63045662` | 0.849 s | 0.87 GiB | completed |

Both scheduler jobs took 17 seconds including environment/model/checkpoint loading. Linear scoring
extrapolation for 2,106 examples × 27 checkpoints is about 291 seconds for Pythia and 754 seconds for
Mamba, before checkpoint loading and I/O. Even a conservative allowance remains well below the
predeclared 2 A100 GPU-hour stop threshold. The full 54-checkpoint Track B matrix is authorized with
the frozen batch size of 32 and 30-minute per-model scheduler caps.
