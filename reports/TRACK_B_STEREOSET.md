# Track B: StereoSet external generalization

**EXPLORATORY. This is external bias behavior, not machine forgetting.**

All 54 Track A checkpoints were evaluated on all 2,106 official StereoSet dev intrasentence examples. SS is neutral near 50; lower SS is not inherently better.

| Model | Method | LMS mean ± SD | SS mean ± SD | |SS−50| | ICAT mean ± SD | Score s | Peak GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pythia-160m | untouched | 80.75 ± 0.00 | 58.88 ± 0.00 | 8.88 | 66.41 ± 0.00 | 20.2 | 1.50 |
| pythia-160m | trained_full | 75.69 ± 0.64 | 57.85 ± 0.23 | 7.85 | 63.80 ± 0.19 | 20.2 | 1.50 |
| pythia-160m | exact_retrain | 76.20 ± 0.26 | 56.66 ± 0.26 | 6.66 | 66.05 ± 0.55 | 20.2 | 1.50 |
| pythia-160m | continued_retain | 76.64 ± 1.54 | 58.67 ± 0.63 | 8.67 | 63.34 ± 1.07 | 20.2 | 1.50 |
| pythia-160m | sham | 75.68 ± 0.63 | 57.87 ± 0.22 | 7.87 | 63.77 ± 0.20 | 20.2 | 1.50 |
| pythia-160m | counterfactual | 74.72 ± 0.17 | 56.85 ± 0.82 | 6.85 | 64.47 ± 1.11 | 20.2 | 1.50 |
| pythia-160m | gradient_ascent | 75.41 ± 0.64 | 57.77 ± 0.49 | 7.77 | 63.69 ± 0.43 | 20.2 | 1.50 |
| pythia-160m | npo | 75.40 ± 0.67 | 57.76 ± 0.48 | 7.76 | 63.70 ± 0.53 | 20.2 | 1.50 |
| pythia-160m | pcgu | 75.50 ± 0.62 | 57.72 ± 0.43 | 7.72 | 63.83 ± 0.13 | 20.2 | 1.50 |
| mamba-130m-hf | untouched | 82.53 ± 0.00 | 60.83 ± 0.00 | 10.83 | 64.66 ± 0.00 | 75.3 | 1.38 |
| mamba-130m-hf | trained_full | 81.98 ± 0.05 | 60.89 ± 0.05 | 10.89 | 64.13 ± 0.12 | 75.2 | 1.38 |
| mamba-130m-hf | exact_retrain | 81.62 ± 0.04 | 60.72 ± 0.15 | 10.72 | 64.13 ± 0.27 | 75.2 | 1.38 |
| mamba-130m-hf | continued_retain | 81.86 ± 0.30 | 61.27 ± 0.15 | 11.27 | 63.41 ± 0.28 | 75.4 | 1.38 |
| mamba-130m-hf | sham | 81.98 ± 0.05 | 60.89 ± 0.05 | 10.89 | 64.13 ± 0.12 | 75.2 | 1.38 |
| mamba-130m-hf | counterfactual | 81.16 ± 0.62 | 61.51 ± 0.24 | 11.51 | 62.49 ± 0.79 | 75.2 | 1.38 |
| mamba-130m-hf | gradient_ascent | 81.97 ± 0.01 | 60.84 ± 0.19 | 10.84 | 64.20 ± 0.30 | 75.2 | 1.38 |
| mamba-130m-hf | npo | 81.99 ± 0.05 | 60.81 ± 0.17 | 10.81 | 64.26 ± 0.27 | 75.2 | 1.38 |
| mamba-130m-hf | pcgu | 81.71 ± 0.19 | 60.89 ± 0.17 | 10.89 | 63.91 ± 0.25 | 75.2 | 1.38 |

## Paired changes from full-trained

Negative Δ|SS−50| moves toward neutrality; positive ΔLMS and ΔICAT are improvements. Values are means across the three paired seeds, with no confirmatory hypothesis tests.

| Model | Method | ΔLMS | ΔSS | Δ|SS−50| | ΔICAT |
| --- | --- | ---: | ---: | ---: | ---: |
| pythia-160m | untouched | 5.057 | 1.029 | 1.029 | 2.604 |
| pythia-160m | exact_retrain | 0.514 | -1.187 | -1.187 | 2.245 |
| pythia-160m | continued_retain | 0.950 | 0.823 | 0.823 | -0.467 |
| pythia-160m | sham | -0.008 | 0.016 | 0.016 | -0.030 |
| pythia-160m | counterfactual | -0.973 | -0.997 | -0.997 | 0.670 |
| pythia-160m | gradient_ascent | -0.277 | -0.079 | -0.079 | -0.116 |
| pythia-160m | npo | -0.285 | -0.095 | -0.095 | -0.098 |
| pythia-160m | pcgu | -0.190 | -0.127 | -0.127 | 0.030 |
| mamba-130m-hf | untouched | 0.546 | -0.063 | -0.063 | 0.532 |
| mamba-130m-hf | exact_retrain | -0.364 | -0.174 | -0.174 | -0.001 |
| mamba-130m-hf | continued_retain | -0.119 | 0.380 | 0.380 | -0.715 |
| mamba-130m-hf | sham | 0.000 | 0.000 | 0.000 | 0.000 |
| mamba-130m-hf | counterfactual | -0.815 | 0.617 | 0.617 | -1.638 |
| mamba-130m-hf | gradient_ascent | -0.008 | -0.047 | -0.047 | 0.072 |
| mamba-130m-hf | npo | 0.008 | -0.079 | -0.079 | 0.136 |
| mamba-130m-hf | pcgu | -0.269 | 0.000 | 0.000 | -0.211 |

## Category heterogeneity

The rows below are selected mechanically as the largest movement toward and away from SS neutrality within each model. Exact retraining and untouched are excluded from this intervention diagnostic.

| Model | Direction | Method | Category | Δ|SS−50| | ΔLMS |
| --- | --- | --- | --- | ---: | ---: |
| pythia-160m | toward neutrality | counterfactual | gender | -2.222 | -1.111 |
| pythia-160m | away from neutrality | continued_retain | profession | 1.481 | 0.576 |
| mamba-130m-hf | toward neutrality | counterfactual | gender | -1.046 | -1.699 |
| mamba-130m-hf | away from neutrality | counterfactual | religion | 5.063 | -0.633 |

Category-level values are in `reports/tables/track_b_categories.csv`. These exploratory results do not establish deletion, training-data membership, or causal architecture effects. Track A utility must be consulted alongside any apparent bias movement.
