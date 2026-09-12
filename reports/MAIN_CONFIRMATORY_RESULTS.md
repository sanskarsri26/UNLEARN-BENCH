# Main confirmatory results

**Preregistration:** `005db27a9f0b427bfa3fcc1cbcc3cc19da0c67a6`

All 54 frozen cells completed and passed manifest, prediction, metric, and checkpoint-hash validation. Lower is better for both primary endpoints.

## Primary seed-aggregated results

| Model | Method | Forget KL mean ± SD | Utility NLL mean ± SD | Retain NLL mean ± SD | Mean optimize s | Peak GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| pythia-160m | untouched | 7.8183 ± 0.4172 | 6.7628 ± 0.0000 | 13.4526 ± 0.0000 | 0.013 | 6.71 |
| pythia-160m | trained_full | 5.1226 ± 0.2911 | 4.7019 ± 0.0491 | 2.4096 ± 0.3227 | 1.629 | 6.71 |
| pythia-160m | exact_retrain | 0.0000 ± 0.0000 | 4.9272 ± 0.0094 | 1.3014 ± 0.0212 | 1.572 | 6.72 |
| pythia-160m | continued_retain | 6.0955 ± 0.6819 | 5.0347 ± 0.2725 | 1.8063 ± 0.0731 | 0.976 | 9.13 |
| pythia-160m | sham | 5.1225 ± 0.2909 | 4.7022 ± 0.0490 | 2.4099 ± 0.3228 | 0.943 | 8.07 |
| pythia-160m | counterfactual | 3.6046 ± 0.3276 | 4.9039 ± 0.0677 | 1.8681 ± 0.4331 | 0.972 | 9.13 |
| pythia-160m | gradient_ascent | 4.8803 ± 0.3524 | 4.6788 ± 0.0550 | 2.3170 ± 0.3904 | 0.514 | 9.74 |
| pythia-160m | npo | 4.8626 ± 0.3450 | 4.6803 ± 0.0562 | 2.3346 ± 0.3863 | 0.529 | 9.74 |
| pythia-160m | pcgu | 4.8189 ± 0.2094 | 4.6810 ± 0.0430 | 2.4378 ± 0.3198 | 1.191 | 9.41 |
| mamba-130m-hf | untouched | 3.6077 ± 0.0595 | 5.2944 ± 0.0000 | 13.5816 ± 0.0000 | 0.019 | 15.47 |
| mamba-130m-hf | trained_full | 1.8494 ± 0.0791 | 4.3615 ± 0.0604 | 3.4823 ± 0.2549 | 6.308 | 15.84 |
| mamba-130m-hf | exact_retrain | 0.0000 ± 0.0000 | 4.3740 ± 0.0434 | 2.8771 ± 0.0451 | 5.939 | 7.77 |
| mamba-130m-hf | continued_retain | 2.0953 ± 0.3085 | 4.2835 ± 0.0926 | 2.9479 ± 0.1837 | 3.581 | 10.16 |
| mamba-130m-hf | sham | 1.8493 ± 0.0788 | 4.3614 ± 0.0605 | 3.4823 ± 0.2550 | 1.011 | 9.83 |
| mamba-130m-hf | counterfactual | 1.7188 ± 0.1723 | 4.2635 ± 0.1050 | 2.5775 ± 0.2250 | 3.786 | 11.61 |
| mamba-130m-hf | gradient_ascent | 1.7968 ± 0.0542 | 4.3669 ± 0.0668 | 3.4548 ± 0.2956 | 2.462 | 12.76 |
| mamba-130m-hf | npo | 1.7952 ± 0.0523 | 4.3705 ± 0.0668 | 3.4666 ± 0.2940 | 2.504 | 13.72 |
| mamba-130m-hf | pcgu | 1.7727 ± 0.0379 | 4.3683 ± 0.0846 | 3.4919 ± 0.2718 | 4.493 | 14.42 |

## Prespecified paired comparisons

Effects are method minus full-trained. Negative effects favor the method. Confidence intervals are the preregistered hierarchical paired bootstrap; p-values use exact content-unit sign flips and Holm correction within each model × endpoint family.

| Model | Endpoint | Method | Effect | 95% CI | Exact p | Holm p |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| pythia-160m | forget_oracle_kl | continued_retain | 0.9729 | [0.1613, 2.0618] | 0.1250 | 0.7500 |
| pythia-160m | forget_oracle_kl | sham | -0.0001 | [-0.0009, 0.0007] | 0.7500 | 0.7500 |
| pythia-160m | forget_oracle_kl | counterfactual | -1.5179 | [-2.5617, -0.5393] | 0.1250 | 0.7500 |
| pythia-160m | forget_oracle_kl | gradient_ascent | -0.2423 | [-0.4550, -0.0232] | 0.1250 | 0.7500 |
| pythia-160m | forget_oracle_kl | npo | -0.2600 | [-0.4817, -0.0325] | 0.1250 | 0.7500 |
| pythia-160m | forget_oracle_kl | pcgu | -0.3037 | [-0.4253, -0.2245] | 0.1250 | 0.7500 |
| pythia-160m | utility_nll | continued_retain | 0.3327 | [0.0186, 0.6264] | 0.0352 | 0.1758 |
| pythia-160m | utility_nll | sham | 0.0002 | [-0.0005, 0.0009] | 0.3359 | 0.5117 |
| pythia-160m | utility_nll | counterfactual | 0.2020 | [0.0956, 0.3300] | 0.0254 | 0.1523 |
| pythia-160m | utility_nll | gradient_ascent | -0.0232 | [-0.0428, -0.0054] | 0.0957 | 0.3828 |
| pythia-160m | utility_nll | npo | -0.0216 | [-0.0428, -0.0031] | 0.1113 | 0.3828 |
| pythia-160m | utility_nll | pcgu | -0.0210 | [-0.0749, 0.0324] | 0.2559 | 0.5117 |
| mamba-130m-hf | forget_oracle_kl | continued_retain | 0.2459 | [0.0253, 0.5721] | 0.1250 | 0.7500 |
| mamba-130m-hf | forget_oracle_kl | sham | -0.0001 | [-0.0003, 0.0001] | 0.7500 | 1.0000 |
| mamba-130m-hf | forget_oracle_kl | counterfactual | -0.1306 | [-0.4784, 0.1772] | 0.6250 | 1.0000 |
| mamba-130m-hf | forget_oracle_kl | gradient_ascent | -0.0525 | [-0.1250, 0.0039] | 0.5000 | 1.0000 |
| mamba-130m-hf | forget_oracle_kl | npo | -0.0542 | [-0.1342, 0.0070] | 0.5000 | 1.0000 |
| mamba-130m-hf | forget_oracle_kl | pcgu | -0.0767 | [-0.1208, -0.0390] | 0.1250 | 0.7500 |
| mamba-130m-hf | utility_nll | continued_retain | -0.0780 | [-0.1670, 0.0118] | 0.2012 | 1.0000 |
| mamba-130m-hf | utility_nll | sham | -0.0000 | [-0.0003, 0.0003] | 0.8398 | 1.0000 |
| mamba-130m-hf | utility_nll | counterfactual | -0.0980 | [-0.2319, 0.0114] | 0.3516 | 1.0000 |
| mamba-130m-hf | utility_nll | gradient_ascent | 0.0054 | [-0.0024, 0.0159] | 0.2324 | 1.0000 |
| mamba-130m-hf | utility_nll | npo | 0.0090 | [0.0003, 0.0203] | 0.0762 | 0.4570 |
| mamba-130m-hf | utility_nll | pcgu | 0.0068 | [-0.0387, 0.0350] | 0.7637 | 1.0000 |

## Pareto and joint-criterion summary

- **pythia-160m:** forgetting–utility frontier: exact_retrain, counterfactual, gradient_ascent, npo, pcgu; forgetting–compute frontier: untouched, exact_retrain, counterfactual, gradient_ascent, npo.
- **pythia-160m joint preregistered criterion:** gradient_ascent, npo.
- **mamba-130m-hf:** forgetting–utility frontier: exact_retrain, counterfactual; forgetting–compute frontier: untouched, exact_retrain, sham, counterfactual, gradient_ascent, npo.
- **mamba-130m-hf joint preregistered criterion:** no method met the criterion.

## Confirmatory interpretation

For Pythia, counterfactual training produced the largest mean forgetting change from full-trained (-1.5179) but worsened utility (0.2020). Gradient ascent and NPO met the preregistered joint criterion; PCGU improved mean forgetting with an interval below zero, but its utility interval crossed zero.

For Mamba, no method met the joint criterion. PCGU's forgetting interval was below zero, while its utility interval crossed zero. Counterfactual training had the largest mean forgetting and utility improvements, but both intervals crossed zero. Continued retain training moved both models farther from the oracle on forget examples.

No Holm-adjusted comparison reached 0.05 (smallest adjusted p = 0.1523). This is consistent with the limited four-group forget holdout and prevents strong significance claims despite several bootstrap intervals excluding zero.

## Compute, failures, and deviations

All 54 cells completed with no technical, OOM, unsupported-device, numerical, or method failures. Scheduler elapsed time totaled 413 A100-seconds (0.115 GPU-hours). Peak allocation was 9.74 GiB for Pythia and 15.84 GiB for Mamba. The only execution deviation was moving two zero-runtime pending jobs from `htc` to the available `lightwork` A100 MIG pool; frozen experiment settings did not change. Mamba used the predeclared sequential eager fallback.

## Interpretation discipline

These results establish behavior only for the controlled synthetic associations, two small pinned checkpoints, and the fixed short optimization budget. Cross-model differences are observational, not causal architecture effects. Exact-retrain KL is zero by definition. Track B external behavior, ablations, and error analysis are separate exploratory analyses.

## Reproduction

Regenerate this report, CSV tables, and figures with `python scripts/analyze_confirmatory.py`. Individual seed values are stored in `reports/tables/main_seed_results.csv`.
