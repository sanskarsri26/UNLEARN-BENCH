# Post-confirmatory error analysis

**EXPLORATORY.** Ranking rules below were applied to every eligible example; examples were not selected by narrative convenience.

## Hardest forget associations

A hard association has the largest residual KL distance to the exact-retrain oracle, averaged over all six approximate methods, three surface forms, and three seeds.

| Model | Rank | Association | Residual KL |
| --- | ---: | --- | ---: |
| pythia-160m | 1 | association-ivy | 9.9191 |
| pythia-160m | 2 | association-ginkgo | 6.6650 |
| pythia-160m | 3 | association-juniper | 1.6147 |
| pythia-160m | 4 | association-hazel | 1.3909 |
| mamba-130m-hf | 1 | association-ivy | 2.8571 |
| mamba-130m-hf | 2 | association-ginkgo | 1.5136 |
| mamba-130m-hf | 3 | association-juniper | 1.5031 |
| mamba-130m-hf | 4 | association-hazel | 1.4783 |

## Largest observed retain damage

Rows are the five largest positive paired NLL changes from full-trained, averaged over seeds. Negative values indicate improvement rather than damage.

| Model | Method | Example | Target | ΔNLL |
| --- | --- | --- | --- | ---: |
| pythia-160m | continued_retain | test-t2-retain-dogwood | violet | 0.4905 |
| mamba-130m-hf | counterfactual | test-t1-retain-elm | quartz | 0.4890 |
| pythia-160m | continued_retain | test-t1-retain-elm | quartz | 0.3092 |
| pythia-160m | pcgu | test-t2-retain-cedar | marble | 0.2519 |
| pythia-160m | pcgu | test-t2-retain-elm | quartz | 0.2465 |

## Seed instability

The table reports the three largest seed SDs for each primary endpoint. With only three seeds, SD is descriptive and tail behavior is not well characterized.

| Endpoint | Model | Method | Seed SD |
| --- | --- | --- | ---: |
| forget_oracle_kl | pythia-160m | continued_retain | 0.6819 |
| forget_oracle_kl | pythia-160m | gradient_ascent | 0.3524 |
| forget_oracle_kl | pythia-160m | npo | 0.3450 |
| utility_nll | pythia-160m | continued_retain | 0.2725 |
| utility_nll | mamba-130m-hf | counterfactual | 0.1050 |
| utility_nll | mamba-130m-hf | continued_retain | 0.0926 |

## Failure modes and interpretation

- Counterfactual fine-tuning produced the largest Pythia movement toward the oracle but damaged held-out utility; this is the central utility-collapse warning.
- Several methods changed retain NLL unevenly across associations, so a favorable mean can conceal localized damage; all paired rows are retained in the CSV.
- Mamba effects were smaller under this frozen protocol. This is observational and does not identify architecture as the cause.
- Exact retraining's maximum forget-set oracle KL was 0, as expected when the reference is compared with itself.

Full group, example, and seed results are in `reports/tables/error_*.csv`. The synthetic task has only four independent forget associations and cannot establish behavior on natural memorization or legal deletion requests.
