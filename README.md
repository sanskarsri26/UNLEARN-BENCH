# UNLEARN-BENCH

A reproducible benchmark for measuring whether approximate parameter interventions remove known
learned associations while preserving retained knowledge and language-model utility.

## Research Question

Under matched compute budgets, which approximate parameter-editing or unlearning methods most
effectively remove known learned associations while preserving retained knowledge and general
language-model utility?

A secondary, exploratory question compares forgetting–utility–compute trade-offs across transformer
and state-space families. Those model families are not sufficiently matched for causal architecture
claims.

## Experimental Design

The research reboot is separate from the preserved historical implementation. Track A performs
controlled machine unlearning with a known forget set. Track B, when enabled, measures external bias
generalization and uses the term **unlearning-inspired debiasing** for inherited pretrained bias.

Main experiments are configuration-gated until GPU smoke calibration is reviewed. Smoke results are
exploratory; no historical headline result is reused as evidence.

## Controlled Unlearning Setup

The primary pipeline is:

```text
base model
  -> train on retain + forget examples
  -> apply an intervention
  -> compare with an exact-retrain model trained from the same base on retain only
```

Versioned JSONL manifests declare every train/validation/test example, retain/forget/utility
partition, source, category, group, ID, and SHA-256 content hash. Different held-out phrasings test
association generalization without exact-example leakage.

## Methods

The frozen matrix includes the untouched base, retain+forget trained model, exact retraining,
continued retain fine-tuning, a seeded random-update sham, counterfactual fine-tuning, gradient
ascent, NPO, and PCGU. PCGU implements contrastive gradients, vector partitioning, cosine ranking,
selection masks, and masked updates. Its causal-LM adaptation and differences from the original
masked-LM method are documented in [PCGU verification](docs/pcgu_verification.md).

## Models

The local `tiny-association-fixture` makes the complete pipeline and CI CPU-runnable. It is not a
scientific model. The causal-LM adapter supports revision-pinned Pythia-160M, Mamba-130M,
Pythia-410M, and Mamba-370M configurations. Main configurations remain gated until real-model
device calibration and preregistration are reviewed.

## Evaluation

The predeclared primary forgetting endpoint is forget-set categorical KL divergence to the
exact-retrain oracle. The primary utility endpoint is held-out utility negative log likelihood.
Secondary outputs include forget/retain loss, target probability and log-probability margin,
accuracy, perplexity, runtime, VRAM, checkpoint size, and trainable parameter count.

StereoSet support reports LMS, SS, and ICAT, with SS interpreted as neutral near 50 rather than
“lower is better.” All causal-LM candidates must be scored conditionally on continuation tokens.

## Results

The nine-method CPU fixture smoke pipeline passes end to end. It generated raw predictions,
machine-readable manifests, regenerated metrics, an aggregate table, and a forgetting–utility plot.
These are engineering validation artifacts, classified **EXPLORATORY**, and support no substantive
model-unlearning claim. No confirmatory main result exists yet.

Canonical artifacts: [aggregate table](reports/tables/summary.md),
[trade-off figure](reports/figures/forgetting_utility_pareto.png), and
[compute calibration](reports/compute_budget.md).

## Reproducibility

```bash
pip install -e '.[dev,figures]'
python scripts/prepare_data.py
python scripts/run_experiment.py --config configs/experiments/smoke.yaml
python scripts/evaluate.py --run RUN_ID
python scripts/aggregate_results.py
python scripts/make_figures.py
```

Each run retains per-example outputs, summary metrics, the final checkpoint, and a JSON manifest with
the git commit, exact configuration, revisions, seeds, hashes, software, hardware, runtime, and
artifact paths. See [reproducibility](docs/reproducibility.md) and
[methodology](docs/methodology.md).

Execution supports explicit `auto`, `cuda`, `mps`, and `cpu` device selection with recorded dtype
and fallback state. See [device portability](docs/device_portability.md); real-model M3 and A100
compatibility remains unverified until hardware calibration is completed.

## Repository Structure

```text
configs/                 versioned dataset, model, method, and experiment settings
data/controlled/v1/      preserved exploratory Track A manifests
data/controlled/v2/      sealed confirmatory Track A holdout and development manifests
docs/                    methodology, audit, reproducibility, limitations, PCGU verification
reports/                 compute budget plus generated figures and tables
results/                 run manifests, raw predictions, checkpoints, and metrics
scripts/                 prepare, run, evaluate, aggregate, figure, and validation entry points
src/unlearn_bench/       data, models, methods, evaluation, statistics, and utilities
tests/                   lightweight behavioral and reproducibility tests
Unlearning/              preserved HISTORICAL / UNVERIFIED implementation and artifacts
```

`Final_Report.pdf` and everything under `Unlearning/` are retained for provenance. The historical
46.82, 46.73, and 46.49 scores and their associated claims are **HISTORICAL / UNVERIFIED** and must
not be cited as reboot results. See the [legacy audit](docs/legacy_audit.md).

## Limitations

The completed smoke uses a synthetic fixture, real-model training is implemented but not yet
calibrated, Track B has no versioned external dataset artifact, and cross-family comparisons will be
observational. Exact scope and deferred evaluations are listed in [limitations](docs/limitations.md).

## Future Work

Calibrate Pythia-160M and Mamba-130M on the target devices; review and freeze the resulting compute
budget; run the three-seed main matrix; then add focused ablations, paired uncertainty,
category-level error analysis, and licensed Track B benchmarks.
