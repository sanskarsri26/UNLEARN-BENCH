# Preregistration: controlled unlearning main experiment

**Status:** frozen before any model evaluation on the `controlled/v2` test split.

This document defines the confirmatory Track A experiment. Its Git commit is the preregistration
identifier. A separate machine-readable authorization marker must name that commit before either
main configuration can execute.

## Research questions and scope

The primary question is: under a matched 48-example intervention budget, which approximate
unlearning methods most closely reproduce exact-retraining behavior on known learned associations
while preserving held-out general language-model utility?

The secondary question is whether the observed forgetting–utility–compute trade-offs differ between
the selected transformer and state-space checkpoints. This comparison is observational. The two
checkpoints differ in more than architecture, so it cannot support an architecture-causality claim.

Track A controlled machine unlearning is the sole confirmatory track. Any later external bias
benchmark is Track B external generalization and must not be described as forgetting.

## Frozen data

`controlled-associations-v1` remains development-only because its test split was inspected during
smoke and calibration. The confirmatory data are `controlled-associations-v2`. The v2 train and
validation splits preserve the controlled task; its newly generated test split has not been scored
by a model before this freeze.

| Split | Examples | SHA-256 |
| --- | ---: | --- |
| Train | 12 | `2315886ab3eb25b2dee72aa5ab997351b531c9825998163c233b94f1713db05c` |
| Validation | 12 | `855f5f9f06eeefd7d18427d20c47453f726974733b3623d15e91046be513d8c3` |
| Test | 40 | `3776547ae88199c9e4bb8ee7e51daafcacba9d9a1ea318741da560edb62c31b1` |

The test split contains 12 forget examples (four association groups, three paraphrases per group),
18 retain examples (six groups, three paraphrases per group), and 10 utility examples. Exact IDs,
prompt/completion pairs, and content hashes do not recur across splits. Association group IDs recur
intentionally so paraphrases are treated as clustered observations.

## Frozen models, device, and seeds

| Model | Family | Repository | Revision | Tokenizer revision |
| --- | --- | --- | --- | --- |
| Pythia-160M | transformer | `EleutherAI/pythia-160m` | `50f5173` | `50f5173` |
| Mamba-130M-HF | state space | `state-spaces/mamba-130m-hf` | `1e76775` | `1e76775` |

Both configurations use seeds 11, 29, and 47, CUDA on an NVIDIA A100, FP32 parameters and
optimization, deterministic algorithms with warnings enabled, completion-token-only loss, and
saved final checkpoints. Mamba uses the Transformers sequential eager implementation because the
optional fused kernels are unavailable in the calibrated environment. No architecture is silently
substituted.

Physical M3/MPS measurements were impossible in the available Linux/x86_64 environment. MPS is
therefore omitted from confirmatory execution rather than guessed; A100 is the predeclared device
for both models. The large 410M/370M tier is deferred because the complete calibrated small-tier
matrix is preferable to an incompletely calibrated large tier.

## Frozen methods and optimization

All nine methods remain in the matrix regardless of outcome:

1. untouched pretrained checkpoint;
2. retain + forget full-trained model;
3. exact retraining from the same pretrained checkpoint on retain + utility data only;
4. retain-only continued training from the full-trained model;
5. seeded random-update sham with global parameter L2 norm 0.05;
6. counterfactual fine-tuning on retain + utility and versioned replacement targets;
7. gradient ascent on forget loss with retain weight 1.0;
8. NPO against the full-trained reference with beta 0.1 and retain weight 1.0; and
9. PCGU with ascending gradient cosine-similarity ranking, replacement-target updates, and a fixed
   selected fraction of 0.1.

Full training and exact retraining each use 20 optimizer steps, batch size 4, and 80 examples
processed. They are training controls, not members of the matched approximate-intervention family.
The untouched and sham controls process zero task examples during intervention.

For each approximate method, matched compute means exactly 48 loss-contributing examples processed,
not equal wall time or FLOPs. Continued-retain and counterfactual training use 12 steps × batch size
4. Gradient ascent and NPO use four full-batch steps over four forget plus eight retain/utility
training examples. PCGU uses six steps over both the original and replacement versions of four
forget examples. Token counts are measured and reported but are not the matching variable.

Pythia learning rates are `1e-5` for full, exact, continued-retain, counterfactual, and PCGU, and
`1e-6` for gradient ascent and NPO. Mamba learning rates are `5e-5` and `5e-6`, respectively. These
values were fixed after exploratory v1 stability/cost calibration; no v2 test outcome informed them.

## Oracle and endpoints

For each seed and architecture, the exact-retrain oracle starts from the identical pretrained
checkpoint and is trained for 20 steps on the eight non-forget training examples. The full-trained
model starts from that checkpoint and is trained for 20 steps on all 12 training examples. Each
intervention starts from the resulting full-trained model, except the explicitly defined untouched
and exact controls.

The primary forgetting endpoint is `forget_oracle_kl`, lower being better. For every teacher-forced
completion token, it is

`KL(p_exact(. | prefix) || p_method(. | prefix))`.

The categorical KL is averaged over completion positions within an example, then over all 12 forget
examples. Exact retraining is zero by construction and is an oracle reference, not an inferential
competitor.

The primary utility endpoint is `utility_nll`, lower being better: mean negative conditional
log-probability over completion tokens within each of the 10 utility examples, then mean across
examples. A method is not considered successful merely because forgetting improves; utility damage
is reported jointly and no arbitrary composite success score is used.

Secondary endpoints are forget NLL, target probability, conditional log-probability margin, retain
NLL and accuracy, utility perplexity and accuracy, distance to exact behavior on retain and utility
examples, parameter distance from the full-trained model, runtime, examples/tokens processed,
throughput, peak memory, trainable parameters, checkpoint size, and checkpoint-write time.

## Confirmatory comparisons and statistics

Every seed value is reported. For each model and method, reports include the arithmetic mean and
sample standard deviation across the three seeds. The primary descriptive result is the two-axis
forgetting–utility trade-off relative to exact retraining and the full-trained model.

The frozen hypotheses are:

- **H-F (forgetting):** for each approximate method or intervention control, the mean paired
  forget-oracle KL difference from the full-trained model is zero; the two-sided alternative is a
  nonzero difference.
- **H-U (utility):** for each such method, the mean paired utility-NLL difference from the
  full-trained model is zero; the two-sided alternative is a nonzero difference.

No direction is selected after calibration. Evidence of the desired joint outcome requires a
negative forgetting difference and an upper utility-difference confidence bound at or below zero.
Otherwise the result is described as a trade-off, not utility-preserving unlearning. The sham is
retained as a negative control, not presumed effective.

For each architecture and primary endpoint separately, one prespecified family compares
continued-retain, sham, counterfactual, gradient ascent, NPO, and PCGU with the full-trained model.
The effect size is the raw paired mean difference (method minus full-trained; negative favors the
method for both primary endpoints). Two-sided 95% percentile confidence intervals use 10,000
hierarchical paired bootstrap draws. Draws resample seeds, then resample association group IDs
within seed for forget results; utility draws resample seeds, then the 10 utility examples within
seed. Retain secondary intervals similarly resample its six association groups. Holm correction is
applied within each six-comparison architecture × primary-endpoint family. Two-sided randomization
p-values average paired effects across seeds within each independent content unit, exhaustively
flip the signs of the four forget-group effects or 10 utility-example effects, and count permuted
absolute means at least as large as the observed absolute mean. These deliberately low-powered
tests supplement effect sizes and intervals; they do not replace them. Exact retraining is not
tested against itself, and cross-architecture differences receive descriptive intervals only.

Pareto frontiers minimize mean forget-oracle KL, mean utility NLL, and, in separate views,
optimization runtime. The two-dimensional forgetting–utility frontier is primary descriptive
analysis; compute frontiers and all post hoc error analyses are secondary or exploratory and labeled
as such. No seed, method, or endpoint will be selected based on favorability.

## Matrix, failure policy, and stopping rule

The fixed matrix is 2 models × 9 methods × 3 seeds = 54 cells. A cell is terminal only when it has a
validated `COMPLETED` manifest or an immutable failure record classified as `TECHNICAL_FAILURE`,
`OOM`, `UNSUPPORTED_DEVICE`, `NUMERICAL_FAILURE`, or `METHOD_FAILURE`. Non-finite parameters or
metrics are numerical failures. Failed cells are not silently overwritten or repeatedly rerun; a
technical correction requires a dated protocol addendum that preserves the original failure.

Execution stops when all 54 cells are terminal, or immediately for a scientific/data-integrity
problem, repeated non-finite behavior, an invalid oracle, or compute substantially beyond the
frozen budget. Missing cells remain visible. Completed cells have deterministic IDs and are not
duplicated on resume.

The calibrated estimate is under 0.10 A100 GPU-hours for computation and checkpoint writes; a
conservative authorization ceiling of 1.0 A100 GPU-hour allows scheduler and I/O variance while
remaining far below the 24 GPU-hour stop threshold. Saving all 54 checkpoints requires about 31.48
GB (29.32 GiB) using measured checkpoint sizes. Checkpoint files remain in ignored artifact storage;
their paths, byte sizes, write times, and SHA-256 hashes are recorded. Raw predictions, metrics,
manifests, failure records, and scheduler logs are preserved for the reproducible report.

## Analysis order and deviations

After the authorization marker is created, the fixed runner may score v2 test data. Analysis first
validates all terminal cells and checkpoint hashes, then computes the preregistered primary metrics,
seed summaries, intervals, corrected comparisons, and Pareto frontiers. Only afterward may error
analysis, Track B, or focused ablations be inspected.

Any implementation or infrastructure correction after this commit is documented in an addendum.
It may not change data, models, seeds, methods, budgets, or endpoints in response to results. The
known pre-execution deviations are the additive v2 holdout necessitated by v1 reuse, omission of
unavailable M3 measurements, use of A100 for both models, sequential eager Mamba execution, FP32
instead of unverified mixed precision, the small-tier choice, and deferral of Track B until after
the Track A primary analysis.
