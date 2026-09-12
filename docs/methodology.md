# Methodology

## Predeclared questions

The primary question is: under matched compute budgets, which approximate parameter-editing or
unlearning methods most effectively remove known learned associations while preserving retained
knowledge and general language-model utility?

The secondary, exploratory question is whether measured forgetting–utility–compute trade-offs
differ between transformer and state-space model families. Any cross-family result is
observational: the planned models are not matched closely enough to identify architecture as a
cause.

## Two tracks and terminology

Track A is controlled machine unlearning. A base model is trained on versioned retain and forget
examples, then an intervention is asked to remove the forget associations. A separately initialized
copy trained from the same base checkpoint on retain data only is the exact-retraining oracle.

Track B is external bias generalization. PCGU or another intervention applied to inherited social
bias is called **unlearning-inspired debiasing**, not proof of machine unlearning. StereoSet results
measure changed behavior on held-out natural examples; they do not establish that those examples
were present in or removed from pretraining.

## Primary endpoints

These endpoints were frozen before the main run:

- Primary forgetting: mean categorical KL divergence from the exact-retrain oracle on held-out
  forget examples (`forget_oracle_kl`, lower is better).
- Primary utility: held-out utility negative log likelihood (`utility_nll`, lower is better).

Forget-set NLL, target probability, conditional log-probability margin, retain NLL/accuracy,
utility perplexity/accuracy, runtime, peak VRAM, checkpoint size, and parameter count are secondary.
A method is not successful when it reduces a forget score by degrading utility broadly.

For causal LMs, completion loss is computed only on continuation tokens. Oracle KL is computed over
the complete next-token vocabulary at each teacher-forced completion position and averaged first
within an example and then across the forget partition. Multi-token target probability is reported
as the geometric mean of its conditional token probabilities. The tiny fixture uses the equivalent
single-position categorical definition.

## Controlled data

`controlled-associations-v1` contains artificial entity–code associations and remains the
development/exploratory fixture. Its test split was inspected during smoke and calibration and is
not eligible for confirmatory claims. `controlled-associations-v2` preserves that history while
adding a new confirmatory holdout with three new association prompt forms and ten utility examples.
The v2 test hash is frozen before preregistration and may not be scored during development.

Association group IDs intentionally recur across surface forms to measure paraphrase generalization;
exact IDs, prompt/completion pairs, and content hashes may not recur across splits. Retain, forget,
and utility partitions are declared in every JSONL record. Counterfactual replacement targets are
versioned rather than generated during training. Confirmatory uncertainty resamples association
groups rather than incorrectly treating paraphrases of the same association as independent.

## Baselines and methods

The matrix contains untouched base, retain+forget trained, exact retrain, continued retain training,
a seeded random-update sham, counterfactual training, gradient ascent with retain regularization,
NPO, and PCGU. NPO is the stronger published approximate-unlearning baseline (Zhang et al.,
[“Negative Preference Optimization”](https://arxiv.org/abs/2404.05868)). Hyperparameters live
under `configs/methods/`; method count will not be expanded without a methodological reason.

PCGU follows Yu et al. (2023) for its defining operations: gradients for a contrastive pair,
input-aggregated parameter-vector partitions, ascending cosine-similarity ranking, a fixed selected
fraction, a boolean parameter mask, and updates only on selected vectors. The original method was
defined for social-bias pairs in masked transformer LMs. Track A substitutes learned targets and
versioned replacements in a causal objective; this is a documented adaptation, not an exact
reproduction of the paper's experiment. See `docs/pcgu_verification.md`.

## Statistical protocol

Main trainable experiments use seeds 11, 29, and 47. Reports include per-seed values, mean, standard
deviation, and confidence intervals where meaningful. The 10,000-draw paired hierarchical bootstrap
resamples seeds and then independent association groups for forget/retain endpoints; utility
examples are their own units. Exact sign-flip tests operate on group-averaged paired effects, and
Holm correction is applied within each model × endpoint family. Four forget groups permit only 16
sign assignments, so inferential resolution is intrinsically low. Smoke and post-confirmatory error
analyses remain exploratory.

The strict joint criterion requires a method-minus-full-trained forgetting interval below zero and
a utility interval with upper bound at or below zero. This is an estimation rule, not a replacement
for the multiplicity-adjusted tests. No best seed is selected.

## Compute matching and oracle construction

The frozen real-model matrix uses FP32, seeds 11/29/47, and revision-pinned checkpoints. Full and
exact training each use 20 steps × batch size four, or 80 examples. Approximate trainable methods use
predeclared method-specific steps/batches totaling 48 examples per cell; untouched and sham are
negative controls rather than optimization-matched algorithms. “Matched” refers to optimization
examples processed, not wall time, parameter count, or FLOPs.

The exact oracle starts from the same pinned pretrained checkpoint and seed-specific initialization
path as the corresponding full-trained condition, but is optimized only on retain examples. Each
method is evaluated against that seed's oracle on the identical held-out records. Oracle KL uses the
full next-token distribution and is zero for oracle self-comparison by construction.

## StereoSet

Candidate continuations are scored conditionally (continuation tokens only). Report LMS, SS, and
ICAT overall and by category. SS has a neutral target of 50; lower is not inherently better. ICAT is
`LMS * min(SS, 100-SS) / 50`. Ties receive half credit, an explicit deterministic convention.

Track B uses all 2,106 intrasentence contexts in the official StereoSet dev artifact, SHA-256
`73a0f31b711688112602e4c3ac6ab1e1a7cadcdd67df6c6fd55501c889676c90`, sourced from upstream commit
`ead7d086a64a192a1eca88e0dd2fd163de375218`. Candidate scores are the mean conditional log
probability across inserted continuation tokens. A context beginning with the blank is seeded with
the pinned tokenizer BOS token (EOS only if BOS is unavailable); this rule was documented after a
zero-cell technical failure and before the successful retry. Track B is descriptive and exploratory.
