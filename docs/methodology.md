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

These endpoints are frozen before any main run:

- Primary forgetting: mean categorical KL divergence from the exact-retrain oracle on held-out
  forget examples (`forget_oracle_kl`, lower is better).
- Primary utility: held-out utility negative log likelihood (`utility_nll`, lower is better).

Forget-set NLL, target probability, conditional log-probability margin, retain NLL/accuracy,
utility perplexity/accuracy, runtime, peak VRAM, checkpoint size, and parameter count are secondary.
A method is not successful when it reduces a forget score by degrading utility broadly.

## Controlled data

`controlled-associations-v1` contains artificial entity–code associations. Train, validation, and
test use different surface forms. Association group IDs intentionally recur to measure paraphrase
generalization; exact IDs, prompt/completion pairs, and content hashes may not recur. Retain, forget,
and utility partitions are declared in every JSONL record. Counterfactual replacement targets are
versioned rather than generated during training.

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
deviation, and confidence intervals where meaningful. Example-level method comparisons use paired
bootstrap intervals. Holm correction is available when a justified family of multiple comparisons
is tested; significance tests will not be added mechanically. Smoke results are exploratory.

## StereoSet

Candidate continuations are scored conditionally (continuation tokens only). Report LMS, SS, and
ICAT overall and by category. SS has a neutral target of 50; lower is not inherently better. ICAT is
`LMS * min(SS, 100-SS) / 50`. Ties receive half credit, an explicit deterministic convention.
