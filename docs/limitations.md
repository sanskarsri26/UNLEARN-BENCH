# Limitations

- **Narrow controlled task.** Track A uses synthetic entity–code associations. It isolates a known
  forget request but does not reproduce natural memorization, long-form generation, copyrighted
  content, personal data, or adversarial extraction.
- **Low independent sample count.** The confirmatory test has only four forget association groups,
  even though each has three paraphrases. Exact sign-flip tests therefore have only 16 assignments;
  no Holm-adjusted comparison reached 0.05. Bootstrap intervals do not remove this limitation.
- **Three training seeds.** Means and SDs expose some instability but cannot characterize tails or
  hardware-level nondeterminism. Untouched outputs are identical across seeds by construction.
- **Behavioral oracle only.** Exact retain-only retraining is the clearest counterfactual behavior
  available here, but proximity to it does not prove parameter-level erasure, privacy protection,
  compliance with a legal deletion standard, or impossibility of information recovery.
- **Small, unmatched models.** Pythia-160M and Mamba-130M-HF differ in training data, tokenizer,
  parameterization, optimization history, and implementation. Cross-model results are observational
  and cannot be attributed causally to transformer versus state-space architecture.
- **Short optimization horizon.** The protocol is deliberately compute-bounded: 80 full/oracle
  examples and 48 intervention examples per approximate cell. Findings may change at larger data,
  model, and compute scales. Wall time and FLOPs are not matched across methods.
- **Single precision/device regime.** Real-model claims use FP32 A100 execution. BF16 training was
  not scientifically validated. No MacBook Air M3 was present, so MPS compatibility, unified-memory
  peaks, fallback performance, and passive-cooling slowdown remain unmeasured.
- **Mamba kernel fallback.** Optional fused selective-scan/causal-convolution kernels were absent;
  the pinned Transformers model used its logged sequential eager implementation. This stayed on
  CUDA and was not an architecture substitution, but performance may differ from fused execution.
- **Externally retained checkpoints.** All 54 Track A checkpoints were saved and SHA-256 checked,
  but their 31.48 GB total is outside normal Git tracking. A clean checkout can validate manifests
  and reproduce reported metrics from committed predictions, while strict checkpoint reanalysis
  requires access to those exact external files.
- **Track B is not unlearning evidence.** StereoSet membership in pretraining is unknown. LMS, SS,
  and ICAT measure external bias behavior only; SS is neutral near 50, not monotonically “lower is
  better.” Tokenization and conditional span scoring affect results, and category shifts were
  heterogeneous.
- **Benchmark scope.** CrowS-Pairs, BBQ, membership inference, privacy attacks, downstream LM suites,
  and larger model tiers were deferred rather than added without frozen protocols and resource
  audits. The absence of these evaluations constrains external validity.
- **Post-confirmatory ablations deferred.** PCGU selected-fraction sensitivity is unresolved. Tuning
  it on the already consumed four-group test would create post hoc overfitting; a future replication
  needs a new holdout or a development-only objective.
- **Legacy evidence excluded.** Historical scripts and a PDF are preserved for provenance, but their
  reported scores were not reproducible under the reboot and are not supporting evidence.
