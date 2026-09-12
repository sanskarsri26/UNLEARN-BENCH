# Final adversarial audit

**Disposition:** no unresolved critical or high-severity issue. One high-severity reproducibility
gap was found and fixed without changing data or results: v2 validation now binds row files to the
sealed manifest's hashes, counts, partition totals, dataset identity, and holdout status. The fix is
commit `0c87657` and includes a manifest-tampering regression test. CI parity also exposed script
import, lint, and late-bound failure-closure regressions; commit `85069bb` fixes all of them.

## ML and machine-unlearning review

- The training intervention begins from the retain-plus-forget checkpoint. The exact oracle begins
  from the same revision-pinned base and sees retain/utility training records but no forget record.
- The primary forgetting endpoint is full-vocabulary teacher-forced KL to the per-seed oracle, not
  forget loss alone. Held-out utility NLL is always reported beside it.
- Exact-retrain self-KL is zero in every stored forget prediction. This is expected by metric
  construction and is not presented as an empirical deletion achievement.
- Continued-retain, sham, untouched, and full-trained controls remain in the complete results even
  where they are unfavorable. No method or seed was removed after inspection.
- NPO implements the reference-relative `-(2/β) log σ(β(log p_ref − log p_model))` forget loss plus
  retain regularization. PCGU's contrastive gradients, last-dimension vector partition, ascending
  cosine ranking, fixed fraction, boolean mask, and replacement-gradient update are tested. The
  causal-LM adaptation from the paper is disclosed rather than described as an exact reproduction.
- Full and exact runs process 80 examples each; approximate trainable methods process 48. The report
  defines this as matched optimization-example budget and does not claim equal wall time or FLOPs.

## Data and leakage review

- v2 train/validation/test IDs and prompt-completion hashes are unique across splits. The test
  contains exactly 12 forget, 18 retain, and 10 utility records; forget paraphrases are grouped into
  four independent association units for inference.
- The old v1 test exposure is documented. v1 remains exploratory and none of its calibration values
  appears as a confirmatory endpoint.
- Frozen v2 hashes still match the preregistration. The hardened validator additionally rejects a
  forged file hash, altered partition total, wrong dataset identity, non-test holdout, bad
  replacement field, or duplicate content.
- Track B uses the byte-verified official StereoSet dev artifact and all 2,106 intrasentence items.
  The first empty-context failure wrote zero cells; BOS handling was documented and tested before a
  complete retry.

## Metrics and statistics review

- All 54 Track A metric files were recomputed from 40 per-example predictions before publication;
  all 54 local checkpoint hashes and sizes were verified during strict analysis.
- The precommitted Track A analyzer uses 10,000-draw paired hierarchical bootstrap intervals, exact
  group-level sign flips, and Holm correction within prespecified families. It reports individual
  seeds, SD, effects, intervals, raw and adjusted p-values, and Pareto frontiers.
- Only four forget groups exist. The report highlights the resulting coarse exact-test resolution
  and does not convert intervals excluding zero into multiplicity-adjusted significance claims.
- Track B hashes and recomputes 54 × 2,106 prediction rows. SS is interpreted by distance from 50;
  lower raw SS is never called universally better. Track B has no confirmatory hypothesis tests.
- Error-analysis examples are selected by deterministic full-table ranking, not narrative
  cherry-picking. The CSVs retain every eligible association/example comparison.

## Engineering and reproducibility review

- A fresh clone at commit `85069bb` installed the project plus missing CI tools into a temporary
  environment, regenerated v1 data without a tracked diff, passed Ruff, passed 45 pytest tests plus
  10 subtests, validated eight configs and 99 committed Track A manifests, recomputed all 54 Track B
  cells, and completed a new CPU/FP32 PCGU smoke cell. The clean checkout remained clean.
- Clean manifest validation accepts absent external checkpoints only when a saved status, SHA-256,
  and positive byte size are present. Strict confirmatory analysis still requires and rehashes the
  actual 31.48 GB checkpoint set.
- Explicit `cpu`, `mps`, `cuda`, and `auto` device behavior is tested. Unavailable explicit devices
  and unapproved MPS fallback fail closed. M3 execution is unmeasured and never inferred.
- Every confirmatory cell records frozen config/preregistration identity, model/tokenizer revision,
  split hashes, method/seed, hardware/software/device/dtype, work, runtime/memory, and checkpoint/
  prediction/metric paths. Source changes invalidate the old execution marker as intended.
- Track B per-cell commit fields vary because documentation and analysis commits occurred while
  asynchronous jobs ran. Scoring code itself remained fixed at `b63deb4`; execution metadata records
  that commit, scheduler jobs, hashes, and the deviation. This is a provenance blemish, not a metric
  ambiguity.

## Skeptical hiring-manager review

The repository now demonstrates actual parameter updates, controls, oracle-based evaluation, GPU
execution, negative results, uncertainty, failure preservation, and clean-checkout reproducibility.
It does not demonstrate production serving, large-scale distributed training, legal deletion
compliance, privacy guarantees, or causal architecture effects. Portfolio language in
`docs/portfolio_package.md` stays within those boundaries.

## Residual risks

Residual risks are scientific scope rather than hidden implementation failures: small synthetic
data, three seeds, unmatched 130–160M models, short optimization, one FP32 GPU regime, an unfused
Mamba path, externally retained weights, and no membership/privacy attack. They are enumerated in
`docs/limitations.md` and define the replication agenda.
