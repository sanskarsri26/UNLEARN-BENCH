# Phase 9 verification status

1. **Phase:** 9 — focused ablations.
2. **Status:** COMPLETE BY PRINCIPLED DEFERRAL. No post-confirmatory ablation was run.
3. **Files changed:** this decision record only. The inherited
   `configs/experiments/ablation.yaml` remains an unexecuted exploratory design.
4. **Commit:** pending this report commit.
5. **Tests:** not applicable; no code or result artifact changed.
6. **Exact test totals:** not applicable for this decision record.
7. **Device used:** none.
8. **Dtype:** not applicable.
9. **Compute time:** zero.
10. **Peak memory:** not applicable.
11. **Artifacts generated:** this decision record; no metric or checkpoint artifacts.
12. **Scientific finding:** the main study justifies interest in PCGU's selected fraction because
    PCGU moved toward the oracle on both models while its utility interval crossed zero. It does not
    justify selecting a fraction on the already consumed confirmatory test set.
13. **Deviation:** the roadmap allowed focused ablations but did not require a specific one. The
    repository's old Pythia-410M/v1, 500-step ablation is not comparable with the frozen small-tier,
    short-budget study and was not launched.
14. **Remaining risks:** PCGU sensitivity to selected fraction is unknown. A future replication
    should freeze 0.01/0.05/0.10 before execution and use a new holdout or an explicitly development-
    only endpoint. It should not select a winner and relabel it confirmatory.
15. **Next phase:** deterministic post-confirmatory error analysis across every stored Track A
    prediction, including hardest forget groups, localized damage, and seed instability.

This deferral avoids a post hoc hyperparameter search whose selection target would be the same four
forget groups used for the headline result. With no unused controlled holdout, running more cells
would add compute and apparent precision without adding independent confirmatory evidence.
