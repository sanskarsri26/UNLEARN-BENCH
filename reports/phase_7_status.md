# Phase 7 verification status

1. **Phase:** 7 — confirmatory Track A analysis.
2. **Status:** COMPLETE. The precommitted analyzer regenerated the primary analysis from all raw
   artifacts without changing the frozen endpoints or matrix.
3. **Files changed:** confirmatory report, three CSV tables, three figures, completion manifest, and
   this status report; the analysis script was precommitted as `2e9f3b8` before results existed.
4. **Analysis commit:** pending this report commit; preregistration commit
   `005db27a9f0b427bfa3fcc1cbcc3cc19da0c67a6`.
5. **Tests:** metric recomputation, manifest/prediction coverage, 54 checkpoint SHA checks,
   deterministic statistics tests, unit suite, config validation, and visual figure inspection.
6. **Exact test totals:** 40 passed, 0 failed after the checkpoint-portability addendum; 54/54
   completed cells and 10,000 bootstrap draws per prespecified comparison.
7. **Device used:** CPU for analysis; source model runs used A100 20 GiB MIG instances.
8. **Dtype:** analysis uses stored FP32 metrics; source model runs used FP32.
9. **Compute time:** 413 A100-seconds for source runs; analysis took about one minute per full
   checkpoint-hash pass on CPU/storage.
10. **Peak memory:** source maxima 9.74 GiB Pythia and 15.84 GiB Mamba; analysis peak not measured.
11. **Artifacts generated:** `reports/MAIN_CONFIRMATORY_RESULTS.md`, seed/summary/comparison CSVs,
    forgetting–utility, forgetting–compute, and seed-variation figures, and completion metadata.
12. **Scientific findings:** Pythia gradient ascent and NPO met the strict joint criterion.
    Counterfactual training had the largest Pythia mean forgetting improvement but damaged utility.
    No Mamba method met the joint criterion. PCGU's forget interval excluded zero on both models but
    its utility interval crossed zero. No Holm-adjusted p-value was below 0.05.
13. **Deviations:** none in analysis. Scheduler relocation and Mamba eager fallback are reported as
    predeclared/execution deviations rather than scientific effects.
14. **Remaining risks:** four forget groups yield low inferential power; three seeds do not support
    strong population claims; synthetic small-model findings may not generalize; cross-model
    differences are observational.
15. **Next phase:** conduct clearly labeled Track B external-generalization work only if a licensed,
    hashed official dataset can be obtained; otherwise document its principled deferral, then run
    focused error analysis and final reproducibility hardening.
