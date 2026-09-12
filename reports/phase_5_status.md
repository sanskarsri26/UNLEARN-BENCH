# Phase 5 verification status

1. **Phase:** 5 — frozen confirmatory research design.
2. **Status:** COMPLETE. No model had scored the `controlled/v2` test split when the protocol was
   committed.
3. **Files changed:** main experiment configs, preregistration, frozen compute budget,
   authorization tooling, manifest/failure schemas, runner gate, validators, and tests.
4. **Preregistration commit:** `005db27a9f0b427bfa3fcc1cbcc3cc19da0c67a6`.
5. **Tests:** unit tests, configuration validation, artifact validation, compilation, diff checks,
   and negative gate/override tests.
6. **Exact test totals:** 36 passed, 0 failed in 3.343 seconds on the final pre-freeze run.
7. **Device used:** CPU for validation; no confirmatory model execution. A100 is frozen for Phase 6.
8. **Dtype:** not applicable to protocol validation; FP32 is frozen for Phase 6.
9. **Compute time:** 0 A100 GPU-hours for Phase 5; the frozen ceiling is 1.0 A100 GPU-hour.
10. **Peak memory:** not measured for documentation/tests; calibrated maxima are 7.32 GiB Pythia
    and 10.57 GiB Mamba.
11. **Artifacts generated:** `docs/preregistration_main.md`, `reports/compute_budget.md`, and a
    machine-readable `results/manifests/CALIBRATION_REVIEWED` authorization record.
12. **Scientific findings:** none; confirmatory results remained sealed throughout design freeze.
13. **Deviations:** v2 replaces the reused v1 holdout additively; M3 and the large model tier are
    excluded; both models use A100 FP32; Mamba uses sequential eager execution.
14. **Remaining risks:** only four independent forget association groups limit inferential power;
    the small synthetic task limits external validity; M3 portability remains physically untested.
15. **Next phase:** execute all 54 frozen Track A cells, preserve terminal failures, and validate raw
    artifacts before primary analysis.
