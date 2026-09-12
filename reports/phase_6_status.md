# Phase 6 verification status

1. **Phase:** 6 — confirmatory Track A execution.
2. **Status:** COMPLETE. All 54 frozen cells have validated `COMPLETED` manifests; zero cells failed
   or remain unexplained.
3. **Files changed:** 54 raw manifest/prediction/metric triplets, two scheduler logs, and this status
   report. The 31.48 GB of model checkpoints remain in ignored scratch artifact storage.
4. **Run commit:** `2e9f3b83c23fd96acf8fb4407b3c20cfd09cf3d7`; preregistration commit
   `005db27a9f0b427bfa3fcc1cbcc3cc19da0c67a6` appears in every run manifest.
5. **Tests:** all unit tests plus config, result-schema, metric-recomputation, prediction-coverage,
   checkpoint-size, and checkpoint-SHA validation.
6. **Exact test totals:** 39 passed, 0 failed before execution; 54/54 checkpoints and 54/54 metric
   files passed the precommitted confirmatory analyzer after execution.
7. **Device used:** NVIDIA A100-SXM4-80GB MIG 1g.20gb for Pythia and 2g.20gb for Mamba, each exposing
   20,937,965,568 bytes to PyTorch. Scheduler jobs: `63044960` and `63044961`.
8. **Dtype:** FP32 for all 54 cells.
9. **Compute time:** 125 seconds Pythia + 288 seconds Mamba = 413 A100-seconds (0.115 A100
   GPU-hours). Attributed optimization time across manifests is 115.33 seconds.
10. **Peak memory:** 9.74 GiB Pythia; 15.84 GiB Mamba; both below the 20 GiB allocation.
11. **Artifacts generated:** 54 manifests, 54 raw 40-example prediction files, 54 metric files, 54
    saved/hash-validated checkpoints totaling 31,481,561,142 bytes, and two scheduler logs.
12. **Scientific findings:** none assigned during execution. The preregistered Phase 7 analysis is
    the source for substantive findings; execution established finite, complete outputs.
13. **Deviations:** the first two zero-runtime pending jobs (`63044644`, `63044645`) were cancelled
    and replaced on the authorized `lightwork` A100 MIG pool to avoid a two-hour queue delay. The
    experiment configs, hardware class, dtype, and resource caps were unchanged. Observed total GPU
    time was slightly above the sub-0.10-hour point estimate but far below the 1.0-hour ceiling.
14. **Remaining risks:** Mamba peak allocation exceeded its 10.57 GiB calibration value but stayed
    safely within the request; checkpoints are intentionally too large for Git and require external
    artifact retention for full weight-level reproduction.
15. **Next phase:** freeze raw metadata in Git, publish the preregistered primary analysis, then run
    clearly labeled secondary/error analyses.
