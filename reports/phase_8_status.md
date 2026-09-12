# Phase 8 verification status

1. **Phase:** 8 — Track B external generalization.
2. **Status:** COMPLETE for StereoSet. CrowS-Pairs and BBQ were not added without a separately
   frozen protocol, dataset audit, and compute calibration.
3. **Files changed:** 54 manifest/metric/prediction triplets, two retry logs, execution metadata,
   aggregation code/test, report, four CSV tables, one figure, and this status report.
4. **Commits:** raw-artifact and analysis commits pending; scorer code was fixed at
   `b63deb4f3be9ebae1e9775287aa0917b8b970965`; analysis was frozen before results at
   `8f9bc8a6547b2a879e7a1b033aba94d3863cfbb3`.
5. **Tests:** 54 prediction and metric hashes; metric recomputation from JSONL; 2,106 unique IDs per
   cell; exact 54-cell matrix check; paired-neutrality unit test; visual figure inspection.
6. **Exact test totals:** 1 focused Track B analysis test passed, 0 failed; repository-wide total is
   reported after final hardening.
7. **Device used:** NVIDIA A100-SXM4-80GB, MIG 1g.20gb and 2g.20gb partitions.
8. **Dtype:** FP32.
9. **Compute time:** retry jobs used 995 scheduler GPU-seconds (0.276 A100-hours); scorer runtime
   summed to 859.03 seconds. Calibration and zero-cell failed attempts are reported separately.
10. **Peak memory:** 1,611,462,656 bytes (1.50 GiB).
11. **Artifacts generated:** `results/track_b/`, `reports/TRACK_B_STEREOSET.md`, summary/category/
    paired-delta CSVs, `reports/figures/track_b_stereoset.png`, and execution metadata.
12. **Scientific findings:** Pythia counterfactual moved mean SS 1.00 points toward 50 while losing
    0.97 LMS points; its ICAT rose 0.67. Pythia PCGU moved only 0.13 toward neutrality. Mamba
    counterfactual moved SS 0.62 away from neutrality and lost 0.82 LMS/1.64 ICAT. NPO's mean Mamba
    changes were small (+0.01 LMS, 0.08 toward neutrality, +0.14 ICAT). These are descriptive,
    exploratory changes, not evidence that StereoSet examples were forgotten.
13. **Deviations:** the first full attempt exposed 19 official items whose context begins with the
    blank; both jobs failed before writing any cell. The documented retry prepended the pinned BOS
    token for empty causal contexts. Because repository documentation/analysis commits occurred
    while the asynchronous jobs ran, per-cell `git_commit` records vary; the scorer and evaluation
    implementation did not change after `b63deb4`, and every artifact records its own hashes.
14. **Remaining risks:** three seeds are descriptive only; candidate span likelihood is sensitive to
    tokenization; StereoSet measures benchmark behavior rather than training-data removal; natural
    bias categories are not comparable to the synthetic Track A associations.
15. **Next phase:** no post hoc tuning on the consumed Track A test; proceed to error analysis,
    clean-checkout reproduction, documentation, and adversarial audit.
