# Phases 10–19 verification status

## Phase 10 — error analysis

1. **Phase:** 10.
2. **Status:** COMPLETE, explicitly exploratory.
3. **Files changed:** `scripts/analyze_errors.py`, error report, three CSVs, one heatmap.
4. **Commit:** `af25ec5`.
5. **Tests:** full prediction coverage and deterministic regeneration; figure visually inspected.
6. **Exact test totals:** final repository suite 45 passed, 0 failed.
7. **Device used:** CPU analysis of stored predictions.
8. **Dtype:** stored FP32 metrics; aggregation uses Python/NumPy floating point.
9. **Compute time:** approximately 3.2 seconds per regeneration.
10. **Peak memory:** not measured; no model loaded.
11. **Artifacts generated:** `reports/ERROR_ANALYSIS.md`, `error_*.csv`, group heatmap.
12. **Scientific findings:** ivy was hardest under the prespecified pooled ranking; the largest
    localized retain damage and highest seed SDs are reported rather than averaged away.
13. **Deviations:** no preregistered claim changed; this is post-confirmatory analysis.
14. **Remaining risks:** rankings pool methods and three seeds on four synthetic associations.
15. **Next phase:** statistical finalization.

## Phase 11 — statistical finalization

1. **Phase:** 11.
2. **Status:** COMPLETE.
3. **Files changed:** precommitted analyzer, main comparisons/seed/summary tables and figures.
4. **Commits:** analyzer `2e9f3b8`; published analysis `66b59a9` and `74a1efb`.
5. **Tests:** deterministic bootstrap/sign-flip/Pareto tests and full raw metric recomputation.
6. **Exact test totals:** 45 final tests passed; 10,000 bootstrap draws per comparison.
7. **Device used:** CPU analysis; source runs used A100 20 GiB MIG.
8. **Dtype:** FP32 source metrics.
9. **Compute time:** 413 A100-seconds for Track A; analysis approximately one minute including
   31.48 GB of checkpoint hashing.
10. **Peak memory:** source maxima 9.74 GiB Pythia and 15.84 GiB Mamba.
11. **Artifacts generated:** confirmatory report, three main CSVs, three main figures.
12. **Scientific findings:** gradient ascent and NPO met Pythia's joint criterion; no Mamba method
    did; no Holm-adjusted p-value was below 0.05.
13. **Deviations:** scheduler partition relocation only; zero-runtime original jobs were cancelled.
14. **Remaining risks:** low exact-test resolution from four independent forget groups.
15. **Next phase:** clean-checkout reproduction.

## Phase 12 — reproducibility hardening

1. **Phase:** 12.
2. **Status:** COMPLETE for available Linux/CPU/A100 evidence; M3 reproduction is documented as
   unmeasured rather than simulated.
3. **Files changed:** single-cell CLI, Track B validate-only mode, CI, data validator/tests,
   reproducibility documentation.
4. **Commits:** `381e0cd`, `0c87657`, `85069bb`, and final documentation commit.
5. **Tests:** fresh clone data build, tests, config validation, Track A manifests, Track B
   recomputation, CPU PCGU smoke, clean tracked status.
6. **Exact test totals:** Ruff passed; 45 pytest tests and 10 subtests passed; 8 configs; 99
   committed Track A manifests; 54 Track B cells; 1 fresh CPU smoke cell.
7. **Device used:** CPU for clean-checkout verification; prior real runs used A100.
8. **Dtype:** CPU FP32 smoke; stored real results FP32.
9. **Compute time:** 29.6 seconds for the final clean-checkout/temporary-environment command.
10. **Peak memory:** not instrumented for checkout; smoke manifest records its own process metrics.
11. **Artifacts generated:** clean clone at `/tmp/unlearn-bench-ci.SOMjpG/repo`, temporary CI-tool
    environment, and validation logs under `/tmp`; these are evidence, not dependencies.
12. **Scientific findings:** no metric changed; reproducibility checks passed without checkpoints.
13. **Deviations:** strict Track A analysis cannot run without externally retained weights by design.
14. **Remaining risks:** M3/macOS and reinstalling the large PyTorch dependency without the host's
    existing packages remain external checks.
15. **Next phase:** publication figures and tables.

## Phase 13 — figures and tables

1. **Phase:** 13.
2. **Status:** COMPLETE.
3. **Files changed:** main, Track B, and error-analysis figures plus their source CSVs/scripts.
4. **Commits:** `66b59a9`, `74a1efb`, `af25ec5`, `afa00b8`.
5. **Tests:** artifact regeneration and visual inspection of every new primary figure.
6. **Exact test totals:** 5 final research figures (3 Track A, 1 Track B, 1 error analysis).
7. **Device used:** CPU for plotting.
8. **Dtype:** stored FP32 metrics.
9. **Compute time:** seconds per plotting pass; source compute reported in Phases 7/8.
10. **Peak memory:** not measured; no model loaded.
11. **Artifacts generated:** forgetting–utility, forgetting–compute, seed variation, StereoSet, and
    forget-group heatmap figures.
12. **Scientific findings:** shared axes/error bars reveal trade-offs without a composite score.
13. **Deviations:** utility–compute remains available in tables/Pareto summaries rather than a sixth
    standalone plot to avoid redundant presentation.
14. **Remaining risks:** small samples make visual uncertainty descriptive.
15. **Next phase:** README finalization.

## Phase 14 — README finalization

1. **Phase:** 14.
2. **Status:** COMPLETE.
3. **Files changed:** `README.md`.
4. **Commit:** `a508559` plus final command/status correction.
5. **Tests:** local links/commands reviewed; clean-checkout commands executed.
6. **Exact test totals:** 45 tests support the documented state.
7. **Device used:** none for prose; command evidence includes CPU/A100.
8. **Dtype:** not applicable; FP32 is documented.
9. **Compute time:** not applicable.
10. **Peak memory:** not applicable.
11. **Artifacts generated:** final research-first README.
12. **Scientific findings:** only frozen Track A results lead; Track B is labeled exploratory.
13. **Deviations:** none.
14. **Remaining risks:** external checkpoint availability must be maintained separately.
15. **Next phase:** methodology documentation.

## Phase 15 — methodology documentation

1. **Phase:** 15.
2. **Status:** COMPLETE.
3. **Files changed:** methodology, reproducibility, limitations, device calibration disposition.
4. **Commit:** `a508559` plus final audit documentation commit.
5. **Tests:** claims cross-checked against configs, manifests, reports, and clean checkout.
6. **Exact test totals:** 45 tests; 54/54 cells in each Track A and Track B matrix.
7. **Device used:** none for prose.
8. **Dtype:** FP32 protocol documented.
9. **Compute time:** not applicable.
10. **Peak memory:** measured source peaks documented, prose generation unmeasured.
11. **Artifacts generated:** complete methods/statistics/compute/reproduction/limitations narrative.
12. **Scientific findings:** behavioral-oracle scope and inferential limits are explicit.
13. **Deviations:** M3 setup is an explicit non-result because hardware was unavailable.
14. **Remaining risks:** external replication may expose environment-specific behavior.
15. **Next phase:** adversarial audit.

## Phase 16 — final adversarial audit

1. **Phase:** 16.
2. **Status:** COMPLETE; no unresolved critical/high issue.
3. **Files changed:** sealed v2 validator/test, CI/import/lint fixes, final audit report.
4. **Commits:** validator fix `0c87657`, CI fix `85069bb`, audit report in finalization commit.
5. **Tests:** leakage, oracle/method/metric/claims/provenance/clean-checkout, Ruff, and pytest audit.
6. **Exact test totals:** Ruff passed; 45 pytest tests and 10 subtests passed.
7. **Device used:** CPU review and tests.
8. **Dtype:** FP32 artifacts reviewed; no new model operation.
9. **Compute time:** approximately 14 seconds for the data-fix validation, 13 seconds for local CI
   parity, and 29.6 seconds for the final clean-checkout pass.
10. **Peak memory:** not measured.
11. **Artifacts generated:** final audit report and hardened validator.
12. **Scientific findings:** no result invalidation; manifest binding was the scientific-integrity
    issue, while script imports/lint/closure binding were reproducibility regressions.
13. **Deviations:** Track B commit-field variability is documented with fixed scoring-code commit.
14. **Remaining risks:** all residual risks are scoped limitations in `docs/limitations.md`.
15. **Next phase:** portfolio packaging.

## Phase 17 — portfolio packaging

1. **Phase:** 17.
2. **Status:** COMPLETE.
3. **Files changed:** `docs/portfolio_package.md`.
4. **Commit:** `88353e6`.
5. **Tests:** every quantitative bullet cross-checked against committed reports/manifests.
6. **Exact test totals:** 45 repository tests; prose has no standalone test.
7. **Device used:** none.
8. **Dtype:** not applicable.
9. **Compute time:** not applicable.
10. **Peak memory:** not applicable.
11. **Artifacts generated:** GitHub, LinkedIn, two resume sets, interview answers.
12. **Scientific findings:** negative/null findings are retained in portfolio language.
13. **Deviations:** none; GitHub URL was taken from configured origin.
14. **Remaining risks:** resume length must be tailored to a specific application.
15. **Next phase:** project ranking.

## Phase 18 — project ranking

1. **Phase:** 18.
2. **Status:** COMPLETE using the provided role descriptions, not unaudited source claims.
3. **Files changed:** ranking section in `docs/portfolio_package.md`.
4. **Commit:** `88353e6`.
5. **Tests:** evidence/rationale review only.
6. **Exact test totals:** six target-role rankings, each with three projects.
7. **Device used:** none.
8. **Dtype:** not applicable.
9. **Compute time:** not applicable.
10. **Peak memory:** not applicable.
11. **Artifacts generated:** role-specific top-three table.
12. **Scientific findings:** UNLEARN-BENCH leads research/model-training roles, not platform/product
    roles where Incident Reporter or LumiNote has more directly relevant stated evidence.
13. **Deviations:** other repositories were not present for source-level audit.
14. **Remaining risks:** rankings should be revisited when job descriptions or other projects change.
15. **Next phase:** research freeze.

## Phase 19 — freeze

1. **Phase:** 19.
2. **Status:** COMPLETE — RESEARCH COMPLETE / MAINTENANCE.
3. **Files changed:** `RESEARCH_STATUS.md`, README status, final audit/status records.
4. **Commit:** finalization commit containing this report.
5. **Tests:** final local and clean-checkout verification.
6. **Exact test totals:** Ruff passed; 45 pytest tests plus 10 subtests; 8 configs; 99 committed
   Track A manifests; 54 Track B cells; clean checkout unchanged after verification.
7. **Device used:** CPU for freeze verification; scientific device history remains A100 FP32.
8. **Dtype:** FP32 for all real-model evidence.
9. **Compute time:** no new GPU compute; verification times reported under Phase 12.
10. **Peak memory:** no new model workload.
11. **Artifacts generated:** research status and final verification record.
12. **Scientific findings:** final claims are frozen to the confirmatory and exploratory reports.
13. **Deviations:** focused ablation was intentionally deferred to avoid post hoc test tuning.
14. **Remaining risks:** future work requires a new independent protocol; see limitations.
15. **Next phase:** maintenance, bug fixes, or genuinely independent external replication only.
