# Phase 3–4 verification status

## Phase 3: device portability

1. **Phase:** 3 — device portability.
2. **Status:** PARTIAL. Unified selection, precision policy, manifest provenance, CPU execution, and
   CUDA execution are verified. Physical MPS verification is outstanding.
3. **Files changed:** device/reproducibility utilities, runner, HF loader, schema, configs, tests,
   Slurm launchers, and device documentation; see commits `67df310`, `c03214b`, and `81beb6d`.
4. **Commit SHA:** latest producing calibration commit `81beb6d`.
5. **Tests:** standard-library unit suite, config validation, result-schema validation, compilation,
   and end-to-end nine-method CPU/CUDA smoke runs.
6. **Exact test totals:** 25 passed, 0 failed in 3.034 seconds on the final pre-report run. `pytest`
   and `ruff` executables are absent from the active environment; `unittest`, `compileall`,
   `git diff --check`, and the repository validators were used.
7. **Device used:** Linux CPU; NVIDIA A100-SXM4-80GB; NVIDIA A100-SXM4-80GB MIG 1g.20gb.
8. **Dtype:** CPU FP32; CUDA BF16 and FP32 fixture smoke; real-model FP32.
9. **Compute time:** CPU schema-v2 smoke 9.42 seconds; CUDA fixture smoke under the 15-minute cap;
   real-model A100 jobs 31 seconds (Pythia) and 60 seconds (Mamba).
10. **Peak memory:** CPU smoke 572,716 KiB RSS; real Pythia 7.32 GiB PyTorch device allocation;
    real Mamba 10.24 GiB PyTorch device allocation.
11. **Artifacts generated:** schema-v2 raw manifests/predictions/metrics, calibration logs,
    `reports/device_calibration.md`, and the updated compute budget.
12. **Scientific findings:** none; device smoke and calibration are exploratory engineering evidence.
13. **Deviations:** MPS could not be tested because the active host is Linux/x86_64, not an M3.
    Real-model BF16 optimization was deferred because FP32 master-weight behavior is unverified.
14. **Remaining risks:** physical MPS operation coverage, fallback profiling, M3 thermals, and
    bitwise reproducibility across repeated A100 runs.
15. **Next phase:** finish M3 verification when that host is available; meanwhile complete bounded
    A100 optimization/checkpoint calibration before freezing Phase 5.

## Phase 4: real-model smoke calibration

1. **Phase:** 4 — real-model smoke calibration.
2. **Status:** PARTIAL. A100 small-tier calibration is complete; M3 and real-checkpoint write tests
   remain outstanding.
3. **Files changed:** real-model calibration configs, causal-LM adapter/evaluator, PCGU memory
   optimization, summarizer, raw artifacts, compute report, and this status report.
4. **Commit SHA:** calibration artifacts were produced from `81beb6d`.
5. **Tests:** all 25 unit tests plus schema validation for every historical and new run artifact.
6. **Exact test totals:** 25 passed, 0 failed; 18/18 real calibration cells completed.
7. **Device used:** A100-SXM4-80GB for Pythia-160M; A100-SXM4-80GB MIG 1g.20gb for Mamba-130M.
8. **Dtype:** FP32 for both real models.
9. **Compute time:** 91 A100-seconds of scheduler wall time total.
10. **Peak memory:** 7.32 GiB Pythia; 10.24 GiB Mamba. PCGU gradient+mask memory was 1.360 GiB
    for Pythia and 1.082 GiB for Mamba.
11. **Artifacts generated:** 18 manifests, 18 prediction files, 18 metric files, 18 explicit
    calibration checkpoint-omission records, two raw scheduler logs, and a generated report.
12. **Scientific findings:** none confirmatory. Exploratory Pythia setup showed catastrophic held-out
    utility damage; Mamba did not show the same gross collapse. This is a calibration warning, not an
    architecture comparison.
13. **Deviations:** Mamba used the Transformers sequential eager implementation because optional
    fused kernels were unavailable. Calibration intentionally omitted full model weights.
14. **Remaining risks:** utility-preserving Pythia training, actual checkpoint time/storage,
    full-length scaling, M3 behavior, and large-tier feasibility.
15. **Next phase:** run bounded optimization/checkpoint calibration, decide small versus large tier,
    then preregister without inspecting any confirmatory results.

## Post-calibration integrity addendum

Seeded optimization calibration measured real checkpoint writes of 0.596 seconds / 649,350,482
bytes for Pythia-160M and 0.445 seconds / 516,633,264 bytes for Mamba-130M. During review, the
existing v1 test split was found to have been repeatedly inspected during exploratory smoke and
calibration, including an adaptive response to Pythia utility collapse. Confirmatory execution is
therefore stopped pending the resolution in `reports/data_integrity_stop.md`.
