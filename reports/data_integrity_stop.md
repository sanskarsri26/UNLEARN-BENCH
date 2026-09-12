# Data-integrity stop before confirmatory execution

## Finding

`data/controlled/v1/test.jsonl` is deterministic and has no exact-example overlap with train or
validation, but it is no longer an uninspected confirmatory holdout. Commit `3db2e24` preserved
canonical exploratory smoke metrics computed on this test split. The A100 calibration artifacts
from commit `81beb6d` also evaluated the same split. Finally, exploratory training calibration from
commit `9023864` evaluated both validation and test, after the first Pythia test evaluation exposed
catastrophic held-out utility damage and motivated a lower-learning-rate calibration.

This is evaluation leakage/adaptive reuse, not train/test record overlap. No existing data file or
hash has changed. All affected outputs remain explicitly exploratory. The problem is that the same
test endpoint can no longer support a strong claim of untouched confirmatory evaluation.

## Why execution stopped

The permanent protocol says to stop when data leakage is discovered and never silently change
deterministic splits. Proceeding with `controlled/v1/test` would weaken claim validity; silently
replacing it would violate split and provenance rules. No `CALIBRATION_REVIEWED` marker or
preregistration freeze has been created, and no confirmatory run has begun.

## Scientifically defensible resolutions requiring explicit direction

1. **Recommended:** preserve controlled/v1 unchanged as development/exploratory data and create a
   versioned `controlled/v2` with a newly generated, sequestered confirmatory holdout. Record this as
   a pre-preregistration protocol correction, commit hashes before further calibration, and prohibit
   all inspection until the frozen analysis.
2. Introduce an independently sourced controlled association dataset with an untouched test set,
   preserving v1 only for fixtures and development.
3. Continue using v1 but label all resulting work exploratory. This cannot satisfy the requested
   confirmatory quality bar and is not recommended.

The first option changes no historical split or artifact; it adds a new dataset version. It still
requires explicit authorization because the handoff names deterministic split changes as a stop
condition and the choice materially changes the confirmatory protocol.
