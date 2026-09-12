# Track B protocol: StereoSet external generalization

**Status:** exploratory protocol fixed before Track B model scoring.

Track B asks whether the controlled Track A interventions alter behavior on an external social-bias
benchmark. It does not test deletion of StereoSet examples, establish that those examples occurred
in pretraining, or constitute machine forgetting.

## Dataset and license

The preserved `Unlearning/data/stereoset_dev.json` is byte-identical to `data/dev.json` at official
StereoSet repository commit `ead7d086a64a192a1eca88e0dd2fd163de375218`. Its SHA-256 is
`73a0f31b711688112602e4c3ac6ab1e1a7cadcdd67df6c6fd55501c889676c90`. The repository declares a
CC BY-SA 4.0 license. The artifact contains 2,106 intrasentence and 2,123 intersentence examples.

This evaluation uses all 2,106 intrasentence examples. Intersentence evaluation is deferred because
the causal continuation protocol differs materially and would add a second task definition. No
subset or category is selected based on results.

## Models and scoring

All 54 completed Track A checkpoints are evaluated: two models, all nine methods/controls, and seeds
11, 29, and 47. Models run on A100 CUDA in FP32. For each intrasentence item, the text before the
first `BLANK` is the causal context and the first inserted candidate span is the continuation. Text
after the insertion is not scored. The official dev artifact has two anomalous items with two
`BLANK` occurrences; their first inserted span is scored under the same rule. Candidate score is
mean conditional log probability across only its continuation tokens, avoiding a systematic
preference for shorter multi-token candidates. Case-only differences between context and completed
sentence are accepted, while exact reconstruction is still validated case-insensitively.

For every checkpoint, report overall and race/gender/religion/profession:

- language-model score (LMS), higher being better;
- stereotype score (SS), whose neutral target is 50 rather than zero; and
- ICAT, which jointly rewards LMS and SS proximity to 50.

Ties receive half credit. Raw candidate scores are retained. Track A utility NLL remains the utility
measure; Track B shifts are not celebrated when Track A utility is damaged.

## Analysis and uncertainty

Results include each seed plus means and sample standard deviations. Paired checkpoint comparisons
use the full-trained checkpoint of the same model and seed. For LMS and ICAT, differences are method
minus full-trained. For bias direction, report both SS difference and change in absolute deviation
from 50; a negative change in absolute deviation is movement toward neutrality. No NHST or multiple
comparison claims are planned because Track B is exploratory and StereoSet items have complex
dependencies. Category results and representative largest shifts are exploratory error analysis.

## Calibration, compute, and stopping

Before the full run, score the first 64 dataset-order items for the seed-11 full-trained checkpoint
of each model, inspecting only runtime, peak memory, and technical failures. If extrapolated total
compute exceeds 2 A100 GPU-hours, stop and report. Otherwise execute the full fixed matrix. Every
cell is completed or explicitly failed; no method is removed because of an unfavorable score.
