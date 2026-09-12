# Legacy audit: historical / unverified

The pre-reboot repository consists of the root `README.md`, `Final_Report.pdf`, and the complete
`Unlearning/` directory. They are preserved in place to retain provenance. Nothing in the reboot
rewrites their adapters, data, scripts, report, or reported numbers.

## What it attempted

The historical work fine-tuned a Mamba-130M LoRA adapter on WinoGender-style data, applied a routine
called PCGU, and reported a single StereoSet-like bias percentage plus qualitative prompt outputs.
It described the intervention as machine unlearning and stated that utility was maintained.

## Material methodological problems

- There is no known pretraining forget set, retain-only exact-retraining oracle, or controlled proof
  that designated examples were learned and then removed. The operation is, at most,
  unlearning-inspired debiasing.
- `Unlearning/scripts/pcgu_unlearn.py` does not implement PCGU's defining gradient-vector partition,
  cosine-similarity ranking, or selected-parameter mask. It updates all trainable LoRA parameters.
- `Unlearning/scripts/evaluate_stereoset.py` compares full prompt losses, counts only cases where the
  stereotype beats both alternatives, omits official LMS and ICAT, and says “lower is better” even
  though StereoSet SS is neutral near 50.
- The historical headline table has no run manifests, model revisions, data hashes, raw predictions,
  uncertainty, seed aggregation, exact oracle, or measured utility evidence.
- Model adapters exist, but the repository does not preserve enough lineage to independently derive
  every reported claim.

## Citation status

The historical 46.82, 46.73, and 46.49 values and associated “best debiasing” or “utility
maintained” statements are **HISTORICAL / UNVERIFIED** and should not be cited as reboot evidence.
The reboot was necessary to distinguish controlled machine unlearning from parameter editing, add an
exact-retrain oracle, predeclare endpoints, retain raw outputs, and enforce reproducibility.
