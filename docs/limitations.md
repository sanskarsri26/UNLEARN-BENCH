# Limitations

- The CPU fixture validates research plumbing and metric behavior, not scientific claims about real
  language models. All smoke results are exploratory.
- The v0.1 runner deliberately rejects the revision-pinned Hugging Face model configurations. A
  checkpoint-efficient causal-LM adapter and GPU calibration must be implemented and reviewed before
  Pythia/Mamba runs. This prevents a scaffold from silently launching an expensive or invalid study.
- The synthetic association task is narrow. Exact retraining is meaningful within this controlled
  task but does not cover every form of memorization or deletion.
- Pythia and Mamba sizes, training corpora, tokenizers, and optimization histories are not fully
  matched. Cross-family differences will be observational.
- Track B currently provides validated StereoSet metric primitives but no downloaded benchmark
  artifact. Dataset access, license, hash, and reference-validation evidence are required before use.
- Membership inference, CrowS-Pairs, BBQ, and downstream LM suites are intentionally deferred rather
  than included without adequate validation.
- Deterministic kernels and seeds reduce variation but do not guarantee bitwise equality across
  hardware/software stacks.
