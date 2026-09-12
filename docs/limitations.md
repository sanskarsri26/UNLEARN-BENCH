# Limitations

- The CPU fixture validates research plumbing and metric behavior, not scientific claims about real
  language models. All smoke results are exploratory.
- The revision-pinned Hugging Face causal-LM adapter is implemented but remains exploratory until
  Pythia/Mamba hardware calibration is reviewed. Calibration may intentionally omit model weights;
  confirmatory runs must save and hash their final checkpoints.
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
- Real-model runs currently use FP32. BF16 loading passes an A100 execution probe, but scientifically
  appropriate mixed-precision optimization with FP32 master weights has not yet been validated.
