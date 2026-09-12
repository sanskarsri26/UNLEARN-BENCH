# Portfolio package

This page translates the finished research evidence into portfolio language. It intentionally omits
the historical report's unsupported claims and does not turn exploratory Track B behavior into a
machine-unlearning claim.

## GitHub

**Description:** Reproducible machine-unlearning benchmark with controlled forget/retain sets, an
exact-retraining oracle, nine interventions, and Pythia/Mamba forgetting–utility–compute trade-offs.

**Repository:** <https://github.com/sanskarsri26/UNLEARN-BENCH>

## LinkedIn project

**Title:** UNLEARN-BENCH — Controlled Machine-Unlearning Benchmark

**Description:** Built and executed a preregistered 54-cell benchmark comparing nine controls and
unlearning interventions on revision-pinned Pythia-160M and Mamba-130M checkpoints. The project uses
versioned forget/retain data, exact retraining as the behavioral oracle, per-example artifacts,
checkpoint hashes, A100 compute accounting, paired uncertainty, multiple-comparison correction,
Pareto analysis, and a separately labeled StereoSet external-behavior track. The confirmatory study
found that gradient ascent and NPO met the prespecified joint forgetting/utility criterion for
Pythia; no method did for Mamba, and no Holm-adjusted comparison reached 0.05.

**Skills:** machine unlearning; PyTorch; Hugging Face Transformers; model fine-tuning; experimental
design; statistical analysis; GPU benchmarking; Slurm; reproducible ML; model evaluation; PCGU;
NPO; transformer models; state-space models.

**GitHub URL:** <https://github.com/sanskarsri26/UNLEARN-BENCH>

## Resume bullets

### ML / Research Engineer

- Designed and executed a preregistered 54-cell controlled machine-unlearning study across two
  revision-pinned 130–160M-parameter language models, nine interventions/controls, and three seeds,
  with exact retraining as the oracle and complete per-example/checkpoint provenance.
- Implemented and verified gradient ascent, NPO, and a causal-LM adaptation of PCGU; measured
  forgetting, held-out utility, retain behavior, runtime, and memory instead of treating increased
  forget loss as sufficient evidence of deletion.
- Built paired 10,000-draw hierarchical bootstrap, exact sign-flip, Holm correction, and Pareto
  analyses; found Pythia gradient ascent and NPO met the prespecified joint criterion, while no
  Mamba method did and no multiplicity-adjusted comparison reached 0.05.
- Hardened research integrity by detecting development exposure of the original test split,
  stopping the study, creating a newly sequestered holdout, freezing hashes and analysis before
  execution, and validating 54/54 result cells plus 31.48 GB of retained checkpoints.

### General AI Engineer

- Built a device-aware PyTorch/Hugging Face evaluation pipeline spanning a CPU fixture and A100
  execution, with explicit CPU/MPS/CUDA policy, pinned model revisions, deterministic run IDs,
  failure records, resumability, and schema-validated artifacts.
- Automated a two-model, nine-method, three-seed Slurm matrix and reproducible reporting from raw
  predictions; the complete confirmatory workload used 413 A100-seconds and peaked at 15.84 GiB.
- Added continuous integration for lint, configuration validation, unit tests, CPU end-to-end
  execution, and artifact validation, while keeping expensive GPU jobs outside ordinary CI.
- Separated controlled unlearning from external bias evaluation and implemented continuation-only
  StereoSet LMS, SS, and ICAT scoring with SS interpreted around its neutral target of 50.

## Interview preparation

### What is machine unlearning?

Machine unlearning aims to remove the influence of specified training data from a trained model.
The difficult part is demonstrating both removal and preservation: a model that simply becomes
worse is not a successful unlearned model. In this benchmark, the request is concrete—a known
forget set—and behavior is compared with retraining from the same base without that set.

### Why is exact retraining the oracle?

Retraining on retain data only constructs the model that the original pipeline would have produced
had the forget examples been absent. It is expensive, but it provides the clearest behavioral
reference for approximate methods. Its KL distance is zero when compared with itself by definition;
that does not imply every aspect of deletion has been proven.

### What are forget and retain sets?

The forget set contains the examples whose learned influence is targeted for removal. The retain
set contains training examples whose behavior should remain useful. A separate held-out utility set
tests broader damage. Their roles are versioned explicitly rather than inferred after training.

### Why is increased forget loss not enough?

Forget loss can rise because the intervention damaged the whole model. A credible evaluation also
checks movement toward an exact-retrain oracle, retain performance, and held-out utility. The
counterfactual Pythia result illustrates this: it moved most toward the oracle but measurably
worsened utility.

### How does PCGU work here?

PCGU computes gradients for an original and contrastive target, partitions parameter vectors,
ranks them by gradient relationship, selects a fixed fraction, and masks the update so only selected
vectors change. The published method targeted masked-LM social bias; this repository documents its
causal-LM controlled-association adaptation and verifies the mask and ranking mechanics separately.

### What does NPO optimize?

Negative Preference Optimization uses a frozen reference model and a preference-style objective to
decrease the relative likelihood of forget examples, with retain regularization in this protocol.
That reference-relative objective is more controlled than unconstrained loss maximization, though
it still requires utility evaluation.

### Why compare Mamba with a transformer?

The comparison asks whether the observed trade-off pattern repeats across two different model
families and exercises the benchmark beyond one implementation. It is useful robustness evidence,
but not a randomized architecture experiment.

### Why can you not make causal architecture claims?

Pythia-160M and Mamba-130M differ in training corpus, tokenizer, parameterization, optimization
history, and implementation—not only architecture. Any outcome difference is therefore
observational and may be explained by those confounders.

### How did you preserve utility?

Utility preservation was an endpoint, not an assumption. Every method was evaluated on held-out
utility NLL and retain NLL, paired to the full-trained baseline across fixed seeds. The project
reports methods that failed the joint criterion and localized retain damage instead of hiding them.

### What did the Mac M3 handle?

No M3 was available in the execution environment, so the repository does not claim measured MPS or
thermal compatibility. The unified device layer supports explicit MPS selection and fail-closed
fallback behavior, but M3 execution remains an external-replication task.

### What required A100?

All real-model calibration, confirmatory training, and external StereoSet scoring used A100
partitions. The final Track A matrix used FP32 on 20 GiB A100 MIG instances, with peak allocation of
9.74 GiB for Pythia and 15.84 GiB for Mamba. Mamba used its documented sequential eager path because
optional fused kernels were unavailable.

### What was the biggest experimental failure?

The most important failure was scientific rather than computational: the original held-out split
had been exposed during development. Work stopped, the contaminated split was retained as
exploratory, and a new deterministic v2 holdout was generated and hash-sealed before the protocol
and analyzer were committed. A later StereoSet retry fixed an empty-context tokenizer edge case;
the failed jobs wrote no scored cells and were preserved in the audit trail.

### What changed from the legacy project?

The reboot replaced unverifiable headline scores with a controlled question, deterministic data,
an exact-retraining oracle, nine explicit methods/controls, pinned real-model adapters, raw
per-example outputs, run manifests, checkpoint hashes, predeclared statistics, device/compute
measurement, CI, and strict separation between machine unlearning and external bias behavior.

## Evidence-based project ranking

This ranking uses the stated portfolio roles for the other projects, not a source audit of those
repositories. Reassess it if their implementation evidence differs materially from those roles.

| Target role | Recommended top three, in order | UNLEARN-BENCH rationale |
| --- | --- | --- |
| Machine Learning Engineer | UNLEARN-BENCH, Dermatology AI, LumiNote | Strongest direct evidence of training, evaluation, experimental controls, and GPU work. |
| AI/ML Engineer | UNLEARN-BENCH, LumiNote, Dermatology AI | Balances model-level work with production and applied-model evidence. |
| LLM Engineer | UNLEARN-BENCH, LumiNote, AI-deception | Direct causal-LM fine-tuning complements RAG and behavioral evaluation. |
| AI Research Engineer | UNLEARN-BENCH, AI-deception, Dermatology AI | Preregistration, negative results, uncertainty, and reproducibility are central evidence. |
| AI Engineer | LumiNote, UNLEARN-BENCH, Incident Reporter | Product delivery leads; this project supplies deeper model internals. |
| AI Platform Engineer | Incident Reporter, LumiNote, UNLEARN-BENCH | Platform/reliability evidence leads; this project adds Slurm and artifact-pipeline depth. |

UNLEARN-BENCH belongs in the top three for all six targets, but should lead only where model
training or research rigor is central. For product- and platform-heavy roles, it supports rather
than replaces the repositories with stronger deployed-system evidence.
