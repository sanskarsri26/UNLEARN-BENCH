# Track B protocol addendum

## 2026-09-11 — empty causal contexts

Full-matrix jobs `63045726` and `63045727` failed after 18 and 20 seconds, respectively, before
writing any Track B result cell. The official StereoSet dev artifact contains 19 intrasentence items
whose first `BLANK` begins the sentence, producing 57 candidates with an empty causal prefix. The
first-64-item calibration subset contained none of these and therefore did not expose the case. The
generic controlled-task batch builder correctly rejects an empty prompt because an autoregressive
model cannot assign a conditional probability to its first token without an initial token.

Before retrying, the Track B scorer is amended to use the pinned tokenizer's BOS token as the causal
context for only these 57 candidates, falling back to EOS only if BOS is undefined. Both selected
tokenizers define the same `<|endoftext|>` BOS/EOS token with ID 0. Candidate spans, score
normalization, dataset order, all 2,106 examples, models, methods, seeds, metrics, and compute caps
remain unchanged. This is a technical coverage correction, not an outcome-adaptive change; no Track
B metric was inspected from the failed jobs.
