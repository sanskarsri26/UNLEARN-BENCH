# PCGU verification

Reference: Charles Yu et al., “Unlearning Bias in Language Models by Partitioning Gradients,”
Findings of ACL 2023, and the authors' `CharlesYu2000/PCGU-UnlearningBias` implementation.

| Requirement | Reboot implementation | Verification |
| --- | --- | --- |
| Starting checkpoint | retain+forget trained checkpoint | runner constructs once per seed and clones it |
| Contrastive gradients | original target and replacement target losses | method test exercises both backward passes |
| Partition | vectors aggregated over final/input dimension | `_vector_partitions`; unit fixture |
| Ranking | lowest cosine similarity first | deterministic known-gradient unit fixture |
| Selected amount | configured fraction, rounded up | manifest records requested and actual fraction |
| Mask | full-shape boolean mask per parameter | test proves unmasked entries do not change |
| Update | minimize replacement-target loss on selected entries | `pcgu_step` |
| Save/load | state dict, vocabulary, model config in checkpoint | run artifact plus manifest path/hash metadata |

Deviation: the paper and official code target masked transformer language models and WinoGender
contrastive pairs. The controlled track uses a causal next-token loss, with the learned association
as one side and its versioned counterfactual as the other. Therefore this implementation preserves
PCGU's defining partition-selection-update mechanics but does not claim exact experimental
reproduction. Applying it to natural pretrained bias must be called unlearning-inspired debiasing.
