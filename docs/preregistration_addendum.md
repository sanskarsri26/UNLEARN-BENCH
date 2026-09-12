# Post-execution preregistration addendum

## 2026-09-11 — checkpoint portability validation

This administrative change was made only after all 54 confirmatory cells completed and the
precommitted analyzer verified every local checkpoint's byte size and SHA-256 hash. It does not
change or rerun any data, model, method, seed, budget, endpoint, metric, statistic, prediction, or
result.

The run manifests and small raw artifacts are committed to Git, but the 54 checkpoints total
31,481,561,142 bytes and are intentionally retained in ignored scratch artifact storage. The generic
manifest validator now accepts an absent saved checkpoint in a clean Git checkout only when its
manifest records `checkpoint_status: saved`, a 64-character SHA-256 digest, and a positive byte
size. Predictions and metrics remain mandatory. The confirmatory analyzer remains stricter: it
requires every checkpoint to be locally present and rehashes it before regenerating analysis.

Because the preregistration marker freezes runner source, this post-execution source change makes
that original marker reject any attempted new execution from the modified worktree. Reproduction of
the frozen run requires checkout of preregistration commit
`005db27a9f0b427bfa3fcc1cbcc3cc19da0c67a6` with the authorization record, or a separately reviewed
and documented replication marker. This fail-closed behavior is intentional.
