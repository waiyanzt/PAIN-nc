# DBLP PAIN link-prediction contract

This experiment predicts Paper-Conference links over DBLP1-3, where Area is
physically represented on Paper, Conference, or Author nodes. It follows the
corrected protocol and leakage constraints in `INV-RGCN-guide`.

## Arms

- Original PAIN: independent models on the `v1`, `v2`, and `v3` physical path
  programs. These match the other GNN baselines: v1/v3 use all available
  auxiliary Area labels, while v2 Conference-Area edges are derived from
  training Paper-Conference blocks only.
- Universal PAIN: an independent model on the leakage-safe union graph.
- PAIN augmentation: one model and optimizer visits physical `v1-v3` once per
  seeded random super-epoch and is evaluated separately on all three graphs.
- Invariant PAIN: three independent same-seed models on a semantic path program
  compiled independently from `v1`, `v2`, and `v3` and stored once after their
  hashes match. These compiler-source graphs use the guide's training-block
  mapping scope and are intentionally distinct from the ordinary baseline
  scope.

The invariant compiler treats a moved Area attachment as context. It projects
the licensed Paper-Area, Conference-Area, and Author-Area relations, excludes
context-only witnesses from propagation sequences, deduplicates relations, and
orders them canonically. PAIN enumerates paths only after this projection.

## Leakage and evaluation

- Paper-Conference edges in every graph are training positives only.
- The v2 Conference-Area relation is derived from training
  Paper-Conference blocks only.
- Validation and test target-edge leakage counts must both be zero.
- Papers are split disjointly using seed `1566911444`.
- Every query is ranked against all 20 Conference nodes with standard filtered
  removal of other known true tails.
- Validation MRR selects the checkpoint.
- Training uses every false conference (19 for the one-venue-per-paper data).
- The budget is 1000 epochs with patience 200.

## Exact versus sampled paths

Exact paths are the preferred headline because they preserve the faithful PAIN
bias. Run `--count-only` before allocating storage. If exact enumeration is not
feasible, `--max-paths-per-root K` retains the root-only path and selects the
other paths by a stable hash of `(sampling_seed, semantic node path)`. Selection
happens after set deduplication.

For a valid sampled comparison, `K`, the sampling seed, training budget,
candidate protocol, and checkpoint selection must be identical across every
original, universal, augmentation, and invariant arm. Results must be labeled sampled PAIN.
Physical-path sampling is allowed for the non-invariant baselines, but the
invariant arm must sample only after semantic compilation.

## Exact-invariance acceptance criteria

1. The three source physical graph hashes differ.
2. The three compiled semantic graph hashes match.
3. Supervision, negatives, and candidate order share one hash/artifact.
4. Same-seed invariant runs use one GPU architecture.
5. Candidate logits and model checkpoints are byte-identical.
6. Every matched-seed pair has Kendall tau exactly 1 and maximum score
   difference 0.
7. Invariant performance is compared with both original and universal PAIN
   under the same path budget.
