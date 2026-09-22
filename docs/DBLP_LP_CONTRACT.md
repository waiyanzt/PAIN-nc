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
- Invariant PAIN: three independent same-seed models on three independently
  materialized conditional path programs, one from each physical `v1`, `v2`,
  and `v3`. Preprocessing refuses to finish unless their semantic edges,
  complete PAIN input tensors, and sampling weights have matching hashes.
  These compiler-source graphs use the guide's training-block mapping scope
  and are intentionally distinct from the ordinary baseline scope.

The invariant compiler treats a moved Area attachment as context. It projects
the licensed Paper-Area, Conference-Area, and Author-Area relations, excludes
context-only witnesses from propagation sequences, deduplicates relations, and
orders them canonically. PAIN enumerates paths only after this projection. By
the paper's model-specific structural-input principle, PAIN's object is the
ordered rooted path,
including node identities, semantic edge types, positions, neighbor marks,
lengths, and inverse-inclusion weights. The LSTM and training objective are
unchanged; they consume those projected paths rather than physical witness
walks. Thus the invariant adaptation occurs before the model but *does*
change the model's structural input.

The conditional projection is block-local. For a training Paper `P` with
certified Area `Ar`, its physical context is `P-Ar` in v1, `P-V-Ar` in v2,
or `P-A-Ar` in v3 (`V` is a training Venue and `A` is an Author). In each
case the compiler requires the physical witness before licensing the same
`P-Ar`, `V-Ar`, and `A-Ar` propagation relations; the base `A-P`, `P-T`,
and training `P-V` relations remain. Every semantic path of at most three
edges is then a simple path through licensed propagation relations. Each of
its virtual edges has a physical context witness, while only the ordered
propagation nodes/edges enter PAIN's LSTM. A path's multiple witnesses may
overlap; set-valued relation/path semantics deduplicate them.

The training-only Paper-Area certificate, derived from author labels, resolves
ambiguous v2 Conference-Area contexts. It is not a model feature or edge, but
it is external side information used by the compiler. The physical graph must
still contain a matching witness; the certificate alone cannot create an
edge. The certificate source, record count, and hash are recorded in metadata.
Any graph-only theoretical claim must explicitly include this certificate in
the transformation contract or replace it with a derivation solely from the
physical graph. No such general graph-only theorem is claimed here.

For the fixed DBLP variant family, equality of the complete path input plus
shared node IDs/features implies equal PAIN representations for any fixed
weights by induction over its layers, and hence equal decoder scores. This is
an exact *fixed-input/fixed-weights* statement, not a claim that independently
trained models are automatically bit-identical on every GPU. Node-ID embeddings
also mean this statement covers the paper's ID-preserving DBLP switchings, not
arbitrary node permutations without consistently permuting the embeddings.

The present semantic closure equals the union of the three *train-scope*
physical edge sets. That fact is audited rather than hidden. Independent
compilation plus identical PAIN path inputs establishes exact input equality
for this DBLP transformation family and fixed sampling seed. It does not by
itself establish Section 4.3's general minimality result, nor the paper's
injective-aggregator expressivity result for sampled LSTM-based PAIN.

Population caveat: this repository filters to the shared one-Area eligible
population (26,076 nodes in the current count-only output), matching the
paper's Appendix Table 4 `DBLP*` node counts. The same table lists larger
`DBLP1`-`DBLP3` populations. The corrected RGCN guide also uses a shared
filtered population for the invariant mapping. The final paper comparison
must explicitly choose and report one population contract; these counts
must not be presented as an exact reproduction of the larger original arms.

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

The DBLP variants contain billions of exact length-three paths, so the runnable
headline is **sampled PAIN**. Exhaustive mode remains available explicitly with
`--max-paths-per-root 0`, but must first be assessed with `--count-only`.

The default `--max-paths-per-root K` policy interprets `K` as the per-root
budget for length-two and length-three paths. It:

1. retains every length-zero and length-one path exactly;
2. allocates `K` proportionally between the nonempty length-two and
   length-three strata for each root, retaining at least one of each when the
   budget permits;
3. samples unique canonical path ranks directly with a repository-local
   SplitMix64/Floyd sampler, without enumerating the length-three population;
4. attaches inverse-inclusion-probability weights per root and length, so sum
   aggregation estimates the exhaustive PAIN sum; and
5. samples the invariant arm only after semantic compilation.

Canonical rank selection depends only on the graph program, root, length, and
sampling seed. Identical compiled programs therefore produce identical sampled
path tensors and weights. Artifact metadata records the policy, selected counts,
and hashes of both the selected path program and weights.

For a valid sampled comparison, `K`, the sampling seed, sampling/weighting policy, training budget,
candidate protocol, and checkpoint selection must be identical across every
original, universal, augmentation, and invariant arm. Results must be labeled sampled PAIN.
Physical-path sampling is allowed for the non-invariant baselines, but the
invariant arm must sample only after semantic compilation.

## Exact-invariance acceptance criteria

1. The three source physical graph hashes differ.
2. The three compiled semantic graph hashes match.
3. Three independently generated semantic path-program and weight hashes match.
4. Supervision, negatives, and candidate order share one hash/artifact.
5. Same-seed invariant runs use one GPU architecture.
6. Candidate logits and model checkpoints are byte-identical.
7. Every matched-seed pair has Kendall tau exactly 1 and maximum score
   difference 0.
8. Invariant performance is compared with both original and universal PAIN
   under the same path budget.

## Safe invariant preprocessing and training

Generate the new invariant artifacts into their own directory. This does not
overwrite the ordinary `data/preprocessed/DBLP` artifacts used by independent
and augmentation jobs:

```bash
python -m preprocessing.dblp_link_prediction \
  --mode invariant --output-dir data/preprocessed/DBLP_invariant \
  --max-paths-per-root 256 --sampling-seed 1566911444
```

Run `--count-only` first if resources need checking. Before training, run:

```bash
python -m experiments.link_prediction.benchmark_dblp \
  --config configs/dblp_lp_invariant.yaml \
  --variants DBLP_invariant_v1 DBLP_invariant_v2 DBLP_invariant_v3 \
  --preflight-only
```

The ordinary and invariant YAML files specify the same PAIN model, optimizer,
budget, and evaluation settings. They differ only in the preprocessed input
directory. Use a fresh output root for the new invariant runs; an existing
checkpoint in `results/dblp_lp` is not automatically retrained by the runner.
