# PAIN architecture fidelity contract

The source of truth is the paper authors' `ExpressivePathGNNs` repository,
principally `Models/path_gnn.py`, `Models/path_conv.py`, and
`Models/compute_paths.py`, audited locally at commit
`4a7e91af4b64b54f145dd60d9b0caed39c7fd76b`. This repository adapts PAIN to heterogeneous node
classification while keeping the path-processing architecture recognizable and
auditable.

## Preserved architecture

| Reference mechanism | Local implementation |
|---|---|
| Every rooted simple path with 0..L edges | `preprocessing/imdb_node_classification.py` |
| Optional leaf-first path reversal | `pain_nc.data.reverse_valid_paths` |
| Node + position/distance + edge + neighbor input | `PathAggregator._encode_chunk` |
| Packed multi-layer LSTM path encoder | `PathAggregator._encode_chunk` |
| Sum/mean aggregation at the root | `PathAggregator.forward` |
| Residual, batch norm, dropout | `PainLayer.forward` |
| `Linear-BN-ReLU-Linear-BN-ReLU` layer MLP | `PainLayer.mlp` |
| Shared or per-layer LSTM | `PainNodeClassifier.__init__` |
| Last/mean/concatenated jumping knowledge | `PainNodeClassifier.node_embeddings` |
| Two-layer prediction MLP | `PainNodeClassifier.classifier` |

The default IMDb configuration follows the authors' ZINC PAIN settings: L=3,
five PAIN layers, hidden size 128, two LSTM layers, shared LSTM, reversed paths,
neighbor marking, sum path aggregation, no dropout, and a two-layer head.

## Intentional task adaptations

- Graph-level pooling is removed; the classifier receives each node's final PAIN
  representation.
- Dense IMDb bag-of-words features use a linear node encoder.
- Heterogeneous node types and relation types receive learned embeddings.
- Exact path processing is chunked, with optional activation checkpointing and
  CPU-to-GPU path streaming. These controls do not sample or discard paths.
- DBLP cannot materialize its billion-path exhaustive programs. Its separately
  labeled sampled-PAIN protocol retains all length-zero/one paths, directly
  samples canonical length-two/three ranks per root, and applies inverse-
  inclusion-probability weights before sum aggregation. This is a scalability
  approximation and is not claimed to preserve PAIN's exact expressivity.
- Root-sorted segment sums replace atomic CUDA scatter-add so repeated runs and
  future invariant comparisons can enforce deterministic algorithms.
- The data container carries shared node supervision separately from each
  physical variant's edge/path program.

## Corrected deviations

The fidelity audit identified and corrected three unintended differences:

1. Learned positional encodings now use Xavier normal initialization.
2. Dropout is applied to the concatenated path representation before the LSTM.
3. The per-layer MLP now includes the reference implementation's final batch
   normalization and activation. The final prediction head also uses the
   reference `Linear-Dropout-Activation` ordering.
4. Edge padding is represented by a non-parameterized zero vector rather than
   an extra learned embedding row, and shortest-path encoding retains the
   reference implementation's extra padding slot. Relation embeddings use the
   reference edge encoder's Xavier uniform initialization.

## Experiment separation

The following must remain separately named and reported:

- **Vanilla PAIN:** independently trained on one physical graph.
- **Universal PAIN:** independently trained on the union graph.
- **PAIN augmentation:** one model and optimizer trained across physical variants.
- **Invariant PAIN:** the DBLP preprocessing-time semantic-path compiler in
  `preprocessing/dblp_link_prediction.py`. It compiles each physical Area
  realization to one canonical relation graph before exact or semantic-keyed
  sampled PAIN paths are enumerated.

Joint augmentation does not imply information invariance. The DBLP invariant
arm enforces context/propagation separation, context-only skip-node exclusion,
semantic deduplication, canonical ordering, and deterministic reductions. Its
preprocessor aborts unless raw physical hashes differ and compiled semantic
hashes match across DBLP1-3.

## Dataset-contract caveat

The current IMDb `v4` intentionally reproduces the legacy DHN graph, which omits
Actor1 while retaining Actor2/3 directly. This is a dataset compatibility choice,
not part of PAIN. A paper-compatible information-preserving `v4` should be added
under a distinct, explicit contract before invariant experiments are claimed.
