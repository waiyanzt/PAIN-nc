# PAIN Node Classification and Link Prediction

This repository adapts the official PAIN (PAth Isomorphism Network) model from
*The Expressive Power of Path-Based Graph Neural Networks* to node
classification. Graph pooling is omitted at the model output; the final PAIN
node embeddings are passed to a node-wise classification head.

The upstream-to-local architecture contract and intentional changes are recorded
in `docs/PAIN_FIDELITY.md`.

The end-to-end benchmarks are IMDb node classification and DBLP
paper-conference link prediction. DBLP includes original physical variants, a
universal union baseline, and a compiled invariant PAIN arm.

## What is faithful and what changed

The default model configuration follows the official PAIN ZINC experiment:
`L=3`, five PAIN layers, a two-layer LSTM, hidden size 128, shared LSTM
weights, reversed paths, neighbor marking, sum path aggregation, no dropout,
and a two-layer prediction MLP.

The task adaptation is intentionally narrow:

- the graph-level pooling/readout is removed;
- the classifier consumes the final embedding of each node;
- IMDb bag-of-words features use a linear encoder, with a learned node-type
  embedding and learned edge-type embeddings;
- exact path aggregation is evaluated in chunks to control peak memory.

Chunking does not sample or discard paths. It produces the same sum as
processing all paths at once.

## Setup

The default HPC target is an NVIDIA V100 (Volta, compute capability 7.0).
`requirements.txt` therefore pins the CUDA 12.6 build of PyTorch 2.13, which
contains V100 kernels. CUDA 13 PyTorch wheels do not support Volta.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

No PyTorch Geometric dependency is required.

Verify the environment on an allocated GPU before preprocessing or training:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0)); print(torch.cuda.get_arch_list()); print(torch.ones(1, device='cuda'))"
```

On a V100, the version should end in `+cu126`, the device capability should be
`(7, 0)`, and the architecture list should contain `sm_70`.

## IMDb preprocessing

The IMDB contract follows the existing `dhn_nclp` node-classification
benchmark: the same 4,180 movie targets, Action/Comedy/Drama labels, node
features, four topology variants, universal union graph, and deterministic
70/10/20 split.

PAIN preprocessing is fixed at path length `L=3`. Every simple path with zero
through three edges is materialized for every root node. Paths are not sampled.

From the repository root:

```bash
python -m preprocessing.imdb_node_classification
```

Outputs are written to `data/preprocessed/IMDB/`:

```text
shared.pt          features, labels, node types, and split masks
v1_L3.pt           v1 edges and PAIN path tensors
v2_L3.pt           v2 edges and PAIN path tensors
v3_L3.pt           v3 edges and PAIN path tensors
v4_L3.pt           v4 edges and PAIN path tensors
universal_L3.pt    union-graph edges and PAIN path tensors
metadata.json      preprocessing counts and provenance
vocabulary.json    plot-keyword feature vocabulary
```

The variant files use the field names from the original PAIN implementation:
`path_index`, `path_lengths`, `mask_index`, `path_edge_idx`,
`neighbor_mask`, and `distances`.

The generated `.pt` files are intentionally ignored by Git; regenerate them
after cloning on the HPC server. Approximate local sizes are 181 MiB for
`shared.pt`, 33–131 MiB for each baseline graph, and 524 MiB for the universal
graph. The universal graph contains 3,801,790 exact rooted paths.

## Benchmark

First validate that every generated artifact is present:

```bash
python -m experiments.node_classification.benchmark_imdb --preflight-only
```

Run the complete five-variant, three-seed benchmark:

```bash
python -m experiments.node_classification.benchmark_imdb \
  --config configs/imdb_nc.yaml \
  --output-root results/imdb_nc
```

The standard seeds are `1566911444`, `20241017`, and `20251017`. A smaller
selection can be used for a shakedown run:

```bash
python -m experiments.node_classification.benchmark_imdb \
  --variants IMDb4 \
  --seeds 1566911444
```

Each run writes `results/imdb_nc/<variant>/seed<seed>.pt`. The artifact includes
the complete configuration, validation-Macro-F1-selected epoch, metrics, test node IDs,
class probabilities (`y_prob`), predictions, and labels. Aggregate run and
mean/sample-standard-deviation tables are written beside those directories.
Existing run artifacts are reused unless `--overwrite` is supplied.

Every new run records mandatory resource telemetry: parameter/buffer/static model
bytes, serialized checkpoint bytes, peak process RSS, training and inference CUDA
allocated/reserved peaks, input artifact sizes, and device/runtime metadata. Older
artifacts without this contract are rejected; use `--overwrite` to regenerate them.

The faithful default is computationally expensive because each of five PAIN
layers processes every rooted path. Important memory controls in
`configs/imdb_nc.yaml` are:

- `path_chunk_size`: reduce it if a chunk itself exhausts memory;
- `checkpoint_chunks: true`: recompute LSTM activations during backward;
- `paths_on_device: false`: stream path indices from CPU if the complete
  universal path artifact does not fit on the GPU.

Changing these controls does not change which paths are used. If a scheduler
time limit is shorter than 12 hours, set `training.max_hours` below the job
limit so the best completed checkpoint is saved cleanly.

The convenience wrapper preprocesses, checks, and benchmarks:

```bash
bash scripts/hpc/run_imdb_nc.sh
```

## Joint graph-variant data augmentation

The augmentation arm trains one PAIN model, optimizer, and checkpoint across
IMDb `v1-v4`. One super-epoch visits every selected physical variant once in a
seeded random order. Checkpoint selection maximizes mean validation Macro-F1;
test metrics and aligned logits are emitted per variant together with pairwise
Kendall tau and exact resume state.

Preflight:

```bash
python -m experiments.node_classification.imdb_augmentation \
  --config configs/imdb_nc_augmentation.yaml \
  --variants v1 v2 v3 v4 \
  --preflight-only
```

Three-seed run:

```bash
python -m experiments.node_classification.imdb_augmentation \
  --config configs/imdb_nc_augmentation.yaml \
  --variants v1 v2 v3 v4 \
  --output-root results/imdb_nc_augmentation
```

Resume an interrupted run with the same configuration and output root by adding
`--resume`. The default augmentation budget is update-matched: 250 super-epochs
times four variants equals the 1,000-update cap of an independent baseline.

## Kendall tau

After all matched seed artifacts exist, compare per-node class-score rankings:

```bash
python -m analysis.kendall_tau_imdb_nc \
  --root results/imdb_nc \
  --output reports/kendall_tau_imdb_nc_test.csv
```

The analysis validates identical test node order and labels before computing
node-level Kendall tau-b, then reports the mean and sample standard deviation
over matched seeds.

## DBLP paper-conference link prediction

DBLP uses one canonical paper-disjoint 70/10/20 split. Message-passing graphs
contain training paper-conference edges only. Evaluation ranks each held-out
paper against all 20 conferences with filtered ranking; checkpoint selection is
validation MRR under that same protocol. Training uses all 19 false conference
candidates per positive. This avoids the malformed sampled-negative protocol
documented in `INV-RGCN-guide`.

Start with an exact path census. It computes counts without materializing path
tensors:

```bash
python -m preprocessing.dblp_link_prediction --mode both --count-only
```

If the reported exact size and expected runtime are acceptable, build all
original, universal, and invariant artifacts:

```bash
python -m preprocessing.dblp_link_prediction --mode both
```

The invariant arm does not train on a renamed physical graph. For each of
DBLP1-3, preprocessing matches the variant-specific Area context, projects it
to a canonical semantic relation program, and verifies that physical hashes
differ while semantic hashes match. Exact rooted PAIN paths are then generated
from the compiled program. One deduplicated `invariant_L3.pt` path store is
shared by the three independently trained invariant runs; its metadata carries
all three source-graph and compiler audits.

The ordinary v1-v3 and augmentation artifacts retain the same Area-information
scope as the existing cross-GNN DBLP datasets (all auxiliary v1/v3 Area labels;
v2 derived from training target blocks). The invariant compiler separately
uses the guide's training-block scope so no held-out Paper-Conference topology
is needed to make its three semantic programs identical.

If exact paths are impractical, use a common deterministic per-root cap:

```bash
python -m preprocessing.dblp_link_prediction \
  --mode both \
  --max-paths-per-root 1000
```

Then set `data.artifact_tag: L3_cap1000` in `configs/dblp_lp.yaml`, or pass
`--artifact-tag L3_cap1000` to the benchmark. The cap is applied after path
deduplication and keyed by semantic node-path identity. Therefore all invariant
variants select the same paths. Use the identical cap and seed for original,
universal, and invariant arms, and report the experiment as sampled PAIN rather
than exact PAIN.

Run the complete seven-arm, three-seed benchmark:

```bash
python -m experiments.link_prediction.benchmark_dblp \
  --config configs/dblp_lp.yaml \
  --output-root results/dblp_lp
```

Keep every matched invariance run on the same GPU architecture (for example,
all V100s); mixing GPU architectures can break bitwise equality even when
semantic hashes match. After all independent runs complete, audit the invariant
arm with:

```bash
python -m analysis.kendall_tau_dblp_lp \
  --root results/dblp_lp \
  --output reports/kendall_tau_dblp_lp_invariant.csv
```

The audit requires aligned candidate scores, model checkpoints, and every
pairwise Kendall tau to be exactly equal across the three invariant runs. It
writes `reports/kendall_tau_dblp_lp_invariant.csv` and fails closed otherwise.

The augmentation arm uses the same seeds and a maximum 1,000-update budget;
early stopping and the exact optimizer-step count are recorded:

```bash
python -m experiments.link_prediction.dblp_augmentation \
  --config configs/dblp_lp_augmentation.yaml \
  --output-root results/dblp_lp_augmentation
```

Add `--resume` after a time-limited interruption.

## Repository layout

```text
preprocessing/                 raw IMDb/DBLP -> shared and L=3 path tensors
pain_nc/                       PAIN node-classification and link models
configs/imdb_nc.yaml           faithful default experiment configuration
experiments/node_classification/
                               training and multi-variant benchmark
experiments/link_prediction/   DBLP training and seven-arm benchmark
configs/dblp_lp.yaml           corrected full-ranking DBLP contract
analysis/                      matched-seed Kendall tau analysis
scripts/hpc/                   cluster-side pipeline wrapper
```
