# PAIN Node Classification

This repository adapts the official PAIN (PAth Isomorphism Network) model from
*The Expressive Power of Path-Based Graph Neural Networks* to node
classification. Graph pooling is omitted at the model output; the final PAIN
node embeddings are passed to a node-wise classification head.

The current end-to-end benchmark is IMDb. DBLP and Freebase raw files are
reserved for later work and are not used by this pipeline.

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
the complete configuration, validation-selected epoch, metrics, test node IDs,
class probabilities (`y_prob`), predictions, and labels. Aggregate run and
mean/sample-standard-deviation tables are written beside those directories.
Existing run artifacts are reused unless `--overwrite` is supplied.

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

## Repository layout

```text
preprocessing/                 raw IMDb -> shared and exact L=3 path tensors
pain_nc/                       loader and PAIN node-classification model
configs/imdb_nc.yaml           faithful default experiment configuration
experiments/node_classification/
                               training and multi-variant benchmark
analysis/                      matched-seed Kendall tau analysis
scripts/hpc/                   cluster-side pipeline wrapper
```
