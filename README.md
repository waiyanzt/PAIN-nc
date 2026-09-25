# PAIN experiments

This repository adapts the official PAIN (PAth Isomorphism Network) model to
the experiments used by this project. Node-classification runs keep PAIN's
node embeddings, omit graph pooling, and classify each target node.

The standard seeds are `1566911444`, `20241017`, and `20251017`. Run commands
from the repository root. Existing completed seed artifacts are skipped unless
`--overwrite` is supplied.

## HPC environment

Load Python and CUDA inside every Slurm job, then activate the repository
environment:

```bash
module add python/3.11 cuda/12.6

cd ~/hpc-share/PAIN-nc
source .venv/bin/activate
```

Create the environment once after cloning or updating the repository:

```bash
module add python/3.11 cuda/12.6
cd ~/hpc-share/PAIN-nc

uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Place the same module and activation commands before the experiment command in
each Slurm job. Freebase preprocessing is CPU- and storage-heavy; its H100 is
mainly useful during training.

## Expected data layout

```text
data/raw/IMDB/movie_metadata.csv
data/raw/DBLP/
data/raw/freebase_node/dataset_variant_3hops_filter/
  unchanged/
  exact_2/
  exact_3/
  union_exact_2_3/
```

For Freebase, `exact_3` and `union_exact_2_3` must be the reconstructed outputs
from `range_2_3`; `range_2_3` itself is not the universal graph. The PAIN
preprocessors below consume those directories but do not rebuild the raw
Freebase variants.

## IMDb node classification

The current IMDb contract contains Movie, Director, Actor, Link, and Year
nodes. Movie--Year is present in every physical variant; only the
Director/Actor attachments move around the Movie--Link bridge.

Preprocess the four physical variants and universal graph:

```bash
python -m preprocessing.imdb_node_classification --no-validate
```

Build the four independently compiled invariant artifacts:

```bash
python -m preprocessing.imdb_invariant \
  --output-dir data/preprocessed/IMDB_invariant \
  --no-validate
```

Run the four independent variants:

```bash
python -m experiments.node_classification.benchmark_imdb \
  --config configs/imdb_nc.yaml \
  --variants IMDb1 IMDb2 IMDb3 IMDb4 \
  --seeds 1566911444 20241017 20251017 \
  --output-root results/imdb_nc_movie_year
```

Run the universal graph:

```bash
python -m experiments.node_classification.benchmark_imdb \
  --config configs/imdb_nc.yaml \
  --variants IMDb_universal \
  --seeds 1566911444 20241017 20251017 \
  --output-root results/imdb_nc_universal_movie_year
```

Run joint v1-v4 augmentation:

```bash
python -m experiments.node_classification.imdb_augmentation \
  --config configs/imdb_nc_augmentation.yaml \
  --variants v1 v2 v3 v4 \
  --seeds 1566911444 20241017 20251017 \
  --output-root results/imdb_nc_augmentation_movie_year
```

Run the invariant arm:

```bash
python -m experiments.node_classification.benchmark_imdb \
  --config configs/imdb_nc_invariant.yaml \
  --variants IMDb_invariant_v1 IMDb_invariant_v2 \
             IMDb_invariant_v3 IMDb_invariant_v4 \
  --seeds 1566911444 20241017 20251017 \
  --output-root results/imdb_nc_invariant_movie_year
```

## DBLP link prediction

> **WIP / optional:** This pipeline is retained for development. It is not
> required and is not included in the current experiment set.

DBLP uses deterministic sampled length-2/3 paths with a budget of 256 per root.
Preprocess the physical variants and universal graph, then preprocess the
invariant artifacts into a separate directory:

```bash
python -m preprocessing.dblp_link_prediction \
  --mode baseline \
  --output-dir data/preprocessed/DBLP \
  --max-paths-per-root 256

python -m preprocessing.dblp_link_prediction \
  --mode invariant \
  --output-dir data/preprocessed/DBLP_invariant \
  --max-paths-per-root 256 \
  --sampling-seed 1566911444
```

Run the independent variants and universal graph:

```bash
python -m experiments.link_prediction.benchmark_dblp \
  --config configs/dblp_lp.yaml \
  --variants DBLP1 DBLP2 DBLP3 DBLP_universal \
  --seeds 1566911444 20241017 20251017 \
  --output-root results/dblp_lp
```

Run augmentation:

```bash
python -m experiments.link_prediction.dblp_augmentation \
  --config configs/dblp_lp_augmentation.yaml \
  --seeds 1566911444 20241017 20251017 \
  --output-root results/dblp_lp_augmentation \
  --resume
```

Run the invariant arm:

```bash
python -m experiments.link_prediction.benchmark_dblp \
  --config configs/dblp_lp_invariant.yaml \
  --variants DBLP_invariant_v1 DBLP_invariant_v2 DBLP_invariant_v3 \
  --seeds 1566911444 20241017 20251017 \
  --output-root results/dblp_lp_invariant
```

## Freebase node classification

Freebase is much larger, so both ordinary and invariant preprocessing use
bounded deterministic path sampling. Report these experiments as **sampled
PAIN**. Preprocessing can be resumed after interruption.

Preprocess the three physical variants and universal graph:

```bash
python -m preprocessing.freebase_node_classification \
  --variants-root data/raw/freebase_node/dataset_variant_3hops_filter \
  --output-dir data/preprocessed/Freebase \
  --variants unchanged exact_2 exact_3 union_exact_2_3 \
  --max-neighbors-per-relation 4 \
  --path-fanout 8 \
  --max-paths-per-root 256 \
  --sampling-seed 1566911444 \
  --split-seed 1566911444 \
  --resume
```

Preprocess the invariant arm:

```bash
python -m preprocessing.freebase_invariant \
  --variants-root data/raw/freebase_node/dataset_variant_3hops_filter \
  --output-dir data/preprocessed/Freebase_invariant \
  --variants unchanged exact_2 exact_3 \
  --path-length 3 \
  --max-neighbors-per-relation 4 \
  --path-fanout 8 \
  --max-paths-per-root 256 \
  --sampling-seed 1566911444 \
  --split-seed 1566911444 \
  --resume
```

Run the three independent variants:

```bash
python -m experiments.node_classification.benchmark_freebase \
  --config configs/freebase_nc.yaml \
  --variants Freebase1 Freebase2 Freebase3 \
  --seeds 1566911444 20241017 20251017 \
  --output-root results/freebase_nc
```

Run joint Freebase augmentation with one shared model per seed. Each
super-epoch makes one optimizer update on each of Freebase1, Freebase2, and
Freebase3, then validates the shared model on all three. The default cap of
100 super-epochs is 300 updates per seed. The runner saves its training state
after every super-epoch, so repeat the same command with `--resume` after a
time limit or interruption. It needs all three physical variant artifacts and
`shared.pt` in `data/preprocessed/Freebase/`. Use `--preflight-only` to check
those artifacts before starting a GPU job.

```bash
python -m experiments.node_classification.freebase_augmentation \
  --config configs/freebase_nc_augmentation.yaml \
  --seeds 1566911444 20241017 20251017 \
  --output-root results/freebase_nc_augmentation \
  --resume
```

Run the universal graph:

```bash
python -m experiments.node_classification.benchmark_freebase \
  --config configs/freebase_nc.yaml \
  --variants Freebase_universal \
  --seeds 1566911444 20241017 20251017 \
  --output-root results/freebase_nc_universal
```

Run the invariant arm:

```bash
python -m experiments.node_classification.freebase_invariant \
  --config configs/freebase_nc_invariant.yaml \
  --variants Freebase_invariant1 Freebase_invariant2 Freebase_invariant3 \
  --seeds 1566911444 20241017 20251017 \
  --output-root results/freebase_nc_invariant
```

## Hyperparameters

The checked-in YAML files are the source of truth. The main settings are:

| Experiment | Training budget | Early stopping | Learning rate | Path program |
|---|---:|---:|---:|---|
| IMDb independent, universal, invariant | 1,000 epochs | 80 epochs | 0.001 | exact length 0-3 |
| IMDb augmentation | 250 super-epochs / 1,000 updates | 20 super-epochs | 0.001 | exact length 0-3 |
| Freebase independent, universal, invariant | 300 epochs | 30 epochs | 0.001 | sampled length 0-3, cap 256/root |
| Freebase augmentation | 100 super-epochs / 300 updates | 10 super-epochs | 0.001 | sampled length 0-3, cap 256/root |
| DBLP independent, universal, invariant (WIP) | 1,000 epochs | 200 epochs | 0.005 | sampled length 0-3, cap 256/root |
| DBLP augmentation (WIP) | 1,000 updates | 67 validation checks | 0.005 | sampled length 0-3, cap 256/root |

All configurations use hidden size 128, five PAIN layers, a two-layer LSTM,
shared LSTM weights, reversed paths, sum aggregation, and gradient clipping at
5.0. IMDb streams chunks of 100,000 paths on the GPU. Freebase and DBLP stream
chunks of 50,000 paths from host memory. Freebase additionally caps neighbors
per semantic relation at 4 and uses path fanout 8.

The wall-clock limits in the configs are 12 hours for IMDb, 48 hours for
Freebase, and 22 hours for the optional DBLP pipeline. The three standard seeds
must use identical preprocessing and hyperparameters.

## Build the paper tables

The builder reads individual per-seed artifacts rather than the aggregate CSV
files written during training. This prevents a later partial run from silently
replacing the complete aggregate. After copying the result directories into
`results/`, run:

```bash
python build_paper_tables.py
```

It writes the following under `paper_tables/pain/`:

- aggregate CSV tables;
- Overleaf-ready LaTeX tables;
- matched-seed Kendall-tau comparisons;
- `PAIN_COMPLETENESS.md`, which lists every loaded, missing, or unreadable run.

Missing seeds remain visibly marked as partial rather than being silently
averaged as a complete result. The default input paths match the result roots
shown in this README. Use `--output-dir <directory>` to change the destination
or `--seeds <seed ...>` to select another seed set.

The current builder covers IMDb node classification and the optional DBLP
link-prediction artifacts. Freebase is not yet included in the table builder.

## Notes

- Do not mix GPU architectures across matched invariant runs.
- Use the same preprocessing budgets and seeds for all compared arms.
- `--overwrite` discards the normal completed-run skip behavior; use it only
  when intentionally replacing an artifact.
- The architecture and adaptation details are recorded in
  [`docs/PAIN_FIDELITY.md`](docs/PAIN_FIDELITY.md).
