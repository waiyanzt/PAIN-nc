#!/usr/bin/env bash
set -euo pipefail

# Activate the HPC Python/CUDA environment before invoking this script.
# Optional positional argument: results directory.
results_dir="${1:-results/imdb_nc_augmentation}"

python -m preprocessing.imdb_node_classification --variants v1 v2 v3 v4
python -m experiments.node_classification.imdb_augmentation \
  --config configs/imdb_nc_augmentation.yaml \
  --variants v1 v2 v3 v4 \
  --preflight-only
python -m experiments.node_classification.imdb_augmentation \
  --config configs/imdb_nc_augmentation.yaml \
  --variants v1 v2 v3 v4 \
  --output-root "$results_dir"
