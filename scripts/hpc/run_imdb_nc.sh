#!/usr/bin/env bash
set -euo pipefail

# Activate the HPC Python/CUDA environment before invoking this script.
# Optional positional argument: results directory.
results_dir="${1:-results/imdb_nc}"

python -m preprocessing.imdb_node_classification
python -m experiments.node_classification.benchmark_imdb \
  --config configs/imdb_nc.yaml \
  --preflight-only
python -m experiments.node_classification.benchmark_imdb \
  --config configs/imdb_nc.yaml \
  --output-root "$results_dir"

