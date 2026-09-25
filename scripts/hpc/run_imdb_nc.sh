#!/usr/bin/env bash
set -euo pipefail

# Activate the HPC Python/CUDA environment before invoking this script.
# Optional positional arguments: independent and universal result directories.
results_dir="${1:-results/imdb_nc_movie_year}"
universal_results_dir="${2:-results/imdb_nc_universal_movie_year}"

python -m preprocessing.imdb_node_classification --no-validate
python -m experiments.node_classification.benchmark_imdb \
  --config configs/imdb_nc.yaml \
  --variants IMDb1 IMDb2 IMDb3 IMDb4 \
  --output-root "$results_dir"
python -m experiments.node_classification.benchmark_imdb \
  --config configs/imdb_nc.yaml \
  --variants IMDb_universal \
  --output-root "$universal_results_dir"

