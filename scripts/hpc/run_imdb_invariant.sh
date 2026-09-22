#!/usr/bin/env bash
set -euo pipefail

# Activate the HPC Python/CUDA environment before invoking this script.
# Optional positional argument: results directory.
results_dir="${1:-results/imdb_nc_invariant}"

variants=(
  IMDb_invariant_v1
  IMDb_invariant_v2
  IMDb_invariant_v3
  IMDb_invariant_v4
)

python -m preprocessing.imdb_invariant \
  --output-dir data/preprocessed/IMDB_invariant
python -m experiments.node_classification.benchmark_imdb \
  --config configs/imdb_nc_invariant.yaml \
  --variants "${variants[@]}" \
  --preflight-only
python -m experiments.node_classification.benchmark_imdb \
  --config configs/imdb_nc_invariant.yaml \
  --variants "${variants[@]}" \
  --output-root "$results_dir"
python -m analysis.validate_imdb_invariant \
  --metadata data/preprocessed/IMDB_invariant/metadata.json \
  --root "$results_dir" \
  --output reports/imdb_nc_invariant_audit.csv
