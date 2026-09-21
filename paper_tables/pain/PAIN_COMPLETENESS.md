# PAIN paper-table completeness

Expected seeds: `1566911444, 20241017, 20251017`.

The builder reads per-run artifacts. It does not use the overwrite-prone top-level aggregate CSV files.

| Dataset/task | Method | Variant | Loaded | Missing or unreadable seeds |
|---|---|---:|---:|---|
| DBLP/LP | Augmentation | v1 | 0/3 | 1566911444 (unreadable), 20241017 (unreadable), 20251017 (unreadable) |
| DBLP/LP | Augmentation | v2 | 0/3 | 1566911444 (unreadable), 20241017 (unreadable), 20251017 (unreadable) |
| DBLP/LP | Augmentation | v3 | 0/3 | 1566911444 (unreadable), 20241017 (unreadable), 20251017 (unreadable) |
| DBLP/LP | Independent | v1 | 0/3 | 1566911444 (unreadable), 20241017 (unreadable), 20251017 (unreadable) |
| DBLP/LP | Independent | v2 | 0/3 | 1566911444 (unreadable), 20241017 (unreadable), 20251017 (unreadable) |
| DBLP/LP | Independent | v3 | 0/3 | 1566911444 (unreadable), 20241017 (unreadable), 20251017 (unreadable) |
| DBLP/LP | Invariant | v1 | 0/3 | 1566911444 (missing), 20241017 (unreadable), 20251017 (missing) |
| DBLP/LP | Invariant | v2 | 0/3 | 1566911444 (missing), 20241017 (unreadable), 20251017 (missing) |
| DBLP/LP | Invariant | v3 | 0/3 | 1566911444 (missing), 20241017 (unreadable), 20251017 (missing) |
| DBLP/LP | Universal graph | universal | 0/3 | 1566911444 (unreadable), 20241017 (unreadable), 20251017 (unreadable) |
| IMDB/NC | Augmentation | v1 | 3/3 | -- |
| IMDB/NC | Augmentation | v2 | 3/3 | -- |
| IMDB/NC | Augmentation | v3 | 3/3 | -- |
| IMDB/NC | Augmentation | v4 | 3/3 | -- |
| IMDB/NC | Independent | v1 | 0/3 | 1566911444 (unreadable), 20241017 (unreadable), 20251017 (unreadable) |
| IMDB/NC | Independent | v2 | 0/3 | 1566911444 (unreadable), 20241017 (unreadable), 20251017 (unreadable) |
| IMDB/NC | Independent | v3 | 0/3 | 1566911444 (unreadable), 20241017 (unreadable), 20251017 (unreadable) |
| IMDB/NC | Independent | v4 | 0/3 | 1566911444 (unreadable), 20241017 (unreadable), 20251017 (unreadable) |
| IMDB/NC | Invariant | v1 | 0/3 | 1566911444 (unreadable), 20241017 (unreadable), 20251017 (missing) |
| IMDB/NC | Invariant | v2 | 0/3 | 1566911444 (missing), 20241017 (missing), 20251017 (missing) |
| IMDB/NC | Invariant | v3 | 0/3 | 1566911444 (missing), 20241017 (missing), 20251017 (missing) |
| IMDB/NC | Invariant | v4 | 0/3 | 1566911444 (missing), 20241017 (missing), 20251017 (missing) |
| IMDB/NC | Universal graph | universal | 0/3 | 1566911444 (unreadable), 20241017 (unreadable), 20251017 (unreadable) |

## Source policy

- IMDb independent v1--v3: `results/imdb_nc/`.
- IMDb independent v4: `results/imdb_nc_v4_fixed/` (the earlier v4 is intentionally ignored).
- IMDb augmentation: `results/imdb_nc_augmentation_v4_fixed/` (the earlier augmentation root is intentionally ignored).
- IMDb universal and invariant: their dedicated result roots.
- DBLP independent and universal: `results/dblp_lp/`; augmentation and invariant: their dedicated result roots.

## LaTeX use

Add `\usepackage{booktabs}` and `\usepackage{graphicx}` to the paper preamble, then use e.g. `\input{tables/PAIN_IMDB_nc_results.tex}` after uploading the desired `.tex` files to Overleaf.

Rows marked with a dagger are provisional because fewer than all requested seeds were available.
