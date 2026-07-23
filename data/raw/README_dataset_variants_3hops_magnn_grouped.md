# Freebase Transitive-Edge Dataset Variants and MAGNN Preprocessing

This package contains legacy Freebase-style node-classification dataset variants and generated MAGNN preprocessing scripts. The goal is to compare the original graph against graph variants where selected transitive relations have been materialized as new one-hop edge types.

The scripts were generated with:

```bash
python generate_magnn_freebase_preprocess_scripts_center_and_allgraph_with_original_grouped.py \
  --variants-root ./dataset_variants_3hops \
  --script-outdir ./dataset_variants_3hops/magnn_preprocess_scripts \
  --center-type 0 \
  --seed 42 \
  --scopes both
```

Here, `center-type 0` means the centered MAGNN node-classification preprocessing is centered on the `BOOK` node type.

---

## 1. What is in this package?

The root directory is expected to look like this:

```text
dataset_variants_3hops/
  unchanged/
    info.dat
    node.dat
    link.dat
    label.dat
    variant_overview.json

  exact_2/
    info.dat
    node.dat
    link.dat
    label.dat
    new_transitive_relations.csv
    new_transitive_relations.json
    variant_overview.json

  exact_3/
    ...

  range_2_3/
    ...

  magnn_preprocess_scripts/
    unchanged/
      all_graph/
        original/preprocess_freebase_node.py

    exact_2/
      center/
        changed/preprocess_freebase_node.py
        unchanged/preprocess_freebase_node.py
      all_graph/
        changed/preprocess_freebase_node.py
        unchanged/preprocess_freebase_node.py

    exact_3/
      center/
        changed/preprocess_freebase_node.py
        unchanged/preprocess_freebase_node.py
      all_graph/
        changed/preprocess_freebase_node.py
        unchanged/preprocess_freebase_node.py

    range_2_3/
      ...
```

The exact list of variant folders depends on which dataset variants were created. For a 3-hop run, the usual variants are:

```text
unchanged/
exact_2/
exact_3/
range_2_3/
```

The `unchanged/` folder contains the original graph. Each changed variant folder contains the original graph plus new materialized transitive edge types.

---

## 2. Dataset file format

Each dataset variant follows the legacy Freebase-style format:

```text
info.dat
node.dat
link.dat
label.dat
```

### `info.dat`

Defines:

1. Node types, for example:

```text
TYPE    MEANING
0       BOOK
1       FILM
2       MUSIC
...
```

2. Link/relation types, for example:

```text
LINK    START   END     MEANING
0       0       0       BOOK-and-BOOK
1       0       1       BOOK-to-FILM
...
```

For changed variants, `info.dat` includes additional relation types for materialized transitive edges.

### `node.dat`

Each row gives a node id, node name, and node type:

```text
node_id    node_name    node_type
```

### `link.dat`

Each row gives a directed edge:

```text
source_node_id    target_node_id    relation_id    weight
```

The generated transitive relations are added as new relation ids in `link.dat`.

### `label.dat`

The original train/test label files were merged into a single `label.dat`. The generated preprocessing scripts create consistent train/validation/test splits from this combined label file using the fixed seed `42`.

---

## 3. What are the dataset variants?

Each changed dataset variant adds new materialized transitive relations.

For example, if a chain exists:

```text
BOOK --BOOK-about-ORGANIZATION--> ORGANIZATION --ORGANIZATION-to-MUSIC--> MUSIC
```

then a new one-hop transitive relation may be added:

```text
BOOK --BOOK-about-ORGANIZATION-to-MUSIC--> MUSIC
```

The reduced transitive relation name combines the chain names. For example:

```text
ORGANIZATION-to-MUSIC ; MUSIC-in-BOOK
```

becomes:

```text
ORGANIZATION-to-MUSIC-in-BOOK
```

The variants are organized by hop length:

| Variant | Meaning |
|---|---|
| `unchanged` | Original dataset only. No materialized transitive edges. |
| `exact_2` | Adds only transitive edges created from exactly 2-hop chains. |
| `exact_3` | Adds only transitive edges created from exactly 3-hop chains. |
| `range_2_3` | Adds transitive edges created from 2-hop and 3-hop chains. |

For larger runs, the same convention extends to `exact_4`, `exact_5`, `range_2_4`, `range_2_5`, etc.

---

## 4. What are the generated MAGNN preprocessing scripts?

For each changed variant, generated preprocessing scripts are provided under:

```text
magnn_preprocess_scripts/{variant}/{scope}/{mode}/preprocess_freebase_node.py
```

where:

- `{variant}` is something like `exact_2`, `exact_3`, or `range_2_3`.
- `{scope}` is either `center` or `all_graph`.
- `{mode}` is either `changed` or `unchanged`.

There is also a standalone all-original all-graph preprocessing script for the true unchanged dataset:

```text
magnn_preprocess_scripts/unchanged/all_graph/original/preprocess_freebase_node.py
```

This script uses only the original graph and only original one-hop metapaths. It does not include any variant-specific transitive chain metapaths.

Example:

```text
magnn_preprocess_scripts/exact_2/center/changed/preprocess_freebase_node.py
magnn_preprocess_scripts/exact_2/center/unchanged/preprocess_freebase_node.py
magnn_preprocess_scripts/exact_2/all_graph/changed/preprocess_freebase_node.py
magnn_preprocess_scripts/exact_2/all_graph/unchanged/preprocess_freebase_node.py
```

Each generated script is self-contained. It includes:

- imports
- relation-aware MAGNN-style preprocessing functions
- `expected_metapaths`
- `etypes_list`
- `save_prefix`
- code to create `.pkl`, `.npz`, `.npy`, `.adjlist`, and index files

The scripts are intended to be run by other users on their own machine/environment.

---

## 5. Center scope vs all-graph scope

Two preprocessing scopes are generated.

### 5.1 `center` scope

The `center` scope follows the usual MAGNN node-classification setup where metapaths start and end at a target node type. Here, the target node type is:

```text
0 = BOOK
```

So centered metapaths look like:

```python
(0, 1, 0)      # BOOK -> FILM -> BOOK
(0, 6, 0)      # BOOK -> ORGANIZATION -> BOOK
(0, 6, 5, 6, 0)  # BOOK -> ORGANIZATION -> LOCATION -> ORGANIZATION -> BOOK
```

This is the recommended mode for BOOK node classification.

### 5.2 `all_graph` scope

The `all_graph` scope creates relation-aware metapaths for the whole graph, not only BOOK-centered metapaths. This includes paths centered on any node type, for example:

```python
(4, 6, 4)      # PEOPLE -> ORGANIZATION -> PEOPLE
(6, 2, 6)      # ORGANIZATION -> MUSIC -> ORGANIZATION
(7, 0, 7)      # BUSINESS -> BOOK -> BUSINESS
```

This mode is useful if you want graph-wide metapath/edge-path files rather than only BOOK-centered node-classification inputs.

---

## 6. Changed vs unchanged preprocessing

For each changed variant, there are two corresponding preprocessing modes.

### 6.1 `changed`

The `changed` preprocessing uses the changed dataset variant. It includes:

1. Original one-hop metapaths.
2. Transitive chain metapaths.
3. Specialized transitive one-hop metapaths.

This means that if the variant added a new transitive relation such as:

```text
BOOK-about-ORGANIZATION-on-LOCATION
```

then the changed preprocessing includes both:

```python
(0, 6, 5, 6, 0)   # full transitive chain
(0, 5, 0)         # specialized transitive one-hop relation
```

Importantly, the specialized one-hop transitive metapath is kept distinct from any original one-hop relation with the same node-type pattern.

### 6.2 `unchanged`

The `unchanged` preprocessing uses the original dataset graph, but includes the transitive chain metapaths corresponding to the changed variant.

It includes:

1. Original one-hop metapaths.
2. Transitive chain metapaths.
3. No specialized transitive one-hop metapaths.

This gives a matched comparison:

- `unchanged`: original graph + chain metapaths only
- `changed`: original graph + chain metapaths + materialized transitive one-hop metapaths


### 6.3 Standalone `unchanged/all_graph/original`

The generator also creates a standalone all-graph preprocessing script for the true unchanged dataset:

```text
magnn_preprocess_scripts/unchanged/all_graph/original/preprocess_freebase_node.py
```

This is different from each variant's `all_graph/unchanged` script. The per-variant `all_graph/unchanged` script is a matched baseline that uses the original graph but still includes that variant's transitive chain metapaths. The standalone `unchanged/all_graph/original` script includes only the original graph and original one-hop metapaths. It is useful as a pure baseline for the entire original graph.

Its default save prefix is:

```text
data/preprocessed/freebase_node/unchanged/all_graph_original
```

---

## 7. Why relation-aware preprocessing is needed

MAGNN metapaths are often represented as node-type sequences. For example:

```python
(0, 5, 0)
```

could mean:

```text
BOOK -> LOCATION -> BOOK
```

However, in this dataset, two different relations can have the same node-type pattern:

```text
Original:    BOOK-on-LOCATION
Transitive:  BOOK-about-ORGANIZATION-on-LOCATION
```

Both may correspond to the node-type metapath:

```python
(0, 5, 0)
```

but they are not the same relation. The transitive one-hop edge is a specialized edge created from a longer chain.

Therefore, the generated preprocessing scripts use both:

```python
expected_metapaths
etypes_list
```

The `expected_metapaths` variable stores node-type sequences. The `etypes_list` variable stores relation/edge-type sequences. This allows the preprocessing code to distinguish:

```text
original BOOK -> LOCATION
```

from:

```text
transitive BOOK -> LOCATION
```

even when the node-type sequence is the same.

### Grouped 2D format

In the newest generated scripts, both variables are **2D lists grouped by starting/target node type**:

```python
expected_metapaths = [
    [... metapaths starting at BOOK/type 0 ...],
    [... metapaths starting at another node type ...],
    ...
]

etypes_list = [
    [... edge-type lists corresponding to expected_metapaths[0] ...],
    [... edge-type lists corresponding to expected_metapaths[1] ...],
    ...
]
```

The first group is always for the BOOK node type:

```python
metapath_group_target_types[0] == 0
expected_metapaths[0]  # BOOK/type-0 metapaths
etypes_list[0]         # edge-type sequences for those BOOK metapaths
```

For `center` scope, this first group is usually the only non-empty group because all metapaths are BOOK-centered. For `all_graph` scope, later groups correspond to the remaining start/target node types, with their ids recorded in:

```python
metapath_group_target_types
metapath_group_indices
```

The generated scripts also keep convenience flat versions for debugging:

```python
expected_metapaths_flat
etypes_list_flat
```

The preprocessing loop still uses `metapath_defs`, so this grouping is mainly for MAGNN-style configuration and downstream model code that expects metapaths organized by target node type.

---

## 8. Running the generated preprocessing scripts

From the repository/package root, run an individual generated script:

```bash
python dataset_variants_3hops/magnn_preprocess_scripts/exact_2/center/changed/preprocess_freebase_node.py
```

Or for the corresponding unchanged version:

```bash
python dataset_variants_3hops/magnn_preprocess_scripts/exact_2/center/unchanged/preprocess_freebase_node.py
```

For all-graph preprocessing of a changed variant:

```bash
python dataset_variants_3hops/magnn_preprocess_scripts/exact_2/all_graph/changed/preprocess_freebase_node.py
```

For all-graph preprocessing of the true original/unchanged dataset with no transitive chain metapaths:

```bash
python dataset_variants_3hops/magnn_preprocess_scripts/unchanged/all_graph/original/preprocess_freebase_node.py
```

Each script writes outputs to a `save_prefix` like:

```text
data/preprocessed/freebase_node/{variant}/{scope}_{mode}
```

For example:

```text
data/preprocessed/freebase_node/exact_2/center_changed
```

---

## 9. Output files created by each preprocessing script

Each generated preprocessing script creates files similar to the original MAGNN preprocessing notebooks.

Expected outputs include:

```text
features_{type}.npz
node_types.npy
type_mask.npy
labels.npy
labels_global.npy
train_val_test_idx.npz
preprocess_outputs.pkl
metapath_config.json
preprocessing_overview.json
```

It also creates metapath-specific files:

```text
adjlists/{metapath}.adjlist
idx/{metapath}_idx.npy
{metapath}_idx.npy
target_idx.npy
target_idx.pickle
target_idx/{metapath}_target_idx.npy
target_idx/{metapath}_target_idx.pickle
```

The `.adjlist` files store metapath-induced adjacency lists. The `idx.npy` files store concrete node-path instances for each metapath.

---

## 10. Features

The generated preprocessing scripts create simple node features for each node type:

```text
features_{type}.npz
```

These are simple identity-style sparse features within each node type. They are intended as a lightweight default so the dataset can be run without additional attribute features.

---

## 11. Labels and splits

The dataset variants use a combined `label.dat`. The generated preprocessing scripts create train/validation/test splits from this combined label file.

The split is deterministic because the generator was run with:

```text
--seed 42
```

As long as the same seed and split ratios are used, the split should be consistent across all variants.

This is important for fair comparisons between:

```text
unchanged_corresponding_preprocessing
variant_preprocessing
```

or between:

```text
center/changed
center/unchanged
```

---

## 12. Self-reference metapaths

Self-reference relations are supported. For example:

```text
BOOK-and-BOOK
```

can produce a centered metapath:

```python
(0, 0, 0)
```

The generated preprocessing code follows the actual relation/edge chain using `etypes_list`, rather than treating `(0, 0, 0)` as a generic type-only pattern.

---

## 13. Recommended workflow for recipients

1. Unzip the package.
2. Install dependencies:

```bash
pip install numpy scipy networkx
```

3. Run the desired generated preprocessing script:

```bash
python dataset_variants_3hops/magnn_preprocess_scripts/exact_2/center/changed/preprocess_freebase_node.py
```

4. Check the output folder:

```text
data/preprocessed/freebase_node/exact_2/center_changed
```

5. Use the generated `expected_metapaths` and `etypes_list` from the generated script or from:

```text
metapath_config.json
```

when configuring MAGNN experiments.

The `metapath_config.json` and `preprocess_outputs.pkl` files contain both grouped and flat metapath variables:

```text
expected_metapaths        # grouped 2D list by target/start node type
etypes_list               # grouped 2D list aligned with expected_metapaths
metapath_group_target_types
metapath_group_indices
expected_metapaths_flat
etypes_list_flat
```

---

## 14. Notes and caveats

- `etypes_list` is intentionally included even though the original MAGNN preprocessing notebooks do not always define it directly.
- The preprocessing is relation-aware because type-only metapaths are insufficient for distinguishing original one-hop edges from specialized transitive one-hop edges.
- The changed and unchanged preprocessing outputs are designed as matched comparisons for each variant.
- For very dense metapaths, preprocessing can produce many concrete path instances. If needed, adjust the path cap in the generated preprocessing script or regenerate scripts with a lower max-path setting.

---

## 15. Quick command examples

Generate BOOK-centered preprocessing for `exact_2` changed:

```bash
python dataset_variants_3hops/magnn_preprocess_scripts/exact_2/center/changed/preprocess_freebase_node.py
```

Generate BOOK-centered preprocessing for `exact_2` unchanged comparison:

```bash
python dataset_variants_3hops/magnn_preprocess_scripts/exact_2/center/unchanged/preprocess_freebase_node.py
```

Generate all-graph preprocessing for `range_2_3` changed:

```bash
python dataset_variants_3hops/magnn_preprocess_scripts/range_2_3/all_graph/changed/preprocess_freebase_node.py
```

Generate all-graph preprocessing for `range_2_3` unchanged comparison:

```bash
python dataset_variants_3hops/magnn_preprocess_scripts/range_2_3/all_graph/unchanged/preprocess_freebase_node.py
```

Generate all-graph preprocessing for the true original unchanged dataset with no transitive chains:

```bash
python dataset_variants_3hops/magnn_preprocess_scripts/unchanged/all_graph/original/preprocess_freebase_node.py
```
