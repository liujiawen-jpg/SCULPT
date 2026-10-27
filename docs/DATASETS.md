# Dataset acquisition and preparation

The J0 benchmark uses five **cell-line contexts from three studies**. It is a
same-perturbation, unseen-context transfer benchmark: the tested perturbation
has measured source-cell-line evidence, and only the target cell line's control
cells are available to the predictor. It is not a novel-target benchmark.

No expression matrix, cell cache, task checkpoint, or model prediction is
distributed in this repository. Downloads are from the original public data
providers. The links, file sizes, and checksums below were checked against their
official records on 2026-10-26.

## Five dataset download addresses

| Context identifier | Study | Exact input | Download |
| --- | --- | --- | --- |
| `K562_essential` | Replogle et al., 2022 | Essential-gene CRISPRi screen, **not** K562 genome-wide | [H5AD, 1.547 GB](https://zenodo.org/records/13350497/files/ReplogleWeissman2022_K562_essential.h5ad?download=1) |
| `RPE1` | Replogle et al., 2022 | RPE1 CRISPRi | [H5AD, 1.237 GB](https://zenodo.org/records/13350497/files/ReplogleWeissman2022_rpe1.h5ad?download=1) |
| `HepG2` | Nadig et al., 2025 | HepG2 CRISPRi; the harmonized filename retains `2024` | [H5AD, 0.851 GB](https://zenodo.org/records/13350497/files/NadigOConner2024_hepg2.h5ad?download=1) |
| `Jurkat` | Nadig et al., 2025 | Jurkat CRISPRi; the harmonized filename retains `2024` | [H5AD, 1.294 GB](https://zenodo.org/records/13350497/files/NadigOConner2024_jurkat.h5ad?download=1) |
| `Orion_HCT116` | Huang et al., 2025, X-Atlas/Orion | All 109 HCT116 Parquet shards, plus the gene-token metadata | [Official dataset](https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion/tree/53a5bc98d49247bcf967500292575c3d3602de31) |

The first four files are pinned to [scPerturb v1.4, Zenodo 13350497](https://zenodo.org/records/13350497).
Their source license is **CC BY 4.0**. Orion is pinned to commit
`53a5bc98d49247bcf967500292575c3d3602de31` of the official Xaira repository;
its license is **CC BY-NC-SA 4.0**, including its noncommercial condition. Source
data retain their original licenses; the software license does not relicense
these datasets. The Orion authors also provide [processed files on Figshare](https://doi.org/10.25452/figshare.plus.29190726),
but these are not substituted for the pinned Parquet input.

### Download commands

```bash
# Inspect URLs, destinations, checksums, and the pinned Orion revision first.
python scripts/download_data.py --config configs/datasets.yaml \
  --raw-root data/raw --dry-run

# All five contexts. This downloads expression data, not model weights.
python scripts/download_data.py --config configs/datasets.yaml --raw-root data/raw

# Download a subset when preparing one context at a time.
python scripts/download_data.py --raw-root data/raw --contexts K562_essential RPE1
```

Existing scPerturb files must pass both size and MD5 checks. New files are
written to `.partial` and renamed only after verification. The Orion downloader
uses `huggingface_hub.snapshot_download` at the pinned revision, includes only
HCT116 shards and necessary gene metadata, and checks that all 109 shards exist.
It does not download the HEK293T screen. The full HCT116 input is substantially
larger than the four H5AD inputs: reserve enough disk space before downloading.

## Expected raw layout

```text
data/raw/
  scPerturb/
    ReplogleWeissman2022_K562_essential.h5ad
    ReplogleWeissman2022_rpe1.h5ad
    NadigOConner2024_hepg2.h5ad
    NadigOConner2024_jurkat.h5ad
  X-Atlas-Orion/
    data/HCT116_Batch1.parquet
    ...                         # all 109 original shards
    metadata/gene_metadata.parquet
```

Already downloaded source files can be reused with symbolic links in this
layout; there is no need to duplicate large files. Every raw source is opened
read-only. Output goes to a separate directory.

## Gene and target identity

The released benchmark axes are ordered stable Ensembl IDs:

* `configs/axes/primary_genes.tsv`: 5,979 jointly measured output genes.
* `configs/axes/targets.tsv`: 1,525 perturbation targets shared across contexts.
* `configs/axes/A2048.tsv`: the fixed 2,048-gene normalization panel.
* `configs/axes/target_symbols.tsv`: the frozen target-symbol-to-stable-ID mapping.
* `configs/mappings/<context>.tsv`: source gene identity metadata for the frozen
  panel. These text files contain identifiers and symbols, **no expression**.

The preparer requires every output gene and every normalization anchor to map
uniquely in each source. It fails on missing or ambiguous identities. It never
imputes absent genes with zero, collapses ambiguous aliases by averaging, or
recomputes the output panel from observed response strength. Ensembl version
suffixes are stripped before checking identity. H5AD perturbations use
`obs['gene_id']`, with the frozen symbol mapping only as a fallback; Orion
perturbations use that same frozen symbol mapping from `gene_target`.

## Expression unit: preserve the A2048 denominator

For raw count vector `x`, output gene `g`, and fixed anchor set `A`, the unit is

```text
y[g] = log1p(10000 * x[g] / sum(x[a] for a in A2048)).
```

In the released J0 panel, all A2048 genes are also included in the 5,979 output
genes. The normalization still sums **only the fixed A2048 panel**, not all
5,979 genes; changing an output query panel must not change this denominator.

The input matrix must contain finite, nonnegative **integer counts**. The four
specified scPerturb files use raw counts in `X`; Orion's `gene_expression` lists
are raw counts paired with `gene_token_id` as documented in the
[official data card](https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion).
Already normalized scPertEval/GCS derivatives, population z-scores, and log
matrices are not interchangeable inputs. The preparer rejects noninteger or
negative counts and excludes rows with a zero anchor denominator.

Do not apply `normalize_total` to the 5,979 output genes, exponentiate the
expression twice, or substitute total native-library counts for the anchor
sum. All methods must be evaluated in the same expression unit.

## Cell QC, control roles, and reproducible finite cell caches

Preparation uses metadata and count validity, never response magnitude or model
performance, to determine eligible cells:

1. H5AD controls have `perturbation == 'control'` and `nperts == 0`; single-target
   cells have `nperts == 1` and a uniquely mapped benchmark target.
2. Counts must be finite nonnegative integers, the native library must be
   nonempty, and both metadata and raw counts must show at least 200 detected
   genes. The A2048 denominator must be positive.
3. Orion additionally requires `pass_guide_filter == 1`. Perturbed cells must
   have exactly two guide entries, both matching the assigned target. Controls
   have `gene_target == 'Non-Targeting'`.
4. The historical E/L control split is reproduced. Only the **E** partition is
   used as model background: for the four H5AD datasets, controls are sorted
   within each batch by `sha256('1701:' + context + ':' + barcode)`, with the
   first `floor(n/2)` assigned E. Orion uses its original sample-stratified
   `v2-seed17` hash and first `ceil(n/2)` assigned E. L is not mixed into E.
5. All eligible E cells determine `control_mean`, `control_variance`, and
   `control_detection`. A capped E cloud supplies higher empirical statistics.
   Pseudo-bulk control statistics are therefore **not** recomputed from only
   the capped cloud.
6. Each perturbation and the E cloud are capped at 256 cells using the original
   seed-17 metadata hash and largest-remainder batch/guide allocation. This
   conserves observed batch/guide proportions. Fewer than 20 eligible cells for
   any fixed target stops preparation instead of silently removing it.

The code supports dense H5AD matrices and CSR count matrices/layers. A custom
layer can be selected with `matrix: layers/counts` in the catalog. CSC matrices
must first be converted to CSR; the distributed source files are dense and do
not require conversion. Preparation streams bounded blocks and does not hold
the full Orion expression matrix in memory.

```bash
python -m cosit prepare --config configs/datasets.yaml \
  --raw-root data/raw --output data/processed
```

## Prepared file contract

Each `data/processed/<context>.h5` contains:

```text
attrs:
  schema = 'cosit_cells_v1'
  units = 'log1p(10000*counts/sum_A2048_counts)'
  genes_json, targets_json, anchors_json
  control_n_full, cell_cap, selection_seed
  provenance_json, complete
controls                         float32 [<=256, 5979]
control_ids                      UTF-8 [n_control_cached]
control_mean                     float32 [5979], full E
control_variance                 float32 [5979], population variance, full E
control_detection                float32 [5979], full E
perturbations/<target_stable_id>  float32 [20..256, 5979]
perturbation_ids/<target_id>      UTF-8 [n_cached]
```

The preparer atomically publishes each complete file and writes a SHA-256
`MANIFEST.json`. Fold membership and task access are defined by the
separate split generator. The existence of a test context in prepared storage
does not authorize its perturbation labels during training or prediction.

## Citations and provenance

Please cite the data-generating studies and the harmonization resource:

* Replogle et al., *Mapping information-rich genotype-phenotype landscapes with
  genome-scale Perturb-seq*, Cell (2022),
  [doi:10.1016/j.cell.2022.05.013](https://doi.org/10.1016/j.cell.2022.05.013).
  [Original processed-data record](https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387).
* Nadig et al., *Transcriptome-wide analysis of differential expression in
  perturbation atlases*, Nature Genetics (2025),
  [doi:10.1038/s41588-025-02169-3](https://doi.org/10.1038/s41588-025-02169-3).
  The exact downloaded versions are the `NadigOConner2024_*` files in the pinned
  scPerturb record; do not change filenames to infer a different release.
* Peidli et al., *scPerturb: harmonized single-cell perturbation data*,
  Nature Methods (2024),
  [doi:10.1038/s41592-023-02144-y](https://doi.org/10.1038/s41592-023-02144-y).
* Huang et al., *X-Atlas/Orion: Genome-wide Perturb-seq Datasets via a Scalable
  Fix-Cryopreserve Platform for Training Dose-Dependent Biological Foundation
  Models* (2025),
  [doi:10.1101/2025.06.11.659105](https://doi.org/10.1101/2025.06.11.659105).
