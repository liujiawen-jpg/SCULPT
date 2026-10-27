# CoSIT — J0

Code for **cross-context perturbation response and single-cell distribution prediction**.
This repository contains the **J0 model with joint training**. It includes data
preparation, reproducible splits, training to convergence, validation selection,
prediction, evaluation, and comparison-method configurations.

Run the commands below from the repository root after activating the environment.

**Scope:** the same perturbation target is observed in source cell lines and
predicted in a held-out cell line, on a fixed set of shared genes. This benchmark
does not establish performance on unseen perturbation targets or unmeasured genes.

[中文入口](docs/README_CN.md) · [Data](docs/DATASETS.md) ·
[Generic priors](docs/PRIORS.md) · [Protocol](docs/BENCHMARK.md) ·
[Metrics](docs/METRICS.md) · [Comparison methods](Compared_methods/README.md)

## Model

```text
Source control + same-target perturbed cells     Target control cells
                 │                                      │
       Per-gene response statistics             Target background
       + sampling precision + joint moments            │
                 │                                      │
     Background-dependent observation operators        │
                 │                                      │
    Selective STRING connection + local PCA frames      │
                 │                                      │
         Joint shared-action inversion ──────────────────┤
                                                        │
                              Target-conditioned hurdle readout
                                         + positive quantile shape
                                         + learned rank permutation
                                                        │
                              512 × 5,979 expression cloud per target
                              + its actual finite-cloud mean
```

The fixed generic prior combines **Tahoe gene embeddings, GenePT, and STRING
diffusion**, projected to 256 dimensions with five availability masks. J0 does
not use WaveGC. Every task-specific weight is initialized from scratch for each
fold and seed. No old predictor or fixed source-mean answer is added to its output.
Measured target controls are legitimate biological inputs and provide the
baseline detection/positive-expression reference, cell prototypes, and shape
templates. Source observations inform the inferred action; they are not a fixed
additive prediction anchor.

All task components learn together in phase **J**. The release does not run the
older sequential A/B/C training procedure.

## Repository layout

```text
src/cosit/                  Portable data, training, inference, evaluation
  core/                     Original J0 mathematical model, solver and objectives
configs/
  j0.json                   Registered architecture and joint training recipe
  datasets.yaml             Verified sources, versions, hashes and preprocessing
  axes/                     Public gene/target/normalization identifiers
  mappings/                 Fixed identifier mappings; no expression measurements
  splits/F1 ... F5/         Frozen metadata-only benchmark partitions
Compared_methods/           TxPert, CPA, PIE and scPILOT configurations/adapters
scripts/                    Download, prior preparation and benchmark entrypoints
tests/                      CPU synthetic, invariance and leakage checks
docs/                       Detailed protocol and implementation notes
```

Expression matrices, cell caches, pretrained weights, checkpoints and experimental
predictions are **not included**. The small coordinate axes and split manifests
are necessary public metadata, not expression data.

## 1. Environment

Use Python **3.11**. For CUDA 12.8-compatible GPUs:

```bash
conda env create -f environment.yml
conda activate cosit-j0
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[dev]'
```

Alternatively, create a Python 3.11 virtual environment and install the PyTorch
wheel followed by `python -m pip install -e '.[dev]'`. The wheel must match your
GPU/driver; see the [official PyTorch installation options](https://pytorch.org/get-started/previous-versions/).
For a CPU installation, replace the wheel index with
`https://download.pytorch.org/whl/cpu`.

The release was checked with Python 3.11.13, PyTorch 2.7.1, NumPy 2.2.6,
SciPy 1.17.1, scikit-learn 1.9.0 and h5py 3.16.0. Formal training uses one CUDA
device at a time. CPU is supported for the synthetic checks; no full-benchmark
CPU runtime is claimed. Lower `execution.microbatch_size` in the configuration
to reduce peak GPU memory while preserving the effective balanced batch.

Check the installation without downloading data:

```bash
python -m cosit --help
python -m pytest -q tests Compared_methods/tests
python -m cosit smoke-test --output work/smoke
```

The smoke command exercises the actual model and complete pipeline on generated
cells. It does not measure biological predictive performance.

## 2. Download the five datasets

| Benchmark context | Study / source | Download |
|---|---|---|
| K562_essential | Replogle et al.; scPerturb | [H5AD](https://zenodo.org/records/13350497/files/ReplogleWeissman2022_K562_essential.h5ad?download=1) |
| RPE1 | Replogle et al.; scPerturb | [H5AD](https://zenodo.org/records/13350497/files/ReplogleWeissman2022_rpe1.h5ad?download=1) |
| HepG2 | Nadig et al.; scPerturb | [H5AD](https://zenodo.org/records/13350497/files/NadigOConner2024_hepg2.h5ad?download=1) |
| Jurkat | Nadig et al.; scPerturb | [H5AD](https://zenodo.org/records/13350497/files/NadigOConner2024_jurkat.h5ad?download=1) |
| Orion_HCT116 | Xaira X-Atlas/Orion | [Official dataset and HCT116 shards](https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion) |

The catalog pins the scPerturb Zenodo revision and Orion repository commit.
The downloader checks the four H5AD file sizes and published MD5 hashes, and
downloads only the HCT116 shards and their required gene metadata for Orion.

```bash
python scripts/download_data.py --help
python scripts/download_data.py --raw-root data/raw --dry-run
python scripts/download_data.py --raw-root data/raw
python -m cosit prepare --config configs/datasets.yaml \
  --raw-root data/raw --output data/processed
python -m cosit audit-data --data data/processed --axes configs/axes
```

For already downloaded files, arrange them as specified in
[DATASETS.md](docs/DATASETS.md) and run `prepare`. Preparation streams the raw
files, checks count semantics and stable gene IDs, applies the frozen control
partition and metadata-only cell sampling, and writes five immutable H5 files.

The expression unit is

\[
x_g=\log\!\left(1+10^4\frac{\mathrm{counts}_g}
{\sum_{a\in A_{2048}}\mathrm{counts}_a}\right).
\]

**Use the fixed A2048 denominator. Do not renormalize using all 5,979 output
genes or each file's gene panel.** Missing anchors, ambiguous IDs and insufficient
condition support raise an error. See the data guide for exact QC and sampling.

Raw source licenses apply independently: the four scPerturb files are CC BY 4.0;
Orion is **CC BY-NC-SA 4.0**. The code license does not grant commercial rights to
Orion data.

## 3. Prepare generic gene priors and STRING

Download the public Tahoe/GenePT resources, HGNC mappings, and STRING files using
the official links in [PRIORS.md](docs/PRIORS.md). The executable preparation
script builds the graph and the five-block generic representation:

```bash
python scripts/prepare_priors.py --help
python scripts/prepare_priors.py string --help
python scripts/prepare_priors.py prior --help
```

The guide provides complete commands and expected filenames. The resulting
inputs used below are:

```text
data/priors/prior.npz                    genes + embedding [genes,261]
data/string/graphs/KS/A.npz              STRING gene adjacency
data/string/source/node_axis.tsv        Graph stable-gene axis
```

These are generic external resources, not task-trained model checkpoints. They
are deliberately downloaded/prepared outside version control.

## 4. Generate and verify the splits

```bash
python -m cosit split --data data/processed --config configs/j0.json \
  --output work/splits --verify-frozen configs/splits
```

| Fold | Training contexts | Validation context | Test context |
|---|---|---|---|
| F1 | K562, HepG2, HCT116 | RPE1 | Jurkat |
| F2 | HepG2, Jurkat, HCT116 | K562 | RPE1 |
| F3 | RPE1, Jurkat, HCT116 | HepG2 | K562 |
| F4 | K562, RPE1, Jurkat | HCT116 | HepG2 |
| F5 | K562, RPE1, HepG2 | Jurkat | HCT116 |

Each fold also holds out 153 complete perturbation conditions from each training
context. These three target sets are disjoint. Held conditions are forbidden in
every gradient role, including source retrieval, negative banks, and fitted
response scales. Each fold has **4,116 legal training query conditions**.
Validation combines one external context (weight 1/2) and three internal held
condition panels (weight 1/6 each). Seeds 17, 29 and 43 change initialization and
training order, **not the data partitions**.

## 5. Train, select and predict

One fold and one initialization:

```bash
python -m cosit train \
  --data data/processed --priors data/priors/prior.npz \
  --graph data/string/graphs/KS/A.npz --graph-axis data/string/source/node_axis.tsv \
  --splits work/splits --fold F1 --seed 17 \
  --config configs/j0.json --output work/j0/F1/seed_17 --device cuda:0
```

Use the identical command with `--resume` to resume its own saved checkpoint.
It restores optimizer, stopping state, schedule position and random generators.
It never imports another method's weights or predictions.

All components train jointly. The five objective groups supervise response,
perturbation identity, marginals, joint structure and shared-action consistency.
Validation runs every two epochs. Selection minimizes the fixed multi-context
output objective; exact ties choose the earlier checkpoint. Training requires
the registered plateau, learning-rate reductions and minimum optimization budget.
An epoch cap produces `needs_decision`, **not a convergence certificate**.

After convergence, generate the held-out test clouds:

```bash
python -m cosit predict \
  --data data/processed --priors data/priors/prior.npz \
  --graph data/string/graphs/KS/A.npz --graph-axis data/string/source/node_axis.tsv \
  --run work/j0/F1/seed_17 --output work/j0/F1/seed_17/test_predictions.h5 --device cuda:0
```

`SELECTION.json` records the actual selected checkpoint path. Each test
perturbation receives **512 cells × 5,979 genes**, not only a mean vector.
The generated population is a deterministic finite quantile cloud, not an iid
sample. Its mean is `round(512*p)/512*r`; `p*r` is only an analytic diagnostic.
Rank permutation preserves every gene's emitted marginal exactly.

All test predictions and the selected checkpoint are sealed before test
responses can be read. Predictions cannot select or update a checkpoint.

## 6. Evaluate response and distribution

Use the checkpoint path written in `SELECTION.json`:

```bash
CHECKPOINT=$(python -c 'import json; print(json.load(open("work/j0/F1/seed_17/SELECTION.json"))["selected_checkpoint"])')
python -m cosit evaluate --data data/processed \
  --predictions work/j0/F1/seed_17/test_predictions.h5 \
  --checkpoint "$CHECKPOINT" --assets work/j0/F1/seed_17/assets \
  --output work/j0/F1/seed_17/test_metrics.json --device cpu
```

Response reporting includes **cosine-ranking PDS, Pearson, Spearman, DEG-AP,
JAC20/50/100, signed Top100 coverage, direction accuracy and amplitude ratio**.
Distribution reporting includes **marginal W1, CRPS, raw/centered sliced W1,
energy distance, Gaussian MMD², PCA covariance error, and support diagnostics**.
Raw MAE/MSE and the historical composite response score are omitted from the
primary report. See [METRICS.md](docs/METRICS.md) for definitions and limitations.

## 7. Run all five folds and three seeds

```bash
python scripts/run_benchmark.py \
  --data data/processed --priors data/priors/prior.npz \
  --graph data/string/graphs/KS/A.npz --graph-axis data/string/source/node_axis.tsv \
  --config configs/j0.json --output work/j0 --device cuda:0
```

The runner executes 15 independent training runs sequentially, resumes existing
runs, predicts only after convergence, and evaluates their selected checkpoints.
It does not stop other projects, choose a GPU automatically, or install a polling
monitor. `--folds F1 --seeds 17` runs a subset, explicitly marked incomplete.

The final `summary.json` first averages the three seeds within each context,
then reports the equally weighted five-context mean, **sample variance and
standard deviation (ddof=1)**. Initialization variance is also reported within
each fold. Missing metrics retain nulls and valid counts; mean-only methods do
not receive invented distribution scores.

## Comparison methods

[`Compared_methods/`](Compared_methods/README.md) contains the registered
**TxPert, CPA, PIE and scPILOT** configurations, pinned official commits,
environment setup, legal training-data export and native prediction bridges.
Upstream code is fetched on request rather than copied into this repository.

TxPert and PIE are mean-only in the registered configuration. CPA and scPILOT
have distribution outputs, with their benchmark boundary adaptations explicitly
documented. The directory distinguishes official APIs from our benchmark
adapters; it does not claim that a generic script faithfully replaces all four
native training pipelines. See each method's guide and license.

## Reproducibility and license

- [Implementation provenance and validation limits](docs/REPRODUCIBILITY.md)
- [Benchmark and access protocol](docs/BENCHMARK.md)
- [Prediction schema](docs/INTERFACE.md)
- [Contributing](CONTRIBUTING.md)

The original CoSIT code is [MIT licensed](LICENSE). External data, pretrained
resources and official comparison implementations retain their own licenses.
This repository is a portable source release: synthetic end-to-end checks and
core equivalence were verified; a new full 15-run benchmark was **not** performed
as part of packaging. No published paper DOI or new performance result is claimed.
