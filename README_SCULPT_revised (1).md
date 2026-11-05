# SCULPT: Joint Inversion of Context-Specific Perturbation Responses for Cross-Context Prediction

**SCULPT** predicts perturbation responses and reconstructs single-cell
populations in an unseen cellular context. Instead of directly
transferring observed source phenotypes, SCULPT jointly infers a shared
latent perturbation action from multiple source contexts and
re-expresses it relative to the target control background.

```{=html}
<!-- TODO: Replace this path with the final overview/method figure. -->
```
![SCULPT overview](fig/SCULPT_overview.png)

## How to run this code

Run all commands below from the repository root after activating the
environment.

The main workflow is:

1.  Install the environment.
2.  Download and preprocess the five benchmark datasets.
3.  Prepare generic gene priors and the STRING network.
4.  Generate and verify the frozen benchmark splits.
5.  Train the model and select a checkpoint.
6.  Predict the held-out cell populations and evaluate the results.

For a concise reproduction checklist and additional details, see:

-   [Quick reproduction checklist](docs/REPRODUCTION_CHECKLIST.md)
-   [中文使用说明](docs/README_CN.md)
-   [Dataset guide](docs/DATASETS.md)
-   [Generic priors](docs/PRIORS.md)
-   [Benchmark protocol](docs/BENCHMARK.md)
-   [Metric definitions](docs/METRICS.md)
-   [Comparison methods](Compared_methods/README.md)

**Benchmark scope.** The same perturbation target is observed in source
cell lines and predicted in a held-out cell line, using a fixed set of
shared genes. This benchmark evaluates cross-context transfer; it does
not establish performance on unseen perturbation targets or unmeasured
genes.

## Model overview

SCULPT treats source responses from different cell lines as
background-conditioned observations of a shared perturbation action. Its
three main components are:

1.  **Selective Subspace Coupling.** Gene-specific local action
    subspaces are constructed from generic gene representations,
    source-control statistics, and a prior STRING network. SVD aligns
    neighboring genes' local subspaces, and directional gates allow
    information sharing only along sufficiently aligned directions.
2.  **Joint Action Inversion.** Baseline-conditioned observation
    operators map the shared latent action to source-specific response
    statistics. A single joint inverse problem combines multi-source
    observation fidelity, selective subspace coupling, and prior
    stabilization to infer the shared action.
3.  **Target-Conditioned Readout.** The inferred action is combined with
    the target control background. A zero-anchored hurdle readout
    predicts gene-wise activation frequency and positive-expression
    mean; positive quantile reshaping and learned rank assignment then
    reconstruct the target perturbed population.

The released implementation contains the **J0 model with joint
training**, named SCULPT in the manuscript. The `cosit` Python namespace
and checkpoint schema are retained for compatibility. All task
components are trained together in phase **J**; the release does not run
the older sequential A/B/C training procedure.

The fixed generic prior combines **Tahoe gene embeddings, GenePT, and
STRING diffusion**, projected to 256 dimensions with five availability
masks. J0 does not use WaveGC. Task-specific weights are initialized
from scratch for every fold and seed. No old predictor or fixed
source-mean answer is added to the model output. Target control cells
are legitimate biological inputs and provide the baseline
detection/positive-expression reference, cell prototypes, and shape
templates. Source observations inform the inferred action; they are not
used as a fixed additive prediction anchor.

```{=html}
<!-- TODO: Replace this path with the final benchmark/results figure, if desired. -->
```
![Benchmark results](fig/benchmark_results.png)

## 1. Environment installation

Use **Python 3.12** for the tested portable core environment.

For a CUDA 12.8-compatible GPU:

``` bash
conda env create -f environment.yml
conda activate sculpt
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[dev]'
```

Alternatively, create a Python 3.12 virtual environment, install a
PyTorch wheel compatible with your GPU/driver, and then run:

``` bash
python -m pip install -e '.[dev]'
```

For CPU installation, use the CPU PyTorch wheel index:

``` bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

This source snapshot was tested with Python 3.12.14, PyTorch 2.7.1,
NumPy 1.26.4, SciPy 1.15.2, scikit-learn 1.5.2, and h5py 3.13.0. Exact
core pins are provided in `requirements-core.txt`. Comparator
environments are managed separately according to their registered
dependencies.

Formal training uses one CUDA device at a time. CPU is supported for
synthetic checks; a full-benchmark CPU runtime is not claimed. To reduce
peak GPU memory, lower `execution.microbatch_size` in the configuration
while preserving the effective balanced batch.

Check the installation without downloading data:

``` bash
python -m cosit --help
python -m pytest -q tests Compared_methods/tests
python -m cosit smoke-test --output work/smoke
```

The smoke test exercises the model and complete pipeline on generated
cells. It is an installation/pipeline check and does not measure
biological predictive performance.

## 2. Dataset

The benchmark uses five cellular contexts:

  -------------------------------------------------------------------------------------------------------------------------------------------------------
  Benchmark context       Study / source          Download
  ----------------------- ----------------------- -------------------------------------------------------------------------------------------------------
  K562_essential          Replogle et al.;        [H5AD](https://zenodo.org/records/13350497/files/ReplogleWeissman2022_K562_essential.h5ad?download=1)
                          scPerturb               

  RPE1                    Replogle et al.;        [H5AD](https://zenodo.org/records/13350497/files/ReplogleWeissman2022_rpe1.h5ad?download=1)
                          scPerturb               

  HepG2                   Nadig et al.; scPerturb [H5AD](https://zenodo.org/records/13350497/files/NadigOConner2024_hepg2.h5ad?download=1)

  Jurkat                  Nadig et al.; scPerturb [H5AD](https://zenodo.org/records/13350497/files/NadigOConner2024_jurkat.h5ad?download=1)

  Orion_HCT116            Xaira X-Atlas/Orion     [Official dataset and HCT116 shards](https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion)
  -------------------------------------------------------------------------------------------------------------------------------------------------------

The dataset catalog pins the scPerturb Zenodo revision and the Orion
repository commit. The downloader checks the four H5AD file sizes and
published MD5 hashes, and downloads the HCT116 shards together with the
required gene metadata.

Download and prepare the data:

``` bash
python scripts/download_data.py --help
python scripts/download_data.py --raw-root data/raw --dry-run
python scripts/download_data.py --raw-root data/raw

python -m cosit prepare --config configs/datasets.yaml \
  --raw-root data/raw --output data/processed

python -m cosit audit-data --data data/processed --axes configs/axes
```

For files downloaded manually, arrange them as described in
[DATASETS.md](docs/DATASETS.md), then run the `prepare` and `audit-data`
commands above. Preparation streams the raw files, checks count
semantics and stable gene IDs, applies the frozen control partition and
metadata-only cell sampling, and writes five immutable H5 files.

The expression unit is:

\[
x_g=`\log`{=tex}!`\left`{=tex}(1+10\^4`\frac{\mathrm{counts}_g}`{=tex}
{`\sum`{=tex}*{a`\in `{=tex}A*{2048}}`\mathrm{counts}`{=tex}\_a}`\right`{=tex}).
\]

**Important:** use the fixed (A\_{2048}) denominator. Do not renormalize
using all 5,979 output genes or each file's gene panel. Missing anchors,
ambiguous IDs, and insufficient condition support raise an error. See
[DATASETS.md](docs/DATASETS.md) for quality-control and sampling
details.

The four scPerturb files are licensed under CC BY 4.0; Orion is licensed
under CC BY-NC-SA 4.0. The code license does not grant commercial rights
to Orion data. Please check the original dataset licenses before reuse.

## 3. Prepare generic gene priors and STRING

SCULPT uses public Tahoe and GenePT resources together with STRING. The
preparation scripts build the STRING graph and the generic gene
representation used by the model.

Download the public resources, HGNC mappings, and STRING files using the
links and version requirements in [PRIORS.md](docs/PRIORS.md). Preview
the resource download plan and inspect the available commands:

``` bash
python scripts/download_resources.py --dry-run
python scripts/prepare_priors.py --help
python scripts/prepare_priors.py string --help
python scripts/prepare_priors.py prior --help
```

The resulting model inputs include:

``` text
data/priors/prior.npz              genes + embedding [genes, 261]
data/string/graphs/KS/A.npz        STRING gene adjacency
data/string/source/node_axis.tsv  stable gene axis for the graph
```

The downloader pins the Tahoe revision and verifies registered
STRING/Tahoe hashes. The HGNC version must be explicitly pinned. These
generic resources are prepared separately from model training and are
not task-trained checkpoints. See [PRIORS.md](docs/PRIORS.md) for
complete commands, expected files, and provenance.

## 4. Benchmark protocol and data splits

SCULPT uses a five-fold context-holdout protocol. Each cell line serves
as the held-out test context once.

  Fold   Training contexts       Validation context   Test context
  ------ ----------------------- -------------------- --------------
  F1     K562, HepG2, HCT116     RPE1                 Jurkat
  F2     HepG2, Jurkat, HCT116   K562                 RPE1
  F3     RPE1, Jurkat, HCT116    HepG2                K562
  F4     K562, RPE1, Jurkat      HCT116               HepG2
  F5     K562, RPE1, HepG2       Jurkat               HCT116

Generate and verify the frozen splits:

``` bash
python -m cosit split --data data/processed --config configs/j0.json \
  --output work/splits --verify-frozen configs/splits
```

Each fold also holds out 153 complete perturbation conditions from each
training context. These three target sets are disjoint. Held conditions
are excluded from every gradient role, including source retrieval,
negative banks, and fitted response scales. Each fold has **4,116 legal
training query conditions**.

Validation combines one external context (weight 1/2) and three internal
held-condition panels (weight 1/6 each). Seeds 17, 29, and 43 change
initialization and training order, not the data partitions.

## 5. Training and validation

Train one fold with one initialization:

``` bash
python -m cosit train \
  --data data/processed --priors data/priors/prior.npz \
  --graph data/string/graphs/KS/A.npz --graph-axis data/string/source/node_axis.tsv \
  --splits work/splits --fold F1 --seed 17 \
  --config configs/j0.json --output work/j0/F1/seed_17 --device cuda:0
```

To resume the same run, repeat the command with `--resume`. The
checkpoint restores the optimizer, stopping state, schedule position,
and random generators. It does not import weights or predictions from
another method.

All model components train jointly. The five objective groups supervise
response, perturbation identity, marginals, joint structure, and
shared-action consistency. Validation runs every two epochs. Checkpoint
selection minimizes the registered multi-context output objective; exact
ties select the earlier checkpoint.

Training follows the registered plateau, learning-rate reductions, and
minimum optimization budget. Reaching an epoch cap produces
`needs_decision`, **not** a convergence certificate. See
[BENCHMARK.md](docs/BENCHMARK.md) and
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the complete selection
and validation rules.

## 6. Prediction

After convergence and checkpoint selection, generate the held-out test
populations:

``` bash
python -m cosit predict \
  --data data/processed --priors data/priors/prior.npz \
  --graph data/string/graphs/KS/A.npz --graph-axis data/string/source/node_axis.tsv \
  --run work/j0/F1/seed_17 \
  --output work/j0/F1/seed_17/test_predictions.h5 --device cuda:0
```

`SELECTION.json` records the selected checkpoint path. Each test
perturbation receives a predicted population of **512 cells × 5,979
genes**, rather than only a mean-response vector.

The generated population is a deterministic finite quantile cloud, not
an i.i.d. sample. Its finite-cloud mean is `round(512*p)/512*r`; (p r)
is an analytic diagnostic. Learned rank permutation preserves every
gene's emitted marginal exactly.

All test predictions and the selected checkpoint are sealed before test
responses can be read. Test responses cannot be used to select or update
a checkpoint.

## 7. Evaluation

Evaluate using the checkpoint path recorded in `SELECTION.json`:

``` bash
CHECKPOINT=$(python -c 'import json; print(json.load(open("work/j0/F1/seed_17/SELECTION.json"))["selected_checkpoint"])')

python -m cosit evaluate --data data/processed \
  --predictions work/j0/F1/seed_17/test_predictions.h5 \
  --checkpoint "$CHECKPOINT" \
  --assets work/j0/F1/seed_17/assets \
  --output work/j0/F1/seed_17/test_metrics.json --device cpu
```

Response-recovery metrics include cosine-ranking PDS, Pearson, Spearman,
DEG-AP, JAC20/50/100, signed Top100 coverage, direction accuracy, and
amplitude ratio.

Distributional metrics include marginal W1, CRPS, raw and centered
sliced W1, energy distance, Gaussian MMD², PCA covariance error, and
support diagnostics. Raw MAE/MSE and the historical composite response
score are omitted from the primary report. See
[METRICS.md](docs/METRICS.md) for definitions and limitations.

## 8. Run the complete benchmark

Run all five folds and three seeds:

``` bash
python scripts/run_benchmark.py \
  --data data/processed --priors data/priors/prior.npz \
  --graph data/string/graphs/KS/A.npz --graph-axis data/string/source/node_axis.tsv \
  --config configs/j0.json --output work/j0 --device cuda:0
```

The runner executes 15 independent training runs sequentially, resumes
existing runs, predicts only after convergence, and evaluates the
selected checkpoints. A subset can be run explicitly, for example:

``` bash
python scripts/run_benchmark.py \
  --data data/processed --priors data/priors/prior.npz \
  --graph data/string/graphs/KS/A.npz --graph-axis data/string/source/node_axis.tsv \
  --config configs/j0.json --output work/j0 --device cuda:0 \
  --folds F1 --seeds 17
```

A subset run is marked incomplete. The runner does not stop other
projects, select a GPU automatically, or install a polling monitor.

The final `summary.json` first averages the three seeds within each
context, then reports the equally weighted five-context mean, sample
variance, and standard deviation (ddof=1). Initialization variance is
also reported within each fold. Missing metrics retain null values and
valid counts; mean-only methods are not assigned distribution scores.

## 9. Comparison methods and reproducibility

The [`Compared_methods/`](Compared_methods/README.md) directory contains
the registered configurations for **TxPert, CPA, PIE, scPILOT, STATE,
CellFlow, and PopPert**, including pinned official commits, environment
setup, legal training-data export, and native prediction bridges.
Upstream code is fetched on request rather than copied into this
repository.

In the registered configuration, TxPert, PIE, and Source-Mean are
mean-only methods. CPA, scPILOT, STATE, CellFlow, PopPert, and
Target-Control produce distribution outputs, with benchmark adaptations
documented in their respective guides.

Source-Mean uses equal-weight means of source **absolute perturbed
expressions**, not source response differences. The two simple baselines
run once per fold. The repository distinguishes official APIs from
benchmark adapters and does not claim that a generic script faithfully
replaces the native training pipelines of all comparison methods.

Additional implementation and protocol documentation:

-   [Implementation provenance and validation
    limits](docs/REPRODUCIBILITY.md)
-   [J0 fidelity audit](docs/J0_FIDELITY.md)
-   [Anonymous packaging and release checks](docs/ANONYMOUS_RELEASE.md)
-   [Benchmark and access protocol](docs/BENCHMARK.md)
-   [Prediction schema](docs/INTERFACE.md)
-   [Comparison-method preprocessing](Compared_methods/PREPROCESSING.md)
-   [Contributing](CONTRIBUTING.md)

```{=html}
<!-- TODO: Replace this path with the final interpretability figure (e.g., the subspace-overlap and directional-coupling analyses). -->
```
![Interpretability analysis](fig/interpretability_analysis.png)

The interpretability analyses examine principal-subspace overlap on
STRING edges versus non-edges, the fraction of active directional
couplings, cross-source consistency of inverted actions, target-context
perturbation discrimination across functional modules, gene-pair
co-expression sign accuracy, and module-score distribution fidelity.

## Citation

If you use this code, please cite the SCULPT manuscript. Add the final
BibTeX entry here when the review/publication details are available.

``` bibtex
@misc{sculpt2026,
  title  = {SCULPT: Joint Inversion of Context-Specific Perturbation Responses for Cross-Context Prediction},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## License

The original SCULPT code is released under the [MIT License](LICENSE).
External datasets, pretrained resources, and official comparison
implementations retain their own licenses. In particular, the Orion
dataset is licensed under CC BY-NC-SA 4.0; the code license does not
grant commercial rights to that dataset.

This repository is a portable source release. Synthetic end-to-end
checks and core equivalence were verified; a new full 15-run benchmark
was **not** performed as part of packaging. No new performance result is
claimed by the source release.
