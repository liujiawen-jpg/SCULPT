# Compared methods

This directory records the **actual registered configurations** used for the
SCULPT/J0 comparison: five fixed context folds and initialization seeds 17, 29, and 43
for learned methods. Target-Control and Source-Mean run once per fold.
It contains configuration files and our small integration helpers. Official
third-party source, pretrained weights, cell data, and experimental outputs are
not redistributed here.

## Scope and licenses

| Method | Pinned official source | Native output used here | Upstream license |
|---|---|---|---|
| TxPert public MG | [valence-labs/TxPert](https://github.com/valence-labs/TxPert/tree/08d82eea86746b044cf7531f4ec8c5f60e1cb73f) | Absolute expression mean | Recursion Non-Commercial EULA |
| CPA 0.8.8 | [theislab/cpa](https://github.com/theislab/cpa/tree/fbd7c0250edc23eff003a10c99655579c53afd63) | Native conditional Gaussian cell cloud; disclosed censoring at zero | Repository LICENSE: BSD-3-Clause; package metadata says MIT |
| PIE | [ArcInstitute/pie](https://github.com/ArcInstitute/pie/tree/21a526add9ff3a81b4f962d9b16bbae570052c63) | Absolute expression mean | CC-BY-NC-SA-4.0 |
| scPILOT | [yongzhuangliulab/scPILOT](https://github.com/yongzhuangliulab/scPILOT/tree/ff83e3283f091324107a55f12f6801c4e72dfb05) | Native signed cell cloud from latent OT, Leiden, and adaptive weighting | MIT |
| STATE | [ArcInstitute/state](https://github.com/ArcInstitute/state/tree/9bbfe78a434a55205e4de834e1ea99f85f7a3add) | Native ReLU cell-set prediction | CC-BY-NC-SA-4.0 |
| CellFlow | [theislab/CellFlow](https://github.com/theislab/CellFlow/tree/446ed6073c60ac2e8db13c4ea096a43cdec288b2) | Native flow samples, inverse-PCA expression cloud | PolyForm Noncommercial |
| PopPert | [whd1125/PopPert](https://github.com/whd1125/PopPert/tree/248fc3e1428de96335bef1d0271d30ca1e4d5565) | ZIGMM marginals and low-rank Gaussian-copula cloud | No license chosen upstream; implementation not redistributed |
| Source-Mean | Built-in deterministic baseline | Mean of source absolute perturbed expressions; **mean only** | This repository |
| Target-Control | Built-in deterministic baseline | Full control mean for response; empirical control population for distribution | This repository |

The SCULPT repository license does not replace these independent upstream
licenses. Read each downloaded license before using its implementation.
TxPert's public MG model contains STRING and GO; the proprietary PxMap and
TxMap resources are unavailable and are not claimed to be reproduced.

## 1. Download official code into a separate working directory

Run from the repository root:

```bash
python Compared_methods/prepare.py txpert --directory external/TxPert
python Compared_methods/prepare.py cpa --directory external/CPA
python Compared_methods/prepare.py pie --directory external/PIE
python Compared_methods/prepare.py scpilot --directory external/scPILOT
python Compared_methods/prepare.py state --directory external/state
python Compared_methods/prepare.py cellflow --directory external/CellFlow
python Compared_methods/prepare.py poppert --directory external/PopPert
```

This checks out exactly the commits in `registry.json`. It does not install,
train, fetch weights, stop jobs, or overwrite an existing different checkout.
Keep each method in its **own environment**. Installation instructions are in
the corresponding pinned official README. In particular:

The four original installation scripts below recreate their registered computational
runtime, require Python 3.12, and refuse to modify an existing environment:

```bash
bash Compared_methods/txpert/install.sh external/TxPert work/envs/txpert
bash Compared_methods/cpa/install.sh external/CPA work/envs/cpa
bash Compared_methods/pie/install.sh external/PIE work/envs/pie
bash Compared_methods/scpilot/install.sh external/scPILOT work/envs/scpilot
```

They use CUDA 12.8 PyTorch wheels and expect a compatible NVIDIA driver. Install
the main CoSIT package in its own environment for common input export and
evaluation. The comparison helpers can be imported from the repository root;
the third-party packages remain isolated from CoSIT dependencies. These
scripts are versioned environment recipes; the release smoke test does not
reinstall the GPU environments or retrain the comparison methods.

- TxPert's pinned project targets Python 3.12 and PyTorch 2.6.0. The benchmark
  used PyTorch 2.7.1/CUDA 12.8 for Blackwell compatibility. This is an explicitly
  disclosed runtime override, not a change to its native model or loss.
- CPA's original dependency metadata targets Python below 3.11 and old
  scvi-tools. The recorded benchmark used Python 3.12, PyTorch 2.7.1,
  scvi-tools 1.3.3, and only the official core modules. Do not install its old
  constraints into the CoSIT environment or assume this is bitwise identical
  to the legacy software stack.
- PIE's pinned `pyproject.toml` requires Python 3.12. Use its own dependency
  resolver and CUDA instructions; its requirements differ from CoSIT's.
- scPILOT additionally needs POT, Leiden/igraph, anndata, scanpy,
  and Lightning. The registered training uses full float32 without TF32.

### Added methods and no-training baselines

Use the per-method guides for complete environment commands and native inputs:

- [STATE](state/README.md): exact Transformer and set-loss configuration, native helper and pinned source checks.
- [CellFlow](cellflow/README.md): separate JAX environment, training-only PCA, target-control condition inputs and flow sampling.
- [PopPert](poppert/README.md): separate PyTorch environment, distribution preprocessing, native likelihood and copula sampling.
- [Target-Control / Source-Mean](naive_baselines/README.md): executable frozen-input generation and evaluation; no model optimization and no three-seed replication.

The three added learned methods include native computational helpers and the
actual benchmark configurations, **not complete portable benchmark training
controllers**. Their method-specific data-loading, stopping and selection
controllers must follow the documented recipes. The complete SCULPT training,
prediction and evaluation CLI is in `src/cosit`; do not confuse those two scopes.
No new comparison training is claimed as part of packaging.

Method-specific preparation CLI and exact commands are in [PREPROCESSING.md](PREPROCESSING.md).

## 2. Reuse the exact data/split protocol

Generate CoSIT's splits with the main repository CLI. All methods use the same
ordered 5,979-gene and 1,525-target axes and the same supplied cells (up to 256
observations per condition). Preserve the fixed A2048 normalization; do not
renormalize the 5,979-gene output subset, select new HVGs, or silently drop
targets with ambiguous symbols.

Export a legal native training AnnData file without loading the complete dense
population in memory:

```bash
python Compared_methods/export_training.py --data data/processed \
  --splits work/splits --fold F1 --output work/native/F1/train.h5ad
```

This streams only `TRAINING_TARGETS.json.allowed_targets` from the three
training environments, includes their measured controls, and writes an access
audit beside the file. Metadata are `obs.environment`, `obs.perturbation`, and
`obs.control`; full pooled control means are in
`uns['control_mean_by_environment']`. The export is a common cell input,
**not** PIE's method-specific DE/evidence conversion. Never concatenate query
perturbed cells into this file.

The public processed input is `processed/<context>.h5`:

```text
attrs: genes_json, targets_json, control_n_full
controls: [at most 256, G]
control_ids
control_mean, control_variance, control_detection: [G] from all eligible E
perturbations/<stable_target_id>: [at most 256, G]
```

Training may read only the active fold's allowed training conditions. Fixed
internal held conditions are validation labels, never gradient supervision.
At inference, active-fold source conditions for the requested same target may
be used where that native method supports them. The query environment supplies
only controls to the model. Every selected checkpoint and all test predictions
must be sealed before the test-label reader is enabled. Initialization seeds
do not change the data split.

Native input interfaces differ. This is a **registered common-benchmark
adaptation**, not a claim of using identical original-paper preprocessing:

| Method | Training/inference interface | Exact configuration |
|---|---|---|
| TxPert | Cell-level expression supervision; full pooled matched control mean; graph target index | [`txpert/benchmark_config.json`](txpert/benchmark_config.json), [`upstream_config.yaml`](txpert/upstream_config.yaml) |
| CPA | Cell-level expression, perturbation class and dose 1; target-control states at prediction | [`cpa/benchmark_config.json`](cpa/benchmark_config.json) |
| PIE | Native DE/log2FC/binned-delta-p labels and pooled source evidence; six official knowledge sources | [`pie/benchmark_config.json`](pie/benchmark_config.json) |
| scPILOT | Shared unconditional representation over all authorized source perturbations; single-target native transfer at inference | [`scpilot/benchmark_config.json`](scpilot/benchmark_config.json) |

## 3. Native computational helpers

These are small callable integrations, **not fully ported training CLIs for all methods**:

- `txpert/native_api.py`: builds the unchanged public MG model, performs the
  vectorized native evaluation forward, and explicitly checkpoints graph
  tensors that upstream does not register as buffers. Set `PYTHONPATH` to the
  downloaded TxPert root and provide the official graph cache and an audited
  stable-ID/HGNC map. Missing targets remain individual isolated learned nodes.
- `cpa/native_api.py`: calls official `CPAModule` and
  `CPATrainingPlan.training_step`; its `predict_cloud` uses the native learned
  Normal likelihood mixed over target-control states. This is the registered
  cloud adapter, with raw negative-rate diagnostics; it is not handcrafted
  Gaussian noise around a separately predicted mean.
- `pie/native_api.py`: creates the full official recipe (768 width, 512
  latents, encoder/processor/decoder 4/6/4) and native model. Label conversion
  must retain native MWU/BH DE, DE-masked LFC, two-hot delta-p, missing-source
  masks and full NCBI token sequences. `pie/mwu_compat.py` handles only the
  all-pooled-tied MWU case (U = n*m/2, p = 1); other nonfinite results fail.
- `scpilot/native_training.py`, `native_inference.py`, and
  `stable_adaptive_weights.py`: invoke the original two-Adam training step and
  original latent OT/Leiden prediction. The shifted exponential prevents
  numerical 0/0 while preserving normalized exponential weights. No clipping
  or mean replacement is applied to native signed outputs.

Training schedules, convergence budgets, losses, source permissions and
validation selectors are fully specified in each `benchmark_config.json`.
TxPert's public repository publishes inference, not its original training
driver; our registration therefore explicitly names a benchmark training
wrapper using its official model/loss/AdamW. A shared one-line native command
would hide these differences and is deliberately not claimed here.

For an independent new reproduction, implement the native data-loader and
training orchestration against the shared protected split contract, or use a
separately published complete method-specific runner. The helpers here are
provided to retain native computational behavior without distributing
machine-specific historical experiment drivers.

## 4. Export native outputs for common evaluation

After validation selects the checkpoint, export `native_predictions.npz`:

```python
# Mean-only TxPert/PIE:
np.savez("native_predictions.npz", mean=absolute_expression_mean)
# Native CPA/scPILOT populations:
np.savez("native_predictions.npz", support=cells,
         mean=cells.astype("float64").mean(1).astype("float32"))
# cells.shape == (number_of_targets, 512, number_of_genes)
```

Then run (one stable ID per line in the axis text files):

```bash
python Compared_methods/convert_predictions.py \
  --source native_predictions.npz --output work/PIE/F1/seed17/predictions.h5 \
  --method pie --environment Jurkat --fold F1 --seed 17 \
  --genes genes.txt --targets targets.txt \
  --checkpoint work/PIE/F1/seed17/selected.ckpt
```

For CPA/scPILOT/STATE/CellFlow/PopPert additionally provide `--support-semantics iid_samples` (the
native scPILOT posterior/bootstrap cloud) and use the appropriate `--method`.
The converter verifies axis dimensions, finiteness, the 512-cell budget and
cloud/mean agreement; records checkpoint and export SHA-256; and seals the
HDF5 (`complete=True`) without touching test labels. It also calls CoSIT's
`seal_predictions` and writes `predictions.seal.json` with the checkpoint and
prediction-file hashes and allowed condition list. Use `--purpose validation`
for a validation export; test is the default. It performs no clipping or recalibration.
Mean-only methods are refused if a synthetic `support` array is supplied.

Use the main `cosit evaluate` command against these sealed predictions. Native
checkpoint files are bound by hash, not deserialized as CoSIT weights. Supply
the corresponding J0 fold's **training-only evaluation geometry**:

```bash
cosit evaluate --data data/processed \
  --predictions work/PIE/F1/seed17/predictions.h5 \
  --checkpoint work/PIE/F1/seed17/selected.ckpt \
  --assets work/J0/F1/seed_17/assets --output work/PIE/F1/seed17/metrics.json
```

Report
mean-only methods as **N/A** for distribution metrics. Retain native signed
scPILOT outputs when computing distances rather than censoring them silently.

## 5. Statistics

Target-Control and Source-Mean have one deterministic result per fold. Their
uncertainty is computed over five fold values; do not manufacture three identical
seed results. Source-Mean is an arithmetic mean of **absolute source expressions**,
not response deltas and not a population mixture. Target-Control preserves its
full pooled control mean for response scores and empirical whole-cell support
for distribution scores; its dedicated adapter records why those means can differ.

For learned methods: 
Average the three seeds **within each context fold**, then report the equal
five-fold mean and the sample variance/standard deviation (`ddof=1`) of the
five context means. Also report seed variability separately. Fifteen runs are
not fifteen independent held-out environments. The common task is same-target
cross-context prediction on shared genes, not double OOD or unseen output
genes. Do not claim equal inference information where native interfaces differ.

## Integration checks

```bash
python -m unittest discover -s Compared_methods/tests -v
```

These check configuration/provenance and prediction conversion on synthetic
arrays. They do not claim to have retrained all four official models after
packaging this release.
