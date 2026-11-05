# Method-specific preprocessing on the shared benchmark

All commands run from the repository root. First run the main dataset preparer
and frozen split verification. Never run upstream HVG selection or a second
normalization over these shared inputs. Every preprocessing receipt binds the
prepared-data manifest and fold permissions by SHA256.

## Shared single-cell input

```bash
python -m Compared_methods.export_training --data data/processed \
  --splits work/splits --fold F1 --output work/native/F1/train.h5ad
```

The training AnnData contains only legal source P conditions and their E
controls. Target/validation P cells and internal-held source P are excluded.
Full pooled control means are retained separately from capped control cells.
TxPert, CPA, scPILOT and STATE can use this common cell interface or their native
API against the protected `cosit.data.CellStore`; their model/loss differences
remain in their method-specific helpers.

## PIE native labels

In the isolated PIE environment with the pinned upstream checkout:

```bash
PYTHONPATH=src python -m Compared_methods.preprocess pie \
  --data data/processed --splits work/splits --fold F1 \
  --upstream external/PIE --output work/native/F1/PIE
python -m Compared_methods.download_knowledge pie \
  --directory data/comparison_knowledge/PIE --dry-run
```

Remove `--dry-run` only when ready for the potentially large public knowledge
download. All six source modalities plus the gene-query source must be present;
do not replace missing modalities by random vectors. Label preprocessing emits
the official `pie.data.preprocessed` schema and a native training split. It
preserves MWU/BH, the tested-gene cutoff, cellwise-linear fold changes and the
mean log-expression shift. It never reads query P. Official evidence/grids and
fitted scales must be built from these legal training rows only, using the
native data module and the registered config. Do not run the raw-count upstream
normalizer on these already normalized cells.

## PopPert populations and training-only copula

```bash
PYTHONPATH=src python -m Compared_methods.preprocess poppert \
  --data data/processed --splits work/splits --fold F1 \
  --upstream external/PopPert --output work/native/F1/PopPert
```

Outputs are `copula.npz`, `vocab.json`, `train_params.npy`,
`train_factor_var.npy`, `train_index.json`, `control_params.npy`,
`control_factor_var.npy` and `control_environments.json`.
Copula directions fit only the three legal training contexts' controls.
All five contexts' controls are transformed under that fixed basis. No held
perturbations are fitted. The vocabulary uses output-gene order followed by
new target IDs, preserving the registered indexing rule. The native GMM and
copula functions are imported from the separately obtained official code.
This preparation is computationally substantial; it is not run on import.

## CellFlow PCA

In the isolated JAX/CellFlow environment:

```bash
PYTHONPATH=src python -m Compared_methods.preprocess cellflow \
  --data data/processed --splits work/splits --fold F1 \
  --output work/native/F1/CellFlow
```

The 100-component centered PCA uses all legal training P and training E rows,
without held-context controls. Covariance is accumulated in registered 4096-row
FP32 JAX blocks, then diagonalized in FP64. Source-cell order, final padding,
component sign and genes are fixed. `pca.npz` can then be used with
`cellflow.native_api.condition`, `make_solver`, and `predict`. Inverse PCA is the
native expression reconstruction, not an additional baseline correction.

## Cautions about scope

These commands prepare source-only native inputs. They do not train comparison
models or manufacture scores. They use the shared public cell-bank schema and
are a portability layer; new full-scale runs must record actual data, runtime
and asset hashes. The per-method registered model, loss, selection and seed
policy must still be followed. See the comparison README for the complete
distinction between callable native primitives and a training controller.
