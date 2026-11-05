# Reproduction checklist

## What is included

- The actual J0 shared-action solver, background operators, target readout,
  marginal shaping, learned rank rearrangement and joint objective.
- Public raw-data download code, count/QC checks, A2048 normalization, stable-ID
  mappings and metadata-only cell sampling for all five contexts.
- Frozen gene/target axes, five folds and validation panels; seeds 17/29/43
  affect initialization and order, not the fold assignments.
- Complete SCULPT training/resume, validation selection, generation, sealing,
  test evaluation and fold/seed aggregation entrypoints.
- Registered comparator environments, pinned official-source acquisition,
  native interfaces, specialized preprocessing and common input/output bridges.
- Unit checks and a synthetic end-to-end exercise of the actual model.

No training outputs or datasets are embedded in the source tree. Downloaded
data, generic pretrained assets, task weights and caches stay in ignored folders.

## Installation and a data-free first check

Use an isolated Python 3.12 environment. Install PyTorch for your hardware,
then the tested dependencies and editable package:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-core.txt
python -m pip install -e . --no-deps
make test
make smoke
```

For real training, replace the CPU wheel by the CUDA wheel in the main README.
The smoke run uses synthetic cells and fresh task weights; its numbers are not
biological benchmark results. It writes only under `work/smoke`.

## Full SCULPT benchmark

1. `make download-plan`, then `make download`. Dataset addresses and exact
   source semantics are in [DATASETS.md](DATASETS.md).
2. `make prepare`, then `make splits`.
3. Download and prepare generic priors according to [PRIORS.md](PRIORS.md).
   `scripts/download_resources.py --dry-run` prints the download plan without
   network access. Its Tahoe revision is immutable; STRING files have expected
   hashes. HGNC requires an explicit archive URL and expected hash.
4. `make train FOLD=F1 SEED=17 DEVICE=cuda:0` runs one independent model.
   See the README for predict/evaluate commands after convergence.
5. `make benchmark DEVICE=cuda:0` executes the complete five-fold/three-seed
   SCULPT protocol, including selection, prediction and evaluation.
6. Report seed means within each fold first; calculate the equal-fold mean
   and sample standard deviation across the five contexts. Preserve failed or
   unconverged runs as such; do not average only successful seeds silently.

Every test prediction must be sealed before its labels are opened. Test labels
are not permitted in checkpoint selection. An epoch cap is not convergence.

## Comparison-method reproduction

Each method uses a separate environment and a fixed official revision. Start
with [Compared_methods/README.md](../Compared_methods/README.md).

| Step | Included entrypoint | Purpose |
|---|---|---|
| Fetch scientific implementation | `python -m Compared_methods.prepare METHOD --directory external/METHOD` | Download pinned upstream code, not weights |
| Install isolated runtime | `Compared_methods/METHOD/install.sh` | Registered dependencies and disclosed overrides |
| Export authorized cells | `python -m Compared_methods.export_training ...` | AnnData for legal training conditions only |
| Prepare native labels/populations | `python -m Compared_methods.preprocess pie/poppert/cellflow ...` | PIE native label directory; PopPert population/coupling assets; CellFlow PCA |
| Native optimization/prediction | Per-method `native_api.py`, registered config, and official trainer | Preserve scientific model and native loss |
| Convert selected native predictions | `python -m Compared_methods.convert_predictions ...` | Seal means/clouds without inventing distributions |
| Score selected predictions | `python -m cosit evaluate ...` | Common response/distribution definitions |
| Parameter-free baselines | `python -m Compared_methods.naive_baselines.run ...` | Full generation/evaluation/aggregation commands |

**Scope:** SCULPT has a complete portable training controller in this package.
For learned comparators, the provided native APIs and inputs are integrations
with the pinned official implementations; they are not seven independently
ported end-to-end training controllers. Native optimization must follow each
registered configuration. This distinction is explicit so a helper or a config
is never presented as a completed benchmark run. No new comparison training is
claimed by the packaging checks.

PIE's binned mean-shift output is not a single-cell distribution. TxPert and PIE
remain mean-only. Source-Mean averages source absolute perturbed means with
equal source weights; it has no distribution. Target-Control uses measured
target controls and runs once per fold. See the per-method licenses and guides.

## Publication and anonymity

The ZIP is a deterministic source snapshot with fixed archive timestamps, no
Git history and relative-path checksums. `scripts/check_anonymity.py` scans the
clean source tree. Run it before submission, excluding generated work folders.
No hosting account or commit author has been created by this release process.
Keep the hosting account, repository metadata and submission link anonymous.
