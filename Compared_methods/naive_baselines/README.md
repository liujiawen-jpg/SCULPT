# Target-Control and Source-Mean

These are independent parameter-free comparators, never components or anchors
of SCULPT. Each runs once for each of the five folds: no fitting, model selection,
training seeds, or artificial seed replicates.

## Exact definitions

**Source-Mean** is the equal arithmetic average of the three source contexts'
**absolute perturbed-expression means** on the common normalized-log gene axis:

`prediction[p,g] = (mean(X[source1,p]) + mean(X[source2,p]) + mean(X[source3,p])) / 3`.

Source cell counts do not alter the three context weights. No source control
is subtracted, and no target control is added to this prediction. A target
control is subtracted later only by the shared response evaluator, as for all
methods. This release implements the requested **mean-only** Source-Mean:
distribution metrics are N/A. An older registered diagnostic also emitted an
empirical source mixture; that is a different distribution baseline and is not
silently attached to this mean-only method.

**Target-Control** uses the exact immutable full-pooled target-control mean for
response metrics (hence an exactly zero response). For distribution metrics it
uses 512 unchanged whole target-control cell rows, selected deterministically
with indices `floor((2*i+1)*N/(2*512))`. Small datasets repeat whole rows. It never
rescales or shifts cells. The finite cloud's mean can differ from the full
pooled mean; both are saved and their discrepancy is reported. This is why the
baseline has a dedicated evaluator rather than bypassing the shared converter's
usual mean/cloud equality requirement.

## Commands

After preparing common data and the corresponding fold's SCULPT metric assets,
run from the repository root (CPU is sufficient):

```bash
python -m Compared_methods.naive_baselines.run generate \
  --data data/processed --splits configs/splits --fold F1 \
  --output work/baselines/F1
python -m Compared_methods.naive_baselines.run evaluate \
  --data data/processed --directory work/baselines/F1 \
  --assets work/j0/F1/seed_17/assets --device cpu
```

Repeat for F2–F5 using the matching fold assets. Aggregate each method once:

```bash
python -m Compared_methods.naive_baselines.run aggregate \
  --metrics work/baselines/F{1,2,3,4,5}/source_mean.metrics.json \
  --output work/baselines/source_mean.summary.json
python -m Compared_methods.naive_baselines.run aggregate \
  --metrics work/baselines/F{1,2,3,4,5}/target_control.metrics.json \
  --output work/baselines/target_control.summary.json
```

These commands intentionally require a fresh generation directory and refuse
to overwrite completed results. `ALGORITHM.json` fingerprints the baseline
implementation/config, fold and prepared data. `SEAL.json` binds that algorithm
and both full-fold prediction files before any target perturbation is read.
The source reader uses `CellStore`'s inference-only, no-gradient authorization;
it can never open target-response labels as prediction inputs.

Response metrics use the shared SCULPT metric implementation (full-panel cosine
PDS, own-target-excluded correlations/TopK, MWU/BH DEG-AP). Distribution metrics
use the same frozen training-control geometry and common implementations.
The evaluator independently checks that Target-Control's analytic mean equals
the full control mean and its support consists of the prescribed untouched
control rows. The final summary reports equal-fold mean, **sample variance** and
**sample SD** (`ddof=1`) separately.

The accompanying tests use only synthetic cells. No private cells, real
predictions or checkpoint files are included.
