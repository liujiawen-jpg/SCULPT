# Portable input/output contract

## Prepared observations

One immutable `CONTEXT.h5` file for each of the five context names:

```text
attrs:
  schema = cosit_cells_v1
  complete = true
  genes_json = ordered stable-gene ID list
  targets_json = ordered perturbation-target ID list
  control_n_full = number of eligible E controls
datasets:
  controls                 [E <= 256, G] float32
  control_ids              [E] UTF-8
  control_mean             [G] float32, full eligible E mean
  control_variance         [G] float32, full eligible E population variance
  control_detection        [G] float32, full eligible E positive fraction
  perturbations/TARGET_ID  [P <= 256, G] float32
```

`MANIFEST.json` binds each file by SHA-256. `.cosit_integrity.json` is a generated
hash/identity receipt, not source code. A read capability is checked before
every response array access, including memory-cache hits. `CellStore` objects
for training, inference sources and evaluation are independent.

The data-preparation guide specifies raw schemas, QC, control partitions and
proportional/hash cell sampling. Stable gene IDs, matrix semantics and the fixed
A2048 denominator must be resolved before creating these files.

## Generic input assets

`prior.npz` contains non-object arrays `genes` (unique stable ID strings) and
`embedding` of shape `[genes,261]`. It contains only frozen generic prior
features. Missing prior rows are represented by zeros and availability masks.

The graph is a SciPy CSR adjacency `.npz` accompanied by a TSV with a `gene_id`
column. The loader aligns graph nodes by stable ID, removes self-loops and uses
the actual output-gene subgraph; no target response is used to choose graph edges.

## Prediction file

```text
attrs:
  schema = cosit_predictions_v1
  complete = true
  environment = held context
  genes_json = ordered output IDs
  targets_json = ordered query IDs
  distribution_targets_json = ordered cloud-query IDs
  method = CoSIT-J0 or registered comparison name
datasets:
  mean                     [T,G]
  support                  [T_distribution,512,G]  # absent for mean-only methods
```

J0 additionally writes `p`, `r`, positive-shape diagnostics and rank penalties.
The generated population mean must match `mean` within the registered floating
point tolerance. A comparator export adds `fold` and its actual initialization
`seed`; it never adds arbitrary noise to invent a population.

## Seals and metrics

`*.seal.json` uses schema `cosit_prediction_seal_v1`. It binds completed
prediction files and the selected checkpoint by SHA-256, declares evaluation
purpose and lists allowed environment/target queries. Evaluator capabilities
recheck these bytes/identities before exposing response labels. This is an
accidental-leakage guard, not an adversarial security boundary.

`test_metrics.json` uses schema `cosit_metrics_v1` with method/fold/seed,
context-level response/distribution summaries, per-condition metrics and actual
checkpoint/prediction hashes. `summary.json` uses `cosit_summary_v1` and stores
within-fold seed mean/variance/SD plus macro context mean/variance/SD.

For a native comparator checkpoint, `--assets` points at the corresponding
J0 run's **training-fitted statistics only**. The evaluator does not deserialize
or run the competitor's checkpoint. It hashes that native checkpoint and uses
its already frozen prediction; native selection must follow that comparator's
registered validation protocol.
