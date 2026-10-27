# Implementation provenance and validation limits

## What J0 means

J0 is the original non-WaveGC, open-zero model trained with a single joint
optimization phase. `src/cosit/core/PORTABILITY.json` records SHA-256 fingerprints
of the frozen J0 mathematical modules and the corresponding packaged files.
No task-trained weights, model outputs, datasets or historical fitted geometry
are included in this source release.

The model, connection solve, hurdle/shape generation, rank permutation,
objective formulas and statistical observation recipes were retained. Imports
are package-relative. The graph and data locations are explicit inputs.
Server-specific historical cache discovery, symlinks, process launchers and
fixed absolute machine paths are replaced by the portable data/CLI layer.
Training uses bounded observational caches and grouped transfers rather than
the original server's large resident-memory orchestration.

The release evaluator has explicit extensions: native competitors may opt into
signed distribution supports, and a supplemental empirical energy statistic is
reported on the same fixed subsampling protocol as MMD (see METRICS.md).
J0's default nonnegative contract and model objectives are unchanged. The DE
metric also handles the mathematical all-pooled-values-tied U-test case with
p=1 across supported SciPy versions. Historical raw mean-error metrics and
legacy composites are omitted from the primary output schema.

## Checks performed during packaging

- Original frozen model and packaged model were given identical synthetic
  tensors and weights. Shared action, detection, positive mean, final support,
  finite-cloud mean and positive quantiles matched exactly on CPU float32.
  All parameter gradients also matched in that fixture.
- The package test suite checks batched/single-condition equivalence,
  finite-cloud means, rank-marginal preservation, full joint-loss gradients,
  bootstrap precision and the five-block prior formula.
- All five frozen folds, held-condition training axes and exact validation
  panels regenerate from identifiers. Permission tests reject held responses
  and changed sealed predictions, including cache-hit requests.
- Synthetic H5AD and Orion raw schemas exercise actual preparation, anchor
  normalization, pooled/capped control separation and required support checks.
- Five real contexts' capped E-control UID orders matched their archived
  256-control rosters. Three P conditions per context (15 total) also matched
  archived cell UID orders, including a capped condition. These checks read
  metadata only, not real expression arrays or new test scores.
- A CPU-only end-to-end synthetic run exercises fresh joint training,
  four-panel validation, checkpoint selection/restoration, 512-cell generation,
  sealing, response evaluation and distribution evaluation.
- Native comparison bridges were checked on synthetic mean-only and signed
  cloud outputs with foreign-format checkpoint files.

The tests and GitHub Actions configuration are included. Generated test cells,
checkpoints and numerical reports are excluded from the repository.

## What these checks do not establish

Packaging did **not** run a new full five-fold, three-seed training benchmark.
The source release does not contain historical numerical result files and does
not claim new biological accuracy from synthetic checks.

The architecture and scientific recipe are reproducible from the public
sources. Bitwise identity of a newly trained full benchmark additionally depends
on original public-file bytes, generic identifier mappings, software versions,
hardware and numerical library behavior. Current HGNC/Tahoe downloads can
differ from past snapshots: the prior guide records the original hashes and
reports actual input hashes rather than calling updated assets identical.
Do not silently replace mismatching public sources to reproduce a preferred score.

Checkpoints use trusted local PyTorch serialization to restore optimizer/RNG
state. Public generic GenePT pickle loading is restricted by known source hash
or an explicit trust flag; safe NPZ input is also supported. No dataset or
pretrained resource is redistributed under the repository's MIT license.
