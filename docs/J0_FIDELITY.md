# Fidelity to the frozen J0 joint-training implementation

This release implements **SCULPT, benchmark identifier J0**: the original
non-WaveGC model with all task modules trained jointly. The release does not
replace it with a newer variant, a phased trainer, or a predictor that adds a
learned correction to an old model's predictions.

## What was retained

| Part | Released behavior |
| --- | --- |
| Scientific model | Background-dependent source operators, shared-action connection solve, target readout, positive quantile shaping and learned rank rearrangement |
| Connection geometry | Training-control statistics and generic gene priors; local rank 4, ambient dimension 32; frozen partial graph restrictions |
| Output | 512-point deterministic quantile cloud over the registered gene axis; reported mean is the actual finite-cloud mean |
| Task initialization | Fresh task parameters for each fold and initialization seed; no old task weights or prediction inputs |
| Optimization | One joint phase, AdamW, effective 48-condition batches balanced over three query environments; gradient accumulation retains condition independence |
| Loss | Registered response, perturbation-discrimination, marginal, joint-structure and shared-action groups; two-epoch distribution-weight warmup |
| Training sources | One or two other legal training environments; held training conditions are excluded from every gradient role |
| Validation | One external panel weighted 1/2 and three internal held-condition panels weighted 1/6 each; every two epochs |
| Selection | Minimum fixed validation output objective; exact ties select the earlier checkpoint |
| Convergence | Registered plateau, learning-rate reductions, minimum epochs and optimizer steps; reaching a resource cap alone does not permit test evaluation |
| Final evaluation | Test labels remain unavailable to training and inference; scoring requires a sealed prediction file and selected checkpoint |

PDS is an evaluation score. J0 also retains the response-discrimination loss
used in its actual joint run; it has not been silently removed or substituted.
The target control is a legitimate biological reference. A source mean or an
old predictor's output is never added as the model's prediction backbone.

## Direct comparison with the frozen implementation

The release audit compared the original mathematical source and this package
in a separate local verification workspace. Original files were read only.

- Twelve core modules have identical parsed Python logic after normalizing
  package-relative imports and the extracted memory-accounting helper import.
- Twenty-eight registered scientific configuration fields match, including
  the model, losses, batch recipe, folds, seeds, validation and stopping budgets.
- On the same synthetic geometry, inputs and weights, the frozen implementation
  and the package produce identical shared actions, hurdle parameters,
  positive quantiles, final clouds, finite-cloud means, loss terms and all
  parameter gradients on CPU float32.
- The portable array-based geometry constructor matches the original
  file-based constructor exactly on a synthetic aligned graph.
- The joint training-loss assembly, including a nonzero discrimination loss,
  is compared against the frozen episode routine, rather than only against
  another function inside the package.
- The plateau and joint-plateau classes retain the original source logic.

These are implementation parity checks on small inputs, not a new biological
benchmark or a claim of identical GPU training trajectories.

## Included tests

Run `python -m pytest -q`. The suite includes a small scientific end-to-end
exercise and focused safeguards. Two additional independent checks cover:

1. **Dense normal-equation reference:** the matrix-free connection solution
   and its derivatives with respect to source operators, observations,
   observation precisions and prior centers agree with a separately assembled
   dense positive-definite system.
2. **Condition isolation:** changing one query's inputs leaves the other
   query's shared action and generated distribution unchanged.

Existing checks also cover batched versus single-condition output, exact
marginal preservation under rearrangement, bootstrap precision, input/asset
binding, role-restricted labels, immutable evaluation seals and fold-before-seed
aggregation semantics.

## Portability changes and limits

Server paths, historical cache discovery and resident-memory orchestration are
replaced by explicit data paths, bounded caches and a portable CLI. This changes
execution and serialization plumbing, not the model equations. Final scoring
also checks the current prepared-data manifest against the frozen fitted-asset
binding before opening prediction files or granting label access. This added
integrity check rejects replacement observations with unchanged gene/target
names; it does not change any numerical metric or prediction formula. The package
checkpoint format is a portable format; it is not claimed to load an arbitrary
historical server checkpoint without an explicit converter.

The release does not bundle datasets, task checkpoints, predictions or fitted
geometry. Downloaded input bytes, identifier mapping, preprocessing, dependency
versions and numerical hardware affect a fresh reproduction. No new complete
five-fold, three-seed benchmark was launched while updating this release.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md), [METRICS.md](METRICS.md) and the
core portability manifest for the broader protocol and documented evaluator
extensions. Those extensions do not change J0's training objective.
