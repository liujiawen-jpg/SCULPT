# CellFlow: registered whole-context comparison

**Scope:** exact registered settings plus callable native numerical helpers.
This folder is **not a standalone reproduction/training CLI**: the protected
fold reader, streaming PCA builder, training controller, native checkpoint
storage and common evaluator orchestration must be connected separately.
No historical machine paths, task weights, data or upstream code are bundled.

## Official source and environment

- [theislab/CellFlow](https://github.com/theislab/CellFlow/tree/446ed6073c60ac2e8db13c4ea096a43cdec288b2)
- Commit: `446ed6073c60ac2e8db13c4ea096a43cdec288b2`.
- License: upstream `LICENSE`, **PolyForm-Noncommercial-1.0.0**.
- Python 3.11; JAX/JAXlib 0.6.2, Flax 0.10.7, Optax 0.2.5, Diffrax 0.7.0.
  GPU recipe requires a compatible CUDA12 driver. The recorded run used CUDA12.8.
  This isolated environment differs from SCULPT's PyTorch environment.

```bash
python Compared_methods/prepare.py cellflow --directory external/CellFlow
bash Compared_methods/cellflow/install.sh external/CellFlow work/envs/cellflow
```

Install recipes are supplied for reproducibility; packaging verification does
not claim to have reinstalled environments or retrained finished experiments.

## Preserved model and disclosed adaptation

`native_api.py` invokes official `ConditionalVelocityField`, `OTFlowMatching`,
`ConstantNoiseFlow(0.5)`, balanced linear OT (`epsilon=0.5`) and the native ODE
prediction. Architecture and settings are in `benchmark_config.json`.
The PBMC tutorial's cytokine embeddings are replaced by the officially
supported one-hot genetic-target input; matched-control expression means
encode background. No custom response loss, prediction residual, previous
SCULPT weights or previous predictions enter the model.

The five immutable folds, 5,979 genes, 1,525 known targets and three seeds
17/29/43 come from the common protocol. Only the 4,116 authorized training
conditions and three training control groups enter training/PCA fitting.
Use the exact existing A2048 normalized expression; do not choose new HVGs.
Fit centered PCA100 on **all legal cached training E and P cells**, with no
held-context controls or held perturbations. Held control cells are only
transformed. The registered builder uses float64 sums, float32 covariance
accumulation, symmetric float64 eigendecomposition, descending eigenvalues,
and signs fixed by each loading's largest absolute entry. Save its gene axis.

Each iteration chooses one legal condition uniformly and independently samples
1,024 same-context control and perturbation cells with replacement. Adam
uses LR `5e-5`, with20-iteration gradient accumulation. Validate every10,000
iterations, minimum50,000, maximum500,000, patience6. Patience counts absent
0.2% relative score improvement; checkpoint selection still retains every
strictly lower score. Iterations are not optimizer updates. Use the fixed
weighted validation selector `R_delta+CRPS_normalized+0.25*centered_SW1_normalized`;
never select on test outcomes. Budget exhaustion is not claimed convergence.

At prediction, independently bootstrap512 observed target control cells using
`generation_seed(seed,environment,target)`, apply train-fit PCA, native ODE,
and the official linear inverse PCA. **Signed outputs are retained.** Do not
clip, add the source mean, or restore the discarded PCA residual. The reported
mean is the float64 mean of the exported cloud, then cast to float32.

```python
from Compared_methods.cellflow.native_api import make_solver, condition, predict
# pca has mean[G], components[100,G], fitted on authorized training cells.
solver, config = make_solver(seed=17, n_targets=1525, n_genes=5979)
# Train solver.step_fn with protected data and select a checkpoint first.
# Here solver must already contain that selected checkpoint's native weights.
cloud = predict(solver, controls512, pca,
                condition(target_index, full_target_control_mean, 1525))
```

Export cloud `[targets,512,5979]` plus its exact mean through the common converter,
using `--method cellflow --support-semantics iid_samples`; here samples are the
empirical-control bootstrap pushed through the flow, not added output noise.
Record raw signed-output fraction. Use the same training-only evaluation
geometry as the other methods; average seeds within folds before fold SD.

## Checks and limitations

`check_helpers.py` checks registered settings, target/context shapes and the
native PCA/inverse-PCA data path on synthetic arrays. `--native-smoke` also
constructs a deliberately tiny official network on CPU without training it.

```bash
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu \
  work/envs/cellflow/bin/python -B -m Compared_methods.cellflow.check_helpers
```

The portable functions preserve the completed benchmark's numerical model
configuration. They do not reproduce hardware-specific CUDA library paths,
controller scheduling, filesystem caches or bitwise weights across hardware.
`source_provenance.json` records source fingerprints without private paths.
