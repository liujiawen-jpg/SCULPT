# STATE (official State Transition model)

This comparator uses Arc Institute's official `StateTransitionPerturbationModel`
from revision `9bbfe78a434a55205e4de834e1ea99f85f7a3add`. Third-party project:
<https://github.com/ArcInstitute/state>, **CC-BY-NC-SA-4.0**. The official model
source is fetched separately; it is not relicensed under this repository's MIT
license. The upstream authors' notices remain in the official checkout.

## Installation

Run from this repository root, using a new environment:

```bash
python -m Compared_methods.prepare state --directory external/state
bash Compared_methods/state/install.sh external/state work/envs/state
export STATE_UPSTREAM="$PWD/external/state"
```

The bridge verifies the revision and six upstream scientific-source SHA-256
hashes from `upstream_provenance.json`. It refuses another installed `state`
package. The actual recorded runtime used Python 3.11, CUDA PyTorch 2.7.1,
Transformers 4.52.3 and GeomLoss 0.3.1. This release contains no pretrained
State Embedding (SE) weights and does not use SE pretraining.

## Scientific configuration and API

`benchmark_config.json` records the actual comparison, including all
benchmark adaptations. `native_api.py` preserves the official model construction,
forward call, native energy loss and Adam optimizer. It does not replace STATE
with a smaller network. The base model has hidden size 768, eight transformer
layers and twelve attention heads. It receives control sets and one-hot target
identities; the target-control background is not an untrained cell-line ID.

- `build_model(G, P + 1, gene_names=genes, device=...)`: fresh task weights.
- `make_batch(...)`: homogeneous condition sets of shape `(B, 512, G)`;
  control perturbation category is index `P`, perturbed categories are `0..P-1`.
- `model.configure_optimizers()`: official Adam, LR `1e-4`, no scheduler.
- `train_step(...)`: native energy loss in FP32, CUDA forward BF16, clip norm 10.
- `sampling.training_conditions(store)` and `draw_training_batch(...)`: legal
  fold conditions only; independently resample 512 control and 512 response
  cells with replacement. One shuffled traversal of 4,116 P groups plus three
  E groups is an epoch; condition batch size is 16.
- `sampling.predict_condition(...)`: the official 512-row ReLU output, using
  the registered per-condition control bootstrap RNG key. No new generator,
  censoring, correction, or old prediction anchor is attached.

The official YAML calls the fixed expression input `X_hvg`. Here this is the
shared 5,979-gene axis, **not** a new high-variance gene selection step. Native
`predict_residual=True` is an internal residual on observed controls, not a
residual on an old predictor. The official auxiliary decoder is retained but
unused for these `X_hvg`/gene batches.

The packaged bridge and sampler are callable primitives, not a second
self-contained 15-job orchestration system. Follow the benchmark validation
panels and stopping policy in the config when integrating a trainer. The
original comparison used full validation every two epochs, minimum ten epochs
and 1,500 optimizer updates, six nonimproving checks, relative delta 0.002,
60/120-epoch initial/extended budgets. Selection minimizes the fixed weighted
`R_delta + CRPS_normalized + 0.25 * centered_joint_SW1_normalized`; this is a
**benchmark adaptation**, not an upstream default. Test labels are opened only
after convergence, checkpoint selection and completed prediction sealing.

Export native arrays `mean[T,G]` and `support[T,512,G]` to the shared converter
with method `state`; mean must be the float64-accumulated cloud mean cast to
float32. Do not assign a distribution to mean-only comparison methods.
