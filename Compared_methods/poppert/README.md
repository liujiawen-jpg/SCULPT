# PopPert: registered whole-context comparison

**Scope:** exact settings, native model/objective/sampler wrappers and population
preparation primitives. This is **not an end-to-end training CLI**. The protected
fold reader, population cache writer, checkpoint/controller and evaluation
orchestration must be connected separately. No upstream code, model weights,
data, historical author identities or machine paths are redistributed.

## Official source and environment

- [whd1125/PopPert](https://github.com/whd1125/PopPert/tree/248fc3e1428de96335bef1d0271d30ca1e4d5565).
- Commit: `248fc3e1428de96335bef1d0271d30ca1e4d5565`.
- **Upstream has not selected a software license** at this revision; see its
  README License section. Our repository license does not grant upstream reuse
  or redistribution rights. Obtain official sources separately under applicable
  upstream terms; this folder contains only our integration code and settings.
- Recorded runtime: Python3.12, PyTorch2.7.1/CUDA12.8, NumPy1.26.4,
  SciPy1.15.2, scikit-learn1.5.2. Use an isolated environment.

```bash
# Given a separately obtained pinned checkout:
bash Compared_methods/poppert/install.sh external/PopPert work/envs/poppert
```

`configure_upstream()` verifies hashes of official `model.py`, `sampling.py`
and `gmm_utils.py`. It fails on a different checkout or conflicting previously
imported generic modules. Upstream source is never copied into this package.

## Preserved model and disclosed adaptations

The unchanged official gene-mode `DistributionTransformer` uses width256,
4heads, 4layers, feedforward1024 and dropout0.1. It predicts two-component
zero-inflated Gaussian marginal parameters and rank32 Gaussian-copula factor
variances. The native sampled truncated-ZIGMM NLL plus0.2 times the log factor
variance objective is used. There is no custom PDS loss or old predictor anchor.

The benchmark uses five fixed whole-context folds and seeds17/29/43, all5,979
output genes and1,525 targets. The stable-ID union forms6,333 trainable gene
and target embeddings. The optional external pretrained embedding was not
available and is not replaced. The optional cell-identity branch is disabled
(`n_cell_types=0`) to avoid unseen, untrained context-ID embeddings. Target
context is provided through its measured control distribution parameters.

`population_helpers.py` preserves these preparation rules:

1. Only4,116 legal training conditions enter response distribution fitting.
   Keep A2048 normalization and the frozen gene order; do not select HVGs.
2. Native128-cell populations use `max(5,floor(n/128))` groups: natural
   contiguous blocks first, then seeded sampling (replacement only if n<128).
   Frozen observed-cell pools are capped at256, producing five populations
   per condition and20,580 training populations per fold.
3. Native `fit_batch_gmm` fits2mixtures; native `expr_to_gaussian` and
   `learn_factor_loading` fit rank32 copula on **training controls only**.
   Held validation/test controls are transformed by that frozen basis.
   Native `factor_variance` is used without additional optimization.
4. Pair each training population with a uniformly selected one of five
   same-context control populations. Training order is `torch.randperm`.

AdamW uses LR1e-4, weight decay1e-4, batch16 and gradient clipping1.0.
Transformer computation uses BF16 on CUDA; parameters passed to native
likelihood/sampling are FP32. Validate every epoch, minimum20, patience15,
maximum100, with ReduceLROnPlateau(factor0.5,patience3). Select using the same
fixed weighted response+distribution score as other methods (see config).
A strict score decrease resets patience; test labels never select weights.
Small nondeterministic GPU numerical drift was observed, so bitwise retraining
identity is not claimed. Independent process parallelism changed no model,
loss, sampling, validation frequency or scientific hyperparameters.

## Callable interface

```python
from Compared_methods.poppert import native_api as api
api.configure_upstream('external/PopPert')
config = api.configuration()
model = api.create_model(vocab_size=6333, seed=17, cfg=config, device='cuda')
# x_ctrl[B,G,7], y_target[B,G,7]; gene_indices[G], target labels[B];
# s_ctrl/s_target[B,32] under a training-control-only copula basis.
value, diagnostics = api.loss(model, x_ctrl, y_target, gene_indices,
                              labels, s_ctrl, s_target, config)
# After legal training and validation selection, native sampling returns[B,512,G].
cloud = api.cloud(model, x_ctrl, gene_indices, labels, s_ctrl,
                  factor_loadings, ctrl_psi, config, generation_seed)
```

Prediction uses one deterministic target-control population per environment:
`generation_seed(seed,environment,'control_population') %5`. Keep the official
`[0,14.9]` export clipping, and compute the reported mean from those512 exported
samples in float64. Use the common converter with
`--method poppert --support-semantics iid_samples`, and the frozen common metric
geometry. Do not compare only analytic means while scoring a different cloud.
Average seeds within each fold, then compute the five-fold mean and sample SD.

## Checks

```bash
PYTHONDONTWRITEBYTECODE=1 work/envs/poppert/bin/python -B \
  -m Compared_methods.poppert.check_helpers --upstream external/PopPert
```

These are finite synthetic CPU shape/loss/sampling checks, not new training or
a repeat of the full five-fold study. Fitting helpers rely on their caller to
enforce fold access; they do not themselves inspect a split manifest.
`source_provenance.json` records original adapter hashes without private paths.

### Explicit-device portability correction

The published wrapper resolves the requested GPU (`cuda:2`, for example) for
memory limits and sampling RNG context. Cloud generation saves/restores the
CPU and selected-GPU RNG streams, including on exceptions, and does not seed
unrelated visible GPUs. Model construction deliberately initializes host RNGs
and that run's chosen GPU; it does not reset other GPUs. Historical execution
exposed one GPU as `cuda:0`; its native network, seed values, loss, sample law,
clipping and output order are unchanged. These device guards are an explicit
portability exception to function-level source identity, recorded separately
in `fidelity_check.json`. They are tested without reserving a real GPU:

```bash
PYTHONDONTWRITEBYTECODE=1 work/envs/poppert/bin/python -B \
  -m Compared_methods.poppert.check_device_portability
```
