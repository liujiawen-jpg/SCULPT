# scPILOT: native latent transport and adaptive source weighting

```bash
python -m Compared_methods.prepare scpilot --directory external/scPILOT
bash Compared_methods/scpilot/install.sh external/scPILOT work/envs/scpilot
python -m Compared_methods.export_training --data data/processed \
  --splits work/splits --fold F1 --output work/native/F1/train.h5ad
```

`native_training.make_plan` and `one_step` preserve the official EGD network,
losses and two-Adam training step. The 800-hidden/100-latent architecture,
32-cell minibatches and full FP32 runtime are registered in the configuration.
Only the detached diagnostic MMD calculation and logging plumbing are adapted.
Do not train separate shortcut regressors for each perturbation.

Use the fixed source-context training cells to fit the representation. At
inference, the native pipeline uses latent OT, Leiden structure and adaptive
weights to transfer the same perturbation to the measured target controls.
`native_inference.py` and `stable_adaptive_weights.py` preserve this mechanism;
the stable shifted exponential avoids numerical 0/0 without changing the
normalized exponential weights. Follow the configuration's allowed source
conditions, sample counts, checkpoint selection and convergence policy.

The native cell cloud can be signed. Do not clip it, replace it by a source mean,
or silently calibrate it. Export 512 native cells and their actual mean with
the declared signed-support semantics. The common converter/evaluator supports
this output without inventing nonnegative cells.
