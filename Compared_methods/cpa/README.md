# CPA: native compositional perturbation autoencoder

```bash
python -m Compared_methods.prepare cpa --directory external/CPA
bash Compared_methods/cpa/install.sh external/CPA work/envs/cpa
python -m Compared_methods.export_training --data data/processed \
  --splits work/splits --fold F1 --output work/native/F1/train.h5ad
```

`native_api.make_plan` loads the pinned, unchanged official core modules and
configures its native training plan. `train_step` uses target IDs from the
shared ordered axis (`-1` means control), dose 1 for a genetic perturbation,
and the registered Gaussian nonvariational likelihood. Native optimizers and
epoch counters belong to the plan; use the schedule in `benchmark_config.json`.
Optional top-level plotting, tuning and chemistry modules are not imported.
This runtime compatibility adaptation is disclosed in the configuration.

At prediction, `predict_cloud` draws from the native conditional Normal
likelihood over measured target-control states, using the condition-specific
registered random seed. The recorded export censors negative samples at zero
and retains raw negative-rate diagnostics. This is a disclosed adapter, not
handcrafted noise around an independently predicted mean.

Use 512 exported cells and report their actual float64-accumulated mean. Native
checkpoint selection follows the registered validation panels; do not use test
labels. The converter and common evaluator use these frozen outputs. Keep CPA
in its own environment because its legacy package metadata conflicts with the
newer compatibility runtime used by this benchmark.
