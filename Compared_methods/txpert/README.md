# TxPert: public multi-graph model

Use the exact source revision and independent runtime from the shared registry:

```bash
python -m Compared_methods.prepare txpert --directory external/TxPert
bash Compared_methods/txpert/install.sh external/TxPert work/envs/txpert
python -m Compared_methods.download_knowledge txpert \
  --directory data/comparison_knowledge/TxPert --dry-run
```

Remove `--dry-run` to obtain the public GO and STRING v11.5 graph files at the
fixed official commit, with the registered SHA256 checks. These are distinct
from SCULPT's STRING v12 graph. No proprietary PxMap/TxMap resource is claimed.

Export authorized single cells using `Compared_methods.export_training`.
Use the full pooled E mean of each cell's own environment as the control input;
do not substitute the mean of the capped control cloud. Build an unambiguous
stable-ID-to-symbol dictionary from the fixed mapping TSVs. `native_api.make_graph`
retains unmapped targets as distinct isolated nodes instead of dropping them.

The native API exposes `make_graph`, `make_model`, `train_step`,
`predict_mean`, `graph_state` and `restore_graph`. It preserves the official
public MG model and native loss. Save both the model state and `graph_state`:
the official module has graph tensors that are not registered buffers.
Follow the training and selection recipe in `benchmark_config.json`; the
upstream repository does not publish its original training controller.

Only absolute-expression means are exported. No single-cell distribution is
invented. Native means pass through `Compared_methods.convert_predictions`
unchanged, then the common sealed evaluator. Keep the official upstream
noncommercial license separate from this repository's MIT license.
