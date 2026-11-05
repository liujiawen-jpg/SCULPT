# PIE: full native architecture and multi-source knowledge

```bash
python -m Compared_methods.prepare pie --directory external/PIE
bash Compared_methods/pie/install.sh external/PIE work/envs/pie
python -m Compared_methods.download_knowledge pie \
  --directory data/comparison_knowledge/PIE --dry-run
PYTHONPATH=src python -m Compared_methods.preprocess pie \
  --data data/processed --splits work/splits --fold F1 \
  --upstream external/PIE --output work/native/F1/PIE
```

Knowledge files must come from the pinned PIE_sources revision. Preserve all
six source modalities, complete NCBI token sequences and the official missing
source mask. `gene_text` is also required for all output-gene queries. Use
stable gene identity and audited public aliases for remapping; do not truncate
the architecture or synthesize embeddings for missing sources.

The preprocessing command writes official native label arrays with only legal
training conditions. Follow [PREPROCESSING.md](../PREPROCESSING.md) for the
fixed-normalization adaptation. Native evidence, DE/delta-p statistics, clipping
scales and bins fit only these legal rows, never internal-held or query labels.

`native_api` exposes the complete official 768-wide model (512 latents,
4/6/4 encoder/processor/decoder blocks), native configuration, optimizer and
four-term objective. `benchmark_config.json` fixes the effective batch 128,
5000-update schedule, source use and validation-AUPRC selector. Do not replace
this by a regression head with the same method name.

The upstream train/data interfaces consume native labels, evidence and source
lookups. These orchestration layers are separate from the supplied model API.
The native categorical delta-p head predicts a mean shift, not individual-cell
variability. Read out its official bin center and reconstruct the nonnegative
absolute mean under the registered target-control rule; report distribution
metrics as unavailable. Do not attach a new cell generator to this comparison.
