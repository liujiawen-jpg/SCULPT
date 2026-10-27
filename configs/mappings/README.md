# Frozen identifier metadata

These small TSV files map the source feature's Ensembl ID and gene symbol to
the fixed benchmark's stable gene identifier. They contain only public gene
identity metadata for the output/normalization panel, no expression values,
cells, task weights, or model results. All five source-specific tables currently
have identity Ensembl mappings on this panel; explicitly keeping the mapping
also lets the preparer detect future source changes and reject ambiguities.

The four scPerturb sources are the version-pinned public files in
`../datasets.yaml` (CC BY 4.0). Orion identity metadata originate from the
official Xaira `metadata/gene_metadata.parquet` (CC BY-NC-SA 4.0); data-source
licenses and citations remain applicable. See `../../docs/DATASETS.md`.
