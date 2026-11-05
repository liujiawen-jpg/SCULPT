# Public priors and STRING preparation

J0 uses **frozen generic gene embeddings**, not a fine-tuned foundation-model
encoder or a historical perturbation predictor. No pretrained weights or
embedding arrays are included in this repository. All task parameters are
initialized afresh for each fold and seed.

## Resources

| Resource | Official source | Required version / artifact | Terms |
|---|---|---|---|
| Tahoe-x1 | [Official model repository](https://huggingface.co/tahoebio/Tahoe-x1/tree/main/1b-model), [code](https://github.com/tahoebio/tahoe-x1) | `1b-model/model.safetensors`, `1b-model/vocab.json`; the gene embedding must have 2,048 columns | Apache-2.0, as stated in the [model card](https://huggingface.co/tahoebio/Tahoe-x1) |
| GenePT | [Official repository](https://github.com/yiqunchen/GenePT), [Zenodo v2](https://zenodo.org/records/10833191) | Extract `GenePT_gene_embedding_ada_text.pickle` from `GenePT_emebdding_v2.zip` (the published spelling is intentional); 1,536 columns | CC-BY-4.0 for the Zenodo data; see the deposited record. No upstream code is copied here. |
| STRING | [Version 12.0](https://version-12-0.string-db.org/), [downloads](https://version-12-0.string-db.org/cgi/download) | Human `9606.protein.links.v12.0.txt.gz`, `9606.protein.info.v12.0.txt.gz`, `9606.protein.aliases.v12.0.txt.gz` | [CC-BY-4.0](https://version-12-0.string-db.org/cgi/info) |
| HGNC | [Official download documentation](https://hgnc.genenames.org/download/), [current complete TSV](https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt), [archives](https://hgnc.genenames.org/download/archive/) | An explicitly archived complete HGNC TSV; changing this file can change identifier mappings | [CC0](https://hgnc.genenames.org/about/license/) |

The repository's code license does not replace these resource licenses.
Cite the resource papers and retain attribution when distributing derived assets.
The GenePT archive already contains the required vectors; no paid embedding API
or text-model request is needed.

Download resources into a user-chosen folder, for example `data/external/`.
The repository does not download multi-GB checkpoints automatically.

```bash
python scripts/download_resources.py --resources tahoe genept string --output data/external --dry-run
python scripts/download_resources.py --resources tahoe genept string --output data/external
unzip data/external/GenePT_emebdding_v2.zip -d data/external/
```

HGNC must be an archived complete TSV, with its independently checked hash:

```bash
python scripts/download_resources.py --resources hgnc --output data/external \
  --hgnc-url "$HGNC_ARCHIVE_URL" --hgnc-sha256 "$HGNC_ARCHIVE_SHA256"
```

Set these two variables to the chosen official archive URL and SHA256 before
running. The original mapping hash is documented below. A current HGNC TSV
can produce a new mapping, but must not be described as the historical one.

Use a fixed Hugging Face revision and an HGNC archive for a new reproducible
release. The exact original inputs had the following SHA-256 values. The script
records all actual input hashes and whether the Tahoe/GenePT inputs match them;
it does not falsely label updated HGNC mappings as bitwise historical replication.

| Original artifact | SHA-256 |
|---|---|
| Tahoe 1B safetensors, 5,415,458,164 bytes | `21f2afc7ee5ec88cadddd6743188aea590b89306c8f2538e137bf9dcacfc0260` |
| GenePT ada-text pickle | `fd297510ddd3040744033fde0b0f2cf15a40ac8b2fd2fb02f10667295e55c862` |
| HGNC complete TSV | `6f43d6ff43aa9fdfa5fb2f20a20a7cace66e6e02e2a0dcf19d9b726e2e248d20` |
| STRING links v12.0 | `3e22f32572211aa341d5b4bd08d30c32e693e294603202120936872f87719d4f` |
| STRING info v12.0 | `144de4b0d98c6a7dfde6ddc2591cf88657f27b989eadff4f501450c3ed1f0f1c` |
| STRING aliases v12.0 | `b65f730b993ed0c1bd72edf4565d3d425db42861101b29699704810e8f125680` |

## 1. Build STRING gene graph

```bash
python scripts/prepare_priors.py string \
  --hgnc data/external/hgnc_complete_set.txt \
  --links data/external/9606.protein.links.v12.0.txt.gz \
  --info data/external/9606.protein.info.v12.0.txt.gz \
  --aliases data/external/9606.protein.aliases.v12.0.txt.gz \
  --genes configs/axes/primary_genes.tsv \
  --targets configs/axes/targets.tsv --out data/string
```

Mapping uses approved HGNC identities and supported stable STRING alias tags.
Conflicting or ambiguous mappings are rejected. Exact approved symbols are used
only when stable evidence is absent and unambiguous. No fuzzy aliases are used.
The node axis contains all approved, uniquely identified HGNC genes, plus the
requested benchmark genes and targets. Edges satisfy `combined_score >= 700`;
gene-level duplicate edges retain their maximum score; self-loops are removed.
`A` is symmetric with weights `score/1000`. `P` is row-normalized `A`, with zero
rows for isolates, no self-loops and no teleportation. STRING associations are
undirected functional evidence, not causal regulatory edges.

Outputs include `source/node_axis.tsv`, `graphs/KS/A.npz`, `graphs/KS/P.npz`,
protein-mapping audit and input-hash provenance. These generated data belong
outside source control.

## 2. Build 261-dimensional prior

Adjust the extracted GenePT path if the archive creates an additional subfolder.

```bash
python scripts/prepare_priors.py prior \
  --genes configs/axes/primary_genes.tsv --targets configs/axes/targets.tsv \
  --tahoe-weights data/external/model.safetensors \
  --tahoe-vocab data/external/vocab.json \
  --genept data/external/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle \
  --hgnc data/external/hgnc_complete_set.txt \
  --graph-directory data/string --out data/priors
```

`prior.npz` stores ordered `genes` and `embedding` `[genes,261]`. It covers the
union of output genes and perturbation targets. The last five columns are
**availability flags**, with one meaning a nonzero vector is available.

For each gene, five blocks are used: Tahoe 2,048D, GenePT 1,536D, STRING
`F0` 1,536D, `P F0` 1,536D and `P² F0` 1,536D. `F0` is the L2-normalized public
GenePT vector on the entire graph. A missing or ambiguous GenePT mapping yields
a zero vector. Each block is independently L2-normalized and projected with
`R_b ~ Normal(0,1)/sqrt(d_b*5)` into 256 columns; the five results are summed.
The random generator is NumPy `default_rng(17)`, in the stated block order.
Five availability flags are then appended. No response values fit this prior.

Propagation uses all graph nodes, including public off-panel neighbors. The
archived `P F0` is stored as float32; the original prior builder computed `P² F0`
from that stored matrix using float64 sparse multiplication, then cast to float32.
This rounding order is retained. A missing direct embedding may acquire a
nonzero propagated block; no experimental expression is used to invent a vector.

For exact old generic mappings, `--frozen-genept-features` optionally accepts a
public-feature NPZ with `target_ids`, normalized `l2` rows and `missing_mask`.
It preserves the original generic target rows in graph `F0`; it must never
contain task-trained weights or perturbation predictions. This optional external
file is not included in the repository. Without it, an updated HGNC snapshot
regenerates mappings and must be reported as such.

The verified official GenePT pickle is accepted by its fixed hash. Other trusted
pickle files require `--trust-genept-pickle`; alternatively pass a safe NPZ with
unique `symbols` and `[N,1536]` `embedding` arrays. Generated `H1_all.npy` and
`H2_all.npy` are temporary reusable public-feature assets, not repository files.

## Model API

```python
from cosit.priors import build_prior

prior_path = build_prior(
    gene_ids,
    tahoe_weights="data/external/model.safetensors",
    tahoe_vocab="data/external/vocab.json",
    genept="data/external/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle",
    hgnc="data/external/hgnc_complete_set.txt",
    graph_directory="data/string",
    out="data/priors",
)
```

## Automated acquisition

```bash
python scripts/download_resources.py --resources tahoe genept string --output data/external --dry-run
python scripts/download_resources.py --resources tahoe genept string --output data/external
```

The default immutable Tahoe revision is `d218a580b9c2500ae9dfc8367a398545e6f017a8`.
Downloads with registered expected hashes fail rather than silently substituting
changed assets. For HGNC, supply both an official archived URL and its SHA256:
`--resources hgnc --hgnc-url URL --hgnc-sha256 SHA256`. No latest-HGNC fallback
is performed by the downloader. GenePT ZIP extraction and the prior builder
retain the pickle hash verification described above. A successful download is
not itself a successful benchmark reproduction; record actual mappings.
