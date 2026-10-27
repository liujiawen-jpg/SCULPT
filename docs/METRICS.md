# Evaluation definitions

Metrics are internal normalized-log expression metrics, **not official VCC
leaderboard metrics**. `evaluate` never changes a prediction or checkpoint.
Primary reporting separates perturbation identity/response from cell-distribution
fit. Raw MAE and MSE are not included in the primary report, although their
mathematical training/audit helpers remain in `core` for provenance.

## Perturbation response

For target `p` in held context `c`, subtract the **full eligible E control mean**
from both the predicted mean and the observed condition mean. This gives the two
response vectors. The primary per-condition metrics exclude the directly
perturbed gene when it is on the output axis.

| Metric | Definition / interpretation | Better |
|---|---|---|
| PDS | Rank of the correct real response among candidate real responses using raw response cosine | Higher |
| Pearson | Centered response-vector correlation | Higher |
| Spearman | Pearson correlation of average ranks | Higher |
| DEG-AP | AP of absolute predicted response against observed DE labels | Higher |
| JAC20/50/100 | Set Jaccard of largest absolute predicted/real responses | Higher |
| Signed Top100 coverage | Correctly signed overlap divided by the larger selected set size | Higher |
| Direction accuracy | Sign agreement over nonzero real responses | Higher |
| RMS amplitude ratio | Predicted response norm / real response norm | Near 1 |
| Paired/wrong cosine and margin | Identity-separation diagnostics | Paired/margin higher |

### Correct PDS computation

PDS is **not** the cosine of a single paired prediction and truth. In one held
context, let `s[p,q] = cosine(predicted_delta[p], real_delta[q])`. All target
genes belonging to the candidate panel are excluded from the common competition
gene axis. For each informative true condition, compute

```text
rank = number(s[p,q] > s[p,p]) + (number(s[p,q] == s[p,p]) - 1)/2
PDS[p] = 1 - rank/(number_of_informative_candidates - 1)
```

Cosine uses raw response vectors without gene-wise centering, whitening or
standardization. Exact equality receives midrank ties. A zero predictor has PDS
0.5 when at least two informative candidates exist. Zero-energy truths and
single-candidate panels are undefined, not scored as successes. PDS changes when
candidate panels change; compare methods only on the identical panel.

### DEG labels and top genes

DEG-AP uses a two-sided, tie-corrected asymptotic Mann–Whitney U test with
continuity correction, then Benjamini–Hochberg FDR 0.05 over eligible genes.
The perturbed gene is removed before BH. The observed condition and the fixed
capped target E sample define DE labels, independent of predictions. All-tied
pooled columns are retained with p=1; other nonfinite test outputs fail.
No positive DE label means AP is undefined. AP scores absolute predicted response
and does not by itself measure direction correctness.

Top-K ties are broken by stable gene ID. Exact zero predicted responses do not
become responsive discoveries by alphabetical tie-breaking. JAC100 is an
absolute-response Top100 set score, not a claim to the official VCC JAC.

## Predicted cell distributions

Use the **complete emitted 512-cell cloud** and all immutable real condition
cache cells (typically 20–256). This cache is not necessarily the complete raw
experimental population. Include all 5,979 genes, including the perturbation
target. All scales, PCA directions, state prototypes and MMD bandwidths are
fixed from permitted training information.

| Metric family | Definition | Better |
|---|---|---|
| Marginal W1 | Exact 1D empirical Wasserstein distance per gene, divided by training control SD (floor 0.05) | Lower |
| CRPS | Expected absolute observation error minus half the predicted pairwise absolute distance | Lower |
| Sliced W1 | Empirical W1 along 128 fixed random full-gene projections, seed 29 | Lower |
| Centered sliced W1 | Same after removing each population's own mean, for scoring only | Lower |
| Energy distance | Empirical energy V-statistic on fixed full-gene subsamples, with normalized Euclidean distances | Lower |
| Gaussian MMD² | Unbiased multi-kernel estimate on three fixed subsampling repeats | Lower |
| PCA covariance error | Relative Frobenius error of covariance in training-control PCA coordinates | Lower |
| Detection/variance/quantile/state diagnostics | Complementary support and shape checks | Metric-dependent |

MMD uses full genes, per-gene training-control scales, up to 128 cells from each
population per repeat, and three deterministic repeats (seed 20261021). Kernel
sigmas are 0.5×, 1× and 2× the median positive pairwise distance among a fixed
sample of at most 256 training controls. Bandwidths are never fitted from test
cells. Unbiased MMD² can be negative; it is not clipped to zero.

The release also provides a supplemental energy statistic (not a historical
training objective): `2 mean d(X,Y) - mean d(X,X) - mean d(Y,Y)`, with self-pairs
included and `d(x,y)=||(x-y)/scale||₂/sqrt(G)`. It uses the same deterministic
up-to-128-cell, three-repeat sampling protocol as MMD. This is the energy
V-statistic without taking its square root; compare only identically defined
implementations. Marginal, sliced and covariance metrics use the complete
emitted support rather than this subsample.

Generated supports are deterministic quantile clouds. Their cells are not
claimed to be iid biological samples. A learned rank permutation changes joint
structure while preserving every gene's marginal, including its finite-support
mean. This is checked before scoring. Good mean or W1 performance alone does
not establish covariance or dependence recovery.

Native competitor outputs retain their declared boundary convention. The
registered CPA Gaussian adaptation censors sampled negative expression at zero
and reports that adaptation separately. Native signed scPILOT predictions are
not silently clipped; the common distance evaluator allows signed competitor
supports and reports their negative fraction. J0 predictions remain strictly
nonnegative. Mean-only TxPert/PIE outputs have no distribution metrics.

## Averaging and missing values

Metrics are first averaged across test conditions within a context. Three seed
results are then averaged within each context, followed by equal five-context
aggregation. Report sample SD/variance (ddof=1) across context means, and seed
variability separately. Null metrics retain valid counts. Do not replace
undefined correlations/AP/covariance ratios with favorable zeros.
