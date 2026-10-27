"""Internal response and finite-support distribution evaluation.

JAC is absolute-response TopK Jaccard, not the official VCC JAC. Primary
response metrics exclude each condition's own perturbed gene; PDS additionally
excludes all panel targets. TopK ties are resolved by stable gene identifier.
Zero-valued entries are not claimed as response discoveries, and zero-truth
conditions have undefined JAC/relative error rather than artificial perfect
scores. No dataset is opened, no scale fitted, and no checkpoint is selected.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from scipy.stats import rankdata

EPS = 1e-12


def _average(values):
    valid = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(valid)) if valid else None


def _correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.ptp(y) == 0:
        return None
    if np.ptp(x) == 0:
        return 0.0
    xc, yc = x - x.mean(), y - y.mean()
    return float(np.clip(xc @ yc / (np.linalg.norm(xc) * np.linalg.norm(yc)), -1, 1))


def _top_indices(values, valid, lex, k, threshold=None):
    indices = lex[valid[lex]]
    # Zero entries have no responsive-gene identity; do not reward arbitrary
    # lexical ties between zero predictions and zero ground truth.
    indices = indices[np.abs(values[indices]) > 0]
    if threshold is not None:
        indices = indices[np.abs(values[indices]) >= threshold]
    return indices[np.argsort(-np.abs(values[indices]), kind='stable')[:k]]


def _top_metrics(pred, truth, valid, lex, k, threshold=None):
    yt = _top_indices(truth, valid, lex, k, threshold)
    ft = _top_indices(pred, valid, lex, k, threshold)
    ys, fs = set(yt.tolist()), set(ft.tolist())
    if not ys:
        return None, None, len(ys), len(fs)
    common = ys & fs
    jac = len(common) / len(ys | fs)
    signed = sum(np.sign(pred[j]) == np.sign(truth[j]) for j in common) / max(len(ys), len(fs))
    return float(jac), float(signed), len(ys), len(fs)


def cosine_retrieval_metrics(pred, truth, target_gene_ids, gene_ids, *, compute_pds=True):
    """Raw response cosine; panel-target exclusion; exact equality midrank ties.

    The control subtraction is supplied by the caller. No gene-wise centering,
    fitted whitening, or Pearson normalization is applied here. Zero predictions
    have cosine zero and chance PDS; zero true vectors are uninformative.
    """
    f, y = np.asarray(pred, np.float64), np.asarray(truth, np.float64)
    if f.ndim != 2 or y.shape != f.shape or not np.isfinite(f).all() or not np.isfinite(y).all():
        raise ValueError('finite aligned response matrices required')
    targets, genes = list(map(str, target_gene_ids)), list(map(str, gene_ids))
    if len(targets) != len(f) or len(genes) != f.shape[1] or (compute_pds and len(set(targets)) != len(targets)):
        raise ValueError('PDS requires one unique target per condition in one environment')
    panel_targets = set(targets)
    mask = np.asarray([g not in panel_targets for g in genes])
    if not mask.any():
        raise ValueError('PDS target exclusion leaves no competition genes')
    fp, yp = f[:, mask], y[:, mask]
    nx, ny = np.linalg.norm(fp, axis=1), np.linalg.norm(yp, axis=1)
    ids = np.flatnonzero(ny > EPS)
    pair = np.divide(np.einsum('ij,ij->i', fp, yp), nx * ny,
                     out=np.zeros(len(f)), where=nx * ny > 0)
    paired = [float(np.clip(pair[i], -1, 1)) if ny[i] > EPS else None for i in range(len(f))]
    pds, wrong, margin = [None] * len(f), [None] * len(f), [None] * len(f)
    if compute_pds and len(ids) >= 2:
        # Exact zero predictor shortcut, including the zero-response baseline.
        if not np.any(fp):
            for row in ids:
                pds[row], wrong[row], margin[row] = .5, 0., 0.
        else:
            den = nx[:, None] * ny[ids][None, :]
            sim = np.divide(fp @ yp[ids].T, den,
                            out=np.zeros((len(f), len(ids))), where=den > 0)
            sim = np.clip(sim, -1, 1)
            for column, row in enumerate(ids):
                diagonal = sim[row, column]
                rank = np.sum(sim[row] > diagonal) + (np.sum(sim[row] == diagonal) - 1) / 2
                pds[row] = float(1 - rank / (len(ids) - 1))
                # Use the same matrix diagonal as PDS for exact comparability.
                paired[row] = float(diagonal)
                wrong[row] = float((sim[row].sum() - diagonal) / (len(ids) - 1))
                margin[row] = float(diagonal - wrong[row])
    return dict(PDS_delta=pds, paired_raw_cosine=paired, wrong_pair_raw_cosine=wrong,
                raw_cosine_margin=margin, competition_genes=int(mask.sum()),
                informative_candidates=len(ids))


def internal_pds(pred, truth, target_gene_ids, gene_ids):
    return cosine_retrieval_metrics(pred, truth, target_gene_ids, gene_ids)['PDS_delta']


def response_metrics(pred, truth, target_gene_ids=None, gene_ids=None, tau=None,
                     *, top_ks=(20, 50, 100), compute_pds=False,
                     compute_global=False):
    """Evaluate aligned [conditions,genes] normalized-log responses.

    ``tau`` is an optional positive strong-response threshold supplied from
    TRAINING ONLY. Nonfinite input is rejected. ``top_ks`` may extend (20,50,
    100), but these standard three are always reported. Gene identifiers must
    be unique; omission uses fixed column indices as the tie-breaking axis.
    Output contains historical aggregate keys plus ``aggregate``, condition
    arrays under ``condition_metrics`` and records under ``per_condition``.
    Composite selection lives in evaluation.py; this function only measures.
    """
    f, y = np.asarray(pred, np.float64), np.asarray(truth, np.float64)
    if f.ndim != 2 or f.shape != y.shape or len(f) < 1 or f.shape[1] < 2:
        raise ValueError('aligned nonempty response matrices with >=2 genes required')
    if not np.isfinite(f).all() or not np.isfinite(y).all():
        raise ValueError('nonfinite response values')
    n, g = f.shape
    genes = np.asarray(list(map(str, gene_ids))) if gene_ids is not None else np.asarray([f'{j:012d}' for j in range(g)])
    targets = np.asarray(list(map(str, target_gene_ids))) if target_gene_ids is not None else np.asarray(['__no_target__'] * n)
    if genes.shape != (g,) or targets.shape != (n,) or len(set(genes.tolist())) != g:
        raise ValueError('target/gene axes must match and gene identifiers must be unique')
    if tau is not None and (not np.isfinite(tau) or tau <= 0):
        raise ValueError('tau must be positive, finite and estimated on training data only')
    ks = sorted(set((20, 50, 100)) | set(top_ks))
    if any(not isinstance(k, (int, np.integer)) or k <= 0 for k in ks):
        raise ValueError('TopK values must be positive integers')
    own = genes[None, :] != targets[:, None]
    if (own.sum(1) < 2).any():
        raise ValueError('own-target exclusion leaves fewer than two genes')
    lex = np.argsort(genes, kind='stable')
    relation = cosine_retrieval_metrics(f, y, targets, genes, compute_pds=compute_pds)
    pds = relation['PDS_delta']
    records = []
    for row in range(n):
        valid = own[row]
        x, v = f[row, valid], y[row, valid]
        mse = float(np.square(x - v).mean())
        zero = float(np.square(v).mean())
        allzero = float(np.square(y[row]).mean())
        truth_nonzero = v != 0
        rec = {
            'target': str(targets[row]), 'mse': mse, 'zero_mse': zero,
            'R_delta': mse / zero if zero > EPS else None,
            'R_all': float(np.square(f[row] - y[row]).mean() / allzero) if allzero > EPS else None,
            'mae': float(np.abs(x - v).mean()), 'rmse': float(np.sqrt(mse)),
            'Pearson': _correlation(x, v), 'Spearman': _correlation(rankdata(x), rankdata(v)),
            'RMS_amplitude_ratio': float(np.linalg.norm(x) / np.linalg.norm(v)) if zero > EPS else None,
            'direction_accuracy_nonzero_truth': float(np.mean(np.sign(x[truth_nonzero]) == np.sign(v[truth_nonzero]))) if truth_nonzero.any() else None,
            'PDS_delta': pds[row],
            'paired_raw_cosine': relation['paired_raw_cosine'][row],
            'wrong_pair_raw_cosine': relation['wrong_pair_raw_cosine'][row],
            'raw_cosine_margin': relation['raw_cosine_margin'][row],
            'zero_truth_energy': bool(zero <= EPS),
            'truth_nonconstant': bool(np.ptp(v) > 0), 'prediction_constant': bool(np.ptp(x) == 0),
            'valid_genes': int(valid.sum()), 'own_target_present': bool(np.any(genes == targets[row])),
        }
        for k in ks:
            jac, signed, nt, nf = _top_metrics(f[row], y[row], valid, lex, k)
            rec[f'JAC{k}'] = jac
            rec[f'signed_top{k}_coverage'] = signed
            rec[f'truth_top{k}_count'] = nt
            rec[f'prediction_top{k}_count'] = nf
        rec['pure_top100_jaccard'] = rec['JAC100']
        rec['pure_signed_top100_coverage'] = rec['signed_top100_coverage']
        rec['top100_jaccard'] = rec['JAC100']
        rec['strong_top100_count'] = None
        rec['SignedTop100Coverage'] = rec['Top100_Jaccard'] = rec['NMAE_strong_100'] = None
        if tau is not None:
            jac, signed, nt, nf = _top_metrics(f[row], y[row], valid, lex, 100, tau)
            indices = _top_indices(y[row], valid, lex, 100, tau)
            rec.update(Top100_Jaccard=jac, SignedTop100Coverage=signed, strong_top100_count=nt)
            if len(indices):
                rec['NMAE_strong_100'] = float(np.abs(f[row, indices] - y[row, indices]).sum() / np.abs(y[row, indices]).sum())
        records.append(rec)
    names = {
        'macro_mse': 'mse', 'macro_zero_mse': 'zero_mse', 'R_delta': 'R_delta', 'R_all_genes': 'R_all',
        'MAE': 'mae', 'RMSE_macro': 'rmse', 'Pearson_macro': 'Pearson', 'Spearman_macro': 'Spearman',
        'PDS_delta': 'PDS_delta', 'paired_raw_cosine': 'paired_raw_cosine',
        'wrong_pair_raw_cosine': 'wrong_pair_raw_cosine', 'raw_cosine_margin': 'raw_cosine_margin',
        'RMS_amplitude_ratio_macro': 'RMS_amplitude_ratio',
        'direction_accuracy_nonzero_truth': 'direction_accuracy_nonzero_truth',
        'pure_top100_jaccard': 'pure_top100_jaccard', 'pure_signed_top100_coverage': 'pure_signed_top100_coverage',
        'Top100_Jaccard': 'Top100_Jaccard', 'SignedTop100Coverage': 'SignedTop100Coverage', 'NMAE_strong_100': 'NMAE_strong_100',
        **{f'JAC{k}': f'JAC{k}' for k in ks},
        **{f'signed_top{k}_coverage': f'signed_top{k}_coverage' for k in ks},
    }
    aggregate = {name: _average([r[key] for r in records]) for name, key in names.items()}
    aggregate.update(MSE=aggregate['macro_mse'], Pearson=aggregate['Pearson_macro'],
                     Spearman=aggregate['Spearman_macro'], top100_jaccard=aggregate['JAC100'],
                     JAC=aggregate['JAC100'], JAC20_macro=aggregate['JAC20'],
                     JAC50_macro=aggregate['JAC50'], JAC100_macro=aggregate['JAC100'])
    aggregate['legacy_response_score_not_for_selection'] = selection_score(aggregate)
    out = dict(aggregate)
    out.update(aggregate=dict(aggregate), per_condition=records,
               condition_metrics={key: [r.get(key) for r in records] for key in records[0] if key != 'target'},
               conditions=n, genes=g, PDS_computed=bool(compute_pds),
               PDS_competition_genes=relation['competition_genes'],
               PDS_informative_candidates=relation['informative_candidates'],
               strong_threshold=float(tau) if tau is not None else None,
               strong_metrics_available=tau is not None,
               zero_energy_conditions=sum(r['zero_truth_energy'] for r in records),
               undefined_truth_correlation_conditions=sum(not r['truth_nonconstant'] for r in records),
               constant_prediction_conditions=sum(r['prediction_constant'] for r in records),
               JAC_defined_conditions=sum(r['JAC100'] is not None for r in records),
               informative_validation_panel=bool(any(not r['zero_truth_energy'] for r in records)),
               legacy_ratio_of_mean_mse=aggregate['macro_mse'] / aggregate['macro_zero_mse'] if aggregate['macro_zero_mse'] > EPS else None,
               global_metrics_computed=bool(compute_global),
               metric_scope='Internal normalized-log response metrics; JAC=TopK absolute-response Jaccard, not official VCC JAC.',
               metric_definition='Own-target excluded; stable identifier ties; zero entries excluded from TopK; zero-truth relative ratios/JAC undefined; constant prediction correlation=0 for nonconstant truth; PDS panel-target-excluded raw cosine midrank; paired_raw_cosine uses the identical mask and no centering.')
    if compute_global:
        xx, yy = f[own], y[own]
        out.update(Pearson_global=_correlation(xx, yy), Spearman_global=_correlation(rankdata(xx), rankdata(yy)),
                   MSE_global=float(np.square(xx - yy).mean()), MAE_global=float(np.abs(xx - yy).mean()),
                   RMSE_global=float(np.sqrt(np.square(xx - yy).mean())))
    return out


def selection_score(metrics):
    """Historical no-PDS selection score; JAC is reported independently."""
    r, p, t = (metrics.get(k) for k in ('R_delta', 'Pearson_macro', 'pure_signed_top100_coverage'))
    if any(v is None or not np.isfinite(v) for v in (r, p, t)):
        return None
    return float(.5 * (1 - r) + .25 * p + .25 * t)


def evaluate_predictions(pred, truth, sources_mean, target_ids, gene_ids, tau,
                         *, compute_pds=True, compute_global=True):
    """Compatibility interface for historical response evaluation callers."""
    out = response_metrics(pred, truth, target_ids, gene_ids, tau,
                           compute_pds=compute_pds, compute_global=compute_global)
    if sources_mean is not None:
        base = response_metrics(sources_mean, truth, target_ids, gene_ids, tau)
        bm = base['macro_mse']
        dif = np.asarray([a['mse'] - b['mse'] for a, b in zip(out['per_condition'], base['per_condition'])])
        out['source_mean_comparison'] = {
            'baseline_MSE_macro': bm, 'baseline_MAE_macro': base['MAE'],
            'relative_MSE_improvement': float(1 - out['macro_mse'] / bm) if bm > EPS else None,
            'relative_MAE_improvement': float(1 - out['MAE'] / base['MAE']) if base['MAE'] > EPS else None,
            'MSE_difference_model_minus_source_mean': float(dif.mean()),
            'MAE_difference_model_minus_source_mean': float(out['MAE'] - base['MAE']),
            'condition_MSE_negative_transfer_fraction': float(np.mean(dif > 0)),
            'condition_MSE_improvement_fraction': float(np.mean(dif < 0)),
        }
    return out


# Audited distribution mathematics copied locally, with no model dependency.
import torch
from torch import Tensor

def normalized_weights(weights: Tensor | None, n: int, reference: Tensor) -> Tensor:
    if n < 1:
        raise ValueError("empty empirical distribution")
    if weights is None:
        return reference.new_full((n,), 1.0 / n)
    w = weights.to(device=reference.device, dtype=reference.dtype)
    if w.shape != (n,) or not bool(torch.isfinite(w).all()) or bool((w < 0).any()) or not bool(w.sum() > 0):
        raise ValueError("weights must be finite nonnegative row weights with positive sum")
    return w / w.sum()


@lru_cache(maxsize=64)
def _uniform_intervals(n: int, m: int):
    # Exact rational breakpoint ordering: products avoid float tie artefacts.
    grid = sorted(set([i * m for i in range(n + 1)] + [j * n for j in range(m + 1)]))
    widths, ix, iy = [], [], []
    for lo, hi in zip(grid[:-1], grid[1:]):
        widths.append((hi - lo) / (n * m))
        ix.append(min((lo + hi) // (2 * m), n - 1))
        iy.append(min((lo + hi) // (2 * n), m - 1))
    return widths, ix, iy


def empirical_w1(x: Tensor, y: Tensor, x_weights: Tensor | None = None,
                 y_weights: Tensor | None = None, *, y_sorted: Tensor | None = None,
                 x_sorted: Tensor | None = None) -> Tensor:
    """Exact W1 per column for samples [N,D], arbitrary unequal N and M.

    Uniform supports use the quantile-integral formula. General shared row
    weights use the integral of the absolute empirical CDF difference.
    Gradients are piecewise valid through support values, including tied values
    (where Wasserstein is itself nondifferentiable and a subgradient is used).
    """
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or min(len(x), len(y)) < 1:
        raise ValueError("W1 requires nonempty supports with matching columns")
    if x_weights is None and y_weights is None:
        a = x.sort(0).values if x_sorted is None else x_sorted
        b = y.sort(0).values if y_sorted is None else y_sorted
        widths, ix, iy = _uniform_intervals(len(x), len(y))
        ix = torch.tensor(ix, dtype=torch.long, device=x.device)
        iy = torch.tensor(iy, dtype=torch.long, device=x.device)
        w = x.new_tensor(widths)[:, None]
        return ((a[ix] - b[iy]).abs() * w).sum(0)
    wx = normalized_weights(x_weights, len(x), x)
    wy = normalized_weights(y_weights, len(y), y)
    joined = torch.cat((x, y), 0)
    values, order = joined.sort(0)
    increments = torch.cat((wx, -wy))[:, None].expand_as(joined).gather(0, order)
    cdf_difference = increments.cumsum(0)
    return (cdf_difference[:-1].abs() * (values[1:] - values[:-1])).sum(0)


def empirical_w1_conditions(x: Tensor, y: Tensor, *, x_is_sorted: bool = False) -> Tensor:
    """Uniform W1 independently for [Q,N,G] and [Q,M,G], returning [Q,G].

    x_is_sorted is a trusted internal contract of the monotone B quantile
    decoder. It must never be used for arbitrary joint cell clouds.
    """
    if (x.ndim != 3 or y.ndim != 3 or x.shape[0] != y.shape[0] or
            x.shape[2] != y.shape[2] or min(x.shape[0],x.shape[1],y.shape[1],x.shape[2]) < 1):
        raise ValueError('Batched W1 requires nonempty matching condition/gene axes')
    a = x if x_is_sorted else x.sort(1).values
    b = y.sort(1).values
    if x.shape[1] == y.shape[1]:
        return (a-b).abs().mean(1)
    widths, ix, iy = _uniform_intervals(x.shape[1],y.shape[1])
    ix = torch.tensor(ix,dtype=torch.long,device=x.device)
    iy = torch.tensor(iy,dtype=torch.long,device=x.device)
    w = x.new_tensor(widths)[None,:,None]
    return ((a[:,ix]-b[:,iy]).abs()*w).sum(1)


def fixed_projections(genes: int, count: int, seed: int, *, device=None,
                      dtype=torch.float32) -> Tensor:
    """Returns [genes,count] fixed Gaussian unit-L2 directions, generated on CPU."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = torch.randn(genes, count, generator=generator, dtype=torch.float64)
    result = result / result.norm(dim=0, keepdim=True)
    return result.to(device=device, dtype=dtype)


def joint_sw1(x: Tensor, y: Tensor, gene_scale: Tensor, projections: Tensor,
              x_weights: Tensor | None = None, y_weights: Tensor | None = None) -> Tensor:
    if projections.shape[0] != x.shape[1] or gene_scale.shape != (x.shape[1],):
        raise ValueError("joint projection must cover the full gene panel")
    if bool((gene_scale <= 0).any()):
        raise ValueError("gene scales must be positive")
    return empirical_w1((x / gene_scale) @ projections, (y / gene_scale) @ projections,
                        x_weights, y_weights).mean()



def weighted_quantiles(values: Tensor, quantiles: Tensor,
                       weights: Tensor | None = None) -> Tensor:
    """Left-continuous empirical inverse CDF at q>0; output [Q,G]."""
    if values.ndim != 2:
        raise ValueError("expected cell-by-gene values")
    w = normalized_weights(weights, len(values), values)
    ordered, index = values.sort(0)
    sw = w[:, None].expand_as(values).gather(0, index)
    cdf = sw.cumsum(0).T.contiguous()
    q = quantiles.to(values).reshape(1, -1).expand(values.shape[1], -1).contiguous()
    indices = torch.searchsorted(cdf, q, right=False).clamp_max(len(values) - 1)
    return ordered.T.gather(1, indices).T


def empirical_crps(prediction: Tensor, observed: Tensor,
                   prediction_weights: Tensor | None = None,
                   observation_weights: Tensor | None = None, *, prepared_observation=None) -> Tensor:
    """CRPS per gene, averaged over observed cells with their shared row weights.

    Computes E|X-Y|-0.5 E|X-X'| using sorted cumulative weighted values,
    O((N+M)*G) working memory. This is the usual finite predictive-distribution
    CRPS, without an ensemble finite-sample 'fair score' adjustment.
    """
    if prediction.ndim != 2 or observed.ndim != 2 or prediction.shape[1] != observed.shape[1]:
        raise ValueError("CRPS requires matching gene columns")
    wx = normalized_weights(prediction_weights, len(prediction), prediction)
    wy = normalized_weights(observation_weights, len(observed), observed)
    xs, xi = prediction.sort(0)
    xw = wx[:, None].expand_as(prediction).gather(0, xi)
    if prepared_observation is None:
        ys, yi = observed.sort(0)
        yw = wy[:, None].expand_as(observed).gather(0, yi)
        yc = yw.cumsum(0)
        yv = (yw * ys).cumsum(0)
    else:
        ys = prepared_observation['sorted']
        yc, yv = prepared_observation['cdf'], prepared_observation['cumulative_value']
    zero = ys.new_zeros((1, ys.shape[1]))
    # Number of observed positions <= each predicted position.
    pos = torch.searchsorted(ys.T.contiguous(), xs.T.contiguous(), right=True).T
    mass = torch.cat((zero, yc), 0).gather(0, pos)
    value = torch.cat((zero, yv), 0).gather(0, pos)
    cross = (xw * (xs * (2 * mass - 1) + yv[-1:] - 2 * value)).sum(0)
    half_self = (xw * xs * (2 * xw.cumsum(0) - xw - 1)).sum(0)
    # Analytical nonnegativity; FP32 cancellation can produce small negatives.
    return (cross - half_self).clamp_min(0)


def _moments(x: Tensor, w: Tensor):
    mean = (x * w[:, None]).sum(0)
    variance = ((x - mean).square() * w[:, None]).sum(0)
    return mean, variance


def _covariance(x: Tensor, w: Tensor) -> Tensor:
    mean = (x * w[:, None]).sum(0)
    residual = x - mean
    return residual.T @ (residual * w[:, None])


def _number(x: Tensor) -> float:
    if x.numel() != 1 or not bool(torch.isfinite(x)):
        raise ValueError("metric is not a finite scalar")
    return float(x.detach().cpu())


@torch.no_grad()
def prepare_distribution_reference(real_cells: Tensor, gene_scale: Tensor, projections: Tensor,
                                   *, pca_components=None, cluster_centers=None, pca_offset=None):
    """Uniform-weight real statistics. Call only inside the authorized evaluator.

    The reference binds its original real tensor and geometry; it is never an
    input to the predictor. It may be moved together to CPU between evaluations.
    """
    y = real_cells
    w = normalized_weights(None, len(y), y)
    mean, variance = _moments(y, w)
    ordered, order = y.sort(0)
    ordered_w = w[:, None].expand_as(y).gather(0, order)
    ref = dict(real_cells=y, mean=mean, variance=variance,
               sorted=ordered, cdf=ordered_w.cumsum(0),
               cumulative_value=(ordered_w*ordered).cumsum(0),
               detection=((y>0).to(y.dtype)*w[:,None]).sum(0),
               quantiles=weighted_quantiles(y, y.new_tensor([.05,.10,.25,.50,.75,.90,.95])),
               geometry=dict(gene_scale=gene_scale, projections=projections,
                             pca_components=pca_components, cluster_centers=cluster_centers,
                             pca_offset=pca_offset))
    if pca_components is not None:
        offset = y.new_zeros(y.shape[1]) if pca_offset is None else pca_offset
        py = (y/gene_scale-offset) @ pca_components.T
        ref['pca'], ref['covariance'] = py, _covariance(py,w)
        if cluster_centers is not None:
            dy, iy = torch.cdist(py,cluster_centers).min(1)
            ref['state_distance'], ref['state_index'] = dy, iy
            ref['state_fractions'] = y.new_zeros(len(cluster_centers)).scatter_add_(0,iy,w)
    return ref


def _evaluate_distribution(support: Tensor, real_cells: Tensor, gene_scale: Tensor,
                          projections: Tensor, support_weights: Tensor | None = None,
                          real_weights: Tensor | None = None,
                          pca_components: Tensor | None = None,
                          cluster_centers: Tensor | None = None,
                          pca_offset: Tensor | None = None,
                          module_projections: Tensor | None = None,
                          cluster_distance_threshold: float | None = None, prepared_real=None,
                          *, allow_signed=False) -> dict:
    """Full distribution metrics, all genes and complete shared-row supports.

    projections is [G,K]. PCA components are [D,G]; latent representation is
    (expression / gene_scale - pca_offset) @ pca_components.T. No target-label
    centering is fitted. pca_offset must be supplied consistently with the
    frozen training E-residual PCA and cluster assets.
    Missing optional geometry assets produce None plus availability flags.
    """
    x, y = support, real_cells
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("predicted and real distributions must have matching genes")
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        raise ValueError("nonfinite expression")
    if (bool((x < 0).any()) or bool((y < 0).any())) and not allow_signed:
        raise ValueError("expression distributions must be nonnegative")
    if gene_scale.shape != (x.shape[1],) or bool((gene_scale <= 0).any()):
        raise ValueError("positive fixed gene scales are required")
    wx = normalized_weights(support_weights, len(x), x)
    wy = normalized_weights(real_weights, len(y), y)
    mx, vx = _moments(x, wx)
    if prepared_real is not None:
        if real_weights is not None or prepared_real['real_cells'].data_ptr() != y.data_ptr():
            raise ValueError('Prepared reference must match this uniform-weight real tensor')
        my, vy = prepared_real['mean'], prepared_real['variance']
    else:
        my, vy = _moments(y, wy)
    marginal = empirical_w1(x, y, support_weights, real_weights,
                            y_sorted=None if prepared_real is None else prepared_real['sorted'])
    joint = joint_sw1(x, y, gene_scale, projections, support_weights, real_weights)
    crps = empirical_crps(x, y, support_weights, real_weights, prepared_observation=prepared_real)
    detection_x = ((x > 0).to(x.dtype) * wx[:, None]).sum(0)
    detection_y = ((y > 0).to(y.dtype) * wy[:, None]).sum(0) if prepared_real is None else prepared_real['detection']
    q_levels = x.new_tensor([.05, .10, .25, .50, .75, .90, .95])
    qx = weighted_quantiles(x, q_levels, support_weights)
    qy = weighted_quantiles(y, q_levels, real_weights) if prepared_real is None else prepared_real['quantiles']
    r = {
        "marginal_W1_normalized": _number((marginal / gene_scale).mean()),
        "marginal_W1": _number(marginal.mean()),
        "joint_SW1_normalized": _number(joint),
        "detection_MAE": _number((detection_x - detection_y).abs().mean()),
        "variance_MAE": _number((vx - vy).abs().mean()),
        "variance_MAE_normalized": _number(((vx - vy).abs() / gene_scale.square()).mean()),
        "CRPS": _number(crps.mean()),
        "CRPS_normalized": _number((crps / gene_scale).mean()),
        "absolute_mean_MSE": _number((mx - my).square().mean()),
        "absolute_mean_MAE": _number((mx - my).abs().mean()),
        "absolute_mean_RMSE": _number((mx - my).square().mean().sqrt()),
        "support_rows": len(x), "observed_rows": len(y),
        "support_row_weight_ESS": _number(1 / wx.square().sum()),
        "real_row_weight_ESS": _number(1 / wy.square().sum()),
        "generated_nonzero_count_mean": _number((x > 0).sum(0).float().mean()),
        "generated_nonzero_fraction_mean": _number(detection_x.mean()),
        "observed_nonzero_fraction_mean": _number(detection_y.mean()),
        "generated_max_value": _number(x.max()),
        "generated_mean_variance": _number(vx.mean()),
        "observed_mean_variance": _number(vy.mean()),
        "degenerate_zero_mean_gene_count": int((mx == 0).sum().cpu()),
    }
    r["legacy_distribution_composite_not_for_selection"] = .5 * r["marginal_W1_normalized"] + .5 * r["joint_SW1_normalized"]
    for index, label in [(1, "10"), (2, "25"), (3, "50"), (4, "75"), (5, "90")]:
        error = (qx[index] - qy[index]).abs()
        r[f"quantile_{label}_MAE"] = _number(error.mean())
        r[f"quantile_{label}_MAE_normalized"] = _number((error / gene_scale).mean())
    for nominal, lo, hi in [(80, 1, 5), (90, 0, 6)]:
        inside = ((y >= qx[lo]) & (y <= qx[hi])).to(y.dtype)
        coverage = (inside * wy[:, None]).sum(0)
        width = qx[hi] - qx[lo]
        r[f"coverage_{nominal}"] = _number(coverage.mean())
        r[f"coverage_{nominal}_absolute_calibration_error"] = _number((coverage - nominal / 100).abs().mean())
        r[f"interval_{nominal}_width"] = _number(width.mean())
        r[f"interval_{nominal}_width_normalized"] = _number((width / gene_scale).mean())
    # Expression-share ESS differs from row-probability ESS. Zero-mean genes
    # contain no shape information and are excluded with an explicit count.
    positive = mx > 0
    if bool(positive.any()):
        shares = x[:, positive] * wx[:, None] / mx[positive]
        r["expression_share_ESS_mean"] = _number((1 / shares.square().sum(0)).mean())
        r["expression_share_max_mean"] = _number(shares.max(0).values.mean())
    else:
        r["expression_share_ESS_mean"] = None
        r["expression_share_max_mean"] = None
    r["shape_identifiable_gene_count"] = int(positive.sum().cpu())
    r.update(PCA_covariance_MAE=None, PCA_covariance_relative_frobenius=None,
             state_proportion_L1=None, state_proportion_TV=None,
             state_nearest_distance_prediction=None, state_nearest_distance_observed=None,
             state_prediction_far_fraction=None, state_observed_far_fraction=None,
             module_covariance_MAE=None, PCA_metrics_available=pca_components is not None,
             state_metrics_available=cluster_centers is not None and pca_components is not None,
             module_metrics_available=module_projections is not None)
    if pca_components is not None:
        if pca_components.ndim != 2 or pca_components.shape[1] != x.shape[1]:
            raise ValueError("pca_components must be [PCA dimensions, all genes]")
        offset = x.new_zeros(x.shape[1]) if pca_offset is None else pca_offset
        if offset.shape != (x.shape[1],):
            raise ValueError("pca_offset must match all genes")
        px = (x / gene_scale - offset) @ pca_components.T
        py = (y / gene_scale - offset) @ pca_components.T if prepared_real is None else prepared_real['pca']
        cx = _covariance(px, wx)
        cy = _covariance(py, wy) if prepared_real is None else prepared_real['covariance']
        r["PCA_covariance_MAE"] = _number((cx - cy).abs().mean())
        norm_true = cy.norm()
        r["PCA_covariance_relative_frobenius"] = _number((cx - cy).norm() / norm_true) if bool(norm_true > 1e-12) else None
        if cluster_centers is not None:
            if cluster_centers.ndim != 2 or cluster_centers.shape[1] != px.shape[1]:
                raise ValueError("cluster centers must use the supplied frozen PCA space")
            dx, ix = torch.cdist(px, cluster_centers).min(1)
            dy, iy = torch.cdist(py, cluster_centers).min(1) if prepared_real is None else (prepared_real['state_distance'], prepared_real['state_index'])
            fx = x.new_zeros(len(cluster_centers)).scatter_add_(0, ix, wx)
            fy = y.new_zeros(len(cluster_centers)).scatter_add_(0, iy, wy) if prepared_real is None else prepared_real['state_fractions']
            r["state_proportion_L1"] = _number((fx - fy).abs().sum())
            r["state_proportion_TV"] = .5 * r["state_proportion_L1"]
            r["state_nearest_distance_prediction"] = _number((dx * wx).sum())
            r["state_nearest_distance_observed"] = _number((dy * wy).sum())
            if cluster_distance_threshold is not None:
                r["state_prediction_far_fraction"] = _number(((dx > cluster_distance_threshold) * wx).sum())
                r["state_observed_far_fraction"] = _number(((dy > cluster_distance_threshold) * wy).sum())
    if module_projections is not None:
        if module_projections.ndim != 2 or module_projections.shape[0] != x.shape[1]:
            raise ValueError("module projections must be [all genes, modules]")
        a = (x / gene_scale) @ module_projections
        b = (y / gene_scale) @ module_projections
        r["module_covariance_MAE"] = _number((_covariance(a, wx) - _covariance(b, wy)).abs().mean())
    return r


def distribution_metrics(support, real_cells, gene_scale, projections=None,
                         *, projection_count=64, projection_seed=17, **kwargs):
    """Pure local full-gene metrics; every geometric asset is train-fitted.

    Reuses audited mathematical definitions, never old model imports or weights.
    Centered SW1 removes each distribution's own mean only for scoring; it
    never fits or modifies the sealed prediction.
    """
    generated_projections = projections is None
    if generated_projections:
        projections = fixed_projections(support.shape[1], projection_count, projection_seed,
                                        device=support.device, dtype=support.dtype)
    out = _evaluate_distribution(support, real_cells, gene_scale, projections, **kwargs)
    wx = normalized_weights(kwargs.get('support_weights'), len(support), support)
    wy = normalized_weights(kwargs.get('real_weights'), len(real_cells), real_cells)
    centered = joint_sw1(support - (support * wx[:, None]).sum(0),
                        real_cells - (real_cells * wy[:, None]).sum(0),
                        gene_scale, projections, kwargs.get('support_weights'),
                        kwargs.get('real_weights'))
    out['centered_joint_SW1_normalized'] = _number(centered)
    out.update(metric_scope='Internal normalized-log empirical metrics; not official VCC scores.',
               projection_count=int(projections.shape[1]),
               projection_seed=int(projection_seed) if generated_projections else None,
               metric_implementation='local pure math, no historical model imports or weights')
    return out

evaluate_distribution = distribution_metrics
