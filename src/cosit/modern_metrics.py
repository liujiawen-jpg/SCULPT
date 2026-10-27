"""Evaluator-only DE recovery and joint-distribution metrics.

No model, checkpoint, training label selection, or prediction is modified here.
DE truth is computed from observed perturbed cells and matched controls only.
All MMD features and kernel bandwidths must be frozen from legal training data.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score


def _matrix(value: Any, name: str) -> np.ndarray:
    result = np.asanyarray(value)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty cell-by-gene matrix")
    if not np.issubdtype(result.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    return result


def _bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini--Hochberg q values over exactly the eligible hypotheses."""
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1 or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("BH requires finite p values in [0, 1]")
    if p.size == 0:
        return p.copy()
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * p.size / np.arange(1, p.size + 1, dtype=np.float64)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def build_deg_reference(
    real: Any,
    control: Any,
    exclude_index: int | None = None,
    *,
    fdr: float = 0.05,
    gene_chunk: int = 256,
) -> dict[str, Any]:
    """Build prediction-independent, cacheable true-DE labels and eligible mask.

    True DE is a two-sided, tie-corrected asymptotic Mann--Whitney U test
    (continuity correction enabled), followed by BH across all eligible genes.
    The perturbed gene is excluded *before* BH. A gene containing a nonfinite
    observed/control value is not a computable hypothesis and is counted below;
    All-tied genes remain eligible with p = 1. ``eligible_mask`` and ``labels``
    both span the original gene axis; labels outside the mask are always False.
    The result contains no predictions and can be reused for every method and
    seed on the same observed condition/control, gene axis, and DE protocol.
    """
    x = _matrix(real, "real")
    y = _matrix(control, "control")
    if x.shape[1] != y.shape[1]:
        raise ValueError("real and control gene axes differ")
    if not 0 < fdr < 1 or gene_chunk < 1:
        raise ValueError("fdr must be in (0, 1) and gene_chunk positive")
    if exclude_index is not None:
        if isinstance(exclude_index, bool) or int(exclude_index) != exclude_index:
            raise ValueError("exclude_index must be an integer or None")
        exclude_index = int(exclude_index)
        if not 0 <= exclude_index < x.shape[1]:
            raise ValueError("exclude_index is outside the gene axis")

    # scipy's ranking/tie correction has several temporary arrays. Keep the
    # cell-by-gene work block small even when controls contain many cells.
    budget = 64 * 1024 * 1024
    cells = x.shape[0] + y.shape[0]
    width = max(1, min(gene_chunk, budget // max(1, cells * 8 * 16)))
    eligible = np.ones(x.shape[1], dtype=bool)
    if exclude_index is not None:
        eligible[exclude_index] = False
    pvalues = np.ones(x.shape[1], dtype=np.float64)
    for start in range(0, x.shape[1], width):
        end = min(x.shape[1], start + width)
        xb = np.asarray(x[:, start:end], dtype=np.float64)
        yb = np.asarray(y[:, start:end], dtype=np.float64)
        ok = np.isfinite(xb).all(axis=0) & np.isfinite(yb).all(axis=0)
        eligible[start:end] &= ok
        selected = np.flatnonzero(eligible[start:end])
        if selected.size:
            tested = mannwhitneyu(
                xb[:, selected],
                yb[:, selected],
                axis=0,
                alternative="two-sided",
                method="asymptotic",
                use_continuity=True,
            ).pvalue
            # Some SciPy releases return NaN for zero tie-corrected variance.
            # All pooled observations identical means exactly no evidence of
            # differential expression. Keep these hypotheses in BH with p=1.
            pooled_min=np.minimum(xb[:,selected].min(0),yb[:,selected].min(0))
            pooled_max=np.maximum(xb[:,selected].max(0),yb[:,selected].max(0))
            tested=np.where(pooled_min==pooled_max,1.,tested)
            if not np.isfinite(tested).all():
                raise ValueError("Mann--Whitney returned a nonfinite p value")
            pvalues[start + selected] = tested

    n_genes = int(eligible.sum())
    qvalues = _bh_adjust(pvalues[eligible])
    labels = np.zeros(x.shape[1], dtype=bool)
    labels[eligible] = qvalues < fdr
    n_de = int(labels.sum())
    return {
        "eligible_mask": eligible,
        "labels": labels,
        "DEG_prevalence": float(n_de / n_genes) if n_genes else float("nan"),
        "DEG_n_true": n_de,
        "DEG_n_genes": n_genes,
        "DEG_n_uncomputable": int((~eligible).sum()) - int(exclude_index is not None),
        "DEG_self_excluded": exclude_index is not None,
        "DEG_fdr": float(fdr),
        "DEG_test": "two-sided_asymptotic_MWU_tie_corrected_BH",
    }


def score_deg_reference(reference: dict[str, Any], pred_delta: Any) -> dict[str, Any]:
    """Score |predicted response| against a previously frozen DE reference.

    This is unsigned DEG recovery; use a separate signed metric for direction.
    AP is undefined when no true DE exists. Nonfinite eligible predictions raise
    instead of being dropped, preventing a failed prediction from inflating AP.
    """
    mask = np.asarray(reference["eligible_mask"], dtype=bool)
    labels = np.asarray(reference["labels"], dtype=bool)
    delta = np.asarray(pred_delta, dtype=np.float64)
    if mask.ndim != 1 or labels.shape != mask.shape or delta.shape != mask.shape:
        raise ValueError("reference mask, labels and prediction gene axes differ")
    if np.any(labels & ~mask):
        raise ValueError("DE reference contains a positive label outside eligibility")
    if not np.isfinite(delta[mask]).all():
        raise ValueError("nonfinite predicted response on an eligible gene")
    n_de = int(labels.sum())
    n_genes = int(mask.sum())
    if n_de != int(reference["DEG_n_true"]) or n_genes != int(reference["DEG_n_genes"]):
        raise ValueError("DE reference counts disagree with labels/mask")
    return {
        "DEG_AP": (
            float(average_precision_score(labels[mask], np.abs(delta[mask])))
            if n_de and n_genes
            else float("nan")
        ),
        "DEG_prevalence": float(n_de / n_genes) if n_genes else float("nan"),
        "DEG_n_true": n_de,
        "DEG_n_genes": n_genes,
        "DEG_n_uncomputable": int(reference["DEG_n_uncomputable"]),
        "DEG_self_excluded": bool(reference["DEG_self_excluded"]),
        "DEG_fdr": float(reference["DEG_fdr"]),
        "DEG_test": str(reference["DEG_test"]),
    }


def deg_ap(
    real: Any,
    control: Any,
    pred_delta: Any,
    exclude_index: int | None = None,
    *,
    fdr: float = 0.05,
    gene_chunk: int = 256,
) -> dict[str, Any]:
    """Convenience combination of building DE truth and scoring a prediction."""
    reference = build_deg_reference(
        real, control, exclude_index, fdr=fdr, gene_chunk=gene_chunk,
    )
    return score_deg_reference(reference, pred_delta)


def _stable_seed(seed: int, condition_key: str, repeat: int, population: str) -> int:
    payload = json.dumps(
        [int(seed), str(condition_key), int(repeat), population],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def mmd_subsample_indices(
    n_real: int,
    n_generated: int,
    condition_key: str,
    *,
    max_samples: int = 128,
    repeats: int = 3,
    seed: int = 20261021,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Method-independent, no-replacement cell indices for each repeat."""
    if min(n_real, n_generated, max_samples, repeats) < 1:
        raise ValueError("sample counts and repeats must be positive")
    m = min(int(max_samples), int(n_real), int(n_generated))
    result = []
    for repeat in range(repeats):
        real_rng = np.random.default_rng(_stable_seed(seed, condition_key, repeat, "real"))
        pred_rng = np.random.default_rng(_stable_seed(seed, condition_key, repeat, "pred"))
        ri = real_rng.choice(n_real, size=m, replace=False)
        pi = pred_rng.choice(n_generated, size=m, replace=False)
        result.append((ri, pi))
    return result


def _kernel_sums(
    a: np.ndarray,
    b: np.ndarray,
    scale: np.ndarray,
    bandwidths: np.ndarray,
    row_chunk: int,
    feature_chunk: int,
    same_population: bool,
) -> float:
    """Sum mixed Gaussian kernels, omitting same-population self pairs.

    Squared distances use direct double-precision differences in scipy.cdist,
    not norm(x)^2 + norm(y)^2 - 2xy, avoiding cancellation at large offsets.
    Feature and row blocking bounds workspace independently of gene count.
    """
    total = 0.0
    for i in range(0, a.shape[0], row_chunk):
        ie = min(a.shape[0], i + row_chunk)
        for j in range(0, b.shape[0], row_chunk):
            je = min(b.shape[0], j + row_chunk)
            distances = np.zeros((ie - i, je - j), dtype=np.float64)
            for g in range(0, a.shape[1], feature_chunk):
                ge = min(a.shape[1], g + feature_chunk)
                ab = np.asarray(a[i:ie, g:ge], dtype=np.float64) / scale[g:ge]
                bb = np.asarray(b[j:je, g:ge], dtype=np.float64) / scale[g:ge]
                if not np.isfinite(ab).all() or not np.isfinite(bb).all():
                    raise ValueError("MMD encountered nonfinite sampled expression")
                distances += cdist(ab, bb, metric="sqeuclidean")
            if not np.isfinite(distances).all():
                raise ValueError("nonfinite squared distance")
            kernel = np.zeros_like(distances)
            for sigma in bandwidths:
                kernel += np.exp(-distances / (2.0 * sigma * sigma))
            kernel /= bandwidths.size
            if same_population:
                left = max(i, j)
                right = min(ie, je)
                if right > left:
                    diagonal = np.arange(left, right)
                    kernel[diagonal - i, diagonal - j] = 0.0
            total += float(kernel.sum(dtype=np.float64))
    return total


def gaussian_mmd2_unbiased(
    real: Any,
    pred: Any,
    scale: Any,
    bandwidths: Sequence[float],
    condition_key: str,
    *,
    max_samples: int = 128,
    repeats: int = 3,
    seed: int = 20261021,
    row_chunk: int = 64,
    feature_chunk: int = 256,
) -> dict[str, Any]:
    """Unbiased full-gene MMD² using an equal mixture of Gaussian kernels.

    bandwidths are Euclidean-distance sigmas, not squared bandwidths. Do not
    estimate them or scale from evaluation cells. Uses three deterministic
    independent subsampling repeats by default, m=min(128,n_real,n_generated).
    A finite-sample unbiased estimate can be negative and is never clamped.
    With m < 2 it is undefined, recorded explicitly with NaN and sample counts.
    """
    x = _matrix(real, "real")
    y = _matrix(pred, "pred")
    if x.shape[1] != y.shape[1]:
        raise ValueError("real and pred gene axes differ")
    s = np.asarray(scale, dtype=np.float64)
    sigmas = np.asarray(bandwidths, dtype=np.float64)
    if s.shape != (x.shape[1],) or np.any(~np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("scale must contain one positive finite value per gene")
    if sigmas.ndim != 1 or sigmas.size == 0 or np.any(~np.isfinite(sigmas)) or np.any(sigmas <= 0):
        raise ValueError("bandwidths must be positive finite Euclidean sigmas")
    if row_chunk < 1 or feature_chunk < 1:
        raise ValueError("chunk sizes must be positive")
    indices = mmd_subsample_indices(
        x.shape[0], y.shape[0], condition_key,
        max_samples=max_samples, repeats=repeats, seed=seed,
    )
    m = len(indices[0][0])
    values = []
    index_hash = hashlib.sha256()
    for real_indices, pred_indices in indices:
        index_hash.update(np.asarray(real_indices, dtype="<i8").tobytes())
        index_hash.update(np.asarray(pred_indices, dtype="<i8").tobytes())
        if m < 2:
            values.append(float("nan"))
            continue
        # m <= 128 for the registered protocol; keep original input dtype here.
        # Arithmetic promotion happens only inside feature/row work blocks.
        a, b = x[real_indices], y[pred_indices]
        xx = _kernel_sums(a, a, s, sigmas, row_chunk, feature_chunk, True)
        yy = _kernel_sums(b, b, s, sigmas, row_chunk, feature_chunk, True)
        xy = _kernel_sums(a, b, s, sigmas, row_chunk, feature_chunk, False)
        values.append(float((xx + yy) / (m * (m - 1)) - 2.0 * xy / (m * m)))
    return {
        "MMD2": float(np.mean(values)),
        "MMD2_repeats": values,
        "MMD2_subsample_size": m,
        "MMD2_bandwidths": sigmas.tolist(),
        "MMD2_seed": int(seed),
        "MMD2_repeat_count": int(repeats),
        "MMD2_n_real": int(x.shape[0]),
        "MMD2_n_generated": int(y.shape[0]),
        "sample_index_sha256": index_hash.hexdigest(),
    }


def energy_distance_normalized(real: Any, pred: Any, scale: Any,
                               condition_key: str, *, max_samples: int = 128,
                               repeats: int = 3, seed: int = 20261021,
                               row_chunk: int = 64, feature_chunk: int = 256) -> dict[str, Any]:
    """Empirical energy V-statistic on fixed subsamples, without a square root.

    Distances are ||(x-y)/scale||_2 / sqrt(G). The reported statistic is
    2 E_xy[d] - E_xx[d] - E_yy[d], including within-population self pairs.
    It is a supplemental release evaluator, not an added training loss.
    """
    x,y=_matrix(real,'real'),_matrix(pred,'pred')
    if x.shape[1]!=y.shape[1]:raise ValueError('real and pred gene axes differ')
    s=np.asarray(scale,dtype=np.float64)
    if s.shape!=(x.shape[1],) or not np.isfinite(s).all() or np.any(s<=0):
        raise ValueError('scale must contain one positive finite value per gene')
    if row_chunk<1 or feature_chunk<1:raise ValueError('chunk sizes must be positive')

    def mean_distance(a,b):
        total=0.
        for i in range(0,len(a),row_chunk):
            for j in range(0,len(b),row_chunk):
                distances=np.zeros((min(row_chunk,len(a)-i),min(row_chunk,len(b)-j)),dtype='f8')
                for g in range(0,a.shape[1],feature_chunk):
                    ab=np.asarray(a[i:i+row_chunk,g:g+feature_chunk],dtype='f8')/s[g:g+feature_chunk]
                    bb=np.asarray(b[j:j+row_chunk,g:g+feature_chunk],dtype='f8')/s[g:g+feature_chunk]
                    if not np.isfinite(ab).all() or not np.isfinite(bb).all():
                        raise ValueError('Energy distance encountered nonfinite expression')
                    distances+=cdist(ab,bb,metric='sqeuclidean')
                total+=float(np.sqrt(distances/a.shape[1]).sum(dtype='f8'))
        return total/(len(a)*len(b))

    values=[]
    indices=mmd_subsample_indices(len(x),len(y),condition_key,
        max_samples=max_samples,repeats=repeats,seed=seed)
    for ri,pi in indices:
        a,b=x[ri],y[pi]
        value=2*mean_distance(a,b)-mean_distance(a,a)-mean_distance(b,b)
        if value < -1e-10:raise ArithmeticError('Negative empirical energy V-statistic')
        values.append(max(value,0.))
    return dict(Energy_distance_normalized=float(np.mean(values)),
        Energy_distance_repeats=values,Energy_distance_subsample_size=len(indices[0][0]),
        Energy_distance_repeat_count=repeats,Energy_distance_seed=seed)
