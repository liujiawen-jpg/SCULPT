"""Prepare method-specific inputs from the shared protected cell store.

Only legal source perturbations are read. Held-context controls are lawful
transform inputs; no held perturbation is used to fit a label/representation.
This command prepares inputs, not models or fabricated comparison predictions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np

from cosit.data import CellStore
from cosit.io import read_json, sha256, write_json
from cosit.splits import CONTEXTS


def legal_rows(store):
    return [(e, t) for e in store.training_environments for t in store.training_targets(e)]


def supervision(store, environment, target):
    store.authorize_episode(environment, [e for e in store.training_environments if e != environment])
    try:
        return np.array(store.get_p(environment, target, 'supervision'), copy=True)
    finally:
        store.clear_episode()


def bh(p):
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(p, kind='stable')
    ranks = p[order] * len(p) / np.arange(1, len(p) + 1)
    result = np.empty_like(ranks)
    result[order] = np.minimum.accumulate(ranks[::-1])[::-1].clip(0, 1)
    return result.astype(np.float32)


def pie_labels(cells, control_cells, pooled_control_mean):
    """Registered MWU/BH, cellwise-linear FC, and full-control mean shift."""
    from scipy.stats import mannwhitneyu
    x, e = np.asarray(cells, dtype=np.float64), np.asarray(control_cells, dtype=np.float64)
    mu = np.asarray(pooled_control_mean, dtype=np.float64)
    if x.ndim != 2 or e.ndim != 2 or x.shape[1] != e.shape[1] or mu.shape != (x.shape[1],):
        raise ValueError('Mismatched expression axes')
    if min(len(x), len(e)) < 2 or not np.isfinite(x).all() or not np.isfinite(e).all() or not np.isfinite(mu).all() or min(x.min(), e.min()) < 0:
        raise ValueError('Need finite nonnegative benchmark cells and pooled control means')
    xp, ep = np.expm1(x).mean(0), np.expm1(e).mean(0)
    tested = (xp > .05) | (ep > .05)
    fdr = np.ones(x.shape[1], dtype='f4')
    if tested.any():
        p = mannwhitneyu(x[:, tested], e[:, tested], axis=0, alternative='two-sided',
                         method='asymptotic', use_continuity=True).pvalue
        # The original all-tied case is p=1, never evidence of differential expression.
        if not np.isfinite(p).all():
            tied = np.ptp(np.concatenate([x[:, tested], e[:, tested]]), axis=0) == 0
            if not np.all(np.isfinite(p) | tied):
                raise FloatingPointError('Nonfinite MWU outside the all-tied case')
            p = np.where(tied, 1., p)
        fdr[tested] = bh(p)
    fc = (xp + 1e-9) / (ep + 1e-9)
    return dict(fold_changes=np.where(tested, fc, 0).astype('f4'), fdr=fdr,
                tested=tested, lfc_true=np.where(tested, np.log2(fc), np.nan),
                delta_p=(x.mean(0) - mu).astype('f4'))


def prepare_pie(store, output, upstream):
    import sys
    sys.path.insert(0, str(Path(upstream).resolve() / 'src'))
    from pie.data import preprocessed as pp
    rows = sorted(legal_rows(store))
    n, g = len(rows), len(store.genes)
    contexts = list(CONTEXTS)
    with tempfile.TemporaryDirectory(prefix='pie-labels-', dir=output) as scratch:
        fields = [pp.FOLD_CHANGES, pp.FDR, pp.TESTED, pp.LFC_TRUE, pp.DELTA_P]
        arrays = {k: np.lib.format.open_memmap(Path(scratch) / k, mode='w+',
                  dtype=pp.ARRAY_DTYPES[k], shape=(n, g)) for k in fields}
        for i, (e, t) in enumerate(rows):
            labels = pie_labels(supervision(store, e, t), store.get_e(e), store.get_e_mean(e))
            for k in fields:
                arrays[k][i] = labels[Path(k).stem]
        arrays[pp.CTRL_MEANS] = np.stack([store.get_e_mean(e) for e in contexts]).astype('f4')
        ci, ti = {e: i for i, e in enumerate(contexts)}, {t: i for i, t in enumerate(store.targets)}
        arrays[pp.CTX_IDS] = np.asarray([ci[e] for e, _ in rows], dtype='i4')
        arrays[pp.PERT_IDS] = np.asarray([ti[t] for _, t in rows], dtype='i4')
        meta = pp.PreprocessedMeta(format_version=1, dataset='J0', genes=store.genes,
            context_to_id=ci, pert_to_id=ti, pert_ensembl={t: t for t in store.targets},
            pert_kind='gene', control_label='E', num_rows=n, num_genes=g,
            num_contexts=len(contexts), num_perts=len(ti), controls_only=False,
            tool_version='SCULPT_PIE_native_labels_v1', array_sha256={})
        pp.write_preprocessed(output / 'preprocessed', meta, arrays)
        arrays.clear()
    write_json(output / 'train.json', {f'J0.{e}': store.training_targets(e) for e in store.training_environments})
    return dict(training_conditions=n, native_format='pie.data.preprocessed v1',
                expression_normalization='unchanged fixed A2048', de='asymptotic MWU + within-condition BH')


def prepare_poppert(store, output, upstream):
    from .poppert.population_helpers import fit_train_copula, fit_population
    from numpy.lib.format import open_memmap
    rows = legal_rows(store)
    controls = [store.get_e(e) for e in store.training_environments]
    assets = fit_train_copula(controls, upstream)
    np.savez(output / 'copula.npz', **assets)
    genes = list(dict.fromkeys(store.genes + store.targets))
    vocab = {g: i for i, g in enumerate(genes)}
    write_json(output / 'vocab.json', dict(vocab=vocab, genes=store.genes, targets=store.targets,
        gene_indices=[vocab[g] for g in store.genes], target_indices=[vocab[t] for t in store.targets]))
    g = len(store.genes)
    params = open_memmap(output / 'train_params.npy', mode='w+', dtype='f4', shape=(len(rows)*5, g, 7))
    factors = open_memmap(output / 'train_factor_var.npy', mode='w+', dtype='f4', shape=(len(rows)*5, 32))
    index = []
    for i, (e, t) in enumerate(rows):
        x = supervision(store, e, t)
        p, s, ids = fit_population(x, e, t, assets, upstream)
        if len(p) != 5:
            raise ValueError('Registered benchmark requires five populations per <=256-cell condition')
        params[i*5:(i+1)*5], factors[i*5:(i+1)*5] = p, s
        index.extend(dict(environment=e, target=t, label=vocab[t], population_index=j,
                          condition_index=i, observed_cell_count=len(x), fitted_cell_count=128) for j in range(5))
    params.flush(); factors.flush()
    control_params, control_factors = [], []
    for e in CONTEXTS:
        p, s, _ = fit_population(store.get_e(e), e, 'control', assets, upstream)
        control_params.append(p); control_factors.append(s)
    np.save(output / 'control_params.npy', np.stack(control_params))
    np.save(output / 'control_factor_var.npy', np.stack(control_factors))
    write_json(output / 'train_index.json', index)
    write_json(output / 'control_environments.json', list(CONTEXTS))
    return dict(training_conditions=len(rows), copula_fit_environments=list(store.training_environments),
                held_controls='transform only', held_perturbations_used=0)


def prepare_cellflow(store, output, dimensions=100):
    """Exact centered PCA; native JAX FP32 accumulation and CPU eigensolver."""
    import jax
    import jax.numpy as jnp
    from scipy.linalg import eigh
    rows = legal_rows(store) + [(e, None) for e in store.training_environments]
    if not 1 <= dimensions < len(store.genes):
        raise ValueError('PCA dimension must be smaller than the gene count')
    def values():
        for e, t in rows:
            yield store.get_e(e) if t is None else supervision(store, e, t)
    count = 0
    total = np.zeros(len(store.genes), dtype='f8')
    for x in values():
        total += x.astype('f8').sum(0); count += len(x)
    mean = (total / count).astype('f4')
    update = jax.jit(lambda c, x: c + x.T @ x)
    covariance = jnp.zeros((len(mean), len(mean)), dtype=jnp.float32)
    # Match the registered contiguous 4096-cell pool chunks, including final padding.
    pending = np.empty((0, len(mean)), dtype='f4')
    for x in values():
        pending = np.concatenate([pending, x])
        while len(pending) >= 4096:
            covariance = update(covariance, jnp.asarray(pending[:4096] - mean))
            pending = pending[4096:]
    if len(pending):
        centered = np.pad(pending - mean, ((0, 4096-len(pending)), (0, 0)))
        covariance = update(covariance, jnp.asarray(centered))
    c = np.asarray(covariance, dtype='f8') / (count-1)
    c = (c+c.T)/2
    v, u = eigh(c, subset_by_index=(len(mean)-dimensions, len(mean)-1), check_finite=False, driver='evr')
    order = np.argsort(v)[::-1]
    components = u[:, order].T.astype('f4')
    for row in components:
        if row[np.argmax(np.abs(row))] < 0: row *= -1
    np.savez(output / 'pca.npz', mean=mean, components=components, eigenvalues=v[order],
             explained_variance_ratio=v[order]/np.trace(c), genes=np.asarray(store.genes))
    return dict(fit_rows=count, dimensions=dimensions, fit_environments=list(store.training_environments),
                held_control_rows_used=0, held_perturbation_rows_used=0,
                numerics='registered FP32 JAX scatter-free covariance; FP64 eigh')


def prepare(method, data, splits, fold, output, upstream=None):
    output = Path(output)
    if output.exists(): raise FileExistsError('Choose a fresh preprocessing directory')
    output.mkdir(parents=True)
    fold_dir = Path(splits) / fold
    spec = read_json(fold_dir / 'FOLD.json')
    if spec['id'] != fold: raise ValueError('Wrong fold metadata')
    store = CellStore(data, spec, role='train')
    try:
        allowed = read_json(fold_dir / 'TRAINING_TARGETS.json')['allowed_targets']
        if any(store.training_targets(e) != allowed[e] for e in store.training_environments):
            raise PermissionError('Fold permissions and training target lists disagree')
        if method in ('pie', 'poppert') and upstream is None: raise ValueError('--upstream is required')
        if method == 'pie': report = prepare_pie(store, output, upstream)
        elif method == 'poppert': report = prepare_poppert(store, output, upstream)
        else: report = prepare_cellflow(store, output)
        files = [p for p in sorted(output.rglob('*')) if p.is_file()]
        report.update(status='completed', method=method, fold=fold, data_manifest_sha256=sha256(Path(data)/'MANIFEST.json'),
            fold_sha256=sha256(fold_dir/'FOLD.json'), allowed_targets_sha256=sha256(fold_dir/'TRAINING_TARGETS.json'),
            source_read_audit=dict(store.audit), labels_from_held_conditions=0,
            files={str(p.relative_to(output)): sha256(p) for p in files})
        write_json(output / 'PREPROCESSING.json', report)
        return report
    finally: store.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('method', choices=['pie', 'poppert', 'cellflow'])
    for name in ('data', 'splits', 'fold', 'output'): p.add_argument('--'+name, required=True)
    p.add_argument('--upstream', type=Path)
    a = p.parse_args()
    prepare(a.method, a.data, a.splits, a.fold, a.output, a.upstream)


if __name__ == '__main__': main()
