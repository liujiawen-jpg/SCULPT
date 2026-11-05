"""Portable native population preparation; caller must enforce split permissions.

Fit the copula on legal training controls only. Transform held controls under
that frozen basis. These helpers never choose folds or open label files.
"""
import hashlib
import numpy as np
import torch
from .native_api import configure_upstream

SET_SIZE = 128
MIN_BATCHES = 5
COMPONENTS = 2
RANK = 32
PREPROCESS_SEED = 42

def population_indices(n_cells, seed):
    """Native preprocess.py rule, with explicit per-condition reproducibility."""
    if n_cells < 2 * COMPONENTS:
        raise ValueError('Official native preprocessing requires >=4 observed cells; no silent skip')
    rng = np.random.RandomState(seed)
    natural = n_cells // SET_SIZE
    rows = []
    for i in range(max(MIN_BATCHES, natural)):
        if i < natural:
            selected = np.arange(i * SET_SIZE, (i + 1) * SET_SIZE)
        else:
            selected = rng.choice(n_cells, SET_SIZE, replace=n_cells < SET_SIZE)
        rows.append(selected)
    return np.asarray(rows, dtype=np.int64)

def condition_seed(environment, target):
    s = f'PopPert-native-batches-v1|{PREPROCESS_SEED}|{environment}|{target}'
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], 'little')

def validate_expression(x):
    if x.ndim != 2 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Input must be unchanged finite nonnegative benchmark expression')


def fit_train_copula(training_controls, upstream_directory):
    native = configure_upstream(upstream_directory)
    x = np.concatenate(training_controls, axis=0).astype(np.float32)
    validate_expression(x)
    p = native.fit_batch_gmm(x, n_components=COMPONENTS, n_jobs=1)
    z = native.expr_to_gaussian(x, p, eps=1e-6, seed=0, truncated=True)
    A, cm, cs, psi, d = native.learn_factor_loading(z, RANK, eps=1e-6)
    assets = dict(factor_loadings=A.numpy(), copula_ctrl_mean=cm.numpy(),
                  copula_ctrl_std=cs.numpy(), ctrl_psi=psi.numpy(), copula_d=d.numpy(),
                  pooled_control_params=p)
    if not all(np.isfinite(v).all() for v in assets.values()):
        raise ValueError('Nonfinite train-only copula fit')
    if not np.allclose(A.T @ A, np.eye(RANK), atol=2e-5):
        raise ValueError('Copula basis not orthonormal')
    return assets


def fit_population(x, environment, target, assets, upstream_directory):
    """Native marginal/factor fitting without extra normalization or HVGs."""
    native = configure_upstream(upstream_directory)
    x = np.asarray(x, dtype=np.float32)
    validate_expression(x)
    selected = population_indices(len(x), condition_seed(environment, target))
    params = np.stack([native.fit_batch_gmm(x[ids], n_components=COMPONENTS, n_jobs=1)
                       for ids in selected]).astype(np.float32)
    A, cm, cs, d = [torch.as_tensor(assets[k], dtype=torch.float32)
                    for k in ['factor_loadings', 'copula_ctrl_mean', 'copula_ctrl_std', 'copula_d']]
    fvs = []
    for ids, p in zip(selected, params):
        z = native.expr_to_gaussian(x[ids], p, eps=1e-6, seed=0, truncated=True)
        fvs.append(native.factor_variance(z, A, cm, cs, eps=1e-6, d=d).numpy())
    fv = np.stack(fvs).astype(np.float32)
    if not np.isfinite(params).all() or not np.isfinite(fv).all() or (fv < 1e-6).any():
        raise ValueError('Invalid native fitted population')
    return params, fv, selected
