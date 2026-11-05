"""Benchmark condition sets for the official STATE model (no normalization)."""
from __future__ import annotations

import hashlib
import numpy as np
import torch

from .native_api import make_batch, predict_cells


def training_conditions(store):
    """Legal P sets first, then one E set for each training context."""
    return [(e, t) for e in store.training_environments for t in store.training_targets(e)] + [
        (e, None) for e in store.training_environments]


def draw_training_batch(store, conditions, indices, rng, *, device="cpu"):
    """Independent 512-control and 512-P draws with replacement per set.

    Caller shuffles training_conditions once per epoch and calls this function
    with consecutive batches of 16 conditions; the final smaller batch remains.
    """
    controls, labels, ids = [], [], []
    lookup = {t: i for i, t in enumerate(store.targets)}
    for index in indices:
        environment, target = conditions[int(index)]
        if environment not in store.training_environments:
            raise PermissionError("STATE supervision must be a training context")
        other_sources = tuple(e for e in store.training_environments if e != environment)
        store.authorize_episode(environment, other_sources)
        try:
            e = store.get_e(environment)
            p = e if target is None else store.get_p(environment, target, "supervision")
        finally:
            store.clear_episode()
        controls.append(e[rng.integers(0, len(e), size=512)])
        labels.append(p[rng.integers(0, len(p), size=512)])
        ids.append(len(store.targets) if target is None else lookup[target])
    if not controls:
        raise ValueError("A batch must contain at least one condition")
    x = torch.as_tensor(np.stack(controls), dtype=torch.float32, device=device)
    y = torch.as_tensor(np.stack(labels), dtype=torch.float32, device=device)
    return make_batch(x, torch.as_tensor(ids, device=device), len(store.targets) + 1, y)


def bootstrap_controls(controls, *, seed, environment, target):
    """Match the registered STATE-native-cell-set condition RNG key."""
    controls = np.asarray(controls, dtype=np.float32)
    if controls.ndim != 2 or not len(controls) or not np.isfinite(controls).all():
        raise ValueError("Finite control cells required")
    key = f"STATE-native-cell-set|{seed}|{environment}|{target}".encode()
    local_seed = int.from_bytes(hashlib.sha256(key).digest()[:4], "little")
    rng = np.random.default_rng(local_seed)
    return controls[rng.integers(0, len(controls), size=512)]


@torch.no_grad()
def predict_condition(model, controls, target_index, *, seed, environment, target):
    """512 native predictions; no extra random noise or output calibration."""
    model.eval()
    device = next(model.parameters()).device
    x = bootstrap_controls(controls, seed=seed, environment=environment, target=target)
    x = torch.as_tensor(x[None], device=device)
    result = predict_cells(model, x, torch.tensor([target_index], device=device))[0]
    cloud = result.float().cpu().numpy()
    if cloud.shape != (512, controls.shape[1]) or not np.isfinite(cloud).all() or (cloud < 0).any():
        raise ValueError("Official STATE returned an invalid native expression cloud")
    return cloud, cloud.mean(axis=0, dtype=np.float64).astype(np.float32)
