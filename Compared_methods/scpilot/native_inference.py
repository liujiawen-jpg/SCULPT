"""Thin, audited access to the complete upstream scPILOT transfer algorithm.

The benchmark trains one unconditional EGD encoder/generator/discriminator on
all legal training perturbations.  At inference, the atlas is restricted to
one target.  This is a registered *multi-perturbation training extension* of
the paper's individual-perturbation experiments, not a claim that upstream
trained one model on the entire benchmark.

No old predictor, source expression mean, numerical calibration, clipping or
mean-shift surrogate is used. scPILOT's latent OT, Leiden grouping, posterior
sampling and generator are retained. Its normalized exp(-OT cost) weights use
an audited, mathematically equivalent shifted exponential to avoid 0/0.
"""
from __future__ import annotations

import contextlib
import hashlib
import random
from collections.abc import Mapping, Sequence
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch

from .stable_adaptive_weights import stable_adaptive_prediction


CONTROL_LABEL = "non-targeting"
CELL_LABEL_KEY = "cell_type"
CONDITION_KEY = "target_gene"
OFFICIAL_FLAGS = {
    "use_latent_ot": True,
    "use_leiden": True,
    "use_adaptive_weighting": True,
}


def _dense_cells(value: Any, *, label: str, n_genes: int) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    out = np.asarray(value, dtype=np.float32)
    if out.ndim != 2 or out.shape[0] == 0 or out.shape[1] != n_genes:
        raise ValueError(f"{label}: expected nonempty cells x {n_genes}; got {out.shape}")
    if not np.isfinite(out).all():
        raise ValueError(f"{label}: input contains nonfinite expression")
    return out


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(repr(array.shape).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


@contextlib.contextmanager
def _isolated_rng(seed: int):
    """Validation/inference must not consume the training RNG stream."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if cuda_states:
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)


def predict_target_distribution(
    model: Any,
    source_controls: Mapping[str, Any],
    source_perturbed: Mapping[str, Any],
    target_controls: Any,
    target_stable_id: str,
    genes: Sequence[str],
    seed: int,
    *,
    target_environment: str,
    n_generated: int = 512,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict cells with native OT/Leiden, without access to query P labels.

    ``source_perturbed[env]`` MUST contain only this target's lawful source
    cells. ``target_controls`` MUST be lawful target E cells. Authorization
    and source cache limits belong to the caller's protected data reader;
    this function does not open datasets or read any test expression labels.

    All supplied source cells are retained. Target controls are resampled
    with replacement to exactly ``n_generated`` rows. Posterior randomness is
    seeded separately from this bootstrap. Generated cells remain the native
    signed outputs; negative entries are recorded, never silently censored.
    The mean prediction is to be computed from these exact returned rows.
    """
    gene_ids = [str(g) for g in genes]
    if not gene_ids or len(set(gene_ids)) != len(gene_ids):
        raise ValueError("genes must be a nonempty unique, ordered stable-ID axis")
    if not isinstance(target_stable_id, str) or not target_stable_id:
        raise ValueError("target_stable_id must be a nonempty string")
    if target_stable_id == CONTROL_LABEL:
        raise ValueError("a perturbation target must differ from the control label")
    if not isinstance(target_environment, str) or not target_environment:
        raise ValueError("target_environment must be a nonempty string")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(n_generated, (int, np.integer)) or n_generated < 2:
        raise ValueError("n_generated must be an integer >= 2")
    if not source_controls or set(source_controls) != set(source_perturbed):
        raise ValueError("nonempty source E and P mappings must have identical environment keys")
    if any(not isinstance(env, str) or not env for env in source_controls):
        raise ValueError("source environment names must be nonempty strings")
    if target_environment in source_controls:
        raise ValueError("query environment cannot appear among source perturbation environments")
    previous_atlas = model.adata
    if list(previous_atlas.var_names.astype(str)) != gene_ids:
        raise ValueError("model training gene axis differs from the requested prediction gene axis")
    if not callable(getattr(model, "predict_new", None)):
        raise TypeError("model must expose the official EGD_model.predict_new")

    n_genes = len(gene_ids)
    query_e = _dense_cells(target_controls, label="target E", n_genes=n_genes)
    # A separate Generator makes bootstrap independent of how much posterior
    # randomness upstream uses in its per-source and per-cluster encodings.
    bootstrap_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 512, 20261022]))
    query_indices = bootstrap_rng.choice(len(query_e), size=int(n_generated), replace=True)
    query_names = pd.Index([f"query_{i:06d}" for i in range(int(n_generated))])
    query_obs = pd.DataFrame(
        {CELL_LABEL_KEY: target_environment, CONDITION_KEY: CONTROL_LABEL},
        index=query_names,
    )
    var = pd.DataFrame(index=pd.Index(gene_ids))
    query = ad.AnnData(X=query_e[query_indices].copy(), obs=query_obs, var=var.copy())

    source_parts = []
    counts = {}
    for i, environment in enumerate(sorted(source_controls)):
        counts[environment] = {}
        for family, condition, mapping in (
            ("E", CONTROL_LABEL, source_controls),
            ("P", target_stable_id, source_perturbed),
        ):
            cells = _dense_cells(mapping[environment], label=f"source {environment}/{family}", n_genes=n_genes)
            counts[environment][family] = int(len(cells))
            obs = pd.DataFrame(
                {CELL_LABEL_KEY: environment, CONDITION_KEY: condition},
                index=[f"source_{i:03d}_{family}_{j:06d}" for j in range(len(cells))],
            )
            source_parts.append(ad.AnnData(X=cells, obs=obs, var=var.copy()))
    atlas = ad.concat(source_parts, join="inner", merge="same", index_unique=None)
    if list(atlas.var_names.astype(str)) != gene_ids:
        raise ValueError("atlas concatenation unexpectedly changed the gene axis")

    was_training = bool(model.module.training)
    try:
        model.adata = atlas
        model.module.eval()
        with _isolated_rng(int(seed)), torch.inference_mode(), stable_adaptive_prediction(model) as weight_diagnostics:
            prediction, native_delta = model.predict_new(
                cell_label_key=CELL_LABEL_KEY,
                cond_key=CONDITION_KEY,
                ctrl_key=CONTROL_LABEL,
                stim_key=target_stable_id,
                adata_to_predict=query,
                **OFFICIAL_FLAGS,
            )
        if prediction.n_obs != int(n_generated):
            raise RuntimeError(f"native predictor returned {prediction.n_obs} cells, expected {n_generated}")
        if not prediction.obs_names.is_unique or set(prediction.obs_names) != set(query_names):
            raise RuntimeError("native prediction lost or duplicated bootstrap row identities")
        if list(prediction.var_names.astype(str)) != gene_ids:
            raise RuntimeError("native prediction changed output gene identities or order")
        native_order = prediction.obs_names.tolist()
        # Leiden concatenates clusters. Restore original bootstrap order by
        # unique row identity, rather than accidentally comparing wrong cells.
        aligned = prediction[query_names].copy()
        output = np.asarray(aligned.X, dtype=np.float32)
        if output.shape != (int(n_generated), n_genes) or not np.isfinite(output).all():
            raise FloatingPointError(
                "native scPILOT generated nonfinite expression; no clipping or fallback is permitted. "
                "Stable native weighting is enabled; inspect finite OT costs and model states."
            )
        native_delta = np.asarray(native_delta)
        if native_delta.shape[0] != int(n_generated) or not np.isfinite(native_delta).all():
            raise FloatingPointError("native scPILOT returned invalid latent OT deltas")
        diagnostics = {
            "method": "scPILOT-MT-native-OT-Leiden",
            "training_extension": "one shared unconditional EGD model across lawful training perturbations",
            "inference_entrypoint": "EGD_model.predict_new",
            "inference_flags": dict(OFFICIAL_FLAGS),
            "target_stable_id": target_stable_id,
            "target_environment": target_environment,
            "source_environments": sorted(source_controls),
            "source_cell_counts": counts,
            "source_cells_subsampled_by_adapter": False,
            "target_input_family": "E_only",
            "target_P_read_by_inference": False,
            "query_control_cells_available": int(len(query_e)),
            "query_bootstrap_with_replacement": True,
            "query_bootstrap_unique_cells": int(len(np.unique(query_indices))),
            "query_bootstrap_indices_sha256": _array_sha256(query_indices),
            "query_bootstrap_indices": query_indices.tolist(),
            "seed": int(seed),
            "shape": list(output.shape),
            "output_sha256": _array_sha256(output),
            "native_output_row_order_changed": native_order != query_names.tolist(),
            "output_reordered_by_unique_query_identity": True,
            "latent_delta_shape": list(native_delta.shape),
            "minimum": float(output.min()),
            "maximum": float(output.max()),
            "negative_values": int((output < 0).sum()),
            "negative_fraction": float((output < 0).mean()),
            "mean_gene_variance": float(output.var(axis=0, dtype=np.float64).mean()),
            "output_domain": "native_signed_normalized_expression",
            "postprocessing": "identity-based row reordering only",
            "clipping_applied": False,
            "mean_correction_applied": False,
            "adaptive_weight_numerics": weight_diagnostics,
            "previous_model_atlas_restored": True,
        }
        return output, diagnostics
    finally:
        model.adata = previous_atlas
        model.module.train(was_training)


# Stable short alias for callers; the implementation above is the only path.
predict_distribution = predict_target_distribution
