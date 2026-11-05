"""Thin, pinned bridge to Arc Institute STATE's actual State Transition model.

This module contains no replacement network.  It adapts the benchmark's cell
sets and perturbation IDs to the official class and uses its distribution loss.
Inputs and outputs are the existing 5,979-gene log-normalized expression axis;
``X_hvg`` is the upstream API name, not a new feature-selection operation.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
import torch.nn.functional as F
import yaml


PINNED_COMMIT = "9bbfe78a434a55205e4de834e1ea99f85f7a3add"


def _load(upstream=None):
    """Load verified upstream source; never fall back to an unrelated install."""
    import json
    import os
    if upstream is None:
        upstream = os.environ.get("STATE_UPSTREAM")
    if not upstream:
        raise ValueError("Set STATE_UPSTREAM to the pinned official checkout")
    root = Path(upstream).resolve()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError("The STATE checkout has the wrong revision")
    proof = json.loads(Path(__file__).with_name("upstream_provenance.json").read_text())
    for filename, expected in proof["files_sha256"].items():
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != expected:
            raise RuntimeError("STATE upstream scientific source differs: " + filename)
    location = str(root / "src")
    if location not in sys.path:
        sys.path.insert(0, location)
    from state.tx.models.state_transition import StateTransitionPerturbationModel
    actual = Path(sys.modules[StateTransitionPerturbationModel.__module__].__file__).resolve()
    if actual != (root / "src/state/tx/models/state_transition.py").resolve():
        raise RuntimeError("STATE imported from a different installation")
    return root, StateTransitionPerturbationModel


def official_configuration(cell_set_len: int = 512, *, upstream=None) -> dict[str, Any]:
    """Resolve the official ``model=state`` YAML without changing its network."""
    VENDOR_ROOT, _ = _load(upstream)
    with (VENDOR_ROOT / "src/state/configs/model/state.yaml").open() as stream:
        config = yaml.safe_load(stream)
    kwargs = config["kwargs"]
    kwargs["cell_set_len"] = int(cell_set_len)
    backbone = kwargs["transformer_backbone_kwargs"]
    backbone["max_position_embeddings"] = int(cell_set_len)
    backbone["hidden_size"] = int(kwargs["hidden_dim"])
    return config


def build_model(
    n_genes: int,
    n_perturbations: int,
    *,
    lr: float = 1e-4,
    batch_size: int = 16,
    cell_set_len: int = 512,
    gene_names: list[str] | None = None,
    device: torch.device | str | None = None,
    verbose: bool = False,
    upstream=None,
) -> StateTransitionPerturbationModel:
    """Construct the official ST-base model from fresh task-specific weights.

    The extra decoder configuration reproduces ``state tx train`` exactly.
    For upstream X_hvg/gene input, cell-load does not supply
    ``pert_cell_counts``; that auxiliary decoder is therefore not supervised or
    used for predictions.  The primary official ``preds`` output is used here.
    """
    if n_genes < 1 or n_perturbations < 2 or cell_set_len < 1:
        raise ValueError("Invalid expression, perturbation, or cell-set dimension")
    if gene_names is not None and len(gene_names) != n_genes:
        raise ValueError("gene_names must match the fixed expression axis")
    VENDOR_ROOT, StateTransitionPerturbationModel = _load(upstream)
    config = official_configuration(cell_set_len, upstream=VENDOR_ROOT)
    kwargs = config["kwargs"]
    kwargs["decoder_cfg"] = {
        "latent_dim": n_genes,
        "gene_dim": n_genes,
        "hidden_dims": [1024, 1024, 512],
        "dropout": 0.1,
        "residual_decoder": False,
    }
    output_context = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with output_context:
        model = StateTransitionPerturbationModel(
            input_dim=n_genes,
            output_dim=n_genes,
            gene_dim=n_genes,
            hvg_dim=n_genes,
            pert_dim=n_perturbations,
            batch_dim=None,
            output_space="gene",
            embed_key="X_hvg",
            gene_names=gene_names,
            basal_mapping_strategy="random",
            control_pert="non-targeting",
            lr=lr,
            batch_size=batch_size,
            **kwargs,
        )
    if Path(sys.modules[StateTransitionPerturbationModel.__module__].__file__).resolve() != (
        VENDOR_ROOT / "src/state/tx/models/state_transition.py"
    ).resolve():
        raise RuntimeError("STATE import did not resolve to the pinned official checkout")
    if device is not None:
        model = model.to(device)
    return model


def make_batch(
    control_cells: torch.Tensor,
    perturbation_ids: torch.Tensor,
    n_perturbations: int,
    target_cells: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """One condition per set; no labels are required or passed at prediction."""
    if control_cells.ndim != 3:
        raise ValueError("control_cells must have shape [conditions, cells, genes]")
    batch_size, n_cells, _ = control_cells.shape
    perturbation_ids = torch.as_tensor(perturbation_ids, device=control_cells.device, dtype=torch.long)
    if perturbation_ids.shape != (batch_size,):
        raise ValueError("One perturbation ID is required per cell set")
    # one_hot itself rejects IDs outside [0, n_perturbations); no host GPU sync.
    one_hot = F.one_hot(perturbation_ids, num_classes=n_perturbations).to(control_cells.dtype)
    result = {
        "ctrl_cell_emb": control_cells,
        "pert_emb": one_hot[:, None, :].expand(-1, n_cells, -1),
    }
    if target_cells is not None:
        if target_cells.shape != control_cells.shape:
            raise ValueError("target_cells must match [conditions, cells, genes]")
        result["pert_cell_emb"] = target_cells
    return result


def forward_cells(model: StateTransitionPerturbationModel, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run the untouched official forward; caller may use bf16 autocast."""
    controls = batch["ctrl_cell_emb"]
    if controls.ndim != 3 or controls.shape[1] != model.cell_sentence_len:
        raise ValueError("Every condition must contain the configured cell-set length")
    return model.forward(batch, padded=True).reshape(controls.shape[0], controls.shape[1], model.output_dim)


def native_loss(
    model: StateTransitionPerturbationModel,
    batch: dict[str, torch.Tensor],
    prediction: torch.Tensor | None = None,
) -> torch.Tensor:
    """Official main energy loss, in fp32, with no surrogate/PDS/MSE loss.

    At the fixed default settings, ``training_step`` has only this active loss:
    confidence/batch/regularization are off, and X_hvg batches have no raw-count
    auxiliary target.  Calling the official loss helper avoids Lightning's
    logging requirement while preserving its set-wise nanmean reduction.
    """
    if "pert_cell_counts" in batch:
        raise ValueError("X_hvg/gene bridge must not introduce an auxiliary gene-decoder target")
    if model.confidence_token is not None or model.regularization != 0 or model.use_batch_token:
        raise ValueError("native_loss bridge is audited only for the official base configuration")
    if prediction is None:
        prediction = forward_cells(model, batch)
    target = batch["pert_cell_emb"]
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target must have identical cell-set axes")
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        return torch.nanmean(model._compute_distribution_loss(prediction.float(), target.float()))


def predict_cells(
    model: StateTransitionPerturbationModel,
    control_cells: torch.Tensor,
    perturbation_ids: torch.Tensor,
    n_perturbations: int | None = None,
) -> torch.Tensor:
    """Native predicted cell cloud; no sampled noise, refitting, or rescaling."""
    batch = make_batch(control_cells, perturbation_ids, n_perturbations or model.pert_dim)
    return forward_cells(model, batch)


def provenance(model: StateTransitionPerturbationModel | None = None, *, upstream=None) -> dict[str, Any]:
    VENDOR_ROOT, _ = _load(upstream)
    paths = [
        "src/state/configs/model/state.yaml",
        "src/state/configs/training/default.yaml",
        "src/state/tx/models/state_transition.py",
        "src/state/tx/models/base.py",
        "src/state/tx/models/utils.py",
        "src/state/_cli/_tx/_train.py",
    ]
    commit = subprocess.check_output(["git", "-C", str(VENDOR_ROOT), "rev-parse", "HEAD"], text=True).strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Unexpected official STATE revision: {commit}")
    result = {
        "repository": "https://github.com/ArcInstitute/state",
        "commit": commit,
        "model_class": "state.tx.models.state_transition.StateTransitionPerturbationModel",
        "input_space": "benchmark log1p normalized expression; official X_hvg API",
        "output_space": "gene",
        "native_distribution_output": True,
        "fresh_task_weights": True,
        "SE_pretrained_weights_used": False,
        "loss": "official geomloss energy; set-wise nanmean; fp32 distances",
        "optimizer": "official configure_optimizers: Adam; no scheduler; no applied weight_decay",
        "auxiliary_decoder": "retained as in official CLI, inactive for X_hvg/gene batches",
        "files_sha256": {p: hashlib.sha256((VENDOR_ROOT / p).read_bytes()).hexdigest() for p in paths},
    }
    if model is not None:
        result["parameters"] = sum(p.numel() for p in model.parameters())
        result["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        result["cell_set_len"] = model.cell_sentence_len
        result["n_genes"] = model.output_dim
        result["n_perturbations"] = model.pert_dim
    return result


def train_step(model, optimizer, batch):
    """One registered native STATE update; benchmark stopping is external."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    device = batch["ctrl_cell_emb"].device
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        prediction = forward_cells(model, batch)
    loss = native_loss(model, batch, prediction=prediction)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite native STATE objective")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("Nonfinite STATE gradient")
    optimizer.step()
    return dict(loss=float(loss.detach()), gradient_norm=float(gradient_norm))
