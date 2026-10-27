"""Complete upstream scPILOT training, with a lightweight resident-pool runner.

The inherited EGDTrainingPlan.training_step is called unchanged.  Only Lightning
plumbing and the detached, diagnostic-only MMD implementation are adapted.
No historical model weights or baseline predictions are used.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any


import numpy as np
import torch
from scpilot.egd_network import EGD_network
from scpilot.egd_training_plan import EGDTrainingPlan

MODEL_CONFIG = dict(n_hidden=800, n_latent=100, n_layers=2,
                    dropout_rate=0.2, use_discriminator=True,
                    lambd1=0.2, lambd2=1.0, lambd3=1e-3, lambd4=0.05,
                    eps=1e-6)
OPTIMIZER_CONFIG = dict(optimizer="Adam", lr=1e-3,
                        weight_decay=1e-6, eps=0.01,
                        reduce_lr_on_plateau=False)
BATCH_SIZE = 32
CHECKPOINT_VERSION = "scpilot-native-dual-adam-v1"
IDENTITY_KEYS = ("fold", "seed", "spec_sha256", "code_sha256")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng() -> dict[str, Any]:
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all()
                if torch.cuda.is_initialized() else None)


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA RNG checkpoint cannot resume without CUDA")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


@torch.no_grad()
def diagnostic_mmd(batch: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    """Official 50-bandwidth biased RBF MMD, vectorized and RNG independent.

    This scalar is detached and is never used in either optimizer objective.
    Float64 direct differences avoid cancellation; every bandwidth and the
    biased V-statistic match upstream's sklearn implementation.
    """
    if batch.ndim != 2 or batch.shape != reconstruction.shape:
        raise ValueError("MMD expects equal, two-dimensional reconstruction shapes")
    x, y = batch.detach().to(torch.float64), reconstruction.detach().to(torch.float64)
    gammas = torch.as_tensor(np.logspace(1, -3, num=50),
                             dtype=torch.float64, device=x.device)
    def rbf_mean(a, b):
        d2 = torch.cdist(a, b, p=2,
                         compute_mode="donot_use_mm_for_euclid_dist").square()
        return torch.exp(-gammas[:, None, None] * d2[None]).mean(dim=(1, 2))
    return (rbf_mean(x, x) + rbf_mean(y, y) - 2 * rbf_mean(x, y)).mean()


class NativeTrainingPlan(EGDTrainingPlan):
    """Keep upstream forward, loss, optimizer creation, and training_step intact."""

    def __init__(self, network: EGD_network, *, seed: int, n_genes: int):
        super().__init__(network, **OPTIMIZER_CONFIG)
        self.native_seed = int(seed)
        self.native_n_genes = int(n_genes)
        self.native_minibatches = 0
        self._native_logs = {}
        configured = EGDTrainingPlan.configure_optimizers(self)
        if not isinstance(configured, list) or len(configured) != 2:
            raise RuntimeError("Complete scPILOT requires the two upstream Adam optimizers")
        self._native_optimizers = configured

    @property
    def use_sync_dist(self):
        return False

    def optimizers(self, *args, **kwargs):
        return self._native_optimizers

    def manual_backward(self, loss, *args, **kwargs):
        loss.backward(*args, **kwargs)

    def log(self, name, value, *args, **kwargs):
        self._native_logs[name] = (value.detach() if torch.is_tensor(value)
                                   else torch.as_tensor(value, device=self.module.device))

    def log_dict(self, dictionary, *args, **kwargs):
        for name, value in dictionary.items():
            self.log(name, value)

    @torch.no_grad()
    def compute_and_log_metrics(self, rl, kld, dl, metrics, mode):
        # Same observation-weighted minibatch quantities as the official logger.
        # Epoch aggregation belongs to the driver, weighted by actual batch size.
        self.log(f"rl_{mode}", rl.mean())
        self.log(f"kld_{mode}", kld.mean())
        self.log(f"elbo_{mode}", rl.mean() + kld.mean())
        self.log(f"dl_{mode}", dl)

    @staticmethod
    def _compute_gene_reconstruction_mmd(batch, generative_outputs):
        return diagnostic_mmd(batch, generative_outputs["xHat"])


def _set_device(network: EGD_network, device: torch.device):
    network.to(device)
    # Upstream helpers store a concrete .device independently of tensor devices.
    for module in network.modules():
        if "device" in module.__dict__:
            module.device = device


def make_plan(seed: int, n_genes: int, device="cuda:0") -> NativeTrainingPlan:
    if int(n_genes) <= 0:
        raise ValueError("n_genes must be positive")
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    seed_all(int(seed))
    network = EGD_network(n_input=int(n_genes), **MODEL_CONFIG)
    _set_device(network, selected_device)
    plan = NativeTrainingPlan(network, seed=int(seed), n_genes=int(n_genes))
    plan.module.train()
    return plan


def one_step(plan: NativeTrainingPlan, batch: torch.Tensor, batch_idx: int) -> dict[str, Any]:
    if batch.ndim != 2 or batch.shape[1] != plan.native_n_genes:
        raise ValueError("Wrong training gene axis or rank")
    if not 2 <= len(batch) <= BATCH_SIZE:
        raise ValueError("Upstream batch size is 32; singleton batches are forbidden")
    if batch.dtype != torch.float32:
        raise ValueError("Official FP32 training requires float32 batches")
    batch = batch.to(plan.module.device)
    if not bool(torch.isfinite(batch).all()):
        raise ValueError("Non-finite training cells")
    plan.module.train()
    plan._native_logs = {}
    # Deliberately invoke the official method, rather than reimplementing updates.
    EGDTrainingPlan.training_step(plan, batch, int(batch_idx))
    names = list(plan._native_logs)
    values = torch.stack([plan._native_logs[n].to(dtype=torch.float64)
                          for n in names]).cpu().tolist()
    logs = dict(zip(names, values))
    if not all(np.isfinite(v) for v in logs.values()):
        raise FloatingPointError("Non-finite official scPILOT training metric")
    plan.native_minibatches += 1
    logs.update(batch_size=len(batch), minibatches=plan.native_minibatches,
                optimizer_steps_per_optimizer=plan.native_minibatches)
    return logs


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_binding() -> dict[str, Any]:
    import scpilot
    upstream = Path(scpilot.__file__).resolve().parent
    upstream_files = {str(p.relative_to(upstream)): _file_sha(p)
                      for p in sorted(upstream.glob("*.py"))}
    return dict(adapter_sha256=_file_sha(__file__), upstream_files=upstream_files)


def _checked_identity(plan, identity):
    if not isinstance(identity, dict) or any(k not in identity for k in IDENTITY_KEYS):
        raise ValueError("Checkpoint identity needs fold, seed, spec_sha256 and code_sha256")
    identity = json.loads(json.dumps(identity, sort_keys=True))
    if identity["seed"] != plan.native_seed:
        raise ValueError("Checkpoint identity seed differs from the plan seed")
    if identity["fold"] not in {f"F{i}" for i in range(1, 6)}:
        raise ValueError("Checkpoint requires a fixed F1..F5 fold identity")
    for name in ("spec_sha256", "code_sha256"):
        value = identity[name]
        if not isinstance(value, str) or len(value) != 64 or any(
                c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Invalid {name}")
    return identity


def save_checkpoint(plan, path, *, identity, progress=None):
    identity = _checked_identity(plan, identity)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = dict(version=CHECKPOINT_VERSION, identity=identity,
                 implementation=implementation_binding(),
                 model_config=dict(n_input=plan.native_n_genes, **MODEL_CONFIG),
                 optimizer_config=OPTIMIZER_CONFIG,
                 model=plan.module.state_dict(),
                 optimizers=[o.state_dict() for o in plan.optimizers()],
                 minibatches=plan.native_minibatches,
                 rng=capture_rng(), progress={} if progress is None else progress)
    temporary = destination.with_name(destination.name + f".partial.{os.getpid()}")
    try:
        torch.save(state, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return dict(path=str(destination.resolve()), sha256=_file_sha(destination),
                minibatches=plan.native_minibatches, optimizer_count=2)


def load_checkpoint(plan, path, *, identity):
    expected = _checked_identity(plan, identity)
    # Only load a checkpoint produced locally by this driver, never downloaded weights.
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if state.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Unknown checkpoint format")
    if state.get("identity") != expected:
        raise ValueError("Refusing checkpoint from another fold, seed, spec or code binding")
    if state.get("implementation") != implementation_binding():
        raise ValueError("Training implementation changed since checkpoint creation")
    if state.get("model_config") != dict(n_input=plan.native_n_genes, **MODEL_CONFIG):
        raise ValueError("Model configuration changed since checkpoint creation")
    if state.get("optimizer_config") != OPTIMIZER_CONFIG or len(state.get("optimizers", [])) != 2:
        raise ValueError("Incomplete or incompatible dual-Adam state")
    if not isinstance(state.get("minibatches"), int) or state["minibatches"] < 0:
        raise ValueError("Invalid checkpoint step count")
    plan.module.load_state_dict(state["model"], strict=True)
    for optimizer, saved in zip(plan.optimizers(), state["optimizers"]):
        optimizer.load_state_dict(saved)
    plan.native_minibatches = state["minibatches"]
    restore_rng(state["rng"])
    return state.get("progress", {})
