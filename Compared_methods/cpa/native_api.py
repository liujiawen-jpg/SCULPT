"""Thin access to the pinned CPA module, native loss, and Gaussian sampling.

Use in CPA's separate environment after installing the official source. No
historical predictor or task checkpoint is imported by these constructors.
"""
from __future__ import annotations

import hashlib
import importlib.util
import random
import sys
import types

import numpy as np
import torch

# The official top-level __init__ imports optional plotting/RDKit/Ray tuners.
# Load the unchanged core package files without importing those unused extras,
# matching the recorded benchmark runtime compatibility adaptation.
if "cpa" not in sys.modules:
    package_spec = importlib.util.find_spec("cpa")
    if package_spec is None or not package_spec.submodule_search_locations:
        raise ImportError("Install the pinned official CPA source first")
    package = types.ModuleType("cpa")
    package.__path__ = list(package_spec.submodule_search_locations)
    package.__spec__ = package_spec
    sys.modules["cpa"] = package
from cpa._module import CPAModule
from cpa._task import CPATrainingPlan
from cpa._utils import CPA_REGISTRY_KEYS as K

K.PERTURBATION_KEY = "perturbation_class"
K.MAX_COMB_LENGTH = 1
K.CONTROL_KEY = "control"


class LoggingFreeModule(CPAModule):
    def r2_metric(self, *args, **kwargs):
        # Logging only: retain every native gradient-bearing objective.
        return 0., 0.


class BenchmarkPlan(CPATrainingPlan):
    @property
    def current_epoch(self):
        return getattr(self, "benchmark_epoch", 0)

    @property
    def global_step(self):
        return getattr(self, "benchmark_step", 0)

    @property
    def kl_weight(self):
        return 0.  # Native nonvariational CPA configuration.

    def optimizers(self):
        return self.benchmark_optimizers

    def manual_backward(self, loss):
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite native CPA objective")
        loss.backward()


def make_plan(n_genes, n_targets, seed=17, device="cuda:0"):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    module = LoggingFreeModule(n_genes=n_genes, n_perts=n_targets+1,
        covars_encoder={}, recon_loss="gauss", variational=False, seed=seed).to(device)
    plan = BenchmarkPlan(module=module,covars_to_ncovars={},n_adv_perts=n_targets+1).to(device)
    plan.benchmark_optimizers, plan.benchmark_schedulers = plan.configure_optimizers()
    plan.benchmark_epoch = 0
    plan.benchmark_step = 0
    return plan


def train_step(plan, cells, target_indices, batch_index=0):
    """target_indices: -1 for measured controls, zero-based target ID otherwise."""
    classes = target_indices.long() + 1
    batch = {K.X_KEY:cells, K.PERTURBATION_KEY:classes,
        K.PERTURBATIONS:classes[:,None],
        K.PERTURBATIONS_DOSAGES:(classes>0).float()[:,None]}
    result = plan.training_step(batch,batch_index)
    plan.benchmark_step += 1
    return result


@torch.no_grad()
def predict_cloud(plan, controls, target_index, *, seed=17, environment="target", target_id=None):
    """512 native conditional Normal samples with the registered zero censoring."""
    device = next(plan.module.parameters()).device
    target_id = str(target_index) if target_id is None else target_id
    key = f"cpa-cloud-v1|{seed}|{environment}|{target_id}".encode()
    local_seed = int.from_bytes(hashlib.sha256(key).digest()[:8],"little") % (2**63-1)
    generator = torch.Generator(device=device).manual_seed(local_seed)
    x = torch.as_tensor(controls,dtype=torch.float32,device=device)
    x = x[torch.randint(len(x),(512,),device=device,generator=generator)]
    classes = torch.full((512,1),target_index+1,device=device,dtype=torch.long)
    perts = {"true":classes,"mixup":classes}
    doses = {"true":torch.ones((512,1),device=device),"mixup":torch.ones((512,1),device=device)}
    gpu_indices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type=="cuda" else []
    with torch.random.fork_rng(devices=gpu_indices):
        torch.manual_seed(local_seed)
        inf = plan.module.inference(x,perts,doses,{},mixup_lambda=1.)
        normal = plan.module.generative(inf["z"])["px"]
    raw = normal.loc + normal.scale * torch.randn(normal.loc.shape,device=device,generator=generator)
    support = raw.clamp_min(0).cpu().numpy()
    return support,dict(raw_negative_fraction=float((raw<0).float().mean()),
        native_unconstrained_mean=normal.loc.mean(0).cpu().numpy(),
        mean_conditional_variance=float(normal.scale.square().mean()))
