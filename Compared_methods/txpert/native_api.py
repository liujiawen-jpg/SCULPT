"""Public two-graph TxPert factory and native forward/loss adapters."""
from __future__ import annotations

import copy
from pathlib import Path
import random

import numpy as np
import torch
import yaml
from gspp.data.graphmodule import GSPGraph
from gspp.models.txpert import TxPert


def make_graph(targets, genes, symbols, graph_cache):
    """symbols must be an audited stable-ID -> unique HGNC symbol mapping.

    Missing or ambiguous symbols retain their unique stable ID; targets are
    never removed. graph_cache contains the official STRING and GO files.
    """
    from collections import Counter
    names = [symbols.get(t) or t for t in targets]
    counts = Counter(names)
    names = [n if counts[n]==1 and n!="ctrl" else t for n,t in zip(names,targets)]
    if len(set(names)) != len(targets):
        raise ValueError("Ambiguous graph name mapping")
    perturbations = {n:i for i,n in enumerate(names)}
    perturbations["ctrl"] = -1
    config = yaml.safe_load((Path(__file__).parent/"upstream_config.yaml").read_text())
    graph = GSPGraph(perturbations,{g:i for i,g in enumerate(genes)},
        graph_cfg=copy.deepcopy(config["graph"]["graph_cfg"]),cache_dir=str(graph_cache))
    return graph


def make_model(graph, n_genes, seed=17, device="cuda:0"):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    config = yaml.safe_load((Path(__file__).parent/"upstream_config.yaml").read_text())
    args = copy.deepcopy(config["model"])
    args.pop("model_type"); args.pop("slow_benchmark")
    controls, perts = args.pop("cntr_model"), args.pop("pert_model")
    return TxPert(input_dim=n_genes,output_dim=n_genes,adata_output_dim=n_genes,
        graph=graph,cntr_model_args=controls,pert_model_args=perts,device=device,**args).to(device)


def train_step(model, optimizer, control_means, target_indices, real_cells):
    """Registered benchmark minibatch through native forward and model.loss.

    Each real cell is paired with its own environment's full pooled control
    mean. Target index -1 denotes an actual control observation.
    """
    device = next(model.parameters()).device
    controls = torch.as_tensor(control_means,dtype=torch.float32,device=device)
    labels = torch.as_tensor(real_cells,dtype=torch.float32,device=device)
    indices = torch.as_tensor(target_indices).tolist()
    perturbations = [[t,-1] if t>=0 else [-1] for t in indices]
    pred,mu,logvar = model(controls,perturbations,torch.zeros((len(indices),2),device=device))
    loss,logs = model.loss(pred,mu,logvar,labels)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite native TxPert loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.detach(),logs


@torch.no_grad()
def predict_mean(model, controls, target_indices):
    """Native eval forward, vectorized only after switching the model to eval."""
    if model.training:
        raise ValueError("Call model.eval() before inference")
    device = next(model.parameters()).device
    x = torch.as_tensor(controls,dtype=torch.float32,device=device)
    idx = torch.as_tensor(target_indices,dtype=torch.long,device=device)
    if x.ndim==1:
        x = x[None].expand(len(idx),-1)
    z_p = model.pert_model.forward()
    z = model.cntr_model.forward(x)[0] + z_p[idx] + z_p[-1]
    return model.decoder(z).cpu().numpy()


def graph_state(model):
    """Upstream graph tensors are attributes and need explicit checkpointing."""
    names = ("edge_index","edge_type","edge_type_multihot","edge_attr",
        "local_edge_index","local_edge_weight","expander_edge_index")
    return {n:v.detach().cpu() for n in names
        if isinstance((v:=getattr(model.pert_model,n,None)),torch.Tensor)}


def restore_graph(model, state):
    device = next(model.parameters()).device
    for name,value in state.items():
        setattr(model.pert_model,name,value.to(device))
