"""Train-only, metadata-defined panels and reliability-balanced RankNet.

These are training surrogates for raw-response cosine PDS, not exact PDS.
The caller must obtain the bank through the authorized training-bank API.
This module never opens label files, checkpoints, or held-environment assets.
"""
from __future__ import annotations
import hashlib
import math
import numpy as np
import torch
from torch.nn import functional as F


def _hash_order(targets, seed, epoch, purpose):
    return sorted(range(len(targets)), key=lambda i: (
        hashlib.sha256(f'{purpose}|{seed}|{epoch}|{targets[i]}'.encode()).digest(), targets[i]))


def _relation_gene_selection(pred, bank, mask, gene_indices):
    """Use cached metadata indices without CUDA's dynamic boolean selection."""
    if gene_indices is None:
        return pred[mask], bank[:, mask].detach(), mask.new_ones(())
    ids = torch.as_tensor(gene_indices, device=pred.device)
    if ids.ndim != 1 or ids.dtype != torch.long or len(ids) > len(pred):
        raise ValueError('Relation gene indices must follow the fixed mask')
    # Invalid caller-supplied indices stay safe until the deferred flag is read.
    # Cached panel indices are already ascending, unique and in range.
    safe_ids = ids.clamp(0, max(0, len(pred) - 1))
    indices_valid = ((ids >= 0) & (ids < len(pred))).all()
    indices_valid = indices_valid & (ids[1:] > ids[:-1]).all()
    indices_valid = indices_valid & (mask.sum() == len(ids)) & mask[safe_ids].all()
    return pred[safe_ids], bank[:, safe_ids].detach(), indices_valid


def _check_relation_inputs(checks):
    """Read one device flag on the valid strict path; retain detailed failures."""
    valid = torch.stack([check for check, _ in checks]).all()
    if not bool(valid):
        for check, message in checks:
            if not bool(check):
                raise ValueError(message)
    return valid


def panel_ranknet_loss(pred, truth, bank, *, own_index, mask, reliability=None,
                       query_weight=None, norm_floor=1e-4, huber_beta=.1,
                       temperature=.05, check_inputs=True, gene_indices=None):
    """Positive paired fit plus zero-margin rank loss against fixed true rows.

    Candidate reliability is in [0,1]. Query weight is independently balanced
    within a training-energy stratum and may exceed one, up to three. Neither
    the candidate rows nor the weights receive gradients. The public default
    checks inputs strictly. A batched caller can defer its single device flag
    by setting check_inputs=False; it must check relation_inputs_valid later.
    """
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError('RankNet temperature must be finite and positive')
    if norm_floor <= 0 or huber_beta <= 0:
        raise ValueError('Positive norm floor and Huber beta required')
    if pred.ndim != 1 or truth.shape != pred.shape or bank.ndim != 2 or bank.shape[1] != len(pred):
        raise ValueError('Relation loss requires aligned response axes')
    if mask.shape != pred.shape or mask.dtype != torch.bool:
        raise ValueError('Relation mask must be one fixed nonempty gene mask')
    if not 0 <= own_index < len(bank):
        raise ValueError('True query missing from training candidate bank')
    x, y, indices_valid = _relation_gene_selection(pred, bank, mask, gene_indices)
    informative = y.norm(dim=1) > norm_floor
    raw_weights = (torch.ones(len(bank), device=pred.device, dtype=pred.dtype)
                   if reliability is None else torch.as_tensor(reliability, device=pred.device, dtype=pred.dtype).detach())
    if raw_weights.shape != (len(bank),):
        raise ValueError('Invalid training-only response reliability')
    weights = raw_weights.clamp(max=1) * informative.to(pred.dtype)
    qweight = (weights[own_index] if query_weight is None else
               torch.as_tensor(query_weight, device=pred.device, dtype=pred.dtype).detach())
    if qweight.ndim != 0:
        raise ValueError('Query weight must be a finite scalar in [0,3]')
    checks = [
        (mask.any(), 'Relation mask must be one fixed nonempty gene mask'),
        (torch.isclose(bank[own_index], truth, atol=2e-6, rtol=1e-5).all(),
         'Candidate diagonal does not match this query truth'),
        ((torch.isfinite(raw_weights) & (raw_weights >= 0)).all(),
         'Invalid training-only response reliability'),
        (torch.isfinite(qweight) & (qweight >= 0) & (qweight <= 3),
         'Query weight must be a finite scalar in [0,3]'),
        (indices_valid, 'Relation gene indices must follow the fixed mask')]
    inputs_valid = (_check_relation_inputs(checks) if check_inputs else
                    torch.stack([check for check, _ in checks]).all())
    # A zero-response query cannot gain weight through stratum normalization.
    qweight = qweight * informative[own_index].to(pred.dtype)
    unit_y = y / y.norm(dim=1, keepdim=True).clamp_min(norm_floor)
    unit_x = x / x.norm().clamp_min(norm_floor)
    sim = (unit_x[None] @ unit_y.T)[0].clamp(-1., 1.)
    own_sim = sim[own_index]
    diagonal = qweight * F.smooth_l1_loss(own_sim, own_sim.new_ones(()), beta=huber_beta)
    # The candidate count and CPU-owned identity fix this exclusion's size.
    other = torch.arange(len(bank), device=pred.device)
    other = torch.cat((other[:own_index], other[own_index + 1:]))
    pair_loss = float(temperature) * F.softplus((sim[other] - own_sim) / float(temperature))
    denominator = weights[other].sum().clamp_min(1e-8)
    off = qweight * (pair_loss * weights[other]).sum() / denominator
    diagnostics = dict(relation_diagonal=diagonal, relation_off_diagonal=off,
        relation_paired_cosine=own_sim.detach(), relation_query_weight=qweight.detach(),
        relation_candidate_weight_sum=weights[other].sum().detach())
    if not check_inputs:
        diagnostics['relation_inputs_valid'] = inputs_valid.detach()
    return .5 * (diagonal + off), diagnostics


class RankingPanels:
    """Each query owns one metadata-only panel; masks follow the full panel.

    Every epoch permutes all targets by a deterministic metadata hash, splits
    into panel_size groups, and pads only the incomplete final group with
    unique targets from other groups. A padded target retains its original
    home panel. Energy strata are fixed by training responses and do not
    depend on epoch, panel scores, held values, or predictions.
    """
    def __init__(self, bank, gene_ids, seed, epoch, panel_size=256,
                 temperature=.05, norm_floor=1e-4):
        self.targets = [str(t) for t in bank['targets']]
        self.genes = [str(g) for g in gene_ids]
        if not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError('Unique nonempty training target axis required')
        if not self.genes or len(set(self.genes)) != len(self.genes):
            raise ValueError('Unique nonempty gene axis required')
        if not isinstance(panel_size, int) or panel_size < 2 or len(self.targets) < 2:
            raise ValueError('A retrieval panel requires at least two candidates')
        self.seed, self.epoch = int(seed), int(epoch)
        self.panel_size = min(panel_size, len(self.targets))
        self.temperature, self.norm_floor = float(temperature), float(norm_floor)
        responses = torch.as_tensor(bank['responses']).detach()
        if not responses.is_floating_point(): responses = responses.float()
        if responses.shape != (len(self.targets), len(self.genes)) or not bool(torch.isfinite(responses).all()):
            raise ValueError('Invalid finite training response bank')
        self.responses = responses
        reliability = torch.as_tensor(bank['reliability'], device=responses.device, dtype=responses.dtype).detach()
        if reliability.shape != (len(self.targets),) or not bool(torch.isfinite(reliability).all()) or bool((reliability < 0).any()):
            raise ValueError('Invalid training reliability')
        self.reliability = reliability.clamp(max=1)
        # Global training mask defines fixed energy strata only. Retrieval
        # itself uses each panel-specific mask below.
        self.explicit_full_target_axis = 'full_target_axis' in bank
        self.energy_target_axis = [str(t) for t in bank.get('full_target_axis',self.targets)]
        if (not self.energy_target_axis or len(set(self.energy_target_axis))!=len(self.energy_target_axis)
                or not set(self.targets)<=set(self.energy_target_axis)):
            raise ValueError('Full target metadata axis must be unique and contain every candidate')
        target_set = set(self.energy_target_axis)
        expected = torch.tensor([g not in target_set for g in self.genes], dtype=torch.bool, device=responses.device)
        energy_mask = torch.as_tensor(bank.get('mask', expected), device=responses.device, dtype=torch.bool)
        if energy_mask.shape != (len(self.genes),) or not bool(energy_mask.any()) or not torch.equal(energy_mask, expected):
            raise ValueError('Stratum energy mask must exclude the complete declared target metadata axis')
        energy = responses[:, energy_mask].double().square().mean(1).cpu().numpy()
        order = np.lexsort((np.asarray(self.targets), energy))
        strata = np.zeros(len(self.targets), dtype=np.int64)
        rel = self.reliability.cpu().double().numpy()
        query_weights = np.zeros(len(self.targets), dtype=np.float64)
        self.stratum_summary = []
        for group, ids in enumerate(np.array_split(order, 4)):
            if not len(ids): continue
            strata[ids] = group
            mean_rel = float(rel[ids].mean())
            query_weights[ids] = np.minimum(3., rel[ids] / max(mean_rel, 1e-12))
            self.stratum_summary.append(dict(stratum=group, count=len(ids),
                minimum_energy=float(energy[ids].min()), maximum_energy=float(energy[ids].max()),
                mean_reliability=mean_rel, mean_query_weight=float(query_weights[ids].mean()),
                capped_queries=int(np.sum(query_weights[ids] == 3.)),
                zero_reliability_queries=int(np.sum(rel[ids] == 0))))
        self.energy, self.strata = energy, strata
        self.query_weights_cpu = query_weights.copy()
        self.query_weights = torch.as_tensor(query_weights, device=responses.device, dtype=responses.dtype)
        self.lookup = {t:i for i,t in enumerate(self.targets)}
        ordered = _hash_order(self.targets, self.seed, self.epoch, 'rank-panel')
        self.panels, self.home = [], {}
        self._panel_cache = {}
        for start in range(0, len(ordered), self.panel_size):
            home_ids = ordered[start:start+self.panel_size]
            members = list(home_ids)
            member_set = set(members)
            if len(members) < self.panel_size:
                padding = _hash_order(self.targets, self.seed, self.epoch, 'rank-panel-padding')
                members += [i for i in padding if i not in member_set][:self.panel_size-len(members)]
            panel_id = len(self.panels)
            for index in home_ids: self.home[index] = panel_id
            self.panels.append(tuple(members))
        for panel_id, members in enumerate(self.panels):
            panel_targets = {self.targets[i] for i in members}
            keep = [g not in panel_targets for g in self.genes]
            gene_ids = [i for i, selected in enumerate(keep) if selected]
            if not gene_ids: raise ValueError('Panel target exclusion leaves no competition genes')
            mask = torch.tensor(keep, dtype=torch.bool, device=responses.device)
            gene_indices = torch.tensor(gene_ids, dtype=torch.long, device=responses.device)
            index = torch.as_tensor(members, device=responses.device)
            self._panel_cache[panel_id] = dict(bank=responses.index_select(0,index), mask=mask,
                                              gene_indices=gene_indices,
                                              reliability=self.reliability.index_select(0,index))

    def relation_inputs(self, target):
        if str(target) not in self.lookup:
            raise PermissionError('Query target outside this training bank')
        own = self.lookup[str(target)]
        panel_id = self.home[own]
        return dict(self._panel_cache[panel_id], own_index=self.panels[panel_id].index(own),
                    query_weight=self.query_weights[own], mode='panel_ranknet',
                    temperature=self.temperature, norm_floor=self.norm_floor)

    def audit(self):
        return dict(schema='training_panel_ranknet_v1', seed=self.seed, epoch=self.epoch,
            target_count=len(self.targets), gene_count=len(self.genes), panel_size=self.panel_size,
            panel_count=len(self.panels), temperature=self.temperature,
            panels=[[self.targets[i] for i in members] for members in self.panels],
            home_panel={self.targets[i]:int(panel) for i,panel in self.home.items()},
            stratum_summary=self.stratum_summary,
            stratum_by_target={t:int(self.strata[i]) for i,t in enumerate(self.targets)},
            query_weight_by_target={t:float(self.query_weights_cpu[i]) for i,t in enumerate(self.targets)},
            energy_mask=('complete explicit target metadata axis excluded; fixed across environments/panels/epochs'
                if self.explicit_full_target_axis else 'all training target genes excluded; fixed across panels/epochs'),
            energy_target_axis_count=len(self.energy_target_axis),
            energy_target_axis_is_explicit_metadata=self.explicit_full_target_axis,
            retrieval_mask='all panel target genes excluded from every candidate and query',
            provenance='caller-supplied authorized training bank; no file or held-label reads by this class',
            reliability_limitation='Inherited signal/(signal+mean-noise) is uncalibrated, omits control-mean noise, and need not vanish for noisy near-null conditions; balancing is not a noise test.',
            query_weight_policy='r / training-energy-stratum mean(r), cap 3; exact zero r stays zero',
            candidate_weight_policy='legacy reliability clipped to [0,1], zero-norm responses excluded',
            panel_membership_uses_response_values=False)
