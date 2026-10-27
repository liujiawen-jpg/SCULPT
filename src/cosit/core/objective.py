"""Legacy phase-specific and end-to-end joint supervision. Held labels never train.

A fits analytic p*r and raw-cosine relationships. B fits positive marginals
with frozen p/r. C fits joint structure by exact column permutations only.
"""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F
from .metrics import empirical_w1, empirical_w1_conditions, joint_sw1
from .ranking import panel_ranknet_loss, _relation_gene_selection, _check_relation_inputs

DEFAULT_WEIGHTS = dict(mean_mse=1., mean_correlation=.15, strong_top100=.15,
    detection=.1, positive_mean=.1, response_relation=.15, target_rule=.1,
    source_fit=.02, marginal_W1=.5, variance=.1, positive_shape=.1, centered_joint_SW1=.15,
    relative_PCA_covariance=.05, rank_penalty=.001)


def covariance(x):
    x = x - x.mean(0, keepdim=True)
    return x.T @ x / len(x)


def raw_cosine_matrix(a, b, norm_floor=1e-4):
    """No centering and no response standardization; matches retrieval geometry."""
    an = a / a.norm(dim=-1, keepdim=True).clamp_min(norm_floor)
    bn = b / b.norm(dim=-1, keepdim=True).clamp_min(norm_floor)
    return (an @ bn.T).clamp(-1., 1.)


def response_relation_loss(pred, truth, bank, *, own_index, mask, reliability=None,
                           norm_floor=1e-4, huber_beta=.1, mode='legacy', query_weight=None,
                           temperature=.05, check_inputs=True, gene_indices=None):
    """One query against authorized same-environment training truths.

    Normalize diagonal and off-diagonal independently: a large candidate bank
    must not drown the paired identity signal. Similar biological truths are
    allowed to be similar; this is neither a fixed-margin nor a repulsion loss.
    """
    if mode == 'panel_ranknet':
        return panel_ranknet_loss(pred, truth, bank, own_index=own_index, mask=mask,
            reliability=reliability, query_weight=query_weight, norm_floor=norm_floor,
            huber_beta=huber_beta, temperature=temperature, check_inputs=check_inputs,
            gene_indices=gene_indices)
    if mode != 'legacy':
        raise ValueError('Unknown relation supervision mode')
    if query_weight is not None:
        raise ValueError('Legacy relation mode does not accept balanced query weights')
    if pred.ndim != 1 or truth.shape != pred.shape or bank.ndim != 2 or bank.shape[1] != len(pred):
        raise ValueError('Relation loss requires aligned response axes')
    if mask.shape != pred.shape or mask.dtype != torch.bool:
        raise ValueError('Relation mask must be one fixed nonempty gene mask')
    if not 0 <= own_index < len(bank):
        raise ValueError('True query missing from training candidate bank')
    x, y, indices_valid = _relation_gene_selection(pred, bank, mask, gene_indices)
    informative = y.norm(dim=1) > norm_floor
    raw_weights = torch.ones(len(bank), device=pred.device, dtype=pred.dtype) if reliability is None else reliability.detach()
    if raw_weights.shape != (len(bank),):
        raise ValueError('Invalid training-only response reliability')
    checks = [
        (mask.any(), 'Relation mask must be one fixed nonempty gene mask'),
        (torch.isclose(bank[own_index], truth, atol=2e-6, rtol=1e-5).all(),
         'Candidate diagonal does not match this query truth'),
        ((torch.isfinite(raw_weights) & (raw_weights >= 0)).all(),
         'Invalid training-only response reliability'),
        (indices_valid, 'Relation gene indices must follow the fixed mask')]
    inputs_valid = (_check_relation_inputs(checks) if check_inputs else
                    torch.stack([check for check, _ in checks]).all())
    weights = raw_weights.clamp(max=1) * informative.to(pred.dtype)
    sim = raw_cosine_matrix(x[None], y, norm_floor)[0]
    true_x, _, _ = _relation_gene_selection(truth, bank, mask, gene_indices)
    target = raw_cosine_matrix(true_x[None].detach(), y, norm_floor)[0].detach()
    err = F.smooth_l1_loss(sim, target, beta=huber_beta, reduction='none')
    diagonal = err[own_index] * weights[own_index]
    other = torch.arange(len(bank), device=pred.device)
    other = torch.cat((other[:own_index], other[own_index + 1:]))
    # Keep absolute query reliability: a weak query should not become full weight
    # after the off-diagonal normalizer cancels a tiny common multiplier.
    off = (err[other] * weights[other]).sum() / weights[other].sum().clamp_min(1e-8)
    off = off * weights[own_index]
    diagnostics = dict(relation_diagonal=diagonal,
        relation_off_diagonal=off, relation_paired_cosine=sim[own_index].detach())
    if not check_inputs:
        diagnostics['relation_inputs_valid'] = inputs_valid.detach()
    return .5 * (diagonal + off), diagnostics


def _source_fit(output):
    info = output['inference']
    precision = info['source_precision']
    return (info['source_residual'].square() * precision).sum() / precision.sum().clamp_min(1e-8)


def objective(output, real, supervision, control_mean, response_scale, gene_scale,
              projections, components, *, own_gene=-1, sampled_genes=None,
              weights=None, relation=None, relation_multiplier=1., control_cells=None,
              real_effective_count=None, covariance_prior_strength=32., check_finite=True):
    phase = output['phase']
    if phase == 'J':
        def lift(value):
            if isinstance(value,torch.Tensor):
                return value.unsqueeze(0)
            if isinstance(value,dict):
                return {key:lift(item) for key,item in value.items()}
            if isinstance(value,(list,tuple)):
                return [lift(item) for item in value]
            return value
        return _objective_j_conditions(lift(output),[real],[supervision],control_mean[None],
            response_scale,gene_scale,projections,components,own_genes=[own_gene],
            sampled_genes=None if sampled_genes is None else [sampled_genes],weights=weights,
            relation=None if relation is None else [relation],relation_multiplier=[relation_multiplier],
            control_cells=None if control_cells is None else [control_cells],
            real_effective_count=None if real_effective_count is None else [real_effective_count],
            covariance_prior_strength=covariance_prior_strength,source_batches=None,
            source_readout_weight=0.,source_readout_detection_weight=.1,
            check_finite=check_finite,return_finite=False)
    if phase not in ('A', 'B', 'C'):
        raise ValueError('Unknown training phase')
    # A supervises analytic marginals and shared action; no cloud generation is needed.
    device = output['analytic_mean'].device
    if phase != 'A':
        y, pre = output['support'], output['pre_reorder_support']
    truth_mean = supervision['mean']
    truth = truth_mean - control_mean
    pred = output['analytic_mean'] - control_mean
    valid = torch.ones_like(pred, dtype=torch.bool)
    if own_gene >= 0:
        valid[own_gene] = False
    # Shape and the CPU-owned target index determine this mask's size.
    # Avoid reducing a device mask or asking nonzero for its dynamic shape.
    if len(pred) - int(own_gene >= 0) < 1:
        raise ValueError('No genes remain after direct-target masking')
    genes = torch.arange(len(pred), device=device)
    if own_gene >= 0:
        genes = torch.cat((genes[:own_gene], genes[own_gene + 1:]))
    ids = sampled_genes if sampled_genes is not None else genes
    terms = {}
    diagnostics = {}
    if phase == 'A':
        error = (pred - truth) / response_scale.clamp_min(1e-6)
        p, t = pred[genes], truth[genes]
        pc, tc = p - p.mean(), t - t.mean()
        terms['mean_mse'] = error[genes].square().mean()
        terms['mean_correlation'] = 1 - (pc * tc).sum() / (pc.norm() * tc.norm()).clamp_min(1e-8)
        top = genes[truth[genes].abs().topk(min(100, len(genes))).indices]
        terms['strong_top100'] = error[top].square().mean()
        # Real population detection is a direct continuous target, not an ST mask.
        terms['detection'] = F.binary_cross_entropy_with_logits(output['detection_logits'][genes],
                                                              supervision['detection'][genes])
        positive = valid & supervision['positive_mask'].bool()
        npos = supervision['count_positive'].to(pred.dtype)
        confidence = npos / (npos + 8.)
        rerr = F.smooth_l1_loss(output['r'] / gene_scale.clamp_min(.05),
                               supervision['positive_mean'] / gene_scale.clamp_min(.05), reduction='none')
        terms['positive_mean'] = (rerr[positive] * confidence[positive]).sum() / positive.sum().clamp_min(1)
        terms['target_rule'] = F.smooth_l1_loss(output['target_statistics'][genes], supervision['statistics'][genes])
        terms['source_fit'] = _source_fit(output)
        if relation is not None and relation_multiplier > 0:
            rl, diagnostics = response_relation_loss(pred, truth,
                **dict(relation, check_inputs=check_finite))
            terms['response_relation'] = float(relation_multiplier) * rl
    elif phase == 'B':
        selected = pre[:, ids]
        terms['marginal_W1'] = (empirical_w1(selected, real[:, ids],
            x_sorted=selected if output.get('marginal_only_training',False) else None) / gene_scale[ids].clamp_min(.05)).mean()
        pred_var = pre[:, ids].var(0, unbiased=False)
        true_var = supervision['variance'][ids]
        terms['variance'] = F.smooth_l1_loss(pred_var / gene_scale[ids].square().clamp_min(.0025),
                                            true_var / gene_scale[ids].square().clamp_min(.0025))
        # A sparse gene can have K=0 or K=1 at training N=64 while the
        # registered N=512 output has several positives. Supervise the SAME
        # conditional quantile shape independently of rounded cloud counts.
        # Normalize both curves: this term must not try to repair frozen r.
        predicted_q = output['positive_quantiles']
        true_q = supervision['positive_quantiles'].detach()
        if predicted_q.shape != (len(pred),33) or true_q.shape != predicted_q.shape:
            raise ValueError('Conditional shape supervision requires aligned Gx33 quantiles')
        count = supervision['count_positive'].to(pred.dtype)
        supported = valid & supervision['positive_mask'].bool() & (count >= 2)
        predicted_shape = predicted_q / output['r'].detach().clamp_min(1e-12)[:,None]
        true_shape = true_q / true_q.mean(1,keepdim=True).clamp_min(1e-12)
        per_gene = (predicted_shape - true_shape).abs().mean(1)
        confidence = count / (count + 8.)
        terms['positive_shape'] = (per_gene[supported]*confidence[supported]).sum()/supported.sum().clamp_min(1)

    else:
        terms['centered_joint_SW1'] = joint_sw1(y-y.mean(0), real-real.mean(0), gene_scale, projections)
        cy = covariance((real / gene_scale) @ components.T).detach()
        cp = covariance((y / gene_scale) @ components.T)
        if control_cells is None or real_effective_count is None:
            raise ValueError('Joint covariance needs real-cell support and target control prior')
        # Control column shape at the SAME generated mean; no target labels enter
        # prediction. Empty control columns get a constant prior, never a fake cell.
        control = control_cells.detach()
        cm = control.mean(0)
        desired = pre.detach().mean(0)
        bound = torch.where(cm[None] > 0, control * (desired / cm.clamp_min(1e-8))[None], desired[None])
        c0 = covariance((bound / gene_scale) @ components.T).detach()
        n = min(int(real_effective_count), int(supervision['n_cells']), len(real))
        if n < 1:
            raise ValueError('Real covariance support must be positive')
        alpha = n / (n + float(covariance_prior_strength))
        target_cov = alpha * cy + (1-alpha) * c0
        terms['relative_PCA_covariance'] = (cp-target_cov).square().mean() / target_cov.square().mean().clamp_min(.01)
        terms['rank_penalty'] = output['rank_penalty']
        diagnostics['covariance_data_weight'] = torch.as_tensor(alpha, device=device)
    effective = dict(DEFAULT_WEIGHTS)
    if weights:
        effective.update(weights)
    loss = sum(effective[k] * v for k, v in terms.items())
    if check_finite and not bool(torch.isfinite(loss)):
        raise RuntimeError('Nonfinite objective')
    # Keep detached device scalars until the runner's logging interval.
    return loss, {k: v.detach() for k, v in {**terms, **diagnostics}.items()}


def _condition_row(value, index, count):
    """Read metadata or one row without transferring device scalars to the host."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _condition_row(item, index, count) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) != count:
            raise ValueError('Per-condition objective field has an incorrect axis')
        return value[index]
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value
        if value.shape[0] != count:
            raise ValueError('Per-condition objective tensor has an incorrect axis')
        return value[index]
    return value


def _condition_stack(value, count):
    if isinstance(value, torch.Tensor):
        if value.ndim < 1 or value.shape[0] != count:
            raise ValueError('Per-condition objective tensor has an incorrect axis')
        return value
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError('Per-condition objective field has an incorrect axis')
    if any(not isinstance(item, torch.Tensor) for item in value):
        raise ValueError('Per-condition objective field requires tensors')
    if any(item.shape != value[0].shape for item in value):
        return None
    return torch.stack(list(value))


def _condition_field(records, name, count):
    value = records[name] if isinstance(records, dict) else [r[name] for r in records]
    result = _condition_stack(value, count)
    if result is None:
        raise ValueError('Per-condition supervision field has incompatible axes: ' + name)
    return result


def _condition_own(own_genes, count, genes):
    # These are CPU-owned schema indices, never learned device values.
    if own_genes is None:
        own = [-1] * count
    elif isinstance(own_genes, torch.Tensor):
        if own_genes.shape != (count,):
            raise ValueError('Per-condition target index has an incorrect axis')
        own = own_genes.detach().cpu().tolist()
    elif isinstance(own_genes, (list, tuple)) and len(own_genes) == count:
        own = [int(i) for i in own_genes]
    else:
        raise ValueError('Per-condition target index has an incorrect axis')
    if any(i >= genes for i in own):
        raise IndexError('Direct-target index is outside the gene axis')
    if any(genes - int(i >= 0) < 1 for i in own):
        raise ValueError('No genes remain after direct-target masking')
    return own


def _condition_finish(terms, diagnostics, weights, *, extra=None, input_flags=(),
                      check_finite=True, return_finite=False):
    """Keep Q equally weighted and audit even zero-weight loss components."""
    effective = dict(DEFAULT_WEIGHTS)
    if weights:
        effective.update(weights)
    losses = sum(effective[name] * value for name, value in terms.items())
    finite = torch.isfinite(torch.stack([losses.detach(),
        *[value.detach() for value in terms.values()],
        *[value.detach() for value in diagnostics.values()]])).all()
    if extra is not None:
        losses = losses + extra
    loss = losses.mean()
    finite = finite & torch.isfinite(losses.detach()).all() & torch.isfinite(loss.detach())
    if input_flags:
        finite = finite & torch.stack([flag.all() for flag in input_flags]).all()
    if check_finite and not bool(finite):
        raise RuntimeError('Nonfinite objective or invalid relation inputs')
    logs = {name: value.detach().mean() for name, value in {**terms, **diagnostics}.items()}
    return (loss, logs, finite) if return_finite else (loss, logs)


def _relation_conditions(pred, truth, relations, multipliers):
    """Vectorize homogeneous RankNet panels; legacy/custom modes keep their API.

    Each condition retains its own true diagonal, exclusions and reliability.
    Grouping uses shapes and CPU configuration only; it does not select labels.
    Cached compact gene indices avoid dynamic CUDA boolean selection.
    """
    q, g = pred.shape
    terms, diagnostics, flags = pred.new_zeros(q), {}, []
    groups = {}
    for i in range(q):
        rel = _condition_row(relations, i, q)
        multiplier = float(_condition_row(multipliers, i, q))
        if rel is None or multiplier <= 0:
            continue
        if rel.get('mode', 'legacy') != 'panel_ranknet':
            value, logs = response_relation_loss(pred[i], truth[i], **dict(rel, check_inputs=False))
            flags.append(logs.pop('relation_inputs_valid'))
            ix = torch.tensor([i], device=pred.device)
            terms = terms.index_copy(0, ix, (value * multiplier)[None])
            for key, value in logs.items():
                diagnostics[key] = diagnostics.get(key, pred.new_zeros(q)).index_copy(0, ix, value[None])
            continue
        bank, mask = rel['bank'], rel['mask']
        own = int(rel['own_index'])
        floor, beta = float(rel.get('norm_floor', 1e-4)), float(rel.get('huber_beta', .1))
        temperature = float(rel.get('temperature', .05))
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('RankNet temperature must be finite and positive')
        if floor <= 0 or beta <= 0:
            raise ValueError('Positive norm floor and Huber beta required')
        if bank.ndim != 2 or bank.shape[1] != g:
            raise ValueError('Relation loss requires aligned response axes')
        if mask.shape != (g,) or mask.dtype != torch.bool:
            raise ValueError('Relation mask must be one fixed nonempty gene mask')
        if not 0 <= own < len(bank):
            raise ValueError('True query missing from training candidate bank')
        indices = rel.get('gene_indices')
        if indices is not None:
            indices = torch.as_tensor(indices, device=pred.device)
            if indices.ndim != 1 or indices.dtype != torch.long or len(indices) > g:
                raise ValueError('Relation gene indices must follow the fixed mask')
        key = (len(bank), None if indices is None else len(indices), floor, beta, temperature)
        groups.setdefault(key, []).append((i, rel, indices, multiplier))
    for (p, width, floor, beta, temperature), records in groups.items():
        ix = torch.tensor([row[0] for row in records], device=pred.device)
        b = len(records)
        banks = torch.stack([row[1]['bank'] for row in records]).detach()
        masks = torch.stack([row[1]['mask'] for row in records])
        own = torch.tensor([row[1]['own_index'] for row in records], device=pred.device)
        x = pred.index_select(0, ix)
        if width is None:
            # Masked zero coordinates contribute neither norms nor dot products.
            x = torch.where(masks, x, 0.)
            y = torch.where(masks[:, None, :], banks, 0.)
            ids_valid = masks.new_ones(b)
        else:
            ids = torch.stack([row[2] for row in records])
            safe = ids.clamp(0, max(0, g-1))
            ids_valid = ((ids >= 0) & (ids < g)).all(1)
            ids_valid = ids_valid & (ids[:, 1:] > ids[:, :-1]).all(1)
            ids_valid = ids_valid & (masks.sum(1) == width) & masks.gather(1, safe).all(1)
            x = x.gather(1, safe)
            y = banks.gather(2, safe[:, None, :].expand(b, p, width))
        raw = []
        for _, rel, _, _ in records:
            value = rel.get('reliability')
            value = (pred.new_ones(p) if value is None else
                     torch.as_tensor(value, device=pred.device, dtype=pred.dtype).detach())
            if value.shape != (p,):
                raise ValueError('Invalid training-only response reliability')
            raw.append(value)
        raw = torch.stack(raw)
        norms = y.norm(dim=2)
        informative = norms > floor
        weights = raw.clamp(max=1) * informative.to(pred.dtype)
        rows = torch.arange(b, device=pred.device)
        query_weights = []
        for j, (_, rel, _, _) in enumerate(records):
            value = rel.get('query_weight')
            value = (weights[j, rel['own_index']] if value is None else
                     torch.as_tensor(value, device=pred.device, dtype=pred.dtype).detach())
            if value.ndim != 0:
                raise ValueError('Query weight must be a finite scalar in [0,3]')
            query_weights.append(value)
        qw = torch.stack(query_weights)
        flags.extend([masks.any(1), ids_valid,
            torch.isclose(banks[rows, own], truth.index_select(0, ix), atol=2e-6, rtol=1e-5).all(1),
            (torch.isfinite(raw) & (raw >= 0)).all(1),
            torch.isfinite(qw) & (qw >= 0) & (qw <= 3)])
        qw = qw * informative[rows, own].to(pred.dtype)
        unit_y = y / norms[:, :, None].clamp_min(floor)
        unit_x = x / x.norm(dim=1, keepdim=True).clamp_min(floor)
        sim = torch.bmm(unit_y, unit_x[:, :, None])[:, :, 0].clamp(-1., 1.)
        paired = sim[rows, own]
        diagonal = qw * F.smooth_l1_loss(paired, torch.ones_like(paired), beta=beta, reduction='none')
        # Compact exclusion keeps the scalar reduction order and P-1 denominator.
        offsets = torch.arange(p-1, device=pred.device)[None, :].expand(b, p-1)
        others = offsets + (offsets >= own[:, None]).long()
        off_weights = weights.gather(1, others)
        weight_sum = off_weights.sum(1)
        pairs = temperature * F.softplus((sim.gather(1, others)-paired[:, None])/temperature)
        off = qw * (pairs * off_weights).sum(1) / weight_sum.clamp_min(1e-8)
        factors = pred.new_tensor([row[3] for row in records])
        terms = terms.index_copy(0, ix, .5 * (diagonal + off) * factors)
        for name, value in dict(relation_diagonal=diagonal, relation_off_diagonal=off,
                relation_paired_cosine=paired.detach(), relation_query_weight=qw.detach(),
                relation_candidate_weight_sum=weight_sum.detach()).items():
            diagnostics[name] = diagnostics.get(name, pred.new_zeros(q)).index_copy(0, ix, value)
    return terms, diagnostics, flags


def _objective_a_conditions(output, supervision, control_mean, response_scale, gene_scale, *,
                            own_genes, weights, relation, relation_multiplier, source_batches,
                            source_readout_weight, source_readout_detection_weight,
                            check_finite, return_finite):
    mean = output['analytic_mean']
    if mean.ndim != 2 or min(mean.shape) < 1:
        raise ValueError('A analytic means require independent [Q,G] axes')
    q, g = mean.shape
    own = _condition_own(own_genes, q, g)
    control = _condition_stack(control_mean, q)
    if control is None or control.shape != mean.shape:
        raise ValueError('A target control means require [Q,G]')
    sup = lambda name: _condition_field(supervision, name, q)
    pred, truth = mean-control, sup('mean')-control
    error = (pred-truth)/response_scale.clamp_min(1e-6)
    device = mean.device
    valid = torch.arange(g, device=device)[None, :] != torch.tensor(own, device=device)[:, None]
    terms = {}
    # Two static-size groups retain exactly the scalar compact gene order. In
    # particular, Top100 tie-breaking never sees padding or masked own genes.
    for masked in (False, True):
        members = [i for i in range(q) if (own[i] >= 0) == masked]
        if not members:
            continue
        ix = torch.tensor(members, device=device)
        width = g-int(masked)
        genes = torch.arange(width, device=device)[None, :].expand(len(members), width)
        if masked:
            genes = genes + (genes >= torch.tensor([own[i] for i in members], device=device)[:, None]).long()
        def selected(value):
            value = value.index_select(0, ix)
            indices = genes.reshape(*genes.shape, *([1]*(value.ndim-2)))
            return value.gather(1, indices.expand(*genes.shape, *value.shape[2:]))
        p, t, err = selected(pred), selected(truth), selected(error)
        pc, tc = p-p.mean(1, keepdim=True), t-t.mean(1, keepdim=True)
        top = t.abs().topk(min(100, width), dim=1).indices
        values = dict(mean_mse=err.square().mean(1),
            mean_correlation=1-(pc*tc).sum(1)/(pc.norm(dim=1)*tc.norm(dim=1)).clamp_min(1e-8),
            strong_top100=err.gather(1, top).square().mean(1),
            detection=F.binary_cross_entropy_with_logits(selected(output['detection_logits']),
                selected(sup('detection')), reduction='none').mean(1),
            target_rule=F.smooth_l1_loss(selected(output['target_statistics']),
                selected(sup('statistics')), reduction='none').flatten(1).mean(1))
        for name, value in values.items():
            terms[name] = terms.get(name, mean.new_zeros(q)).index_copy(0, ix, value)
    positive = valid & sup('positive_mask').bool()
    count = sup('count_positive').to(mean.dtype)
    rerr = F.smooth_l1_loss(output['r']/gene_scale.clamp_min(.05),
        sup('positive_mean')/gene_scale.clamp_min(.05), reduction='none')
    terms['positive_mean'] = torch.where(positive, rerr*(count/(count+8.)), 0.).sum(1)/positive.sum(1).clamp_min(1)
    precision = output['inference']['source_precision']
    residual = output['inference']['source_residual']
    if precision.shape != residual.shape or precision.ndim != 4 or precision.shape[0] != q:
        raise ValueError('Source fit requires independent [Q,S,G,M] axes')
    terms['source_fit'] = (residual.square()*precision).flatten(1).sum(1)/precision.flatten(1).sum(1).clamp_min(1e-8)
    diagnostics, flags = {}, []
    if relation is not None:
        rel, diagnostics, flags = _relation_conditions(pred, truth, relation, relation_multiplier)
        if diagnostics:
            terms['response_relation'] = rel
    extra = None
    if source_readout_weight:
        if source_batches is None:
            raise ValueError('Source readout supervision requires authorized source batches')
        readouts = output['source_readouts']
        observations = _condition_field(source_batches, 'observations', q)
        if not readouts or observations.ndim != 4 or observations.shape[1] != len(readouts):
            raise ValueError('Source readouts differ from authorized source axis')
        source_mean = torch.stack([record['analytic_mean'] for record in readouts], dim=1)
        source_logits = torch.stack([record['detection_logits'] for record in readouts], dim=1)
        true_mean = _condition_field(source_batches, 'source_means', q)
        true_detection = _condition_field(source_batches, 'source_detections', q)
        if source_mean.shape != true_mean.shape or source_logits.shape != true_detection.shape:
            raise ValueError('Source readout supervision differs from source/gene axes')
        selected = valid[:, None, :]
        denominator = valid.sum(1)*len(readouts)
        mse = torch.where(selected, ((source_mean-true_mean)/response_scale.clamp_min(1e-6)).square(), 0.).sum((1,2))/denominator
        det = torch.where(selected, F.binary_cross_entropy_with_logits(source_logits, true_detection, reduction='none'), 0.).sum((1,2))/denominator
        total = mse+float(source_readout_detection_weight)*det
        diagnostics.update(source_readout_mse=mse, source_readout_detection=det, source_readout_total=total)
        extra = float(source_readout_weight)*total
    return _condition_finish(terms, diagnostics, weights, extra=extra, input_flags=flags,
        check_finite=check_finite, return_finite=return_finite)


def _objective_c_conditions(output, real, supervision, gene_scale, projections, components, *,
                            own_genes, weights, control_cells, real_effective_count,
                            covariance_prior_strength, check_finite, return_finite):
    y, pre = output['support'], output['pre_reorder_support']
    if y.ndim != 3 or min(y.shape) < 1 or pre.shape != y.shape:
        raise ValueError('C clouds require independent [Q,N,G] axes')
    q, _, g = y.shape
    _condition_own(own_genes, q, g)
    real = _condition_stack(real, q)
    if control_cells is None or real_effective_count is None:
        raise ValueError('Joint covariance needs real-cell support and target control prior')
    control = _condition_stack(control_cells, q)
    # Ragged real/control clouds retain the original exact empirical W1 path.
    if real is None or control is None:
        return None
    if real.ndim != 3 or real.shape[2] != g or control.ndim != 3 or control.shape[2] != g:
        raise ValueError('C real/control clouds differ from condition/gene axes')
    if gene_scale.shape != (g,) or projections.shape[0] != g:
        raise ValueError('Joint projections differ from gene axes')
    n = [min(int(_condition_row(real_effective_count, i, q)),
             int(_condition_row(supervision, i, q)['n_cells']), real.shape[1]) for i in range(q)]
    if min(n) < 1:
        raise ValueError('Real covariance support must be positive')
    def cov(value):
        value = value-value.mean(1, keepdim=True)
        return value.transpose(1,2) @ value/value.shape[1]
    projected_pred = ((y-y.mean(1, keepdim=True))/gene_scale) @ projections
    projected_real = ((real-real.mean(1, keepdim=True))/gene_scale) @ projections
    sw1 = empirical_w1_conditions(projected_pred, projected_real).mean(1)
    cp = cov((y/gene_scale) @ components.T)
    cy = cov((real/gene_scale) @ components.T).detach()
    control = control.detach()
    cm, desired = control.mean(1), pre.detach().mean(1)
    bound = torch.where(cm[:, None, :] > 0,
        control*(desired/cm.clamp_min(1e-8))[:, None, :], desired[:, None, :])
    c0 = cov((bound/gene_scale) @ components.T).detach()
    alpha_values = [count/(count+float(covariance_prior_strength)) for count in n]
    alpha = y.new_tensor(alpha_values)
    prior_alpha = y.new_tensor([1-value for value in alpha_values])
    target = alpha[:, None, None]*cy+prior_alpha[:, None, None]*c0
    terms = dict(centered_joint_SW1=sw1,
        relative_PCA_covariance=(cp-target).square().mean((1,2))/target.square().mean((1,2)).clamp_min(.01),
        rank_penalty=output['rank_penalty'])
    diagnostics = dict(covariance_data_weight=torch.tensor(alpha_values, device=y.device))
    return _condition_finish(terms, diagnostics, weights, input_flags=[(gene_scale > 0).all()],
        check_finite=check_finite, return_finite=return_finite)


def _objective_b_conditions(output, real, supervision, gene_scale, *, own_genes,
                            sampled_genes, weights, check_finite, return_finite):
    """Vectorized B loss; padding affects neither each gene mean nor Q weights.

    Unequal real-cell counts retain the original scalar implementation. This
    path changes no sampling: callers supply the same authorized real cells
    and gene indices. All positive quantiles are still supervised across G.
    """
    pre = output['pre_reorder_support']
    if pre.ndim != 3:
        raise ValueError('B clouds require independent [Q,N,G] axes')
    q, _, g = pre.shape
    if q < 1 or g < 1 or output['analytic_mean'].shape != (q,g):
        raise ValueError('B analytic means differ from condition/gene axes')

    def stack_rows(value):
        if isinstance(value,torch.Tensor):
            return value
        if not isinstance(value,(list,tuple)) or len(value) != q:
            raise ValueError('Per-condition objective field has an incorrect axis')
        if any(not isinstance(v,torch.Tensor) for v in value):
            raise ValueError('B supervision must contain tensors')
        if any(v.shape != value[0].shape for v in value):
            return None
        return torch.stack(list(value))

    real = stack_rows(real)
    if real is None:
        return None
    if real.ndim != 3 or real.shape[0] != q or real.shape[2] != g:
        raise ValueError('B real cells differ from condition/gene axes')
    if own_genes is None:
        own = [-1]*q
    elif isinstance(own_genes,(list,tuple)):
        if len(own_genes) != q:
            raise ValueError('Per-condition target index has an incorrect axis')
        own = [int(v) for v in own_genes]
    elif isinstance(own_genes,torch.Tensor) and own_genes.shape == (q,):
        own = own_genes.detach().cpu().tolist()
    else:
        raise ValueError('Per-condition target index has an incorrect axis')
    if any(i >= g for i in own):
        raise IndexError('Direct-target index is outside the gene axis')
    if any(g-int(i >= 0) < 1 for i in own):
        raise ValueError('No genes remain after direct-target masking')
    device = pre.device
    all_genes = torch.arange(g,device=device)
    valid = all_genes[None,:] != torch.tensor(own,device=device)[:,None]
    if sampled_genes is None:
        # Full-G gathers with a static mask avoid dynamic CUDA boolean indexing.
        ids = all_genes[None,:].expand(q,g)
        selected = valid
    else:
        if isinstance(sampled_genes,torch.Tensor):
            if sampled_genes.ndim != 2 or sampled_genes.shape[0] != q:
                raise ValueError('Sampled genes require a condition axis')
            rows = list(sampled_genes.unbind(0))
        elif isinstance(sampled_genes,(list,tuple)) and len(sampled_genes) == q:
            rows = list(sampled_genes)
        else:
            raise ValueError('Sampled genes require a condition axis')
        if any(not isinstance(row,torch.Tensor) or row.ndim != 1 for row in rows):
            raise ValueError('Each sampled gene panel must be one-dimensional')
        lengths = [len(row) for row in rows]
        if min(lengths) < 1:
            # Preserve the original nonfinite empty-panel refusal.
            raise RuntimeError('Nonfinite objective: empty sampled gene panel')
        width = max(lengths)
        ids = torch.stack([F.pad(row.to(device=device),(0,width-len(row)),value=0) for row in rows])
        selected = torch.arange(width,device=device)[None,:] < torch.tensor(lengths,device=device)[:,None]
    denom = selected.sum(1).clamp_min(1)
    px = pre.gather(2,ids[:,None,:].expand(q,pre.shape[1],-1))
    tx = real.gather(2,ids[:,None,:].expand(q,real.shape[1],-1))
    scale = gene_scale[ids]
    w1 = empirical_w1_conditions(px,tx,x_is_sorted=bool(output.get('marginal_only_training',False)))
    terms = {'marginal_W1':torch.where(selected,w1/scale.clamp_min(.05),0.).sum(1)/denom}

    def sup(name):
        if isinstance(supervision,dict):
            value = stack_rows(supervision[name])
        elif isinstance(supervision,(list,tuple)) and len(supervision) == q:
            value = stack_rows([record[name] for record in supervision])
        else:
            raise ValueError('Per-condition supervision has an incorrect axis')
        if value is None or value.shape[0] != q:
            raise ValueError('B supervision has incompatible condition axes')
        return value

    variance = sup('variance')
    if variance.shape != (q,g):
        raise ValueError('B variance supervision requires [Q,G]')
    var_loss = F.smooth_l1_loss(px.var(1,unbiased=False)/scale.square().clamp_min(.0025),
        variance.gather(1,ids)/scale.square().clamp_min(.0025),reduction='none')
    terms['variance'] = torch.where(selected,var_loss,0.).sum(1)/denom
    predicted_q = output['positive_quantiles']
    true_q = sup('positive_quantiles').detach()
    if predicted_q.shape != (q,g,33) or true_q.shape != predicted_q.shape:
        raise ValueError('Conditional shape supervision requires aligned QxGx33 quantiles')
    count = sup('count_positive').to(pre.dtype)
    positive = sup('positive_mask').bool()
    if count.shape != (q,g) or positive.shape != (q,g) or output['r'].shape != (q,g):
        raise ValueError('B positive supervision requires [Q,G]')
    supported = valid & positive & (count >= 2)
    predicted_shape = predicted_q/output['r'].detach().clamp_min(1e-12)[...,None]
    true_shape = true_q/true_q.mean(2,keepdim=True).clamp_min(1e-12)
    per_gene = (predicted_shape-true_shape).abs().mean(2)
    weighted = per_gene*(count/(count+8.))
    terms['positive_shape'] = torch.where(supported,weighted,0.).sum(1)/supported.sum(1).clamp_min(1)
    effective = dict(DEFAULT_WEIGHTS)
    if weights:
        effective.update(weights)
    losses = sum(effective[name]*value for name,value in terms.items())
    loss = losses.mean()
    # A zero loss weight must not conceal a nonfinite component/condition.
    finite = torch.isfinite(torch.stack([losses.detach(),*[v.detach() for v in terms.values()]])).all()
    finite = finite & torch.isfinite(loss.detach())
    if check_finite and not bool(finite):
        raise RuntimeError('Nonfinite objective')
    logs = {name:value.detach().mean() for name,value in terms.items()}
    return (loss,logs,finite) if return_finite else (loss,logs)


def objective_conditions(output, real, supervision, control_mean, response_scale, gene_scale,
                         projections, components, *, own_genes=None, sampled_genes=None,
                         weights=None, relation=None, relation_multiplier=1., control_cells=None,
                         real_effective_count=None, covariance_prior_strength=32.,
                         source_batches=None, source_readout_weight=0.,
                         source_readout_detection_weight=.1, check_finite=True,
                         return_finite=False):
    """Equal-condition mean of the original objectives, with independent cell clouds.

    The network and connection solves run with a condition axis. This adapter
    preserves per-query target masks, 64-candidate ranking panels, covariance
    shrinkage, and variable real-cell counts. It never merges cells or genes
    belonging to different perturbations. By default, one device flag checks
    every condition and the aggregate. A runner can request that detached flag
    with return_finite=True and defer its host check with check_finite=False.
    """
    from .shared_path_model import source_readout_loss
    if isinstance(output, dict) and output.get('phase') == 'J':
        return _objective_j_conditions(output, real, supervision, control_mean, response_scale, gene_scale,
            projections, components, own_genes=own_genes, sampled_genes=sampled_genes, weights=weights,
            relation=relation, relation_multiplier=relation_multiplier, control_cells=control_cells,
            real_effective_count=real_effective_count, covariance_prior_strength=covariance_prior_strength,
            source_batches=source_batches, source_readout_weight=source_readout_weight,
            source_readout_detection_weight=source_readout_detection_weight,
            check_finite=check_finite, return_finite=return_finite)
    if isinstance(output, dict) and output.get('phase') == 'A':
        return _objective_a_conditions(output, supervision, control_mean, response_scale, gene_scale,
            own_genes=own_genes, weights=weights, relation=relation,
            relation_multiplier=relation_multiplier, source_batches=source_batches,
            source_readout_weight=source_readout_weight,
            source_readout_detection_weight=source_readout_detection_weight,
            check_finite=check_finite, return_finite=return_finite)
    if isinstance(output, dict) and output.get('phase') == 'C':
        result = _objective_c_conditions(output, real, supervision, gene_scale, projections, components,
            own_genes=own_genes, weights=weights, control_cells=control_cells,
            real_effective_count=real_effective_count,
            covariance_prior_strength=covariance_prior_strength,
            check_finite=check_finite, return_finite=return_finite)
        if result is not None:
            return result
    if isinstance(output,dict) and output.get('phase') == 'B':
        result = _objective_b_conditions(output,real,supervision,gene_scale,
            own_genes=own_genes,sampled_genes=sampled_genes,weights=weights,
            check_finite=check_finite,return_finite=return_finite)
        if result is not None:
            return result
    if isinstance(output, (list, tuple)):
        outputs = list(output)
    else:
        from .batched_model import split_condition_outputs
        outputs = split_condition_outputs(output)
    q = len(outputs)
    if q < 1:
        raise ValueError('An empty condition batch has no objective')

    def row(value, index):
        if value is None:
            return None
        if isinstance(value, dict):
            return {key: row(item, index) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            if len(value) != q:
                raise ValueError('Per-condition objective field has an incorrect axis')
            return value[index]
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value
            if value.shape[0] != q:
                raise ValueError('Per-condition objective tensor has an incorrect axis')
            return value[index]
        return value

    losses, logs, finite_values, input_flags = [], [], [], []
    for index, prediction in enumerate(outputs):
        own = -1 if own_genes is None else int(row(own_genes, index))
        loss, terms = objective(prediction, row(real, index), row(supervision, index),
            row(control_mean, index), response_scale, gene_scale, projections, components,
            own_gene=own, sampled_genes=row(sampled_genes, index), weights=weights,
            relation=row(relation, index), relation_multiplier=float(row(relation_multiplier, index)),
            control_cells=row(control_cells, index), real_effective_count=row(real_effective_count, index),
            covariance_prior_strength=covariance_prior_strength, check_finite=False)
        inputs_valid = terms.pop('relation_inputs_valid', None)
        if inputs_valid is not None:
            input_flags.append(inputs_valid)
        # Retain each unweighted term and each base loss in the validity flag:
        # a disabled weight must never conceal a nonfinite evaluated term.
        finite_values.extend([loss.detach(), *terms.values()])
        if prediction['phase'] == 'A' and source_readout_weight:
            if source_batches is None:
                raise ValueError('Source readout supervision requires authorized source batches')
            extra, extra_terms = source_readout_loss(prediction, row(source_batches, index),
                response_scale, own, detection_weight=source_readout_detection_weight)
            finite_values.extend([extra.detach(), *extra_terms.values()])
            loss = loss + float(source_readout_weight) * extra
            terms.update(extra_terms, source_readout_total=extra.detach())
        finite_values.append(loss.detach())
        losses.append(loss)
        logs.append(terms)
    loss = torch.stack(losses).mean()
    finite_values.append(loss.detach())
    finite = torch.isfinite(torch.stack(finite_values)).all()
    if input_flags:
        finite = finite & torch.stack(input_flags).all()
    if check_finite and not bool(finite):
        raise RuntimeError('Nonfinite objective')
    # Terms absent during a per-query warmup contribute zero to the batch mean.
    keys = set().union(*(record.keys() for record in logs))
    terms = {key: torch.stack([record.get(key, loss.detach().new_zeros(()))
                              for record in logs]).mean() for key in keys}
    return (loss, terms, finite) if return_finite else (loss, terms)


JOINT_WEIGHTS = dict(joint_delta=1., joint_discrimination=.25,
    joint_marginal=1., joint_structure=.25, joint_shared=.05)


def weighted_hurdle_w1(p, positive_quantiles, real):
    """Exact W1 of a finite hurdle law and an empirical law, independently per gene.

    p: [...,G], positive_quantiles: [...,G,K], real: [...,N,G].
    Atom masses are 1-p at zero and p/K at the K positive points. Sorting is
    piecewise differentiable in positions; CDF integration differentiates p
    through masses. No N-by-N transport matrix or rounded detection is used.
    """
    if positive_quantiles.shape[:-1] != p.shape or real.shape[:-2] != p.shape[:-1] or real.shape[-1] != p.shape[-1]:
        raise ValueError('Weighted hurdle W1 requires aligned condition/gene axes')
    k, n = positive_quantiles.shape[-1], real.shape[-2]
    if k < 1 or n < 1:
        raise ValueError('Weighted hurdle W1 needs nonempty supports')
    zero = torch.zeros_like(p[..., None])
    atoms = torch.cat((zero, positive_quantiles, real.transpose(-1, -2)), -1)
    signed_mass = torch.cat(((1-p)[..., None], p[..., None].expand_as(positive_quantiles)/k,
        -torch.ones_like(real.transpose(-1,-2))/n), -1)
    ordered, permutation = atoms.sort(dim=-1, stable=True)
    cdf_difference = signed_mass.gather(-1, permutation).cumsum(-1)
    return (cdf_difference[..., :-1].abs() * (ordered[..., 1:]-ordered[..., :-1])).sum(-1)


def _joint_effective(weights):
    result = dict(JOINT_WEIGHTS)
    if weights:
        result.update({key: float(value) for key, value in weights.items() if key in result})
    if any(not math.isfinite(v) or v < 0 for v in result.values()):
        raise ValueError('Joint group weights must be finite and nonnegative')
    return result


def joint_response_objective_conditions(output, supervision, control_mean, response_scale,
                                       gene_scale=None, *, own_genes=None, relation=None,
                                       relation_multiplier=1., weights=None,
                                       check_finite=True, return_finite=False):
    """Output-only response objective; no encoder internals needed at validation."""
    w = _joint_effective(weights)
    mean = output['mean512_st']
    q,g = mean.shape
    own = _condition_own(own_genes,q,g)
    control = _condition_stack(control_mean,q)
    truth_mean = _condition_field(supervision,'mean',q)
    pred,truth = mean-control,truth_mean-control
    valid = torch.arange(g,device=mean.device)[None] != torch.tensor(own,device=mean.device)[:,None]
    denominator = valid.sum(1)
    error = ((pred-truth)/response_scale.clamp_min(1e-6)).square()
    mse = torch.where(valid,error,0.).sum(1)/denominator
    pc = pred-torch.where(valid,pred,0.).sum(1,keepdim=True)/denominator[:,None]
    tc = truth-torch.where(valid,truth,0.).sum(1,keepdim=True)/denominator[:,None]
    pc,tc = torch.where(valid,pc,0.),torch.where(valid,tc,0.)
    corr = 1-(pc*tc).sum(1)/(pc.norm(dim=1)*tc.norm(dim=1)).clamp_min(1e-8)
    top_values=[]
    # Metadata-owned compact axes preserve own-target exclusion and ties.
    for i in range(q):
        ids=torch.arange(g,device=mean.device)
        if own[i]>=0:
            ids=torch.cat((ids[:own[i]],ids[own[i]+1:]))
        top=ids[truth[i,ids].abs().topk(min(100,len(ids))).indices]
        top_values.append(error[i,top].mean())
    strong=torch.stack(top_values)
    delta=mse+.15*corr+.15*strong
    disc,diagnostics,flags=(mean.new_zeros(q),{},[])
    if relation is not None:
        disc,diagnostics,flags=_relation_conditions(pred,truth,relation,relation_multiplier)
    loss=(w['joint_delta']*delta+w['joint_discrimination']*disc).mean()
    logs={name:value.detach().mean() for name,value in dict(mean_mse=mse,
        mean_correlation=corr,strong_top100=strong,joint_delta=delta,
        response_relation=disc,joint_discrimination=disc,**diagnostics).items()}
    finite=torch.isfinite(loss.detach()) & torch.isfinite(torch.stack(list(logs.values()))).all()
    if flags:
        finite=finite & torch.stack([flag.all() for flag in flags]).all()
    if check_finite and not bool(finite):
        raise RuntimeError('Nonfinite joint response objective')
    return (loss,logs,finite) if return_finite else (loss,logs)


def _objective_j_conditions(output, real, supervision, control_mean, response_scale, gene_scale,
                            projections, components, *, own_genes, sampled_genes, weights,
                            relation, relation_multiplier, control_cells, real_effective_count,
                            covariance_prior_strength, source_batches, source_readout_weight,
                            source_readout_detection_weight, check_finite, return_finite):
    """One end-to-end objective: mean, identity, marginals, copula and shared rule."""
    w = _joint_effective(weights)
    mapped = dict(mean_mse=w['joint_delta'], mean_correlation=.15*w['joint_delta'],
        strong_top100=.15*w['joint_delta'], response_relation=w['joint_discrimination'],
        detection=0., positive_mean=0.,
        target_rule=w['joint_shared'], source_fit=.2*w['joint_shared'])
    adapted = dict(output, analytic_mean=output['mean512_st'])
    base, logs, base_finite = _objective_a_conditions(adapted, supervision, control_mean,
        response_scale, gene_scale, own_genes=own_genes, weights=mapped,
        relation=relation, relation_multiplier=relation_multiplier, source_batches=source_batches,
        source_readout_weight=source_readout_weight*w['joint_shared'],
        source_readout_detection_weight=source_readout_detection_weight,
        check_finite=False, return_finite=True)
    distribution, distribution_logs, distribution_finite = joint_distribution_objective_conditions(
        output,real,supervision,gene_scale,projections,components,own_genes=own_genes,
        sampled_genes=sampled_genes,weights=weights,control_cells=control_cells,
        real_effective_count=real_effective_count,covariance_prior_strength=covariance_prior_strength,
        check_finite=False,return_finite=True)
    loss=base+distribution
    logs.update(distribution_logs)
    logs['joint_delta'] = logs['mean_mse']+.15*logs['mean_correlation']+.15*logs['strong_top100']
    logs['joint_discrimination'] = logs.get('response_relation',loss.detach().new_zeros(()))
    logs['joint_marginal'] = logs['marginal_W1']+.1*(logs['positive_shape']+logs['detection']+logs['positive_mean'])
    logs['joint_shared'] = logs['target_rule']+.2*logs['source_fit']+source_readout_weight*logs.get('source_readout_total',loss.detach().new_zeros(()))
    finite = base_finite & distribution_finite & torch.isfinite(loss.detach()) & torch.isfinite(torch.stack(list(logs.values()))).all()
    if check_finite and not bool(finite):
        raise RuntimeError('Nonfinite joint objective or invalid relation inputs')
    return (loss,logs,finite) if return_finite else (loss,logs)


def joint_distribution_objective_conditions(output, real, supervision, gene_scale, projections,
                                           components, *, own_genes=None, sampled_genes=None,
                                           weights=None, control_cells=None, real_effective_count=None,
                                           covariance_prior_strength=32.,check_finite=True,return_finite=False):
    """Marginal and structure objective from final outputs and lawful evaluation labels."""
    w=_joint_effective(weights)
    y = output['support']
    if y.ndim != 3 or y.shape[1] < 2:
        raise ValueError('J training requires complete independent condition clouds')
    q,n,g = y.shape
    real = _condition_stack(real,q)
    control = _condition_stack(control_cells,q)
    if real is None or control is None or real.ndim != 3 or control.ndim != 3:
        raise ValueError('Joint batches require uniform real/control cell counts')
    if real.shape[0] != q or real.shape[2] != g or control.shape[0] != q or control.shape[2] != g:
        raise ValueError('Joint labels/control differ from output axes')
    own = _condition_own(own_genes,q,g)
    valid = torch.arange(g,device=y.device)[None] != torch.tensor(own,device=y.device)[:,None]
    if sampled_genes is None:
        ids = torch.arange(g,device=y.device)[None].expand(q,-1)
        selected = valid
    else:
        ids = _condition_stack(sampled_genes,q)
        if ids is None or ids.ndim != 2 or ids.shape[1] < 1:
            raise ValueError('Joint sampled genes need uniform nonempty [Q,K] panels')
        selected = valid.gather(1,ids)
    p = output['p'].gather(1,ids)
    quantiles = output['positive_quantiles'].gather(1,ids[...,None].expand(-1,-1,33))
    real_selected = real.gather(2,ids[:,None].expand(-1,real.shape[1],-1))
    marginal_gene = weighted_hurdle_w1(p,quantiles,real_selected)/gene_scale[ids].clamp_min(.05)
    marginal = torch.where(selected,marginal_gene,0.).sum(1)/selected.sum(1).clamp_min(1)
    sup = lambda name: _condition_field(supervision,name,q)
    count = sup('count_positive').to(y.dtype)
    supported = valid & sup('positive_mask').bool() & (count >= 2)
    predicted_shape = output['unit_mean_shape']
    true_quantiles = sup('positive_quantiles').detach()
    true_shape = true_quantiles/true_quantiles.mean(-1,keepdim=True).clamp_min(1e-12)
    shape_error = (predicted_shape-true_shape).abs().mean(-1)*(count/(count+8.))
    shape_loss = torch.where(supported,shape_error,0.).sum(1)/supported.sum(1).clamp_min(1)
    projected_pred = ((y-y.mean(1,keepdim=True))/gene_scale) @ projections
    projected_real = ((real-real.mean(1,keepdim=True))/gene_scale) @ projections
    sw = empirical_w1_conditions(projected_pred,projected_real).mean(1)
    def cov(x):
        centered=x-x.mean(1,keepdim=True)
        return centered.transpose(1,2)@centered/x.shape[1]
    cp = cov((y/gene_scale)@components.T)
    cy = cov((real/gene_scale)@components.T).detach()
    # Crucial: target does not drift with a trainable predicted mean. The
    # prior is control shape at the FIXED lawful training population mean.
    desired = sup('mean').detach()
    cm = control.mean(1)
    bound = torch.where(cm[:,None] > 0, control*(desired/cm.clamp_min(1e-8))[:,None],desired[:,None])
    c0 = cov((bound/gene_scale)@components.T).detach()
    if real_effective_count is None:
        raise ValueError('Joint covariance requires real-cell support counts')
    counts = [min(int(_condition_row(real_effective_count,i,q)),
        int(_condition_row(supervision,i,q)['n_cells']),real.shape[1]) for i in range(q)]
    if min(counts) < 1:
        raise ValueError('Joint covariance has no real-cell support')
    alpha = y.new_tensor([v/(v+float(covariance_prior_strength)) for v in counts])
    target_cov = alpha[:,None,None]*cy+(1-alpha[:,None,None])*c0
    covariance_error = (cp-target_cov).square().mean((1,2))/target_cov.square().mean((1,2)).clamp_min(.01)
    structure = sw+.25*covariance_error+.004*output['rank_penalty']
    detection = F.binary_cross_entropy(output['p'].clamp(1e-7,1-1e-7),sup('detection'),reduction='none')
    detection = torch.where(valid,detection,0.).sum(1)/valid.sum(1)
    positive = valid & sup('positive_mask').bool()
    positive_error = F.smooth_l1_loss(output['r']/gene_scale.clamp_min(.05),
        sup('positive_mean')/gene_scale.clamp_min(.05),reduction='none')
    positive_error = torch.where(positive,positive_error*(count/(count+8.)),0.).sum(1)/positive.sum(1).clamp_min(1)
    marginal_group = marginal+.1*(shape_loss+detection+positive_error)
    loss = (w['joint_marginal']*marginal_group+w['joint_structure']*structure).mean()
    diagnostics = dict(marginal_W1=marginal.mean(),positive_shape=shape_loss.mean(),
        centered_joint_SW1=sw.mean(),relative_PCA_covariance=covariance_error.mean(),
        rank_penalty=output['rank_penalty'].mean(),covariance_data_weight=alpha.mean(),
        detection=detection.mean(),positive_mean=positive_error.mean(),
        joint_marginal=marginal_group.mean(),joint_structure=structure.mean())
    logs={name:value.detach() for name,value in diagnostics.items()}
    finite=torch.isfinite(loss.detach()) & torch.isfinite(torch.stack(list(logs.values()))).all()
    if check_finite and not bool(finite):
        raise RuntimeError('Nonfinite joint distribution objective')
    return (loss,logs,finite) if return_finite else (loss,logs)

