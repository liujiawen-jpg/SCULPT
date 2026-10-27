"""True independent-condition vectorization of the fresh open-zero model.

Q is a distinct condition axis: observations [Q,S,G,7], rules [Q,G,d],
clouds [Q,N,G]. Neither cells, graph nodes nor retrieval identities are pooled
across Q. Single-condition forward and checkpoint parameter names stay intact.
"""
from contextlib import nullcontext
import math
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .shared_path_model import SharedPathModel
from .model import formal_mean
from .batched_connection import BatchedConnectionInference

_COUNT = '_private_condition_count'


def split_condition_outputs(output):
    """Return legacy per-condition objective/cache records without detaching."""
    count = int(output[_COUNT])
    def part(value, i):
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value
            if value.shape[0] != count:
                raise ValueError('Output tensor lost independent condition axis')
            return value[i]
        if isinstance(value, dict):
            return {k: part(v, i) for k, v in value.items() if k != _COUNT}
        if isinstance(value, list):
            return [part(v, i) for v in value]
        if isinstance(value, tuple):
            return tuple(part(v, i) for v in value)
        return value
    return [part(output, i) for i in range(count)]


def _stack_compact(records):
    if isinstance(records, dict):
        return records
    if not records or any(r.get('cache_format') != 'connection_hurdle_v4_compact_1' for r in records):
        raise ValueError('Expected current-run compact condition encodings')
    keys = set(records[0])
    if any(set(r) != keys for r in records):
        raise ValueError('Compact condition fields differ')
    out = {_COUNT: len(records), 'cache_format': records[0]['cache_format']}
    for key in keys - {'cache_format'}:
        values = [r[key] for r in records]
        if isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values)
        elif isinstance(values[0], (int, float)):
            out[key] = torch.as_tensor(values)
        else:
            raise ValueError('Unsupported compact condition field: ' + key)
    return out


def _rank_cells(x):
    order = torch.argsort(x, dim=1, stable=True)
    rows = torch.arange(x.shape[1], device=x.device)[None, :, None].expand_as(order)
    return torch.empty_like(order).scatter_(1, order, rows)


def _smooth_proxy_conditions(values, score, temperature, anchor, rank):
    n = score.shape[1]
    q = torch.sigmoid((score[:, :, None, :] - anchor[:, None, :, :]) / temperature).mean(2)
    qmin = q.min(1, keepdim=True).values.detach()
    qmax = q.max(1, keepdim=True).values.detach()
    span = qmax - qmin
    margin = 1e-3
    regular = margin + (n - 1 - 2 * margin) * (q - qmin) / span.clamp_min(1e-4)
    fallback = margin + (n - 1 - 2 * margin) * rank.to(score.dtype) / (n - 1)
    fallback = fallback + (score - score.detach()) * (n - 1)
    position = torch.where(span >= 1e-4, regular, fallback).clamp(0, n - 1)
    lo = position.floor().long(); hi = (lo + 1).clamp(max=n - 1)
    fraction = position - lo
    return values.gather(1, lo) * (1 - fraction) + values.gather(1, hi) * fraction


class BatchedSharedPathModel(SharedPathModel):
    def __init__(self, geometry, *, route='open_zero', **kwargs):
        super().__init__(geometry, route=route, **kwargs)
        old = self.inference
        replacement = BatchedConnectionInference(geometry, strength=old.strength,
            ridge=old.ridge, tolerance=old.tolerance, max_steps=old.max_steps)
        replacement.load_state_dict(old.state_dict(), strict=True)
        self.inference = replacement
        self._apply_module_modes()

    @staticmethod
    def _batch(records):
        if isinstance(records, (list, tuple)):
            from .condition_batching import collate_condition_batches
            records = collate_condition_batches(list(records))
        observations = records['observations']
        if observations.ndim != 4 or observations.shape[-1] != 7:
            raise ValueError('Expected independent conditions observations [Q,S,G,7]')
        q, s, g, _ = observations.shape
        if min(q, s, g) < 1 or records['source_controls'].shape != observations.shape:
            raise ValueError('Source condition axes differ')
        if records['target_control'].shape != (q, g, 7) or records['target_prior'].ndim != 2:
            raise ValueError('Target background/prior condition axes differ')
        if records['target_prior'].shape[0] != q or records['source_joint'].shape != (q, s, 625):
            raise ValueError('Joint evidence or target prior lost condition axis')
        if records['control_cells'].ndim != 3 or records['control_cells'].shape[0] != q or records['control_cells'].shape[2] != g:
            raise ValueError('Control cells must be [Q,N,G]')
        return records

    def _operator_conditions(self, control, prior):
        leading = control.shape[:-2]
        normalized = (control - self.ctrl_center) / self.ctrl_scale
        expanded = prior.expand(*leading, *prior.shape)
        delta = .5 * torch.tanh(self.background(torch.cat((expanded, normalized), -1)).reshape(
            *control.shape[:-1], 7, self.ambient))
        return torch.einsum('...gma,gad->...gmd', self.observation_base + delta, self.inference.bases)

    def _operators_conditions(self, batch):
        prior = self.prior_norm(self.gene_prior)
        # Controls are population summaries, often repeated across Q. Reuse
        # their operator WITHIN this forward graph only; expand sums gradients.
        # Equality is checked so synthetic or future varying controls cannot be
        # silently collapsed to the first condition's background.
        def compute(control):
            if control.shape[0] > 1 and torch.equal(control,control[:1].expand_as(control)):
                return self._operator_conditions(control[:1],prior).expand(control.shape[0],*control.shape[1:-1],7,self.inference.bases.shape[-1])
            return self._operator_conditions(control,prior)
        return compute(batch['source_controls']),compute(batch['target_control']),prior

    def _fuse_joint_conditions(self, batch):
        joint = batch['source_joint']
        q, sources = joint.shape[:2]
        reliability = batch['source_reliability']
        if reliability.shape not in ((q, sources), (q, sources, len(self.gene_prior))):
            raise ValueError('Reliability must be [Q,S] or [Q,S,G]')
        if not bool(torch.isfinite(joint).all()) or not bool(torch.isfinite(reliability).all()) or bool((reliability < 0).any()):
            raise ValueError('Invalid source joint evidence or reliability')
        embedded = self.joint(joint)
        source_ctrl = ((batch['source_controls'] - self.ctrl_center) / self.ctrl_scale).mean(-2)
        # The accepted SharedPath upstream query deliberately has no target E.
        target_ctrl = torch.zeros((q, 7), device=joint.device, dtype=joint.dtype)
        query = self.joint_query(torch.cat((batch['target_prior'], target_ctrl), -1))
        keys = self.joint_key(torch.cat((embedded, source_ctrl), -1))
        logits = (keys * query[:, None, :]).sum(-1) / math.sqrt(self.state_rank)
        rel = reliability[..., None] if reliability.ndim == 2 else reliability
        valid = rel > 0
        scores = (logits[..., None] + rel.clamp_min(1e-30).log()).masked_fill(~valid, -1e30)
        weights = torch.softmax(scores, dim=1) * valid.to(scores.dtype)
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-12)
        confidence = (rel * weights).sum(1).clamp(0, 1)
        fused = torch.einsum('qsg,qsd->qgd', weights, embedded) * confidence[..., None]
        raw = torch.einsum('qsg,qsd->qgd', weights, joint)
        return fused, raw, weights, confidence

    def _gene_context_conditions(self, batch, rule, operator, fused):
        q, g = rule.shape[:2]
        shared = torch.einsum('gad,qgd->qga', self.inference.bases, rule)
        statistics = torch.einsum('qgmd,qgd->qgm', operator, rule)
        ctrl = (batch['target_control'] - self.ctrl_center) / self.ctrl_scale
        target = batch['target_prior'][:, None, :].expand(q, g, -1)
        joint = fused.expand(q, g, -1)
        if self.route == 'mandatory':
            target = torch.zeros_like(target); joint = torch.zeros_like(joint)
        features = torch.cat((self.gene_prior.expand(q, -1, -1), target, ctrl, shared, statistics, joint), -1)
        return self.gene(features), statistics

    def _readout_conditions(self, batch, rule, operator, fused):
        context, statistics = self._gene_context_conditions(batch, rule, operator, fused)
        raw = self.decoder(context)
        if self.route in ('mandatory', 'open_zero'):
            zero, _ = self._gene_context_conditions(batch, torch.zeros_like(rule), operator, fused)
            raw = raw - self.decoder(zero)
        base_p = batch['target_control_detection'].clamp(1e-6, 1 - 1e-6)
        logit = torch.logit(base_p) + raw[..., 0]
        p = torch.sigmoid(logit)
        mean0 = batch['target_control_mean']
        r0 = torch.where(mean0 > 0, mean0 / base_p, self.expression_scale)
        r = r0 * torch.exp(4 * torch.tanh(raw[..., 1] / 4))
        return dict(p=p, r=r, analytic_mean=p*r, detection_logits=logit,
                    target_statistics=statistics, gene_context=context, raw_action=raw)

    def encode_conditions(self, batches):
        batch = self._batch(batches)
        q, sources, genes, _ = batch['observations'].shape
        scope = nullcontext() if self.phase in ('A', 'J') and self.training else torch.no_grad()
        with scope:
            source_A, target_A, prior = self._operators_conditions(batch)
            fused, raw_joint, weights, confidence = self._fuse_joint_conditions(batch)
            target_code = self.prior_target(self.prior_norm(batch['target_prior']))[:, None, :] + self.joint_to_prior(fused)
            ambient_prior = torch.tanh(self.prior_gene(prior))[None] * torch.tanh(target_code)
            local_prior = torch.einsum('gad,qga->qgd', self.inference.bases, ambient_prior)
            inferred = self.inference.forward_conditions(source_A, batch['observations'], batch['precision'], local_prior)
            decoded = self._readout_conditions(batch, inferred['rule'], target_A, fused)
            source_readouts = []
            if self.phase in ('A', 'J') and self.training:
                # One batched MLP call over Q*S independent source readouts.
                source_batch = dict(batch,
                    target_control=batch['source_controls'].reshape(q*sources, genes, 7),
                    target_control_mean=batch['source_control_means'].reshape(q*sources, genes),
                    target_control_detection=batch['source_control_detections'].reshape(q*sources, genes),
                    target_prior=batch['target_prior'][:, None, :].expand(q, sources, -1).reshape(q*sources, -1))
                srule = inferred['rule'][:, None].expand(q, sources, genes, -1).reshape(q*sources, genes, -1)
                sfused = fused[:, None].expand(q, sources, *fused.shape[1:]).reshape(q*sources, *fused.shape[1:])
                read = self._readout_conditions(source_batch, srule, source_A.reshape(q*sources, genes, 7, -1), sfused)
                for source in range(sources):
                    source_readouts.append({k: v.reshape(q, sources, *v.shape[1:])[:, source] for k, v in read.items()})
            return dict(**decoded, **{_COUNT: q}, rule=inferred['rule'], inference=inferred,
                target_operator=target_A, source_readouts=source_readouts,
                fused_joint=fused, fused_joint_raw=raw_joint, source_joint_weights=weights,
                source_joint_confidence=confidence, deformation=decoded['raw_action'].square().mean((-2, -1)))

    def restore_encoded_conditions(self, encoded, batches, *, lightweight=False):
        batch = self._batch(batches)
        if lightweight and (self.phase not in ('B','C') or not self.training):
            raise ValueError('Lightweight frozen restoration is restricted to B/C training')
        encoded = _stack_compact(encoded)
        if encoded.get('cache_format') != 'connection_hurdle_v4_compact_1':
            return encoded
        if self.phase in ('A', 'J') and self.training:
            raise ValueError('Frozen encoded cache cannot train phase A')
        q = len(batch['observations'])
        if encoded['rule'].shape[0] != q:
            raise ValueError('Cached condition count changed')
        with torch.no_grad():
            if lightweight:
                # A's current-run compact cache already fixes the inferred rule.
                # B/C training uses target context, not source-fit diagnostics;
                # avoid rebuilding source operators and normal equations.
                target_A = self._operator_conditions(batch['target_control'], self.prior_norm(self.gene_prior))
            else:
                source_A, target_A, _ = self._operators_conditions(batch)
            rule = encoded['rule']
            context, statistics = self._gene_context_conditions(batch, rule, target_A, encoded['fused_joint'])
            if lightweight:
                residual = torch.as_tensor(encoded['solve_relative_residual'],device=rule.device).reshape(-1)
                if residual.shape != (q,):
                    raise ValueError('One residual is required per frozen condition')
                encoding_valid = (torch.isfinite(residual) & (residual >= 0)).all()
                inferred = dict(rule=rule, solve_relative_residual=residual)
            else:
                inferred = self.inference.reconstruct_result_conditions(source_A, batch['observations'],
                    batch['precision'], rule, encoded['solve_relative_residual'])
            p, r = encoded['p'], encoded['r']
            return dict(**{_COUNT: q}, p=p, r=r, analytic_mean=p*r,
                detection_logits=encoded['detection_logits'], rule=rule, inference=inferred,
                target_operator=target_A, target_statistics=statistics, gene_context=context,
                fused_joint=encoded['fused_joint'], fused_joint_raw=encoded['fused_joint_raw'],
                source_joint_confidence=encoded['source_joint_confidence'],
                source_joint_weights=encoded['source_joint_weights'], source_readouts=[],
                deformation=p.new_zeros(q), lightweight_training_encoding=bool(lightweight),
                encoding_valid=encoding_valid if lightweight else p.new_ones((),dtype=torch.bool))

    def _positive_reference_conditions(self, batch):
        template = batch['positive_template']
        q = batch['observations'].shape[0]
        if template.shape != (q, len(self.gene_prior), 33):
            raise ValueError('Positive templates must be [Q,G,33]')
        invalid = ~torch.isfinite(template).all(-1) | (template <= 0).any(-1)
        base_r = batch['positive_base_mean']
        base_r = torch.where(torch.isfinite(base_r) & (base_r > 0), base_r, self.expression_scale)
        template = torch.where(invalid[..., None], base_r[..., None], template)
        if bool((template[..., 1:] < template[..., :-1]).any()):
            raise ValueError('Positive template quantiles must be nondecreasing')
        return template, base_r, invalid

    def _shape_parameters_conditions(self, context, joint, confidence):
        q, genes = context.shape[:2]
        embedded = self.shape.joint(joint) * confidence[..., None]
        raw = self.shape.network(torch.cat((context, embedded.expand(q, genes, -1)), -1))
        return raw, dict(power=F.softplus(raw[..., 0] + math.log(math.expm1(1.))),
            spread=F.softplus(raw[..., 1] - 2.), tail=F.softplus(raw[..., 2] - 3.),
            regularization=raw.square().mean((-2, -1)))

    @staticmethod
    def _shape_values_conditions(template_values, u, parameters):
        z = 2*u - 1
        log_values = parameters['power'][:, None] * template_values.clamp_min(1e-6).log()
        log_values = log_values + parameters['spread'][:, None] * z + parameters['tail'][:, None] * z.pow(3)
        log_values = log_values - log_values.max(1, keepdim=True).values
        return torch.exp(log_values.clamp_min(-40.))

    def _reorder_conditions(self, y, cells, context, joint, confidence):
        head = self.reorder
        q, n, genes = y.shape
        ranks = _rank_cells(y)
        base = (ranks.to(y.dtype) + .5) / n
        values = y.sort(1).values
        if head.joint_dim:
            embedded = head.joint(joint) * confidence[..., None]
            context = torch.cat((context, embedded.expand(q, genes, -1)), -1)
        latent, loading = head.cell(cells), head.gene(context)
        raw = (latent @ loading.transpose(-1, -2)) / math.sqrt(head.rank)
        raw = raw - raw.mean(1, keepdim=True)
        shift = head.max_shift * torch.tanh(raw)
        score = base + shift
        rows = torch.arange(n, device=y.device)[None, :, None]
        anchor_ix = torch.linspace(0, n - 1, min(head.anchors, n), device=y.device).round().long()
        outputs, errors, rank_parts = [], [], []
        surrogate = head.training and torch.is_grad_enabled()
        for start in range(0, genes, head.gene_chunk):
            v = values[..., start:start + head.gene_chunk]
            s = score[..., start:start + head.gene_chunk]
            order = torch.argsort(s, dim=1, stable=True)
            rank = torch.empty_like(order).scatter_(1, order, rows.expand_as(order))
            rank_parts.append(rank)
            hard = v.gather(1, rank)
            if surrogate:
                anchors = s.detach().gather(1, order)[:, anchor_ix]
                soft = checkpoint(lambda a,b,c,d: _smooth_proxy_conditions(a,b,head.temperature,c,d),
                                  v.detach(), s, anchors, rank, use_reentrant=False)
                outputs.append(hard + (soft - soft.detach()))
                errors.append((soft.detach() - hard.detach()).abs().mean((1,2)))
            else:
                outputs.append(hard)
        out = torch.cat(outputs, dim=2)
        displacement = (torch.cat(rank_parts, 2) - ranks).abs().to(y.dtype) / n
        return out, dict(rank_penalty=(shift/head.max_shift).square().mean((1,2)),
            mean_normalized_rank_displacement=displacement.mean((1,2)).detach(),
            changed_value_fraction=(out.detach() != y).float().mean((1,2)),
            soft_hard_MAE=torch.stack(errors).mean(0) if errors else y.new_zeros(q))

    def decode_encoded_conditions(self, encoded, batches, *, marginal_only=False, lightweight=False):
        batch = self._batch(batches)
        if marginal_only and (self.phase != 'B' or not self.training):
            raise ValueError('Marginal-only clouds are restricted to B training')
        encoded = self.restore_encoded_conditions(encoded, batch, lightweight=lightweight)
        if encoded.get('lightweight_training_encoding') and not self.training:
            raise ValueError('A training-only encoding cannot serve evaluation')
        control = batch['control_cells']
        q, n, genes = control.shape
        if n < 2 or genes != len(self.gene_prior) or not bool((torch.isfinite(control) & (control >= 0)).all()):
            raise ValueError('Expected finite nonnegative [Q,N>=2,G] controls')
        contexts = (encoded['gene_context'], encoded['fused_joint'].expand(q, genes, -1),
                    encoded['target_statistics'], (batch['target_control'] - self.ctrl_center)/self.ctrl_scale)
        if self.phase in ('B','C'):
            contexts = tuple(x.detach() for x in contexts)
        context = torch.cat(contexts, -1)
        template, base_r, invalid = self._positive_reference_conditions(batch)
        p, r = encoded['p'], encoded['r']
        k = torch.round(p.detach()*n).long().clamp(0,n)
        rows = torch.arange(n,device=p.device)[None,:,None]
        active = rows >= (n-k)[:,None,:]
        u = ((rows-(n-k)[:,None,:]).to(p.dtype)+.5)/k.clamp_min(1)[:,None,:]
        u = u.clamp(0,1)
        pos = u*32; lo = pos.floor().long().clamp(0,32); hi = (lo+1).clamp(max=32)
        fraction = pos-lo
        tt = template.transpose(1,2)
        values = tt.gather(1,lo)*(1-fraction)+tt.gather(1,hi)*fraction
        relative = values/base_r[:,None,:].clamp_min(1e-6)
        relative_q = tt/base_r[:,None,:].clamp_min(1e-6)
        uq = torch.linspace(0,1,33,device=p.device,dtype=p.dtype)[None,:,None].expand_as(relative_q)
        scope = torch.no_grad() if self.phase == 'C' else nullcontext()
        with scope:
            if self.phase in ('B','C','J'):
                _, shape_stats = self._shape_parameters_conditions(context,
                    encoded['fused_joint_raw'] if self.phase == 'J' else encoded['fused_joint_raw'].detach(),
                    encoded['source_joint_confidence'] if self.phase == 'J' else encoded['source_joint_confidence'].detach())
                values = self._shape_values_conditions(relative,u,shape_stats)
                values_q = self._shape_values_conditions(relative_q,uq,shape_stats)
            else:
                values, values_q = relative, relative_q
                shape_stats = dict(regularization=p.new_zeros(q),identity=True)
            mass = torch.where(active,values,torch.zeros_like(values))
            normalizer = mass.sum(1)/k.clamp_min(1).to(values.dtype)
            normalized = torch.where(active,values/normalizer.clamp_min(1e-20)[:,None,:],torch.zeros_like(values))
            sorted_cloud = normalized*r[:,None,:]
            # B supervises only column marginals. Its training fast path keeps
            # the already sorted quantile cloud; validation and C retain the
            # original control-cell ordering and full joint construction.
            pre = sorted_cloud if marginal_only else sorted_cloud.gather(1,_rank_cells(control))
            positive_quantiles = ((values_q/values_q.mean(1,keepdim=True).clamp_min(1e-20))*r[:,None,:]).transpose(1,2)
        if self.phase in ('C', 'J'):
            cells = torch.cat((batch['cell_features'],batch['target_prior'][:,None,:].expand(q,n,-1)),-1)
            support,reorder_stats = self._reorder_conditions(pre if self.phase == 'J' else pre.detach(),cells.detach(),
                context if self.phase == 'J' else context.detach(),
                encoded['fused_joint_raw'] if self.phase == 'J' else encoded['fused_joint_raw'].detach(),
                encoded['source_joint_confidence'] if self.phase == 'J' else encoded['source_joint_confidence'].detach())
        else:
            support = pre
            reorder_stats = {key:p.new_zeros(q) for key in ('rank_penalty','mean_normalized_rank_displacement',
                                                         'changed_value_fraction','soft_hard_MAE')}
        finite_support = (torch.isfinite(support) & (support >= 0)).all()
        if 'encoding_valid' in encoded:
            finite_support = finite_support & encoded['encoding_valid']
        if not bool(finite_support):
            raise RuntimeError('Invalid condition cloud')
        detection = k.to(p.dtype)/n
        shape_stats.update(template_fallback=invalid,positive_missing=batch['positive_missing'],
                           normalized_positive_mean=normalizer)
        result = dict(encoded)
        result.update(support=support,pre_reorder_support=pre,mean=support.mean(1),pre_reorder_mean=pre.mean(1),
            p=p,r=r,analytic_mean=p*r,detection_probability=p,positive_mean=r,detection=detection,
            positive_count=k,quantized_mean=detection*r,positive_quantiles=positive_quantiles,
            unit_mean_shape=(values_q/values_q.mean(1,keepdim=True).clamp_min(1e-20)).transpose(1,2),
            mean512_st=formal_mean(p,r,512,straight_through=True),
            gates=(pre>0).to(pre.dtype),soft_gates=p[:,None,:].expand_as(pre),
            shape=shape_stats,shape_stats=shape_stats,source_count=batch['observations'].shape[1],phase=self.phase,
            reorder_stats=reorder_stats,rank_penalty=reorder_stats['rank_penalty'],
            marginal_only_training=bool(marginal_only))
        return result

    def forward_conditions(self, batches, encoded=None, mean_only=False, *, means_only=None,
                           marginal_only=False, lightweight=False):
        if means_only is not None:
            if mean_only and not means_only:
                raise ValueError('Conflicting mean-only aliases')
            mean_only = bool(means_only)
        batch = self._batch(batches)
        if marginal_only and (self.phase != 'B' or not self.training):
            raise ValueError('Marginal-only clouds are restricted to B training')
        if lightweight and (self.phase not in ('B','C') or not self.training):
            raise ValueError('Lightweight frozen restoration is restricted to B/C training')
        if encoded is not None and self.phase in ('A', 'J') and self.training:
            raise ValueError('Phase A must freshly encode trainable source evidence')
        values = (self.encode_conditions(batch) if encoded is None else
                  self.restore_encoded_conditions(encoded,batch,lightweight=lightweight))
        if mean_only:
            if self.phase != 'A' and not (self.phase == 'J' and not self.training):
                raise ValueError('Only A training or J evaluation may skip cell-cloud construction')
            q,g = values['p'].shape
            result = dict(values,phase=self.phase,source_count=batch['observations'].shape[1],mean_only_training=True)
            result['mean512_st'] = formal_mean(values['p'],values['r'],512,straight_through=True)
            if self.phase == 'J':
                result['mean'] = result['mean512_st']
                result['quantized_mean'] = result['mean512_st']
            # Compatibility for loss consumers that inspect support.device;
            # zero rows allocate no cell cloud and never serve as predictions.
            result['support'] = values['p'].new_empty(q,0,g)
            result['pre_reorder_support'] = result['support']
            return result
        return self.decode_encoded_conditions(values,batch,marginal_only=marginal_only,lightweight=lightweight)
