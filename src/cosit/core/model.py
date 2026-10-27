"""Fresh connection-rule model with separable or end-to-end hurdle/shape training.

A predicts continuous detection probability p and positive conditional mean r.
B changes only normalized positive quantiles. C changes only per-gene cell ranks.
J jointly trains all modules. The emitted cloud is deterministic (not iid):
K=round(N*p), mean=K/N*r.
No historical task weights, predictions or learned mean anchors are accepted.
"""
from __future__ import annotations

from contextlib import nullcontext
import math

import torch
from torch import nn
from torch.nn import functional as F

from .connection import ConnectionInference
from .rearrangement import ReorderHead



def formal_mean(p, r, n=512, *, straight_through=False):
    """Exact finite-cloud mean; optional biased derivative through rounded p.

    The forward value is the emitted N-cell cloud mean. No derivative through
    integer positive counts exists; ST explicitly uses dp=1 as a surrogate.
    """
    if n < 2 or p.shape != r.shape:
        raise ValueError('Formal cloud mean requires aligned p/r and N>=2')
    fraction = torch.round(p * n).clamp(0, n) / n
    if straight_through:
        fraction = p + (fraction - p).detach()
    return fraction * r


def inverse_softplus(x):
    x = x.clamp_min(1e-5)
    return x + torch.log(-torch.expm1(-x))


class PositiveShape(nn.Module):
    """Monotone amplitude deformation, including dispersion of flat templates."""
    def __init__(self, context_dim, width, joint_rank):
        super().__init__()
        # This projection belongs to phase B, so covariance information need
        # not survive an A-phase embedding trained only for mean marginals.
        self.joint = nn.Sequential(nn.LayerNorm(625), nn.Linear(625, width),
                                   nn.GELU(), nn.Linear(width, joint_rank))
        self.network = nn.Sequential(nn.LayerNorm(context_dim + joint_rank), nn.Linear(context_dim + joint_rank, width),
                                     nn.GELU(), nn.Linear(width, 3))
        nn.init.normal_(self.network[-1].weight, std=.02)
        nn.init.normal_(self.network[-1].bias, std=.002)

    def forward(self, template_values, quantiles, context, joint, confidence):
        embedded = self.joint(joint) * confidence[:, None]
        context = torch.cat((context, embedded.expand(len(context), -1)), -1)
        raw = self.network(context)
        # Positive and initially about one, but can approach zero. The old
        # exp(.75*tanh(.)) floor at .472 prevented sufficiently narrowing a
        # broad control-positive template, even with unlimited optimization.
        power = F.softplus(raw[:, 0] + math.log(math.expm1(1.)))
        # The additive monotone log-amplitude terms can introduce dispersion
        # even when every observed positive template quantile is identical.
        spread = F.softplus(raw[:, 1] - 2.)
        tail = F.softplus(raw[:, 2] - 3.)
        z = 2 * quantiles - 1
        log_values = power[None] * template_values.clamp_min(1e-6).log()
        log_values = log_values + spread[None] * z + tail[None] * z.pow(3)
        # A common column shift improves numerical range and cancels exactly
        # in the following positive-mean normalization. Keep its derivative.
        log_values = log_values - log_values.max(0, keepdim=True).values
        values = torch.exp(log_values.clamp_min(-40.))
        return values, dict(power=power, spread=spread, tail=tail,
                            regularization=raw.square().mean())


class ConnectionRuleModel(nn.Module):
    def __init__(self, geometry, prior_dim=261, width=128, state_rank=32,
                 strength=.15, ridge=.2, reorder_max_shift=.2):
        super().__init__()
        self.inference = ConnectionInference(geometry, strength=strength, ridge=ridge)
        self.register_buffer('gene_prior', torch.as_tensor(geometry['prior']).float())
        self.register_buffer('ctrl_center', torch.as_tensor(geometry['ctrl_center']).float())
        self.register_buffer('ctrl_scale', torch.as_tensor(geometry['ctrl_scale']).float().clamp_min(1e-5))
        scales = geometry.get('stat_scale')
        scale = torch.ones(len(geometry['bases'])) if scales is None else torch.as_tensor(scales[:, 0]).float()
        self.register_buffer('expression_scale', scale.clamp_min(.05))
        self.register_buffer('_phase_code', torch.tensor(0, dtype=torch.long))
        self._phase = 'A'
        ambient = self.inference.bases.shape[1]
        self.ambient, self.state_rank = ambient, state_rank
        self.prior_dim, self.width = prior_dim, width
        self.prior_norm = nn.LayerNorm(prior_dim)
        self.observation_base = nn.Parameter(torch.empty(7, ambient))
        nn.init.orthogonal_(self.observation_base)
        self.background = nn.Sequential(nn.Linear(prior_dim + 7, 64), nn.GELU(), nn.Linear(64, 7 * ambient))
        nn.init.normal_(self.background[-1].weight, std=.01)
        nn.init.normal_(self.background[-1].bias, std=.001)
        self.prior_gene = nn.Linear(prior_dim, ambient, bias=False)
        self.prior_target = nn.Linear(prior_dim, ambient, bias=False)
        self.joint = nn.Sequential(nn.LayerNorm(625), nn.Linear(625, width), nn.GELU(),
                                   nn.Linear(width, state_rank))
        self.joint_key = nn.Linear(state_rank + 7, state_rank)
        self.joint_query = nn.Sequential(nn.Linear(prior_dim + 7, state_rank), nn.Tanh())
        gene_dim = 2 * prior_dim + 7 + ambient + 7 + state_rank
        self.gene = nn.Sequential(nn.LayerNorm(gene_dim), nn.Linear(gene_dim, width), nn.GELU(),
                                  nn.Linear(width, width), nn.GELU())
        self.decoder = nn.Linear(width, 2)
        nn.init.normal_(self.decoder.weight, std=.02)
        nn.init.normal_(self.decoder.bias, std=.002)
        self.shape_context_dim = width + state_rank + 7 + 7
        self.shape = PositiveShape(self.shape_context_dim, width, state_rank)
        self.reorder = ReorderHead(cell_dim=48 + prior_dim, gene_dim=self.shape_context_dim,
                                  width=width, rank=state_rank, max_shift=reorder_max_shift,
                                  anchors=32, gene_chunk=256, joint_dim=625)
        self.register_load_state_dict_post_hook(self._restore_phase)
        self.set_phase('A')

    @property
    def phase(self):
        return self._phase

    def _restore_phase(self, module, incompatible_keys):
        code = int(self._phase_code.detach().cpu())
        if code not in (0, 1, 2, 3):
            raise ValueError('Invalid phase in model state')
        self.set_phase(('A', 'B', 'C', 'J')[code])

    def _apply_module_modes(self):
        # There are currently no running batch statistics, but the freeze
        # contract also remains correct if such layers are added later.
        for name, module in self.named_children():
            active = (self.phase == 'J' or (name == 'shape' and self.phase == 'B') or
                      (name == 'reorder' and self.phase == 'C') or
                      (name not in ('shape', 'reorder') and self.phase == 'A'))
            module.train(bool(self.training and active))

    def set_phase(self, phase):
        if phase not in ('A', 'B', 'C', 'J'):
            raise ValueError('Phase must be A, B, C or J')
        self._phase = phase
        self._phase_code.fill_(('A', 'B', 'C', 'J').index(phase))
        for name, parameter in self.named_parameters():
            trainable = (phase == 'J' or (phase == 'B' and name.startswith('shape.')) or
                         (phase == 'C' and name.startswith('reorder.')) or
                         (phase == 'A' and not name.startswith(('shape.', 'reorder.'))))
            parameter.requires_grad_(trainable)
            parameter.grad = None
        self._apply_module_modes()
        return self

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, '_phase'):
            self._apply_module_modes()
        return self

    def operator(self, control, normalized_prior=None):
        normalized = (control - self.ctrl_center) / self.ctrl_scale
        prior = self.prior_norm(self.gene_prior) if normalized_prior is None else normalized_prior
        delta = .5 * torch.tanh(self.background(torch.cat((prior, normalized), -1)).reshape(-1, 7, self.ambient))
        return torch.einsum('gma,gad->gmd', self.observation_base[None] + delta, self.inference.bases)

    def _operators(self, batch):
        prior = self.prior_norm(self.gene_prior)
        return (torch.stack([self.operator(c, prior) for c in batch['source_controls']]),
                self.operator(batch['target_control'], prior), prior)

    def _fuse_joint(self, batch):
        joint = batch['source_joint']
        sources = len(batch['observations'])
        if joint.shape != (sources, 625):
            raise ValueError('source_joint must have shape [sources,625]')
        reliability = batch['source_reliability']
        if reliability.shape not in ((sources,), (sources, len(self.gene_prior))):
            raise ValueError('source_reliability must have shape [S] or [S,G]')
        if not bool(torch.isfinite(joint).all()) or not bool(torch.isfinite(reliability).all()) or bool((reliability < 0).any()):
            raise ValueError('Invalid source joint evidence or reliability')
        embedded = self.joint(joint)
        source_ctrl = ((batch['source_controls'] - self.ctrl_center) / self.ctrl_scale).mean(1)
        target_ctrl = ((batch['target_control'] - self.ctrl_center) / self.ctrl_scale).mean(0)
        query = self.joint_query(torch.cat((batch['target_prior'], target_ctrl)))
        keys = self.joint_key(torch.cat((embedded, source_ctrl), -1))
        logits = (keys * query[None]).sum(-1) / math.sqrt(self.state_rank)
        rel = reliability[:, None] if reliability.ndim == 1 else reliability
        valid = rel > 0
        scores = logits[:, None] + rel.clamp_min(1e-30).log()
        scores = scores.masked_fill(~valid, -1e30)
        weights = torch.softmax(scores, 0) * valid.to(scores.dtype)
        weights = weights / weights.sum(0, keepdim=True).clamp_min(1e-12)
        confidence = (rel * weights).sum(0).clamp(0, 1)
        # For scalar reliability the result is [1,d], not G redundant rows.
        fused = torch.einsum('sg,sd->gd', weights, embedded) * confidence[:, None]
        raw_fused = torch.einsum('sg,sd->gd', weights, joint)
        return fused, raw_fused, weights, confidence

    def _gene_context(self, batch, rule, target_operator, fused_joint):
        shared = torch.einsum('gad,gd->ga', self.inference.bases, rule)
        statistics = torch.einsum('gmd,gd->gm', target_operator, rule)
        ctrl = (batch['target_control'] - self.ctrl_center) / self.ctrl_scale
        joint = fused_joint.expand(len(rule), -1)
        features = torch.cat((self.gene_prior, batch['target_prior'].expand(len(rule), -1),
                              ctrl, shared, statistics, joint), -1)
        context = self.gene(features)
        return context, statistics

    def encode(self, batch):
        """Encode one legal condition; no generated-cell roster enters p or r.

        B/C callers may compress this result and cache it for the same frozen
        A checkpoint. All cached tensors keep their original float32 precision.
        """
        scope = nullcontext() if self.phase in ('A', 'J') and self.training else torch.no_grad()
        with scope:
            source_A, target_A, prior = self._operators(batch)
            target = batch['target_prior']
            ambient_prior = torch.tanh(self.prior_gene(prior)) * torch.tanh(
                self.prior_target(self.prior_norm(target)))[None]
            local_prior = torch.einsum('gad,ga->gd', self.inference.bases, ambient_prior)
            inferred = self.inference(source_A, batch['observations'], batch['precision'], local_prior)
            fused_joint, raw_joint, source_weights, confidence = self._fuse_joint(batch)
            context, target_statistics = self._gene_context(batch, inferred['rule'], target_A, fused_joint)
            raw = self.decoder(context)
            # Population controls define the absolute expression reference;
            # sampled control prototypes are only shape/geometry evidence.
            base_p = batch['target_control_detection'].clamp(1e-6, 1 - 1e-6)
            logit = torch.logit(base_p) + raw[:, 0]
            p = torch.sigmoid(logit)
            base_r = batch['positive_base_mean']
            base_r = torch.where(torch.isfinite(base_r) & (base_r > 0), base_r, self.expression_scale)
            pooled_mean = batch['target_control_mean']
            base_r = torch.where(pooled_mean > 0, pooled_mean / base_p, base_r)
            r = base_r * torch.exp(4 * torch.tanh(raw[:, 1] / 4))
            return dict(p=p, r=r, analytic_mean=p * r, detection_logits=logit,
                        rule=inferred['rule'], inference=inferred, target_operator=target_A,
                        target_statistics=target_statistics, gene_context=context,
                        fused_joint=fused_joint, fused_joint_raw=raw_joint,
                        source_joint_confidence=confidence, source_joint_weights=source_weights,
                        deformation=raw.square().mean())

    @staticmethod
    def compress_encoded(encoded):
        """Drop reconstructible fields, not precision; device/CPU move is caller-owned."""
        return dict(cache_format='connection_hurdle_v4_compact_1',
                    rule=encoded['rule'].detach(), p=encoded['p'].detach(), r=encoded['r'].detach(),
                    detection_logits=encoded['detection_logits'].detach(),
                    fused_joint=encoded['fused_joint'].detach(),
                    fused_joint_raw=encoded['fused_joint_raw'].detach(),
                    source_joint_confidence=encoded['source_joint_confidence'].detach(),
                    source_joint_weights=encoded['source_joint_weights'].detach(),
                    solve_relative_residual=encoded['inference']['solve_relative_residual'])

    def restore_encoded(self, encoded, batch):
        """Rebuild inexpensive frozen operators/MLP context without a PCG solve."""
        if encoded.get('cache_format') != 'connection_hurdle_v4_compact_1':
            return encoded
        if self.phase in ('A', 'J') and self.training:
            raise ValueError('Frozen encoded cache cannot train phase A')
        with torch.no_grad():
            source_A, target_A, _ = self._operators(batch)
            rule = encoded['rule']
            context, statistics = self._gene_context(batch, rule, target_A, encoded['fused_joint'])
            inferred = self.inference.reconstruct_result(
                source_A, batch['observations'], batch['precision'], rule,
                encoded['solve_relative_residual'])
            p, r = encoded['p'], encoded['r']
            return dict(p=p, r=r, analytic_mean=p*r,
                        detection_logits=encoded['detection_logits'], rule=rule,
                        inference=inferred, target_operator=target_A,
                        target_statistics=statistics, gene_context=context,
                        fused_joint=encoded['fused_joint'], fused_joint_raw=encoded['fused_joint_raw'],
                        source_joint_confidence=encoded['source_joint_confidence'],
                        source_joint_weights=encoded['source_joint_weights'],
                        deformation=p.new_zeros(()))

    def _positive_reference(self, batch):
        template = batch['positive_template']
        if template.shape != (len(self.gene_prior), 33):
            raise ValueError('positive_template must have shape [G,33]')
        # Data fallback is normally resolved by runtime. This last fallback
        # keeps a fully absent positive template finite and visibly flagged.
        invalid = ~torch.isfinite(template).all(-1) | (template <= 0).any(-1)
        base_r = batch['positive_base_mean']
        base_r = torch.where(torch.isfinite(base_r) & (base_r > 0), base_r, self.expression_scale)
        template = torch.where(invalid[:, None], base_r[:, None], template)
        if bool((template[:, 1:] < template[:, :-1]).any()):
            raise ValueError('Positive template quantiles must be nondecreasing')
        return template, base_r, invalid

    def _positive_quantiles(self, encoded, shape_context, reference):
        """N-independent positive shape supervision, even when the cloud K is 0/1.

        These are 33 deterministic positive quantile points whose arithmetic
        mean is r; this finite-grid normalization is also used by supervision.
        They are not observed target values and do not alter the emitted cloud.
        """
        template, base_r, _ = reference
        relative = template.T / base_r[None].clamp_min(1e-6)
        u = torch.linspace(0, 1, 33, device=relative.device, dtype=relative.dtype)[:, None].expand_as(relative)
        if self.phase in ('B', 'C', 'J'):
            values, _ = self.shape(relative, u, shape_context,
                encoded['fused_joint_raw'] if self.phase == 'J' else encoded['fused_joint_raw'].detach(),
                encoded['source_joint_confidence'] if self.phase == 'J' else encoded['source_joint_confidence'].detach())
        else:
            values = relative
        normalized = values / values.mean(0, keepdim=True).clamp_min(1e-20)
        return (normalized * encoded['r'][None]).T

    def _positive_cloud(self, encoded, batch, shape_context, reference):
        control = batch['control_cells']
        n, genes = control.shape
        p, r = encoded['p'], encoded['r']
        k = torch.round(p.detach() * n).long().clamp(0, n)
        row = torch.arange(n, device=p.device)[:, None]
        # Positive quantiles cover [0,1] with K midpoint samples. Inactive slots
        # are represented explicitly and stay exact zero, including K=0.
        active = row >= (n-k)[None]
        u = ((row - (n-k)[None]).to(p.dtype) + .5) / k.clamp_min(1)[None]
        u = u.clamp(0, 1)
        template, base_r, invalid = reference
        position = u * 32
        lo = position.floor().long().clamp(0, 32)
        hi = (lo + 1).clamp(max=32)
        fraction = position - lo
        template_t = template.T
        values = template_t.gather(0, lo) * (1-fraction) + template_t.gather(0, hi) * fraction
        relative = values / base_r[None].clamp_min(1e-6)
        if self.phase in ('B', 'C', 'J'):
            values, shape_stats = self.shape(relative, u, shape_context,
                encoded['fused_joint_raw'] if self.phase == 'J' else encoded['fused_joint_raw'].detach(),
                encoded['source_joint_confidence'] if self.phase == 'J' else encoded['source_joint_confidence'].detach())
        else:
            values = relative
            shape_stats = dict(regularization=p.new_zeros(()), identity=True)
        mass = torch.where(active, values, torch.zeros_like(values))
        normalizer = mass.sum(0) / k.clamp_min(1).to(values.dtype)
        normalized = torch.where(active, values / normalizer.clamp_min(1e-20)[None], torch.zeros_like(values))
        # Mean normalization stays in the graph. No clipping is performed
        # afterwards, since that would violate the promised mean invariant.
        sorted_cloud = normalized * r[None]
        # The initial copula is the observed control ordering. Ties use stable
        # row order; phase C learns a different exact column permutation.
        order = torch.argsort(control, dim=0, stable=True)
        ranks = torch.empty_like(order).scatter_(0, order, row.expand_as(order))
        cloud = sorted_cloud.gather(0, ranks)
        return cloud, k, dict(**shape_stats, template_fallback=invalid,
                               positive_missing=batch['positive_missing'],
                               normalized_positive_mean=normalizer)

    def decode_encoded(self, encoded, batch):
        encoded = self.restore_encoded(encoded, batch)
        control = batch['control_cells']
        if control.ndim != 2 or len(control) < 2 or control.shape[1] != len(self.gene_prior):
            raise ValueError('Expected [N>=2,G] control roster')
        if not bool(torch.isfinite(control).all()) or bool((control < 0).any()):
            raise ValueError('Control roster must be finite and nonnegative')
        if self.phase in ('B', 'C'):
            # Even an accidentally non-detached caller cache cannot send B/C
            # gradients back into the frozen marginal feature extractor.
            shape_context = torch.cat((encoded['gene_context'].detach(),
                encoded['fused_joint'].detach().expand(len(self.gene_prior), -1),
                encoded['target_statistics'].detach(),
                ((batch['target_control']-self.ctrl_center)/self.ctrl_scale).detach()), -1)
        else:
            shape_context = torch.cat((encoded['gene_context'],
                encoded['fused_joint'].expand(len(self.gene_prior), -1),
                encoded['target_statistics'], (batch['target_control']-self.ctrl_center)/self.ctrl_scale), -1)
        reference = self._positive_reference(batch)
        shape_scope = torch.no_grad() if self.phase == 'C' else nullcontext()
        with shape_scope:
            pre, k, shape_stats = self._positive_cloud(encoded, batch, shape_context, reference)
            positive_quantiles = self._positive_quantiles(encoded, shape_context, reference)
        if self.phase in ('C', 'J'):
            cells = torch.cat((batch['cell_features'], batch['target_prior'].expand(len(control), -1)), -1)
            support, reorder_stats = self.reorder(pre if self.phase == 'J' else pre.detach(), cells.detach(),
                shape_context if self.phase == 'J' else shape_context.detach(),
                joint=encoded['fused_joint_raw'] if self.phase == 'J' else encoded['fused_joint_raw'].detach(),
                confidence=encoded['source_joint_confidence'] if self.phase == 'J' else encoded['source_joint_confidence'].detach())
        else:
            support = pre
            zero = support.new_zeros(())
            reorder_stats = dict(rank_penalty=zero, mean_normalized_rank_displacement=zero,
                                 changed_value_fraction=zero, soft_hard_MAE=zero)
        if not bool(torch.isfinite(support).all()) or bool((support < 0).any()):
            raise RuntimeError('Invalid nonnegative cloud')
        p, r = encoded['p'], encoded['r']
        detection = k.to(p.dtype) / len(control)
        result = dict(encoded)
        result.update(support=support, pre_reorder_support=pre, mean=support.mean(0),
                      pre_reorder_mean=pre.mean(0), p=p, r=r, analytic_mean=p*r,
                      detection_probability=p, positive_mean=r, detection=detection,
                      positive_count=k, quantized_mean=detection*r, positive_quantiles=positive_quantiles,
                      unit_mean_shape=positive_quantiles / r.clamp_min(1e-12)[:, None],
                      mean512_st=formal_mean(p, r, 512, straight_through=True),
                      # Compatibility only: no ST gate and no gate gradient.
                      gates=(pre > 0).to(pre.dtype), soft_gates=p[None].expand_as(pre),
                      shape=shape_stats, shape_stats=shape_stats,
                      source_count=len(batch['observations']), phase=self.phase,
                      reorder_stats=reorder_stats, rank_penalty=reorder_stats['rank_penalty'])
        return result

    def forward(self, batch, uncertainty_probes=0, encoded=None):
        if encoded is not None and self.phase in ('A', 'J') and self.training:
            raise ValueError('Phase A must freshly encode trainable source evidence')
        result = self.decode_encoded(self.encode(batch) if encoded is None else encoded, batch)
        if uncertainty_probes:
            result['uncertainty'] = self.inference.uncertainty(
                result['inference'], result['target_operator'], probes=uncertainty_probes)
        return result
