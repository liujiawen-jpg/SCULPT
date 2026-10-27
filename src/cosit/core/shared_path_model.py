"""Fresh shared-state model with complete A/B/C distribution training.

All routes have identical parameters, source-only joint evidence prior and
source reconstruction supervision. ``open_zero`` retains every open input and
adds only the same-network differentiable zero-action reference. ``mandatory``
also masks direct target/joint features. No old predictor output or historical
task weight enters. Phase A is unchanged from the registered open-zero pilot;
B learns mean-normalized positive shape, and C learns exact column permutations.
"""
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from .model import ConnectionRuleModel


class SharedPathModel(ConnectionRuleModel):
    def __init__(self, geometry, *, route='mandatory', **kwargs):
        if route not in ('open', 'mandatory', 'open_zero'):
            raise ValueError('Unknown route')
        super().__init__(geometry, **kwargs)
        self.route = route
        # Initialized identically in all fresh arms. Source joint observations
        # contribute to the prior on the inferred state, never a fixed output.
        self.joint_to_prior = nn.Linear(self.state_rank, self.ambient, bias=False)
        nn.init.normal_(self.joint_to_prior.weight, std=.02)

    def set_phase(self, phase):
        # The parent implements exclusive A/backbone, B/shape and C/reorder
        # parameter groups, module modes and checkpoint phase restoration.
        # joint_to_prior is an A parameter through the same namespace rule.
        return super().set_phase(phase)

    def _fuse_joint(self, batch):
        # A shared source state must not change merely because its query
        # destination changes. Target identity is legal; target context is not
        # used to choose this upstream evidence summary in any arm.
        upstream = dict(batch, target_control=self.ctrl_center.expand_as(batch['target_control']))
        return super()._fuse_joint(upstream)

    def _gene_context(self, batch, rule, target_operator, fused_joint):
        shared = torch.einsum('gad,gd->ga', self.inference.bases, rule)
        statistics = torch.einsum('gmd,gd->gm', target_operator, rule)
        ctrl = (batch['target_control'] - self.ctrl_center) / self.ctrl_scale
        target = batch['target_prior'].expand(len(rule), -1)
        joint = fused_joint.expand(len(rule), -1)
        if self.route == 'mandatory':
            # Keep the same feature slots/parameter tensors for a matched
            # randomized initialization, but forbid these direct input paths.
            target = torch.zeros_like(target)
            joint = torch.zeros_like(joint)
        features = torch.cat((self.gene_prior, target, ctrl, shared, statistics, joint), -1)
        return self.gene(features), statistics

    def readout(self, batch, rule, operator, fused_joint):
        context, statistics = self._gene_context(batch, rule, operator, fused_joint)
        raw = self.decoder(context)
        if self.route in ('mandatory', 'open_zero'):
            # The same fresh network, both branches differentiable. This is a
            # zero-action definition, not an old prediction plus a correction.
            zero_context, _ = self._gene_context(batch, torch.zeros_like(rule), operator, fused_joint)
            raw = raw - self.decoder(zero_context)
        base_p = batch['target_control_detection'].clamp(1e-6, 1-1e-6)
        logit = torch.logit(base_p) + raw[:, 0]
        p = torch.sigmoid(logit)
        mean0 = batch['target_control_mean']
        # No source-template fallback can bypass Z to affect the mean.
        r0 = torch.where(mean0 > 0, mean0 / base_p, self.expression_scale)
        r = r0 * torch.exp(4 * torch.tanh(raw[:, 1] / 4))
        return dict(p=p, r=r, analytic_mean=p*r, detection_logits=logit,
                    target_statistics=statistics, gene_context=context, raw_action=raw)

    def encode(self, batch):
        scope = nullcontext() if self.phase in ('A', 'J') and self.training else torch.no_grad()
        with scope:
            source_A, target_A, prior = self._operators(batch)
            fused, raw_joint, weights, confidence = self._fuse_joint(batch)
            target_code = self.prior_target(self.prior_norm(batch['target_prior']))
            target_code = target_code[None] + self.joint_to_prior(fused)
            ambient_prior = torch.tanh(self.prior_gene(prior)) * torch.tanh(target_code)
            local_prior = torch.einsum('gad,ga->gd', self.inference.bases, ambient_prior)
            inferred = self.inference(source_A, batch['observations'], batch['precision'], local_prior)
            decoded = self.readout(batch, inferred['rule'], target_A, fused)
            source_readouts = []
            if self.phase in ('A', 'J') and self.training:
                for j in range(len(source_A)):
                    source_batch = dict(batch,
                        target_control=batch['source_controls'][j],
                        target_control_mean=batch['source_control_means'][j],
                        target_control_detection=batch['source_control_detections'][j])
                    source_readouts.append(self.readout(source_batch, inferred['rule'], source_A[j], fused))
            return dict(**decoded, rule=inferred['rule'], inference=inferred,
                target_operator=target_A, source_readouts=source_readouts,
                fused_joint=fused, fused_joint_raw=raw_joint, source_joint_weights=weights,
                source_joint_confidence=confidence, deformation=decoded['raw_action'].square().mean())


def source_readout_loss(output, batch, response_scale, own_gene=-1, *, detection_weight=.1):
    """Same decoded observation rule on already authorized input sources.

    Average over sources, mask the perturbed output gene, and keep source truth
    outside inference/readout APIs. No query supervision enters this term.
    """
    records = output['source_readouts']
    if len(records) != len(batch['observations']) or not records:
        raise ValueError('Expected one readout per authorized source')
    valid = torch.ones_like(response_scale, dtype=torch.bool)
    if own_gene >= 0:
        valid[own_gene] = False
    mse, detection = [], []
    for j, decoded in enumerate(records):
        err = (decoded['analytic_mean'] - batch['source_means'][j]) / response_scale.clamp_min(1e-6)
        mse.append(err[valid].square().mean())
        detection.append(F.binary_cross_entropy_with_logits(decoded['detection_logits'][valid],
                                                            batch['source_detections'][j, valid]))
    mean_mse, mean_detection = torch.stack(mse).mean(), torch.stack(detection).mean()
    return mean_mse + float(detection_weight)*mean_detection, dict(source_readout_mse=mean_mse.detach(),
                                            source_readout_detection=mean_detection.detach())
