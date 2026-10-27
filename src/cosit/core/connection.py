"""Partial connection energy and differentiable, matrix-free SPD inference.

The bases and restriction maps are frozen. Learned quantities are only the
per-node observation blocks and right-hand side. Backward uses an adjoint
solve, so memory does not grow with the number of conjugate-gradient steps.
"""
from __future__ import annotations

import torch
from torch import nn


def edge_energy_apply(z, src, dst, left, right, weight):
    delta = torch.einsum('ekd,ed->ek', left, z[src]) - torch.einsum('ekd,ed->ek', right, z[dst])
    weighted = weight * delta
    out = torch.zeros_like(z)
    out.index_add_(0, src, torch.einsum('ekd,ek->ed', left, weighted))
    out.index_add_(0, dst, -torch.einsum('ekd,ek->ed', right, weighted))
    return out


def edge_diagonal(nodes, rank, src, dst, left, right, weight):
    out = weight.new_zeros(nodes, rank, rank)
    out.index_add_(0, src, torch.einsum('eki,ek,ekj->eij', left, weight, left))
    out.index_add_(0, dst, torch.einsum('eki,ek,ekj->eij', right, weight, right))
    return out


def pcg(blocks, rhs, src, dst, left, right, weight, strength, tolerance=2e-5, max_steps=100, *, graph_diagonal=None, chol=None):
    """Solve Hx=b; H is node blocks plus a PSD partial-connection energy."""
    def apply(x):
        return torch.einsum('gij,gj->gi', blocks, x) + strength * edge_energy_apply(x, src, dst, left, right, weight)

    if chol is None:
        if graph_diagonal is None:
            graph_diagonal = edge_diagonal(*rhs.shape, src, dst, left, right, weight)
        chol = torch.linalg.cholesky(blocks + strength * graph_diagonal)
    def precondition(r):
        return torch.cholesky_solve(r.unsqueeze(-1), chol).squeeze(-1)

    norm = rhs.norm()
    if float(norm) == 0:
        return torch.zeros_like(rhs), {'iterations': 0, 'relative_residual': 0.0}
    x = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = precondition(residual)
    rz = (residual * direction).sum()
    for step in range(max_steps):
        product = apply(direction)
        denominator = (direction * product).sum()
        if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
            raise RuntimeError('PCG encountered a non-positive/nonfinite curvature')
        alpha = rz / denominator
        x = x + alpha * direction
        residual = residual - alpha * product
        if float(residual.norm() / norm) <= tolerance:
            break
        new = precondition(residual)
        next_rz = (residual * new).sum()
        direction = new + (next_rz / rz) * direction
        rz = next_rz
    # Check the true residual rather than the recursively accumulated residual.
    relative = float((rhs - apply(x)).norm() / norm)
    if not torch.isfinite(x).all() or relative > max(5 * tolerance, 1e-10):
        raise RuntimeError(f'PCG did not converge: residual={relative:.3g}, steps={step + 1}')
    return x, {'iterations': step + 1, 'relative_residual': relative}


class _ImplicitConnectionSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, blocks, rhs, src, dst, left, right, weight, graph_diagonal, strength, tolerance, max_steps):
        # One factorization for this exact H; it is reused by the adjoint and
        # optional uncertainty probes, never across optimizer updates.
        chol = torch.linalg.cholesky(blocks + strength * graph_diagonal)
        z, info = pcg(blocks, rhs, src, dst, left, right, weight, strength, tolerance, max_steps, chol=chol)
        ctx.save_for_backward(blocks, z, src, dst, left, right, weight, chol)
        ctx.settings = strength, tolerance, max_steps
        metadata = torch.tensor([info['relative_residual'], info['iterations']], dtype=torch.float64)
        ctx.mark_non_differentiable(chol, metadata)
        return z, chol, metadata

    @staticmethod
    def backward(ctx, gradient, ignored_chol, ignored_metadata):
        blocks, z, src, dst, left, right, weight, chol = ctx.saved_tensors
        adjoint, _ = pcg(blocks, gradient, src, dst, left, right, weight, *ctx.settings, chol=chol)
        block_gradient = -torch.einsum('gi,gj->gij', adjoint, z)
        block_gradient = (block_gradient + block_gradient.transpose(-1, -2)) / 2
        return block_gradient, adjoint, None, None, None, None, None, None, None, None, None


class ConnectionInference(nn.Module):
    def __init__(self, geometry, strength=0.15, ridge=0.2, tolerance=2e-5, max_steps=100):
        super().__init__()
        if ridge <= 0 or strength < 0:
            raise ValueError('Positive ridge and nonnegative connection strength required')
        for key in ('bases', 'edge_src', 'edge_dst', 'left', 'right', 'weight'):
            x = torch.as_tensor(geometry[key])
            self.register_buffer(key, x.long() if key.startswith('edge_') else x.float())
        self.strength, self.ridge = float(strength), float(ridge)
        self.tolerance, self.max_steps = tolerance, max_steps
        # Frozen geometry only. This buffer follows dtype/device moves with
        # the model, and does not cache any learned operator.
        self.register_buffer('graph_diagonal', edge_diagonal(
            self.bases.shape[0], self.bases.shape[-1], self.edge_src, self.edge_dst,
            self.left, self.right, self.weight))

    def forward(self, operators, observations, precision, prior):
        # Source axis S remains intact until the likelihood's normal equations.
        if operators.ndim != 4 or observations.shape != operators.shape[:-1] or precision.shape != observations.shape:
            raise ValueError('Expected A[S,G,M,d], observations and precision[S,G,M]')
        if not bool((precision > 0).all()) or not bool(torch.isfinite(precision).all()):
            raise ValueError('Observation precision must be positive and finite')
        rank = operators.shape[-1]
        information = torch.einsum('sgmi,sgm,sgmj->gij', operators, precision, operators)
        blocks = information + self.ridge * torch.eye(rank, device=operators.device, dtype=operators.dtype)
        rhs = torch.einsum('sgmi,sgm,sgm->gi', operators, precision, observations) + self.ridge * prior
        z, chol, metadata = _ImplicitConnectionSolve.apply(
            blocks, rhs, self.edge_src, self.edge_dst, self.left, self.right, self.weight, self.graph_diagonal,
            self.strength, self.tolerance, self.max_steps,
        )
        recovered = torch.einsum('sgmd,gd->sgm', operators, z)
        residual = recovered - observations
        return {'rule': z, 'blocks': blocks, 'information': information,
                'source_residual': residual, 'source_precision': precision,
                'solve_relative_residual': float(metadata[0]),
                'solve_iterations': int(metadata[1]), 'preconditioner_cholesky': chol}

    def reconstruct_result(self, operators, observations, precision, rule, relative_residual):
        """Rebuild a frozen cached solution's diagnostics without solving again."""
        rank = operators.shape[-1]
        information = torch.einsum('sgmi,sgm,sgmj->gij', operators, precision, operators)
        blocks = information + self.ridge * torch.eye(rank, device=operators.device, dtype=operators.dtype)
        return dict(rule=rule, blocks=blocks, information=information,
                    source_residual=torch.einsum('sgmd,gd->sgm', operators, rule) - observations,
                    source_precision=precision, solve_relative_residual=float(relative_residual))

    @torch.no_grad()
    def uncertainty(self, result, target_operator, probes=16, seed=17):
        """Hutchinson diagonal estimate of A_target H^-1 A_target^T.

        This is CONDITIONAL rule-estimation uncertainty, not cell variability
        or a calibrated predictive interval. Negative finite-probe estimates
        are explicitly counted, and the raw estimate is retained.
        """
        generator = torch.Generator(device=target_operator.device).manual_seed(seed)
        estimate = target_operator.new_zeros(target_operator.shape[:2])
        chol = result.get('preconditioner_cholesky')
        if chol is None:
            chol = torch.linalg.cholesky(result['blocks'] + self.strength * self.graph_diagonal)
        for _ in range(probes):
            v = torch.randint(0, 2, estimate.shape, generator=generator, device=estimate.device).to(estimate.dtype) * 2 - 1
            rhs = torch.einsum('gmd,gm->gd', target_operator, v)
            sol, _ = pcg(result['blocks'], rhs, self.edge_src, self.edge_dst, self.left, self.right,
                         self.weight, self.strength, self.tolerance, self.max_steps, chol=chol)
            estimate += v * torch.einsum('gmd,gd->gm', target_operator, sol) / probes
        eig = torch.linalg.eigvalsh(result['information'])
        return {'raw_variance': estimate, 'variance': estimate.clamp_min(0),
                'negative_estimate_count': int((estimate < 0).sum()), 'probes': probes,
                'data_only_min_eigenvalue': eig[:, 0],
                'data_only_rank': (eig > 1e-4 * eig[:, -1:].clamp_min(1e-8)).sum(-1),
                'calibrated': False}
