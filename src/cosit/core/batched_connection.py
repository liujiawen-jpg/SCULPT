"""Independent connection solves vectorized over the condition axis.

The graph is shared, but every condition has its own normal equations, PCG
scalars, convergence mask and implicit adjoint. The single-condition inherited
API and state-dict layout are unchanged. No graph/condition approximation is
introduced; acceptance uses the same true-residual rule as connection.pcg.
"""
from __future__ import annotations

import torch

from .connection import ConnectionInference, edge_diagonal


def batched_edge_energy_apply(z, src, dst, left, right, weight):
    """Apply the shared graph to z[Q,G,d], without coupling different Q rows."""
    delta = torch.einsum('ekd,qed->qek', left, z[:, src])
    delta = delta - torch.einsum('ekd,qed->qek', right, z[:, dst])
    weighted = delta * weight.unsqueeze(0)
    result = torch.zeros_like(z)
    result.index_add_(1, src, torch.einsum('ekd,qek->qed', left, weighted))
    result.index_add_(1, dst, -torch.einsum('ekd,qek->qed', right, weighted))
    return result


def batched_pcg(blocks, rhs, src, dst, left, right, weight, strength,
                tolerance=2e-5, max_steps=100, *, graph_diagonal=None, chol=None,
                check_interval=4):
    """Solve independent H_q x_q=b_q, checking all query flags in one sync.

    Recursive convergence is checked on-device EVERY iteration. A converged
    query is immediately masked out, while harder queries continue. Only the
    host decision to stop the whole loop is batched every check_interval steps.
    The final explicitly recomputed residual has exactly the single-solve
    acceptance threshold max(5*tolerance,1e-10).

    Returned diagnostics remain on-device; the implicit wrapper transfers both
    vectors to the CPU together once. This routine, like connection.pcg, is a
    numerical solver. Use the implicit Function below for differentiated solves.
    """
    if rhs.ndim != 3 or blocks.shape != (*rhs.shape, rhs.shape[-1]):
        raise ValueError('Expected blocks[Q,G,d,d] and rhs[Q,G,d]')
    if rhs.shape[0] == 0 or rhs.shape[1] == 0 or rhs.shape[2] == 0:
        raise ValueError('PCG axes must be nonempty')
    if not 0 < tolerance or int(max_steps) < 1 or int(check_interval) < 1:
        raise ValueError('Positive tolerance, max_steps and check_interval required')
    max_steps, check_interval = int(max_steps), int(check_interval)

    def apply(value):
        return torch.einsum('qgij,qgj->qgi', blocks, value) + strength * batched_edge_energy_apply(
            value, src, dst, left, right, weight)

    if chol is None:
        if graph_diagonal is None:
            graph_diagonal = edge_diagonal(rhs.shape[1], rhs.shape[2], src, dst, left, right, weight)
        chol = torch.linalg.cholesky(blocks + strength * graph_diagonal.unsqueeze(0))

    def precondition(value):
        return torch.cholesky_solve(value.unsqueeze(-1), chol).squeeze(-1)

    axes = (-2, -1)
    norm = torch.linalg.vector_norm(rhs, dim=axes)
    zero_rhs = norm == 0
    safe_norm = torch.where(zero_rhs, torch.ones_like(norm), norm)
    x = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = precondition(residual)
    rz = (residual * direction).sum(dim=axes)
    active = ~zero_rhs
    bad_curvature = torch.zeros_like(active)
    iterations = torch.zeros(rhs.shape[0], device=rhs.device, dtype=torch.int64)

    # All-zero adjoints and all-zero forward right-hand sides remain exact zero.
    if bool(active.any()):
        for step in range(max_steps):
            product = apply(direction)
            denominator = (direction * product).sum(dim=axes)
            valid = torch.isfinite(denominator) & (denominator > 0) & torch.isfinite(rz) & (rz > 0)
            bad_curvature = bad_curvature | (active & ~valid)
            active = active & valid
            iterations = torch.where(active, torch.full_like(iterations, step + 1), iterations)
            divisor = torch.where(active, denominator, torch.ones_like(denominator))
            alpha = torch.where(active, rz / divisor, torch.zeros_like(rz))
            # Inactive slices are explicitly zeroed, including possible NaNs in
            # a failed slice; 0*NaN must not contaminate another iteration.
            direction = torch.where(active[:, None, None], direction, torch.zeros_like(direction))
            product = torch.where(active[:, None, None], product, torch.zeros_like(product))
            x = x + alpha[:, None, None] * direction
            residual = residual - alpha[:, None, None] * product
            relative_recursive = torch.linalg.vector_norm(residual, dim=axes) / safe_norm
            active = active & (relative_recursive > tolerance)

            if (step + 1) % check_interval == 0 or step + 1 == max_steps:
                # Two booleans cross the device boundary in a single transfer.
                flags = torch.stack((active.any(), bad_curvature.any())).detach().cpu().tolist()
                if flags[1]:
                    bad = torch.nonzero(bad_curvature, as_tuple=False).flatten().detach().cpu().tolist()
                    raise RuntimeError(f'Batched PCG non-positive/nonfinite curvature in conditions {bad}')
                if not flags[0]:
                    break

            new = precondition(residual)
            next_rz = (residual * new).sum(dim=axes)
            safe_rz = torch.where(active, rz, torch.ones_like(rz))
            beta = torch.where(active, next_rz / safe_rz, torch.zeros_like(rz))
            direction = new + beta[:, None, None] * direction
            direction = torch.where(active[:, None, None], direction, torch.zeros_like(direction))
            rz = next_rz

    # Never accept convergence solely from the recursive residual or by a batch
    # average: every condition must pass its own true residual check.
    relative = torch.linalg.vector_norm(rhs - apply(x), dim=axes) / safe_norm
    relative = torch.where(zero_rhs, torch.zeros_like(relative), relative)
    finite = torch.isfinite(x).all(dim=-1).all(dim=-1) & torch.isfinite(relative)
    failed = ~finite | (relative > max(5 * tolerance, 1e-10)) | bad_curvature
    if bool(failed.any()):
        bad = torch.nonzero(failed, as_tuple=False).flatten()
        detail = dict(conditions=bad.detach().cpu().tolist(),
                      residuals=relative[bad].detach().cpu().tolist(),
                      iterations=iterations[bad].detach().cpu().tolist())
        raise RuntimeError(f'Batched PCG did not converge independently: {detail}')
    return x, dict(iterations=iterations, relative_residual=relative)


class _ImplicitBatchedConnectionSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, blocks, rhs, src, dst, left, right, weight, graph_diagonal,
                strength, tolerance, max_steps, check_interval):
        chol = torch.linalg.cholesky(blocks + strength * graph_diagonal.unsqueeze(0))
        rule, info = batched_pcg(blocks, rhs, src, dst, left, right, weight,
            strength, tolerance, max_steps, chol=chol, check_interval=check_interval)
        ctx.save_for_backward(blocks, rule, src, dst, left, right, weight, chol)
        ctx.settings = strength, tolerance, max_steps, check_interval
        metadata = torch.stack((info['relative_residual'].double(), info['iterations'].double()), dim=-1).cpu()
        ctx.mark_non_differentiable(chol, metadata)
        return rule, chol, metadata

    @staticmethod
    def backward(ctx, gradient, ignored_chol, ignored_metadata):
        blocks, rule, src, dst, left, right, weight, chol = ctx.saved_tensors
        strength, tolerance, max_steps, check_interval = ctx.settings
        adjoint, _ = batched_pcg(blocks, gradient, src, dst, left, right, weight,
            strength, tolerance, max_steps, chol=chol, check_interval=check_interval)
        block_gradient = -torch.einsum('qgi,qgj->qgij', adjoint, rule)
        block_gradient = (block_gradient + block_gradient.transpose(-1, -2)) / 2
        return block_gradient, adjoint, None, None, None, None, None, None, None, None, None, None


class BatchedConnectionInference(ConnectionInference):
    """Drop-in state dict and single forward, plus independent condition batching."""
    def __init__(self, geometry, strength=0.15, ridge=0.2, tolerance=2e-5,
                 max_steps=100, check_interval=4):
        super().__init__(geometry, strength=strength, ridge=ridge,
                         tolerance=tolerance, max_steps=max_steps)
        if int(check_interval) < 1:
            raise ValueError('Positive check_interval required')
        self.check_interval = int(check_interval)

    def _check_condition_axes(self, operators, observations, precision, prior):
        if operators.ndim != 5 or observations.shape != operators.shape[:-1] or precision.shape != observations.shape:
            raise ValueError('Expected A[Q,S,G,M,d], observations and precision[Q,S,G,M]')
        q, sources, genes, statistics, rank = operators.shape
        if min(q, sources, genes, statistics, rank) < 1 or prior.shape != (q, genes, rank):
            raise ValueError('Nonempty condition axes and prior[Q,G,d] required')
        if genes != self.bases.shape[0] or rank != self.bases.shape[-1]:
            raise ValueError('Operators do not match the frozen graph gene/rank axes')
        if any(x.device != operators.device or x.dtype != operators.dtype for x in (observations, precision, prior, self.bases)):
            raise ValueError('Inputs and frozen geometry must share floating dtype/device')
        valid = torch.stack(((precision > 0).all(), torch.isfinite(precision).all(),
            torch.isfinite(operators).all(), torch.isfinite(observations).all(), torch.isfinite(prior).all())).all()
        if not bool(valid):
            raise ValueError('Finite inputs and strictly positive observation precision required')

    def forward_conditions(self, operators, observations, precision, prior):
        self._check_condition_axes(operators, observations, precision, prior)
        rank = operators.shape[-1]
        information = torch.einsum('qsgmi,qsgm,qsgmj->qgij', operators, precision, operators)
        blocks = information + self.ridge * torch.eye(rank, dtype=operators.dtype, device=operators.device)
        rhs = torch.einsum('qsgmi,qsgm,qsgm->qgi', operators, precision, observations) + self.ridge * prior
        rule, chol, metadata = _ImplicitBatchedConnectionSolve.apply(
            blocks, rhs, self.edge_src, self.edge_dst, self.left, self.right, self.weight,
            self.graph_diagonal, self.strength, self.tolerance, self.max_steps, self.check_interval)
        return dict(rule=rule, blocks=blocks, information=information,
            source_residual=torch.einsum('qsgmd,qgd->qsgm', operators, rule) - observations,
            source_precision=precision, solve_relative_residual=metadata[:, 0],
            solve_iterations=metadata[:, 1].long(), preconditioner_cholesky=chol)

    def reconstruct_result_conditions(self, operators, observations, precision, rule, relative_residual):
        """Restore per-condition diagnostics around a frozen cached solution."""
        self._check_condition_axes(operators, observations, precision, rule)
        rank = operators.shape[-1]
        information = torch.einsum('qsgmi,qsgm,qsgmj->qgij', operators, precision, operators)
        blocks = information + self.ridge * torch.eye(rank, dtype=operators.dtype, device=operators.device)
        residual = torch.as_tensor(relative_residual, dtype=torch.float64, device='cpu').reshape(-1)
        if residual.shape != (operators.shape[0],) or not bool(torch.isfinite(residual).all()) or bool((residual < 0).any()):
            raise ValueError('One finite nonnegative relative residual is required per condition')
        return dict(rule=rule, blocks=blocks, information=information,
            source_residual=torch.einsum('qsgmd,qgd->qsgm', operators, rule)-observations,
            source_precision=precision, solve_relative_residual=residual)
