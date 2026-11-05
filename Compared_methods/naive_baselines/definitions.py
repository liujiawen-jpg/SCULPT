"""Independent empirical baselines, with no model fitting or label access.

Analytic means and finite support means are deliberately distinct.  Target
Control uses its supplied pooled control mean; Source Mean equally averages
the three source-condition means.  Supports consist exclusively of unchanged
observed cell rows.  There is no shift, rescaling, clipping, or learned adapter.

The caller is responsible for supplying only legally available source/control
cells in the frozen gene order.  This module does not open any data files.
"""
from __future__ import annotations

import operator
from typing import Sequence

import numpy as np


SUPPORT_SIZE = 512
SAMPLING_RULE = "floor((arange(n)+0.5)*N/n), unchanged whole-cell rows"


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(result)


def _cells(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(f"{name} must be a nonempty cells-by-genes matrix")
    if not (np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.floating)):
        raise ValueError(f"{name} must contain real numeric expression values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite expression values")
    return array


def _systematic_indices(observed: int, requested: int) -> np.ndarray:
    observed = _positive_integer(observed, "observed")
    requested = _positive_integer(requested, "n")
    # This integer expression is exactly floor((i + .5) * N / n).
    return ((2 * np.arange(requested, dtype=np.int64) + 1)
            * observed // (2 * requested))


def deterministic_support(cells, n: int = SUPPORT_SIZE) -> np.ndarray:
    """Select midpoint-systematic whole rows; repeat rows if N is below n.

    No random generator is involved.  The output is a copy, so later handling
    cannot overwrite the input cells.  Values, gene order and within-row joint
    structure are unchanged.
    """
    array = _cells(cells, "cells")
    ids = _systematic_indices(len(array), n)
    return np.array(array[ids], copy=True)


def _mean_diagnostics(analytic_mean, support_mean) -> dict:
    difference = np.asarray(support_mean, dtype=np.float64) - np.asarray(
        analytic_mean, dtype=np.float64)
    return {
        "analytic_support_mean_max_abs": float(np.max(np.abs(difference))),
        "analytic_support_mean_rms": float(np.sqrt(np.mean(difference ** 2))),
        "mean_matching_transform_applied": False,
    }


def target_control(control_cells, pooled_mean) -> dict:
    """Predict unchanged target controls, with an exact pooled-mean output.

    Reuse this one support for every target in a fold.  A pooled mean may use a
    larger legal control roster than the finite cached cells, so its difference
    from the support mean is recorded rather than corrected.
    """
    cells = _cells(control_cells, "control_cells")
    mean = np.asarray(pooled_mean)
    if (mean.shape != (cells.shape[1],)
            or not (np.issubdtype(mean.dtype, np.integer)
                    or np.issubdtype(mean.dtype, np.floating))
            or not np.isfinite(mean).all()):
        raise ValueError("pooled_mean must be one finite real value per gene")
    analytic_mean = np.array(mean, copy=True)
    ids = _systematic_indices(len(cells), SUPPORT_SIZE)
    support = deterministic_support(cells, SUPPORT_SIZE)
    support_mean = support.mean(axis=0, dtype=np.float64)
    return {
        "method": "Target-Control",
        "analytic_mean": analytic_mean,
        "support": support,
        "support_mean": support_mean,
        "observed_count": int(len(cells)),
        "observed_counts": [int(len(cells))],
        "support_count": SUPPORT_SIZE,
        "support_counts": [SUPPORT_SIZE],
        "support_row_indices": ids,
        "sampling_rule": SAMPLING_RULE,
        "analytic_mean_definition": "supplied_full_pooled_target_control_mean",
        "support_definition": "unchanged_target_control_whole_cell_rows",
        **_mean_diagnostics(analytic_mean, support_mean),
    }


def source_mean(source_cells: Sequence[np.ndarray]) -> np.ndarray:
    """Equal context-weighted absolute expression, not a control-subtracted shift.

    Return only a gene vector. Different numbers of observed cells never change
    the source-context weights; no synthetic cell cloud is attached to it.
    """
    if len(source_cells) != 3:
        raise ValueError("Source-Mean requires exactly three ordered sources")
    arrays = [_cells(value, f"source_cells[{i}]") for i, value in enumerate(source_cells)]
    if len({x.shape[1] for x in arrays}) != 1:
        raise ValueError("Source gene axes differ")
    if any((x < 0).any() for x in arrays):
        raise ValueError("Normalized absolute expression must be nonnegative")
    return np.stack([x.mean(axis=0, dtype=np.float64) for x in arrays]).mean(axis=0, dtype=np.float64)
