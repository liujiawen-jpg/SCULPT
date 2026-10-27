"""Runtime-only compatibility for the benchmark's documented all-tied MWU rule.

This does not edit the frozen evaluator, change its eligibility/BH rules, or
replace SciPy for any ordinary comparison. The module-local binding is wrapped
only so finite, pooled-identical observations have U=n*m/2 and p=1, as already
specified by ``build_deg_reference``. Unexpected nonfinite results pass through
to the frozen evaluator's existing refusal.
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import copy
from pathlib import Path
from typing import Any

import numpy as np


_STATE_ATTRIBUTE = "_pie_mwu_all_tied_compat_20261026"


def _record_file(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    before = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size, after.st_mtime_ns, after.st_ino
    ):
        raise RuntimeError("A compatibility provenance file changed while hashing")
    return {"path": str(path), "bytes": after.st_size, "sha256": digest}


def get_state(modern_module: Any) -> dict[str, Any]:
    """Return an independent, JSON-serializable snapshot of live patch state."""
    state = getattr(modern_module.mannwhitneyu, _STATE_ATTRIBUTE, None)
    if state is None:
        raise RuntimeError("The all-tied MWU compatibility wrapper is not installed")
    return copy.deepcopy(state)


def install(modern_module: Any) -> dict[str, Any]:
    """Install once on one evaluator module; return live provenance/counters.

    The returned mapping is shared with the wrapper, so its ``counters`` reflect
    later calls. Repeated installation returns that same mapping and does not
    add another wrapper. SciPy's global function binding remains unchanged.
    """
    original = modern_module.mannwhitneyu
    existing = getattr(original, _STATE_ATTRIBUTE, None)
    if existing is not None:
        return existing
    signature = inspect.signature(original)
    counters = dict(calls=0, scope_calls=0, all_tied_columns=0,
                    nonfinite_all_tied_repaired=0, other_nonfinite_returned=0,
                    exceptions_propagated=0)
    state = dict(
        schema="PIE_frozen_evaluator_all_tied_MWU_runtime_compat_v1",
        rule="finite pooled-identical observations: U=n*m/2 and p=1; remain eligible for BH",
        scope="axis=0, two-sided, asymptotic, continuity=True, nan_policy=propagate; 1D or equal-width 2D arrays",
        frozen_evaluator=_record_file(modern_module.__file__),
        compatibility_code=_record_file(__file__),
        original_callable=dict(module=original.__module__, name=original.__qualname__,
                               implementation=_record_file(inspect.getfile(original))),
        source_files_modified=False,
        counters=counters,
    )

    @functools.wraps(original)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        counters["calls"] += 1
        try:
            result = original(*args, **kwargs)
        except Exception:
            counters["exceptions_propagated"] += 1
            raise
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        params = bound.arguments
        if not (params.get("axis") == 0
                and params.get("alternative") == "two-sided"
                and params.get("method") == "asymptotic"
                and params.get("use_continuity") is True
                and params.get("nan_policy", "propagate") == "propagate"):
            return result
        x, y = np.asarray(params["x"]), np.asarray(params["y"])
        if (x.ndim not in (1, 2) or y.ndim != x.ndim
                or x.shape[1:] != y.shape[1:] or not len(x) or not len(y)
                or not np.issubdtype(x.dtype, np.number)
                or not np.issubdtype(y.dtype, np.number)):
            return result
        counters["scope_calls"] += 1
        all_tied = (np.isfinite(x).all(axis=0) & np.isfinite(y).all(axis=0)
                    & (x == x[0]).all(axis=0) & (y == x[0]).all(axis=0))
        original_p = np.asarray(result.pvalue)
        mask = np.asarray(all_tied)
        if params.get("keepdims", False):
            mask = np.expand_dims(mask, axis=0)
        counters["other_nonfinite_returned"] += int((~np.isfinite(original_p) & ~mask).sum())
        if not np.any(all_tied):
            return result
        counters["all_tied_columns"] += int(np.asarray(all_tied).sum())
        counters["nonfinite_all_tied_repaired"] += int((~np.isfinite(original_p) & mask).sum())
        pvalue = np.where(mask, 1.0, original_p)
        statistic = np.where(mask, len(x) * len(y) / 2.0, result.statistic)
        if pvalue.ndim == 0:
            pvalue = pvalue[()]
        if np.asarray(statistic).ndim == 0:
            statistic = np.asarray(statistic)[()]
        # SciPy 1.18's tuple-like result exposes ``_replace = None``; construct
        # the same two-field result type rather than relying on namedtuple APIs.
        return type(result)(statistic, pvalue)

    setattr(guarded, _STATE_ATTRIBUTE, state)
    modern_module.mannwhitneyu = guarded
    return state
