"""Instance-scoped, algebraically equivalent scPILOT adaptive weights.

The audited upstream file remains byte-for-byte intact. Only its four
``exp(-loss) / sum(exp(-loss))`` expressions are replaced in a compiled copy
of ``EGD_model.predict_new``. OT, Leiden, posterior draws, generator calls,
training and NumPy itself retain their original implementations.
"""
from __future__ import annotations

import ast
import contextlib
import functools
import hashlib
import inspect
from pathlib import Path
import textwrap
import types
from typing import Any

import numpy as np


EXPECTED_UPSTREAM_SHA256 = "b8a8ec39c9ac73032427971fddafb42a5cc16f6abffa88f43c8dafe8b5d405a9"
EXPECTED_PREDICT_SOURCE_SHA256 = "e0dfc37d3527b41e12ae33d81f788860c28418041861253e138e6d992e8c3412"
EXPECTED_LOSS_NAMES = ("matching_loss_mtx", "OTloss_mtx", "OTloss_mtx", "OTloss_mtx")
HELPER_ATTRIBUTE = "_scpilot_stable_adaptive_weight_call"
MAX_DIAGNOSTIC_RECORDS = 32
_MISSING = object()


def _loss_array(loss: Any) -> np.ndarray:
    values = np.asarray(loss, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("adaptive losses must be a nonempty one-dimensional array")
    if not np.isfinite(values).all():
        raise FloatingPointError("adaptive losses must all be finite")
    return values


def stable_negative_loss_weights(loss: Any) -> np.ndarray:
    """Compute softmax(-loss) in float64, with no clipping or fallback.

    Dividing all original exponentials by exp(-min(loss)) preserves their
    normalized values. At least one shifted exponential is exactly one.
    A finite extreme difference may overflow to +inf; its exponential is
    correctly rounded to zero and does not affect that guarantee.
    """
    values = _loss_array(loss)
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp(-(values - values.min()))
    return weights / weights.sum(dtype=np.float64)


def _adaptive_loss_name(node: ast.AST) -> str | None:
    """Recognize only the exact upstream numerator/denominator AST pattern."""
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
        return None
    numerator, denominator = node.left, node.right
    if not (
        isinstance(numerator, ast.Call)
        and isinstance(numerator.func, ast.Attribute)
        and isinstance(numerator.func.value, ast.Name)
        and numerator.func.value.id == "np"
        and numerator.func.attr == "exp"
        and len(numerator.args) == 1
        and not numerator.keywords
        and isinstance(numerator.args[0], ast.UnaryOp)
        and isinstance(numerator.args[0].op, ast.USub)
        and isinstance(numerator.args[0].operand, ast.Name)
        and isinstance(denominator, ast.Call)
        and isinstance(denominator.func, ast.Attribute)
        and isinstance(denominator.func.value, ast.Name)
        and denominator.func.value.id == "np"
        and denominator.func.attr == "sum"
        and len(denominator.args) == 1
        and not denominator.keywords
        and ast.dump(numerator, include_attributes=False)
        == ast.dump(denominator.args[0], include_attributes=False)
    ):
        return None
    return numerator.args[0].operand.id


def _rewrite_adaptive_weights(source: str) -> tuple[ast.Module, list[dict[str, Any]]]:
    """Replace exactly four known expressions; reject any pattern drift."""
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise RuntimeError("expected the single official predict_new function")
    if tree.body[0].name != "predict_new" or tree.body[0].decorator_list:
        raise RuntimeError("unexpected official predict_new declaration")
    replacements: list[dict[str, Any]] = []

    class Rewriter(ast.NodeTransformer):
        def visit_BinOp(self, node):
            name = _adaptive_loss_name(node)
            if name is None:
                return self.generic_visit(node)
            replacements.append({"loss_name": name, "relative_line": node.lineno})
            replacement = ast.Call(
                func=ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()),
                                   attr=HELPER_ATTRIBUTE, ctx=ast.Load()),
                args=[ast.Name(id=name, ctx=ast.Load())], keywords=[],
            )
            return ast.copy_location(replacement, node)

    transformed = Rewriter().visit(tree)
    if tuple(item["loss_name"] for item in replacements) != EXPECTED_LOSS_NAMES:
        raise RuntimeError("official adaptive weighting AST must contain exactly four audited expressions")
    return ast.fix_missing_locations(transformed), replacements


def _verified_original():
    from scpilot.egd_model import EGD_model

    original = EGD_model.predict_new
    if original.__module__ != "scpilot.egd_model" or original.__qualname__ != "EGD_model.predict_new":
        raise RuntimeError("official predict_new identity changed")
    upstream_file = Path(inspect.getsourcefile(original)).resolve()
    upstream_sha = hashlib.sha256(upstream_file.read_bytes()).hexdigest()
    if upstream_sha != EXPECTED_UPSTREAM_SHA256:
        raise RuntimeError("official scPILOT source differs from the audited upstream SHA256")
    source = textwrap.dedent(inspect.getsource(original))
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    if source_sha != EXPECTED_PREDICT_SOURCE_SHA256:
        raise RuntimeError("official predict_new source differs from the audited source SHA256")
    return EGD_model, original, source


@functools.lru_cache(maxsize=1)
def _compile_predict_new(original, source):
    tree, replacements = _rewrite_adaptive_weights(source)
    # Exec has separate locals, so the original globals dictionary is reused
    # without adding, replacing or deleting any global name.
    namespace: dict[str, Any] = {}
    filename = str(inspect.getsourcefile(original)) + "::<stable-adaptive-weights>"
    exec(compile(tree, filename, "exec"), original.__globals__, namespace)
    transformed = namespace["predict_new"]
    transformed.__defaults__ = original.__defaults__
    transformed.__kwdefaults__ = original.__kwdefaults__
    transformed.__annotations__ = original.__annotations__.copy()
    functools.update_wrapper(transformed, original)
    return transformed, tuple((x["loss_name"], x["relative_line"]) for x in replacements)


@contextlib.contextmanager
def stable_adaptive_prediction(model):
    """Bind the equivalent predictor only to this instance, then restore it.

    The yielded dictionary records bounded loss/weight diagnostics without
    retaining cell data. Invalid OT losses still fail rather than triggering
    a substitute prediction. The official class and module globals are never
    mutated, and restoration also runs when native inference raises.
    """
    model_class, original, source = _verified_original()
    if not isinstance(model, model_class):
        raise TypeError("stable adaptive prediction requires an official EGD_model instance")
    transformed, replacements = _compile_predict_new(original, source)
    diagnostics = {
        "algorithm": "native_exp_negative_OT_loss_normalization",
        "numerical_implementation": "float64_subtract_group_min_before_exp",
        "upstream_sha256": EXPECTED_UPSTREAM_SHA256,
        "predict_source_sha256": EXPECTED_PREDICT_SOURCE_SHA256,
        "ast_replacements": len(replacements),
        "ast_sites": [{"loss_name": name, "relative_line": line} for name, line in replacements],
        "weight_groups": 0,
        "original_exp_all_zero_groups": 0,
        "loss_minimum": None,
        "loss_maximum": None,
        "original_exp_all_zero_loss_minimum": None,
        "original_exp_all_zero_loss_maximum": None,
        "records_limit": MAX_DIAGNOSTIC_RECORDS,
        "records": [],
        "records_omitted": 0,
        "instance_method_restored": False,
        "clipping_applied": False,
        "fallback_applied": False,
    }

    def recorded_weights(loss):
        values = _loss_array(loss)
        weights = stable_negative_loss_weights(values)
        with np.errstate(over="ignore", under="ignore"):
            original_exp = np.exp(-values)
        all_zero = bool((original_exp == 0).all())
        loss_minimum, loss_maximum = float(values.min()), float(values.max())
        diagnostics["weight_groups"] += 1
        diagnostics["original_exp_all_zero_groups"] += int(all_zero)
        for key, value, reducer in (("loss_minimum", loss_minimum, min),
                                    ("loss_maximum", loss_maximum, max)):
            diagnostics[key] = value if diagnostics[key] is None else reducer(diagnostics[key], value)
        if all_zero:
            for key, value, reducer in (("original_exp_all_zero_loss_minimum", loss_minimum, min),
                                        ("original_exp_all_zero_loss_maximum", loss_maximum, max)):
                diagnostics[key] = value if diagnostics[key] is None else reducer(diagnostics[key], value)
        if len(diagnostics["records"]) < MAX_DIAGNOSTIC_RECORDS:
            diagnostics["records"].append({
                "group_index": diagnostics["weight_groups"],
                "loss_count": int(values.size),
                "loss_dtype": str(values.dtype),
                "loss_minimum": loss_minimum,
                "loss_maximum": loss_maximum,
                "original_exp_all_zero": all_zero,
                "original_exp_all_finite": bool(np.isfinite(original_exp).all()),
                "weight_minimum": float(weights.min()),
                "weight_maximum": float(weights.max()),
                "normalized_weight_sum": float(weights.sum(dtype=np.float64)),
            })
        else:
            diagnostics["records_omitted"] += 1
        return weights

    old_predict = model.__dict__.get("predict_new", _MISSING)
    old_helper = model.__dict__.get(HELPER_ATTRIBUTE, _MISSING)
    try:
        setattr(model, HELPER_ATTRIBUTE, recorded_weights)
        model.predict_new = types.MethodType(transformed, model)
        yield diagnostics
    finally:
        for name, previous in (("predict_new", old_predict), (HELPER_ATTRIBUTE, old_helper)):
            if previous is _MISSING:
                model.__dict__.pop(name, None)
            else:
                setattr(model, name, previous)
        diagnostics["instance_method_restored"] = True
