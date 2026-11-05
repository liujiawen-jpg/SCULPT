"""Generate, seal, evaluate and aggregate two parameter-free baselines.

Source-Mean is mean-only. Target-Control has an analytic full-pooled control
mean AND an unchanged finite empirical cloud; their means need not be equal.
This standalone evaluator preserves that deliberate historical distinction.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import h5py
import numpy as np
import torch

from cosit.data import CellStore
from cosit.io import read_json, write_json, sha256
from cosit.prediction import seal_predictions, verify_prediction_seal
from .definitions import source_mean, target_control, deterministic_support

HERE = Path(__file__).resolve().parent
METHODS = ("target_control", "source_mean")


def _sanitize(value):
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _avg(values):
    finite = [float(x) for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(finite)) if finite else None


@torch.no_grad()
def generate(data, splits, fold_id, output):
    """No query perturbations are read; both methods sealed before evaluation."""
    fold_path = Path(splits) / fold_id / "FOLD.json"
    fold = read_json(fold_path)
    if fold.get("id") != fold_id or len(fold["train"]) != 3:
        raise ValueError("Expected one registered three-source fold")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh output directory; existing predictions are never overwritten")
    output.mkdir(parents=True)
    algorithm = output / "ALGORITHM.json"
    write_json(algorithm, dict(schema="parameter_free_algorithm_lock_v1", trainable_parameters=0,
        source_files={p.name: sha256(p) for p in [HERE / "definitions.py", HERE / "run.py", HERE / "benchmark_config.json"]},
        configuration=read_json(HERE / "benchmark_config.json"), fold=fold, fold_sha256=sha256(fold_path),
        data_manifest_sha256=sha256(Path(data) / "MANIFEST.json")))
    store = CellStore(data, fold, role="train", evaluation_source=True)
    env = fold["test"]
    sources = tuple(fold["train"])
    paths = []
    try:
        control_mean = store.get_e_mean(env)
        controls = store.get_e(env)
        for method in METHODS:
            destination = output / (method + ".h5")
            partial = destination.with_suffix(".partial.h5")
            with h5py.File(partial, "w") as h:
                h.attrs.update(schema="parameter_free_predictions_v1", method=method, fold=fold_id,
                    environment=env, genes_json=json.dumps(store.genes), targets_json=json.dumps(store.targets),
                    source_environments_json=json.dumps(sources), expression_units="anchor2048_log1p_per10000_v1",
                    training_seed_applicable=False, query_perturbation_reads=0, complete=False,
                    response_mean_definition="full_pooled_control" if method == "target_control" else "equal_source_absolute_expression")
                means = h.create_dataset("mean", (len(store.targets), len(store.genes)), dtype="f4")
                h.create_dataset("control_mean", data=control_mean)
                if method == "target_control":
                    result = target_control(controls, control_mean)
                    means[:] = np.broadcast_to(result["analytic_mean"], means.shape)
                    h.create_dataset("support", data=result["support"], compression="lzf")
                    h.create_dataset("support_mean", data=result["support_mean"])
                    h.attrs.update(support_shared_across_targets=True,
                        support_semantics="unchanged target control whole-cell rows, deterministic midpoint selection",
                        analytic_support_mean_rms=result["analytic_support_mean_rms"],
                        analytic_support_mean_max_abs=result["analytic_support_mean_max_abs"])
                else:
                    for i, target in enumerate(store.targets):
                        store.authorize_episode(env, sources)
                        try:
                            observations = [store.get_p(source, target, "source") for source in sources]
                        finally:
                            store.clear_episode()
                        means[i] = source_mean(observations).astype(np.float32)
                h.attrs["complete"] = True
            partial.replace(destination)
            paths.append(destination)
        seal_predictions(algorithm, paths, {env: store.targets}, output / "SEAL.json", purpose="test")
        write_json(output / "GENERATION_AUDIT.json", dict(fold=fold_id, environments=sources,
            query_perturbation_reads=0, source_read_counts=dict(store.audit), models_trained=0,
            prediction_files=[dict(file=p.name, sha256=sha256(p)) for p in paths],
            algorithm_sha256=sha256(algorithm), seeds=[]))
        return paths
    finally:
        store.close()


def _response_metrics(mean, truth, control, targets, genes, deg):
    from cosit.core.metrics import response_metrics
    r = response_metrics(mean - control, truth - control, targets, genes, compute_pds=True)
    names = ("PDS_delta", "Pearson", "Spearman", "JAC20", "JAC50", "JAC100",
             "signed_top100_coverage", "paired_raw_cosine", "wrong_pair_raw_cosine",
             "raw_cosine_margin", "RMS_amplitude_ratio_macro", "direction_accuracy_nonzero_truth")
    summary = {key: r["aggregate"].get(key) for key in names}
    summary["DEG_AP"] = _avg([x.get("DEG_AP") for x in deg])
    per = [dict(row, **d) for row, d in zip(r["per_condition"], deg)]
    return summary, per


@torch.no_grad()
def evaluate(data, directory, assets, *, device="cpu"):
    """Score shared empirical controls without enforcing an artificial shift."""
    from scipy.spatial.distance import pdist
    from cosit.assets import verify_fitted_assets
    from cosit.modern_metrics import build_deg_reference, score_deg_reference, gaussian_mmd2_unbiased, energy_distance_normalized
    from cosit.core.metrics import distribution_metrics

    directory, assets = Path(directory), Path(assets)
    if (directory / "EVALUATED.json").exists():
        raise FileExistsError("Evaluation already exists")
    verify_fitted_assets(assets)
    binding = read_json(assets / "BINDING.json")
    algorithm = read_json(directory / "ALGORITHM.json")
    current_data_sha256 = sha256(Path(data) / "MANIFEST.json")
    if algorithm.get("data_manifest_sha256") != current_data_sha256:
        raise PermissionError("Prepared data manifest differs from the frozen prediction algorithm")
    if binding.get("data_manifest_sha256") != current_data_sha256:
        raise PermissionError("Prepared data manifest differs from frozen evaluation assets")
    fold = binding["fold"]
    if algorithm.get("fold") != fold:
        raise PermissionError("Prediction fold differs from frozen evaluation geometry")
    paths = [directory / (method + ".h5") for method in METHODS]
    store = CellStore(data, fold, role="evaluation")
    handles = []
    try:
        for path, method in zip(paths, METHODS):
            h = h5py.File(path, "r"); handles.append(h)
            if (not h.attrs.get("complete") or h.attrs.get("schema") != "parameter_free_predictions_v1"
                    or h.attrs.get("method") != method or h.attrs.get("fold") != fold["id"]
                    or h.attrs.get("environment") != fold["test"]
                    or json.loads(h.attrs["source_environments_json"]) != fold["train"]
                    or json.loads(h.attrs["genes_json"]) != store.genes
                    or json.loads(h.attrs["targets_json"]) != store.targets):
                raise ValueError("Baseline identity/axes do not match metric geometry")
        seal = directory / "SEAL.json"
        verify_prediction_seal(directory / "ALGORITHM.json", paths,
            {fold["test"]: store.targets}, seal, purpose="test")
        store.unlock(seal)
        controls = store.get_e(fold["test"])
        control = store.get_e_mean(fold["test"])
        means = {m: np.array(h["mean"]) for m, h in zip(METHODS, handles)}
        np.testing.assert_array_equal(means["target_control"], np.broadcast_to(control, means["target_control"].shape))
        for h in handles:
            np.testing.assert_array_equal(h["control_mean"][:], control)
        cloud = np.array(handles[0]["support"])
        np.testing.assert_array_equal(cloud, deterministic_support(controls))
        np.testing.assert_allclose(cloud.mean(0, dtype=np.float64), handles[0]["support_mean"][:], rtol=0, atol=0)
        if "support" in handles[1]:
            raise ValueError("Source-Mean must remain a mean-only predictor")
        with np.load(assets / "statistics/control_geometry.npz", allow_pickle=False) as a:
            geometry = {k: np.array(a[k]) for k in a.files}
        scale = geometry["scale"]
        train = np.concatenate([store.get_e(e) / scale for e in fold["train"]])
        selected = np.random.default_rng(17).choice(len(train), min(256, len(train)), replace=False)
        distances = pdist(train[selected].astype(np.float64))
        base = max(float(np.median(distances[distances > 0])) if (distances > 0).any() else 1., 1e-6)
        bandwidths = [base * .5, base, base * 2]
        tensor = lambda x: torch.as_tensor(np.array(x, copy=True), dtype=torch.float32, device=device)
        truth, distribution = [], []
        deg = {m: [] for m in METHODS}
        for i, target in enumerate(store.targets):
            real = store.get_p(fold["test"], target, "test")
            truth.append(real.mean(0))
            own = store.genes.index(target) if target in store.genes else None
            reference = build_deg_reference(real, controls, exclude_index=own)
            for method in METHODS:
                deg[method].append(score_deg_reference(reference, means[method][i] - control))
            d = distribution_metrics(tensor(cloud), tensor(real), tensor(scale), tensor(geometry["eval_projections"]),
                pca_components=tensor(geometry["components"]),
                pca_offset=tensor(control / scale + geometry["pca_mean"]),
                cluster_centers=tensor(geometry["cluster_centers"]))
            for key in ("absolute_mean_MSE", "absolute_mean_MAE", "absolute_mean_RMSE", "legacy_distribution_composite_not_for_selection"):
                d.pop(key, None)
            d.update(gaussian_mmd2_unbiased(real, cloud, scale, bandwidths, fold["test"] + "|" + target))
            d.update(energy_distance_normalized(real, cloud, scale, fold["test"] + "|" + target))
            distribution.append(dict(target=target, **d))
        truth = np.stack(truth)
        output_files = []
        for method in METHODS:
            response, per = _response_metrics(means[method], truth, control, store.targets, store.genes, deg[method])
            rows = distribution if method == "target_control" else []
            keys = set().union(*(r.keys() for r in rows)) if rows else set()
            dist_summary = {k: _avg([r.get(k) for r in rows]) for k in keys if k != "target"
                and all(r.get(k) is None or isinstance(r.get(k), (int, float, np.number)) for r in rows)}
            record = dict(schema="parameter_free_metrics_v1", method=method, fold=fold["id"], environment=fold["test"],
                seed=None, trainable_parameters=0, conditions=len(store.targets), genes=len(store.genes),
                response=response, distribution=dist_summary, distribution_available=bool(rows),
                per_condition_response=per, per_condition_distribution=rows,
                prediction_sha256=sha256(directory / (method + ".h5")),
                algorithm_sha256=sha256(directory / "ALGORITHM.json"), mmd_bandwidths=bandwidths,
                response_mean_semantics="exact analytic absolute expression",
                distribution_mean_may_differ=method == "target_control")
            path = directory / (method + ".metrics.json")
            if path.exists():
                raise FileExistsError("Refusing to overwrite metric output")
            write_json(path, _sanitize(record)); output_files.append(path)
        write_json(directory / "EVALUATED.json", dict(status="completed", files=[
            dict(file=p.name, sha256=sha256(p)) for p in output_files]))
        return output_files
    finally:
        for h in handles: h.close()
        store.close()


def aggregate(paths, output):
    """Exactly five independent fold values; no artificial seed replicates."""
    from cosit.splits import FOLDS
    records = [read_json(p) for p in paths]
    expected = {f["id"]: f["test"] for f in FOLDS}
    if len(records) != 5 or {r["fold"] for r in records} != set(expected):
        raise ValueError("Provide each of five folds exactly once")
    methods = {r["method"] for r in records}
    if len(methods) != 1 or not methods <= set(METHODS):
        raise ValueError("Aggregate only one parameter-free method at a time")
    if any(r.get("seed") is not None or r["environment"] != expected[r["fold"]] for r in records):
        raise ValueError("Baseline folds cannot have training seed replicas or different contexts")
    result = dict(schema="parameter_free_summary_v1", method=records[0]["method"], folds=records,
        aggregation="equal five-fold mean; sample variance/SD ddof=1; no seed axis", macro={})
    for family in ("response", "distribution"):
        keys = set().union(*(r[family].keys() for r in records))
        result["macro"][family] = {}
        for key in keys:
            values = [r[family].get(key) for r in records]
            values = [float(x) for x in values if x is not None and np.isfinite(x)]
            result["macro"][family][key] = dict(mean=_avg(values), n_valid=len(values),
                variance=float(np.var(values, ddof=1)) if len(values) > 1 else None,
                std=float(np.std(values, ddof=1)) if len(values) > 1 else None)
    if Path(output).exists(): raise FileExistsError("Existing summary is read-only")
    write_json(output, _sanitize(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("generate"); p.add_argument("--data", required=True); p.add_argument("--splits", required=True)
    p.add_argument("--fold", required=True, choices=[f"F{i}" for i in range(1, 6)]); p.add_argument("--output", required=True)
    p = sub.add_parser("evaluate"); p.add_argument("--data", required=True); p.add_argument("--directory", required=True)
    p.add_argument("--assets", required=True); p.add_argument("--device", default="cpu")
    p = sub.add_parser("aggregate"); p.add_argument("--metrics", nargs=5, required=True); p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "generate": generate(args.data, args.splits, args.fold, args.output)
    elif args.command == "evaluate": evaluate(args.data, args.directory, args.assets, device=args.device)
    else: aggregate(args.metrics, args.output)


if __name__ == "__main__": main()
