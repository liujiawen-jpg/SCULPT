"""Convert native mean/cloud arrays into the shared sealed prediction format.

This bridge never invents a cell population for a mean-only method. It performs
no normalization, clipping, calibration, noise injection, or model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def convert(source, output, *, method, environment, genes, targets, checkpoint, fold, seed,
            support_semantics=None, seal=None, purpose="test"):
    from cosit.prediction import seal_predictions
    registry = json.loads((Path(__file__).parent / "registry.json").read_text())
    config = registry[method]
    if fold not in {f"F{i}" for i in range(1,6)} or seed not in (17,29,43):
        raise ValueError("Formal benchmark requires F1..F5 and initialization seed17/29/43")
    genes, targets = list(genes), list(targets)
    if len(set(genes)) != len(genes) or len(set(targets)) != len(targets):
        raise ValueError("Gene and target axes must be unique and ordered")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a previously sealed prediction file")
    seal = Path(seal) if seal is not None else output.with_suffix(".seal.json")
    if seal.exists():
        raise FileExistsError("Refusing to overwrite an existing evaluation seal")
    if purpose not in ("validation","test"):
        raise ValueError("Invalid evaluation purpose")
    with np.load(source, allow_pickle=False) as native:
        if "mean" not in native:
            raise ValueError("Native export needs an absolute-expression mean array")
        mean = np.asarray(native["mean"], dtype=np.float32)
        if mean.shape != (len(targets), len(genes)) or not np.isfinite(mean).all():
            raise ValueError("Invalid native prediction mean or mismatched axes")
        support = native["support"] if "support" in native else None
        if config["output"] == "mean_only" and support is not None:
            raise ValueError("This method does not have a native single-cell generator")
        if support is not None:
            if support.ndim != 3 or support.shape[0] != len(targets) or support.shape[2] != len(genes):
                raise ValueError("Cloud must be targets x generated cells x genes")
            if support.shape[1] != 512 or not np.isfinite(support).all():
                raise ValueError("Formal benchmark requires 512 finite generated cells")
            np.testing.assert_allclose(support.astype(np.float64).mean(1), mean,
                                       rtol=2e-5, atol=2e-6)
            if support_semantics is None:
                raise ValueError("Declare the native support semantics explicitly")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".partial")
        with h5py.File(temporary, "w") as handle:
            handle.attrs.update(schema="cosit_predictions_v1", method=method,
                environment=environment, genes_json=json.dumps(genes),
                fold=fold,fold_id=fold,seed=int(seed),
                targets_json=json.dumps(targets), expression_units="anchor2048_log1p_per10000_v1",
                checkpoint_sha256=digest(checkpoint), source_export_sha256=digest(source),
                upstream_commit=config["commit"], sealed=False, complete=False,
                distribution_targets_json=json.dumps(targets if support is not None else []),
                generated_cells=512 if support is not None else 0,
                native_output_type=config["output"])
            handle.create_dataset("mean", data=mean)
            if support is not None:
                handle.attrs["support_semantics"] = support_semantics
                handle.create_dataset("support", data=support, chunks=(1,512,min(256,len(genes))),
                                      compression="lzf")
            handle.attrs["sealed"] = True
            handle.attrs["complete"] = True
        temporary.replace(output)
    seal_predictions(checkpoint,[output],{environment:targets},seal,purpose=purpose)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Native .npz with mean and optional support")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=("txpert","cpa","pie","scpilot"))
    parser.add_argument("--environment", required=True)
    parser.add_argument("--fold", required=True, choices=tuple(f"F{i}" for i in range(1,6)))
    parser.add_argument("--seed", required=True, type=int, choices=(17,29,43))
    parser.add_argument("--genes", required=True, type=Path, help="One stable ID per line, no header")
    parser.add_argument("--targets", required=True, type=Path, help="One stable ID per line, no header")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--support-semantics")
    parser.add_argument("--seal",type=Path,help="Defaults to prediction file stem + .seal.json")
    parser.add_argument("--purpose",choices=("validation","test"),default="test")
    args = parser.parse_args()
    print(convert(args.source,args.output,method=args.method,environment=args.environment,
        genes=args.genes.read_text().splitlines(),targets=args.targets.read_text().splitlines(),
        checkpoint=args.checkpoint,fold=args.fold,seed=args.seed,support_semantics=args.support_semantics,
        seal=args.seal,purpose=args.purpose))


if __name__ == "__main__":
    main()
