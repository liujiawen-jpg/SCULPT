"""Stream ONLY authorized fold training conditions into native AnnData input.

This common input bridge does not implement a method's biological label
conversion, optimizer, or inference. It never reads held-out perturbation rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8<<20),b""):
            h.update(block)
    return h.hexdigest()


def export(data_root, splits_root, fold_id, output, *, include_controls=True):
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    data_root, splits_root, output = Path(data_root), Path(splits_root), Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a native training input")
    fold_dir = splits_root / fold_id
    fold = read_json(fold_dir/"FOLD.json")
    allowed = read_json(fold_dir/"TRAINING_TARGETS.json")["allowed_targets"]
    rows, groups, control_means, axes = [], [], {}, None
    for environment in fold["train"]:
        permitted = set(allowed[environment])
        held = set(fold.get("joint_holdout",{}).get("held_targets",{}).get(environment,[]))
        if permitted & held:
            raise PermissionError("Frozen split gives held targets training permission")
        with h5py.File(data_root/f"{environment}.h5","r") as cells:
            genes = json.loads(cells.attrs["genes_json"])
            targets = json.loads(cells.attrs["targets_json"])
            if axes is None:
                axes = genes
            if genes != axes:
                raise ValueError("Processed contexts have different ordered gene axes")
            missing = permitted-set(targets)
            if missing:
                raise ValueError(f"Missing authorized training targets: {sorted(missing)}")
            control_means[environment] = cells["control_mean"][:]
            family = [(t,f"perturbations/{t}") for t in targets if t in permitted]
            if include_controls:
                family.append(("control","controls"))
            for target,dataset in family:
                # Shape inspection and this allowlist precede every numeric P read.
                shape = cells[dataset].shape
                if len(shape)!=2 or shape[1]!=len(genes) or shape[0]<1:
                    raise ValueError("Invalid authorized condition shape")
                groups.append((environment,target,dataset,shape[0]))
                rows.extend((environment,target,target=="control",f"{environment}|{target}|{i}")
                    for i in range(shape[0]))
    obs = pd.DataFrame(rows,columns=["environment","perturbation","control","cell_id"]).set_index("cell_id")
    obs["environment"] = obs["environment"].astype("category")
    obs["perturbation"] = obs["perturbation"].astype("category")
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_suffix(output.suffix+".partial")
    # An empty CSR matrix initializes valid AnnData metadata without allocating
    # the full dense training population in memory. Dense X is then streamed.
    native = ad.AnnData(X=sp.csr_matrix((len(obs),len(axes)),dtype=np.float32),obs=obs,
        var=pd.DataFrame(index=pd.Index(axes,name="gene_id")),
        uns={"normalization":"anchor2048_log1p_per10000_v1", "fold":fold_id,
             "control_mean_by_environment":control_means})
    native.write_h5ad(temporary)
    total = 0
    with h5py.File(temporary,"r+") as dest:
        del dest["X"]
        x = dest.create_dataset("X",shape=(len(obs),len(axes)),dtype="f4",
            chunks=(min(64,len(obs)),min(256,len(axes))),compression="lzf")
        x.attrs.update({"encoding-type":"array","encoding-version":"0.2.0"})
        for environment,target,dataset,count in groups:
            # A second permission check is deliberately before accessing values.
            if target!="control" and target not in set(allowed[environment]):
                raise PermissionError("Unauthorized condition reached numeric reader")
            with h5py.File(data_root/f"{environment}.h5","r") as source:
                for offset in range(0,count,64):
                    block = source[dataset][offset:offset+64]
                    if not np.isfinite(block).all():
                        raise ValueError("Nonfinite native training expression")
                    x[total+offset:total+offset+len(block)] = block
            total += count
    temporary.replace(output)
    audit = dict(schema="native_training_export_v1",fold=fold_id,training_environments=fold["train"],
        conditions=[dict(environment=e,target=t,cells=n) for e,t,_,n in groups],
        perturbed_conditions=sum(t!="control" for _,t,_,_ in groups),cells=total,genes=len(axes),
        held_perturbation_numeric_reads=0,query_perturbation_numeric_reads=0,
        fold_sha256=digest(fold_dir/"FOLD.json"),
        permissions_sha256=digest(fold_dir/"TRAINING_TARGETS.json"),output_sha256=digest(output))
    output.with_suffix(".access.json").write_text(json.dumps(audit,indent=2)+"\n")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True,type=Path)
    parser.add_argument("--splits",required=True,type=Path)
    parser.add_argument("--fold",required=True)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--exclude-controls",action="store_true")
    args = parser.parse_args()
    report = export(args.data,args.splits,args.fold,args.output,include_controls=not args.exclude_controls)
    print(json.dumps({k:v for k,v in report.items() if k!="conditions"},indent=2))


if __name__ == "__main__":
    main()
