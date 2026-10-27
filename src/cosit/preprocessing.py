"""Prepare five public raw-count sources for the fixed CoSIT/J0 benchmark.

Preprocessing is a trusted offline operation: it may inspect all raw cells but
never chooses genes, targets, folds, or cell rosters using response magnitude.
Training/validation/test access is enforced by the separate runtime data layer.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import yaml


UNIT = "log1p(10000*counts/sum_A2048_counts)"


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value) if value is not None else ""


def _stable(value):
    value = _text(value).strip().split(".")[0]
    return value if value.startswith("ENSG") and value[4:].isdigit() else ""


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _table(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _vector(obj):
    if isinstance(obj, h5py.Group):
        if not {"categories", "codes"} <= set(obj):
            raise ValueError(f"Unsupported categorical storage: {obj.name}")
        categories = [_text(x) for x in obj["categories"][:]]
        codes = np.asarray(obj["codes"][:], dtype=np.int64)
        if np.any((codes < -1) | (codes >= len(categories))):
            raise ValueError("Malformed categorical codes")
        return np.asarray([categories[i] if i >= 0 else "" for i in codes], dtype=object)
    return np.asarray([_text(x) for x in obj[:]], dtype=object)


def _numeric(obj):
    return np.asarray(_vector(obj), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Cell:
    file: str
    row: int
    uid: str
    batch: str
    guide: str
    target: str
    # Orion's original E/L partition used the sample identifier, while its
    # later capped-cell cache used the Parquet filename. Preserve both hashes.
    partition_uid: str = ""


def _quota(counts, cap):
    counts = {str(k): int(v) for k, v in counts.items() if v > 0}
    total = sum(counts.values())
    if not total:
        return {}
    cap = min(int(cap), total)
    numer = {k: cap * v for k, v in counts.items()}
    result = {k: numer[k] // total for k in counts}
    remaining = cap - sum(result.values())
    for key in sorted(counts, key=lambda k: (-(numer[k] % total), k))[:remaining]:
        result[key] += 1
    return result


def select_roster(cells, context, target, cap=256, seed=17):
    """Frozen metadata-only batch/guide proportional sampling; no expression."""
    if not 1 <= int(cap) <= 256:
        raise ValueError("Cell cap must be between 1 and 256")
    role = "P" if target else "E"
    by_batch = defaultdict(list)
    keys = {}
    for cell in cells:
        if cell.uid in keys:
            raise ValueError(f"Duplicate cell UID: {cell.uid}")
        message = f"distribution-cells-v1|{seed}|{context}|{role}|{target}|{cell.uid}"
        keys[cell.uid] = hashlib.sha256(message.encode()).hexdigest()
        by_batch[cell.batch].append(cell)
    chosen = []
    for batch, batch_quota in _quota({b: len(v) for b, v in by_batch.items()}, cap).items():
        by_guide = defaultdict(list)
        for cell in by_batch[batch]:
            by_guide[cell.guide].append(cell)
        for guide, guide_quota in _quota({g: len(v) for g, v in by_guide.items()}, batch_quota).items():
            chosen.extend(sorted(by_guide[guide], key=lambda c: (keys[c.uid], c.uid))[:guide_quota])
    return sorted(chosen, key=lambda c: (keys[c.uid], c.uid))


def partition_controls(cells, context, orion=False):
    """Reproduce the original E/L split; only E is used as model background."""
    grouped = defaultdict(list)
    for cell in cells:
        grouped[cell.batch].append(cell)
    encoder = []
    for batch, group in grouped.items():
        if orion:
            stratum = f"{context}:{batch}"
            def key(cell):
                message = "v2-seed17\0" + "\0".join(map(str, (17, "EL", stratum, cell.partition_uid)))
                return hashlib.sha256(message.encode()).hexdigest(), cell.partition_uid
            cut = (len(group) + 1) // 2
        else:
            def key(cell):
                return hashlib.sha256(f"1701:{cell.uid}".encode()).hexdigest(), cell.uid
            cut = len(group) // 2
        encoder.extend(sorted(group, key=key)[:cut])
    return encoder


def anchor_normalize(raw, gene_positions, anchor_positions):
    """Return normalized output genes and a valid-denominator mask.

    The denominator uses *all* frozen A2048 anchors, including anchors outside
    the output gene panel. Missing anchors are an error upstream, never zeros.
    """
    raw = np.asarray(raw, dtype=np.float64)
    denominator = raw[:, np.asarray(anchor_positions, dtype=np.int64)].sum(axis=1)
    valid = np.isfinite(denominator) & (denominator > 0)
    output = np.zeros((len(raw), len(gene_positions)), dtype=np.float64)
    output[valid] = np.log1p(raw[valid][:, gene_positions] * (10000.0 / denominator[valid, None]))
    return output, valid


class H5Matrix:
    """Bounded-memory rows from dense H5AD or standard CSR counts layers."""
    def __init__(self, obj):
        self.obj = obj
        self.dense = isinstance(obj, h5py.Dataset)
        self.shape = obj.shape if self.dense else tuple(map(int, obj.attrs["shape"]))
        if not self.dense and _text(obj.attrs.get("encoding-type")) != "csr_matrix":
            raise ValueError("Counts sparse storage must be CSR; convert CSC to CSR once before preparation")

    def read(self, start, stop):
        if self.dense:
            return np.asarray(self.obj[start:stop, :], dtype=np.float64)
        from scipy import sparse
        pointers = np.asarray(self.obj["indptr"][start:stop + 1], dtype=np.int64)
        left, right = int(pointers[0]), int(pointers[-1])
        return sparse.csr_matrix((np.asarray(self.obj["data"][left:right], dtype=np.float64),
                                  np.asarray(self.obj["indices"][left:right], dtype=np.int64),
                                  pointers - left), shape=(stop - start, self.shape[1])).toarray()

    def blocks(self, block_rows):
        for start in range(0, self.shape[0], block_rows):
            stop = min(start + block_rows, self.shape[0])
            yield start, self.read(start, stop)


def _positions(native_ids, genes, anchors, mapping):
    translation = {row["source_feature_id"]: row["gene_id"] for row in mapping}
    locations = defaultdict(list)
    for position, raw_id in enumerate(native_ids):
        canonical = translation.get(_stable(raw_id), _stable(raw_id))
        if canonical:
            locations[canonical].append(position)
    wanted = set(genes) | set(anchors)
    invalid = {gene: len(locations.get(gene, [])) for gene in wanted if len(locations.get(gene, [])) != 1}
    if invalid:
        raise ValueError(f"Every output gene and A2048 anchor must resolve uniquely; invalid examples: {list(invalid.items())[:12]}")
    return np.asarray([locations[g][0] for g in genes], dtype=np.int64), np.asarray([locations[g][0] for g in anchors], dtype=np.int64)


def _valid_raw(raw):
    finite = np.isfinite(raw).all(axis=1)
    nonnegative = (raw >= 0).all(axis=1)
    integer = (np.abs(raw - np.rint(raw)) <= 1e-5).all(axis=1)
    return finite & nonnegative & integer & (raw.sum(axis=1) > 0)


def _h5_rosters(context, spec, raw_root, genes, anchors, symbols, mapping, prep):
    path = raw_root / spec["path"]
    if spec.get("matrix_semantics") != "raw_counts":
        raise ValueError("Raw integer counts are required; log-normalized inputs must not be normalized twice")
    with h5py.File(path, "r") as handle:
        reader = H5Matrix(handle[spec.get("matrix", "X")])
        native = _vector(handle["var"][spec.get("gene_column", "ensembl_id")])
        gene_positions, anchor_positions = _positions(native, genes, anchors, mapping)
        obs = handle["obs"]
        required = [spec.get("target_column", "gene_id"), spec.get("label_column", "perturbation"), "cell_barcode", "batch", "guide_id", "nperts", "ngenes", "ncounts"]
        missing = set(required) - set(obs)
        if missing:
            raise ValueError(f"{context}: missing metadata columns {sorted(missing)}")
        labels = _vector(obs[spec.get("label_column", "perturbation")])
        ids = _vector(obs[spec.get("target_column", "gene_id")])
        barcodes, batches, guides = (_vector(obs[name]) for name in ("cell_barcode", "batch", "guide_id"))
        nperts, ngenes, ncounts = (_numeric(obs[name]) for name in ("nperts", "ngenes", "ncounts"))
        if len(labels) != reader.shape[0] or len(native) != reader.shape[1]:
            raise ValueError("Source metadata and matrix axes disagree")
        uids = [f"{context}:{barcode}" for barcode in barcodes]
        if len(set(uids)) != len(uids):
            raise ValueError(f"{context}: duplicated cell barcodes require an explicit upstream disambiguation")
        numeric_valid = np.zeros(len(labels), dtype=bool)
        eligible = np.zeros(len(labels), dtype=bool)
        for start, raw in reader.blocks(int(prep["block_rows"])):
            valid = _valid_raw(raw)
            stop = start + len(raw)
            numeric_valid[start:stop] = valid
            eligible[start:stop] = (valid & (np.count_nonzero(raw, axis=1) >= int(prep["min_detected_genes"]))
                                   & (raw[:, anchor_positions].sum(axis=1) > 0)
                                   & (ngenes[start:stop] >= int(prep["min_detected_genes"])) & (ncounts[start:stop] > 0))
        controls, perturbed = [], defaultdict(list)
        target_set = set(symbols.values())
        for row in range(len(labels)):
            if labels[row] == spec.get("control_label", "control") and nperts[row] == 0 and numeric_valid[row]:
                controls.append(Cell(str(path), row, uids[row], str(batches[row]), str(guides[row] or "missing"), ""))
            elif nperts[row] == 1 and labels[row] != spec.get("control_label", "control") and eligible[row]:
                target = _stable(ids[row]) or symbols.get(str(labels[row]), "")
                if target in target_set:
                    perturbed[target].append(Cell(str(path), row, uids[row], str(batches[row]), str(guides[row] or "missing"), target))
        # Original E/L assignment preceded the later minimum-gene/anchor QC.
        encoder = [cell for cell in partition_controls(controls, context) if eligible[cell.row]]
    return encoder, perturbed, {"kind": "h5", "path": path, "matrix": spec.get("matrix", "X"),
                                  "gene_positions": gene_positions, "anchor_positions": anchor_positions,
                                  "native_genes": len(native), "raw_cells": len(labels)}


def _orion_rosters(context, spec, raw_root, genes, anchors, symbols, mapping, prep):
    import pyarrow.parquet as pq
    files = sorted(raw_root.glob(spec["path"]))
    if len(files) != int(spec["expected_shards"]):
        raise ValueError(f"{context}: expected exactly {spec['expected_shards']} original HCT116 shards; got {len(files)}")
    metadata = pq.read_table(raw_root / spec["gene_metadata"]).to_pydict()
    token_ids = np.asarray(metadata["gene_token_id"], dtype=np.int64)
    if len(set(token_ids.tolist())) != len(token_ids) or np.any(token_ids < 0):
        raise ValueError("Orion gene token metadata is ambiguous")
    native_size = int(token_ids.max()) + 1
    native = [""] * native_size
    for token, ensembl in zip(token_ids, metadata["ensembl_id"]):
        native[int(token)] = _text(ensembl)
    gene_positions, anchor_positions = _positions(native, genes, anchors, mapping)
    anchor_flag = np.zeros(native_size, dtype=bool)
    anchor_flag[anchor_positions] = True
    controls, perturbed = [], defaultdict(list)
    total = 0
    columns = ["gene_token_id", "gene_expression", "cell_barcode", "sample", "guide_target", "gene_target", "n_genes_by_counts", "total_counts", "pass_guide_filter"]
    for path in files:
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=int(prep["block_rows"]), columns=columns):
            meta = batch.select(columns[2:]).to_pydict()
            ids, values = batch.column(0), batch.column(1)
            io, vo = ids.offsets.to_numpy(), values.offsets.to_numpy()
            all_ids, all_values = ids.values.to_numpy(), values.values.to_numpy()
            for index in range(batch.num_rows):
                symbol = str(meta["gene_target"][index] or "")
                target = symbols.get(symbol, "")
                control = symbol == "Non-Targeting"
                if not (target or control) or meta["pass_guide_filter"][index] != 1:
                    continue
                if meta["n_genes_by_counts"][index] < int(prep["min_detected_genes"]) or meta["total_counts"][index] <= 0:
                    continue
                guide = str(meta["guide_target"][index] or "")
                if not control and (len(guide.split("|")) != 2 or not all(g.startswith(symbol + "_") for g in guide.split("|"))):
                    continue
                row_ids = np.asarray(all_ids[io[index]:io[index + 1]], dtype=np.int64)
                row_values = np.asarray(all_values[vo[index]:vo[index + 1]], dtype=np.float64)
                if len(row_ids) != len(row_values) or len(set(row_ids.tolist())) != len(row_ids) or np.any((row_ids < 0) | (row_ids >= native_size)):
                    raise ValueError(f"Malformed Orion paired sparse row {path.name}/{offset + index}")
                if not _valid_raw(row_values[None, :])[0] or np.count_nonzero(row_values) < int(prep["min_detected_genes"]) or row_values[anchor_flag[row_ids]].sum() <= 0:
                    continue
                sample = str(meta["sample"][index] or path.stem)
                barcode = str(meta["cell_barcode"][index])
                uid = f"{context}:{path.name}:{barcode}"
                cell = Cell(str(path), offset + index, uid, sample, guide or "missing", target,
                            f"{context}:{sample}:{barcode}")
                (controls if control else perturbed[target]).append(cell)
            offset += batch.num_rows
        total += offset
    encoder = partition_controls(controls, context, orion=True)
    return encoder, perturbed, {"kind": "orion", "files": files, "gene_positions": gene_positions,
                                  "anchor_positions": anchor_positions, "native_genes": native_size, "raw_cells": total}


def _normalized_rows(source, requested, prep):
    """Stream requested rows only into output space; full native values verify QC."""
    gp, ap = source["gene_positions"], source["anchor_positions"]
    by_file = defaultdict(dict)
    for cell in requested:
        if cell.row in by_file[cell.file] and by_file[cell.file][cell.row].uid != cell.uid:
            raise ValueError("One native row has conflicting identities")
        by_file[cell.file][cell.row] = cell
    if source["kind"] == "h5":
        with h5py.File(source["path"], "r") as handle:
            reader = H5Matrix(handle[source["matrix"]])
            for start, raw in reader.blocks(int(prep["block_rows"])):
                rows = [r for r in range(start, start + len(raw)) if r in by_file[str(source["path"])]]
                if not rows:
                    continue
                local = np.asarray(rows) - start
                values, valid = anchor_normalize(raw[local], gp, ap)
                if not valid.all():
                    raise ValueError("Frozen selected cells changed anchor QC between passes")
                for row, vector in zip(rows, values):
                    yield by_file[str(source["path"])][row], vector
    else:
        import pyarrow.parquet as pq
        native_to_output = np.full(source["native_genes"], -1, dtype=np.int64)
        native_to_output[gp] = np.arange(len(gp))
        anchor_flag = np.zeros(source["native_genes"], dtype=bool)
        anchor_flag[ap] = True
        for path in source["files"]:
            rows = by_file[str(path)]
            if not rows:
                continue
            offset = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=int(prep["block_rows"]), columns=["gene_token_id", "gene_expression"]):
                ids, values = batch.column(0), batch.column(1)
                io, vo = ids.offsets.to_numpy(), values.offsets.to_numpy()
                all_ids, all_values = ids.values.to_numpy(), values.values.to_numpy()
                for index in range(batch.num_rows):
                    cell = rows.get(offset + index)
                    if cell is None:
                        continue
                    row_ids = np.asarray(all_ids[io[index]:io[index + 1]], dtype=np.int64)
                    row_values = np.asarray(all_values[vo[index]:vo[index + 1]], dtype=np.float64)
                    den = row_values[anchor_flag[row_ids]].sum()
                    if not np.isfinite(den) or den <= 0:
                        raise ValueError("Frozen Orion selected cells changed anchor QC between passes")
                    positions = native_to_output[row_ids]
                    keep = positions >= 0
                    vector = np.zeros(len(gp), dtype=np.float64)
                    vector[positions[keep]] = np.log1p(row_values[keep] * (10000.0 / den))
                    yield cell, vector
                offset += batch.num_rows


def _write_context(destination, context, encoder, perturbed, source, genes, targets, anchors, prep, provenance):
    cap, seed = int(prep["cell_cap"]), int(prep["selection_seed"])
    missing = {target: len(perturbed.get(target, [])) for target in targets if len(perturbed.get(target, [])) < int(prep["min_cells_per_target"])}
    if missing:
        raise ValueError(f"{context}: fixed targets fail minimum support; examples {list(missing.items())[:12]}")
    if len(encoder) < 32:
        raise ValueError(f"{context}: fewer than 32 eligible E controls")
    controls = select_roster(encoder, context, "", cap, seed)
    selected = {target: select_roster(perturbed[target], context, target, cap, seed) for target in targets}
    selected_uid = {cell.uid: (target, index) for target, cells in selected.items() for index, cell in enumerate(cells)}
    control_uid = {cell.uid: index for index, cell in enumerate(controls)}
    encoder_uid = {cell.uid for cell in encoder}
    wanted = list(encoder) + [cell for cells in selected.values() for cell in cells]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + f".{os.getpid()}.partial")
    total = np.zeros(len(genes), dtype=np.float64)
    square = np.zeros(len(genes), dtype=np.float64)
    detection = np.zeros(len(genes), dtype=np.int64)
    count = 0
    received = Counter()
    strings = h5py.string_dtype("utf-8")
    try:
        with h5py.File(temp, "w") as handle:
            handle.attrs.update(schema="cosit_cells_v1", context=context, units=UNIT,
                                genes_json=json.dumps(genes), targets_json=json.dumps(targets), anchors_json=json.dumps(anchors),
                                control_n_full=len(encoder), cell_cap=cap, selection_seed=seed,
                                provenance_json=json.dumps(provenance, sort_keys=True))
            control_dataset = handle.create_dataset("controls", shape=(len(controls), len(genes)), dtype="f4", compression="lzf")
            handle.create_dataset("control_ids", data=np.asarray([c.uid for c in controls], dtype=object), dtype=strings)
            pgroup = handle.create_group("perturbations")
            idgroup = handle.create_group("perturbation_ids")
            for target, cells in selected.items():
                dataset = pgroup.create_dataset(target, shape=(len(cells), len(genes)), dtype="f4", compression="lzf")
                dataset.attrs["n_full_eligible"] = len(perturbed[target])
                idgroup.create_dataset(target, data=np.asarray([c.uid for c in cells], dtype=object), dtype=strings)
            for cell, vector in _normalized_rows(source, wanted, prep):
                if not np.isfinite(vector).all() or np.any(vector < 0):
                    raise ValueError("Non-finite or negative normalized expression")
                if cell.uid in encoder_uid:
                    total += vector
                    square += vector * vector
                    detection += vector > 0
                    count += 1
                    if cell.uid in control_uid:
                        control_dataset[control_uid[cell.uid]] = vector.astype(np.float32)
                        received["controls"] += 1
                else:
                    target, position = selected_uid[cell.uid]
                    pgroup[target][position] = vector.astype(np.float32)
                    received[target] += 1
            if count != len(encoder) or received["controls"] != len(controls) or any(received[t] != len(selected[t]) for t in targets):
                raise ValueError("Selected cell roster was not completely recovered from raw data")
            mean = total / count
            handle.create_dataset("control_mean", data=mean.astype(np.float32))
            handle.create_dataset("control_variance", data=np.maximum(0, square / count - mean * mean).astype(np.float32))
            handle.create_dataset("control_detection", data=(detection / count).astype(np.float32))
            handle.attrs["complete"] = True
        os.replace(temp, destination)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return {"context": context, "file": destination.name, "sha256": _sha(destination), "genes": len(genes), "targets": len(targets),
            "n_control_full": len(encoder), "n_control_cached": len(controls), "cells_cached": sum(map(len, selected.values())),
            "units": UNIT, "raw_cells": source["raw_cells"]}


def prepare_data(config_path, raw_root, output, contexts=None, overwrite=False):
    """Build portable per-context HDF5 files; never fit an expression encoder.

    Paths in ``axes`` are relative to the catalog directory, and raw paths are
    relative to ``raw_root``. ``contexts`` can prepare a subset without changing
    the frozen common axes. Existing outputs require explicit ``overwrite``.
    """
    config_path, raw_root, output = Path(config_path), Path(raw_root), Path(output)
    config = yaml.safe_load(config_path.read_text())
    base = config_path.parent
    prep = config["preprocessing"]
    if prep["unit"] != UNIT or prep.get("control_partition") != "legacy_E_only":
        raise ValueError("Unsupported normalization or control partition")
    genes = [r["gene_id"] for r in _table(base / config["axes"]["genes"])]
    targets = [r["target_stable_id"] for r in _table(base / config["axes"]["targets"])]
    anchors = [r["gene_id"] for r in _table(base / config["axes"]["anchors"])]
    if len(anchors) != int(prep.get("expected_anchors", 2048)):
        raise ValueError("Frozen A2048 normalization requires exactly 2048 anchors")
    if any(len(axis) != len(set(axis)) for axis in (genes, targets, anchors)):
        raise ValueError("Gene, target, and anchor axes must each be unique")
    symbols = {r["target_symbol"]: r["target_stable_id"] for r in _table(base / config["axes"]["target_symbols"])}
    if set(symbols.values()) != set(targets):
        raise ValueError("Target symbol mapping must cover the fixed target axis")
    selected = list(contexts) if contexts else list(config["datasets"])
    unknown = set(selected) - set(config["datasets"])
    if unknown:
        raise ValueError(f"Unknown contexts: {sorted(unknown)}")
    reports = []
    for context in selected:
        destination = output / f"{context}.h5"
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Refusing to replace prepared data: {destination}")
        spec = config["datasets"][context]
        mapping_path = base / spec.get("gene_mapping", f"mappings/{context}.tsv")
        mapping = _table(mapping_path)
        provenance = {"source_page": spec["source_page"], "source_license": spec["license"], "catalog_sha256": _sha(config_path),
                      "gene_mapping_sha256": _sha(mapping_path), "axes_sha256": {k: _sha(base / v) for k, v in config["axes"].items()},
                      "source_version": spec.get("revision", "Zenodo13350497"), "roster_selection": "metadata_hash_batch_guide_proportional_v1"}
        print(f"Preparing {context}: validating raw counts and metadata", flush=True)
        if spec["format"] == "scperturb_h5ad":
            encoder, perturbed, source = _h5_rosters(context, spec, raw_root, genes, anchors, symbols, mapping, prep)
        elif spec["format"] == "orion_parquet":
            encoder, perturbed, source = _orion_rosters(context, spec, raw_root, genes, anchors, symbols, mapping, prep)
        else:
            raise ValueError(f"Unsupported source format: {spec['format']}")
        reports.append(_write_context(destination, context, encoder, perturbed, source, genes, targets, anchors, prep, provenance))
        print(f"Prepared {context}: {destination}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "MANIFEST.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"schema": "cosit_prepared_data_v1", "contexts": {}}
    previous["contexts"].update({r["context"]: r for r in reports})
    previous["units"] = UNIT
    manifest_path.write_text(json.dumps(previous, indent=2, sort_keys=True) + "\n")
    return previous


prepare = prepare_data
