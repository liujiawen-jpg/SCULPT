"""Small synthetic source tests; no downloaded datasets or task weights."""
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from cosit.preprocessing import Cell, anchor_normalize, partition_controls, prepare_data, select_roster


def _tsv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(columns)
        writer.writerows(rows)


def _fixture(tmp_path, fmt="scperturb_h5ad"):
    context = "Toy"
    cfgdir = tmp_path / "configs"
    rawroot = tmp_path / "raw"
    rawroot.mkdir()
    genes = [f"ENSG{i:011d}" for i in range(1, 7)]
    targets = genes[:2]
    anchors = [genes[0], genes[-1]]
    _tsv(cfgdir / "axes/genes.tsv", ["index", "gene_id"], list(enumerate(genes[:4])))
    _tsv(cfgdir / "axes/targets.tsv", ["index", "target_stable_id"], list(enumerate(targets)))
    _tsv(cfgdir / "axes/anchors.tsv", ["gene_id"], [[g] for g in anchors])
    _tsv(cfgdir / "axes/symbols.tsv", ["target_symbol", "target_stable_id"], [["ONE", targets[0]], ["TWO", targets[1]]])
    _tsv(cfgdir / "mappings/Toy.tsv", ["source_feature_id", "source_symbol", "gene_id"], [[g, f"SYM{i}", g] for i, g in enumerate(genes)])
    rng = np.random.default_rng(17)
    counts = rng.poisson(3, size=(104, 6)).astype(np.int32) + 1
    labels = ["control"] * 64 + ["ONE"] * 20 + ["TWO"] * 20
    barcodes = [f"cell{i:03d}" for i in range(104)]
    common = {"source_page": "https://example.org/test-only", "license": "test_fixture", "matrix_semantics": "raw_counts", "format": fmt}
    strings = h5py.string_dtype("utf-8")
    if fmt == "scperturb_h5ad":
        common.update(path="input.h5ad", matrix="X", gene_column="ensembl_id", target_column="gene_id", label_column="perturbation", control_label="control")
        with h5py.File(rawroot / "input.h5ad", "w") as handle:
            handle.create_dataset("X", data=counts)
            handle.create_group("var").create_dataset("ensembl_id", data=np.asarray(genes, dtype=object), dtype=strings)
            obs = handle.create_group("obs")
            for key, values in {"perturbation": labels, "gene_id": [""] * 64 + [targets[0]] * 20 + [targets[1]] * 20,
                                "cell_barcode": barcodes, "batch": ["batch1"] * 104,
                                "guide_id": ["NT"] * 64 + ["ONE_1|ONE_2"] * 20 + ["TWO_1|TWO_2"] * 20}.items():
                obs.create_dataset(key, data=np.asarray(values, dtype=object), dtype=strings)
            obs.create_dataset("nperts", data=[0] * 64 + [1] * 40)
            obs.create_dataset("ngenes", data=(counts > 0).sum(1))
            obs.create_dataset("ncounts", data=counts.sum(1))
    else:
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        common.update(path="data/HCT116_Batch*.parquet", gene_metadata="gene_metadata.parquet", expected_shards=1)
        (rawroot / "data").mkdir()
        pq.write_table(pa.table({"gene_token_id": list(range(6)), "ensembl_id": genes, "gene_name": [f"SYM{i}" for i in range(6)]}), rawroot / "gene_metadata.parquet")
        pq.write_table(pa.table({"gene_token_id": [list(range(6))] * 104, "gene_expression": counts.tolist(),
                                 "cell_barcode": barcodes, "sample": ["HCT116_Batch1"] * 104,
                                 "guide_target": ["NT1|NT2"] * 64 + ["ONE_1|ONE_2"] * 20 + ["TWO_1|TWO_2"] * 20,
                                 "gene_target": ["Non-Targeting"] * 64 + ["ONE"] * 20 + ["TWO"] * 20,
                                 "n_genes_by_counts": (counts > 0).sum(1).tolist(), "total_counts": counts.sum(1).tolist(),
                                 "pass_guide_filter": [1] * 104}), rawroot / "data/HCT116_Batch1.parquet")
    config = {"axes": {"genes": "axes/genes.tsv", "targets": "axes/targets.tsv", "anchors": "axes/anchors.tsv", "target_symbols": "axes/symbols.tsv"},
              "preprocessing": {"unit": "log1p(10000*counts/sum_A2048_counts)", "control_partition": "legacy_E_only",
                                "expected_anchors": 2, "cell_cap": 8, "selection_seed": 17, "min_cells_per_target": 20,
                                "min_detected_genes": 1, "block_rows": 13},
              "datasets": {context: common}}
    cfgpath = cfgdir / "datasets.yaml"
    cfgpath.write_text(yaml.safe_dump(config))
    return cfgpath, rawroot, counts, genes, targets


def test_anchor_denominator_uses_genes_outside_output():
    raw = np.array([[1, 2, 97], [0, 3, 0]], dtype=float)
    result, valid = anchor_normalize(raw, [0, 1], [0, 2])
    np.testing.assert_allclose(result[0], np.log1p([10000 / 98, 20000 / 98]))
    assert valid.tolist() == [True, False]


def test_roster_hash_selection_order_invariant():
    cells = [Cell("x", i, f"x:{i}", str(i % 3), str(i % 4), "gene") for i in range(80)]
    forward = select_roster(cells, "ctx", "gene", cap=17)
    backward = select_roster(list(reversed(cells)), "ctx", "gene", cap=17)
    assert forward == backward
    assert len(forward) == 17
    with pytest.raises(ValueError, match="Duplicate"):
        select_roster([cells[0], cells[0]], "ctx", "gene")


def test_control_partition_reproduces_half_and_is_order_invariant():
    cells = [Cell("x", i, f"x:{i}", "b", "NT", "", f"sample:{i}") for i in range(65)]
    h5 = partition_controls(cells, "ctx")
    assert len(h5) == 32
    assert h5 == partition_controls(list(reversed(cells)), "ctx")
    assert len(partition_controls(cells, "ctx", orion=True)) == 33


@pytest.mark.parametrize("fmt", ["scperturb_h5ad", "orion_parquet"])
def test_prepare_roundtrip_preserves_full_control_statistics(tmp_path, fmt):
    config, rawroot, counts, genes, targets = _fixture(tmp_path, fmt)
    output = tmp_path / "prepared"
    manifest = prepare_data(config, rawroot, output)
    assert manifest["contexts"]["Toy"]["n_control_full"] == 32
    with h5py.File(output / "Toy.h5", "r") as handle:
        assert handle.attrs["schema"] == "cosit_cells_v1"
        assert bool(handle.attrs["complete"])
        assert json.loads(handle.attrs["genes_json"]) == genes[:4]
        assert handle["controls"].shape == (8, 4)
        assert handle["perturbations"][targets[0]].shape == (8, 4)
        for uid, row in zip(handle["control_ids"].asstr()[:], handle["controls"][:]):
            native_row = int(uid.rsplit("cell", 1)[1])
            expected = np.log1p(counts[native_row, :4] * (10000.0 / counts[native_row, [0, 5]].sum()))
            np.testing.assert_allclose(row, expected, rtol=1e-6)
        assert not np.allclose(handle["controls"][:].mean(0), handle["control_mean"][:])
        assert handle.attrs["control_n_full"] == 32
    with pytest.raises(FileExistsError):
        prepare_data(config, rawroot, output)


def test_missing_normalization_gene_fails_before_publishing(tmp_path):
    config, rawroot, *_ = _fixture(tmp_path)
    with h5py.File(rawroot / "input.h5ad", "r+") as handle:
        handle["var/ensembl_id"][5] = "ENSG99999999999"
    with pytest.raises(ValueError, match="resolve uniquely"):
        prepare_data(config, rawroot, tmp_path / "prepared")
    assert not (tmp_path / "prepared/Toy.h5").exists()


def test_insufficient_perturbation_support_fails_instead_of_dropping_target(tmp_path):
    config, rawroot, *_ = _fixture(tmp_path)
    with h5py.File(rawroot / "input.h5ad", "r+") as handle:
        handle["obs/nperts"][64] = 2
    with pytest.raises(ValueError, match="minimum support"):
        prepare_data(config, rawroot, tmp_path / "prepared")


def test_explicit_csr_counts_layer_is_used_instead_of_X(tmp_path):
    from scipy import sparse
    config, rawroot, counts, *_ = _fixture(tmp_path)
    with h5py.File(rawroot / "input.h5ad", "r+") as handle:
        handle["X"][:] = 0  # The selected counts layer must supply all values.
        matrix = sparse.csr_matrix(counts)
        group = handle.create_group("layers").create_group("counts")
        group.attrs["encoding-type"] = "csr_matrix"
        group.attrs["shape"] = matrix.shape
        group.create_dataset("data", data=matrix.data)
        group.create_dataset("indices", data=matrix.indices)
        group.create_dataset("indptr", data=matrix.indptr)
    settings = yaml.safe_load(config.read_text())
    settings["datasets"]["Toy"]["matrix"] = "layers/counts"
    config.write_text(yaml.safe_dump(settings))
    prepare_data(config, rawroot, tmp_path / "prepared")
    with h5py.File(tmp_path / "prepared/Toy.h5") as handle:
        assert np.all(handle["controls"][:] > 0)
