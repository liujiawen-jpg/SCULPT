from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/"src"))
spec = importlib.util.spec_from_file_location("comparison_converter", ROOT/"convert_predictions.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class ConfigurationTests(unittest.TestCase):
    def test_fixed_revisions_and_configuration_have_no_historical_paths(self):
        registry = json.loads((ROOT/"registry.json").read_text())
        self.assertEqual(set(registry), {"txpert","cpa","pie","scpilot"})
        for name,record in registry.items():
            self.assertRegex(record["commit"],r"^[a-f0-9]{40}$")
            text = (ROOT/name/"benchmark_config.json").read_text()
            for machine_path in ("/root/","/mnt/","/tmp/"):
                self.assertNotIn(machine_path,text)
            json.loads(text)

    def test_native_output_types_are_disclosed(self):
        registry = json.loads((ROOT/"registry.json").read_text())
        self.assertEqual(registry["pie"]["output"],"mean_only")
        self.assertEqual(registry["txpert"]["output"],"mean_only")
        self.assertIn("gaussian",registry["cpa"]["output"])
        self.assertIn("signed",registry["scpilot"]["output"])


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.checkpoint = self.path/"selected.ckpt"
        self.checkpoint.write_bytes(b"synthetic checkpoint identity only")
        self.genes = ["g0","g1","g2"]
        self.targets = ["p0","p1"]

    def convert(self,method,source,**kwargs):
        return converter.convert(source,self.path/"predictions.h5",method=method,
            environment="test",genes=self.genes,targets=self.targets,
            checkpoint=self.checkpoint,fold="F1",seed=17,**kwargs)

    def test_mean_only_export_is_sealed_and_unchanged(self):
        mean = np.arange(6,dtype="f4").reshape(2,3)
        src = self.path/"native.npz"; np.savez(src,mean=mean)
        output = self.convert("pie",src)
        with h5py.File(output) as h:
            self.assertTrue(h.attrs["sealed"])
            self.assertTrue(h.attrs["complete"])
            self.assertEqual(h.attrs["fold"],"F1")
            self.assertEqual(h.attrs["seed"],17)
            np.testing.assert_array_equal(h["mean"][:],mean)
            self.assertNotIn("support",h)
        seal = json.loads(output.with_suffix(".seal.json").read_text())
        self.assertEqual(seal["purpose"],"test")
        self.assertEqual(seal["allowed"],{"test":self.targets})
        with self.assertRaises(FileExistsError):
            self.convert("pie",src)

    def test_mean_only_methods_reject_invented_cloud(self):
        cloud = np.zeros((2,512,3),dtype="f4")
        src = self.path/"native.npz"; np.savez(src,support=cloud,mean=cloud.mean(1))
        for name in ("pie","txpert"):
            with self.assertRaises(ValueError):
                self.convert(name,src,support_semantics="iid_samples")

    def test_native_signed_cloud_is_retained(self):
        cloud = np.random.default_rng(3).normal(size=(2,512,3)).astype("f4")
        src = self.path/"native.npz"; np.savez(src,support=cloud,mean=cloud.astype("f8").mean(1))
        output = self.convert("scpilot",src,support_semantics="iid_samples")
        with h5py.File(output) as h:
            np.testing.assert_array_equal(h["support"][:],cloud)
            self.assertTrue((h["support"][:]<0).any())

    def test_mismatched_cloud_mean_refused(self):
        src = self.path/"native.npz"
        np.savez(src,support=np.ones((2,512,3),dtype="f4"),mean=np.zeros((2,3),dtype="f4"))
        with self.assertRaises(AssertionError):
            self.convert("cpa",src,support_semantics="iid_samples")

    def test_nonfinite_prediction_refused(self):
        src = self.path/"native.npz"
        np.savez(src,mean=np.full((2,3),np.nan,dtype="f4"))
        with self.assertRaises(ValueError):
            self.convert("txpert",src)


class TrainingExportTests(unittest.TestCase):
    def test_native_export_never_reads_held_numeric_values(self):
        try:
            import anndata as ad
        except ImportError:
            self.skipTest("anndata optional comparison input dependency is not installed")
        source_spec = importlib.util.spec_from_file_location("native_training_export",ROOT/"export_training.py")
        exporter = importlib.util.module_from_spec(source_spec)
        source_spec.loader.exec_module(exporter)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data = root/"data"; split = root/"splits"/"F1"
            data.mkdir(); split.mkdir(parents=True)
            (split/"FOLD.json").write_text(json.dumps(dict(train=["A"],validation="B",test="C",
                joint_holdout=dict(held_targets={"A":["held"]}))))
            (split/"TRAINING_TARGETS.json").write_text(json.dumps(dict(allowed_targets={"A":["allowed"]})))
            with h5py.File(data/"A.h5","w") as h:
                h.attrs["genes_json"] = json.dumps(["g0","g1","g2"])
                h.attrs["targets_json"] = json.dumps(["allowed","held"])
                h.create_dataset("controls",data=np.ones((2,3),dtype="f4"))
                h.create_dataset("control_mean",data=np.ones(3,dtype="f4"))
                p = h.create_group("perturbations")
                p.create_dataset("allowed",data=np.full((4,3),2,dtype="f4"))
                # Reading this held condition would cause exporter to fail its
                # finite-input check; neither B nor C files exist at all.
                p.create_dataset("held",data=np.full((7,3),np.nan,dtype="f4"))
            output = root/"train.h5ad"
            audit = exporter.export(data,root/"splits","F1",output)
            self.assertEqual(audit["cells"],6)
            self.assertEqual(audit["held_perturbation_numeric_reads"],0)
            native = ad.read_h5ad(output)
            self.assertEqual(native.shape,(6,3))
            self.assertEqual(set(native.obs["perturbation"]),{"allowed","control"})
            self.assertTrue(np.isfinite(native.X).all())
            np.testing.assert_array_equal(native.uns["control_mean_by_environment"]["A"],np.ones(3))


if __name__ == "__main__":
    unittest.main()
