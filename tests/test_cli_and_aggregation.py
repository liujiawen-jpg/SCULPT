"""Public CLI discovery and correct context/seed aggregation boundaries."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from cosit.aggregate import aggregate
from cosit.splits import FOLDS


@pytest.mark.parametrize("command",[None,"prepare","audit-data","split","train",
    "predict","evaluate","aggregate","smoke-test"])
def test_public_help_has_no_execution_side_effects(tmp_path,command):
    source_root = Path(__file__).resolve().parents[1]/"src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command_line = [sys.executable,"-m","cosit"]
    if command is not None:
        command_line.append(command)
    command_line.append("--help")
    result = subprocess.run(command_line,cwd=tmp_path,env=env,text=True,
                            capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert "usage:" in result.stdout
    assert list(tmp_path.iterdir())==[]


def _write_runs(directory):
    paths=[]
    for fold_index in range(1,6):
        for seed,offset in zip((17,29,43),(0.,.3,.6)):
            value=dict(fold=f"F{fold_index}",seed=seed,method='CoSIT-J0',environment=FOLDS[fold_index-1]['test'],
                response={"Pearson":fold_index+offset},
                distribution={"W1":2*fold_index+offset})
            path=directory/f"F{fold_index}_seed{seed}.json"
            path.write_text(json.dumps(value))
            paths.append(path)
    return paths


def test_seed_means_precede_equal_context_aggregation(tmp_path):
    paths=_write_runs(tmp_path)
    result=aggregate(paths,tmp_path/"summary.json")
    assert result["complete"]
    np.testing.assert_allclose(result["folds"]["F1"]["response"]["Pearson"]["mean"],1.3)
    np.testing.assert_allclose(result["folds"]["F1"]["response"]["Pearson"]["variance"],.09)
    np.testing.assert_allclose(result["macro"]["response"]["Pearson"]["mean"],3.3)
    # Five context means: [1.3,2.3,3.3,4.3,5.3], sample variance ddof=1.
    np.testing.assert_allclose(result["macro"]["response"]["Pearson"]["variance"],2.5)
    assert result["macro"]["response"]["Pearson"]["n_valid"]==5


def test_primary_aggregate_refuses_missing_seed(tmp_path):
    paths=_write_runs(tmp_path)
    output=tmp_path/"summary.json"
    with pytest.raises(ValueError,match="five folds"):
        aggregate(paths[:-1],output)
    assert not output.exists()


def test_primary_aggregate_refuses_missing_context(tmp_path):
    paths=_write_runs(tmp_path)
    output=tmp_path/"summary.json"
    with pytest.raises(ValueError,match="five folds"):
        aggregate(paths[:-3],output)
    assert not output.exists()


def test_aggregate_refuses_duplicate_fold_seed(tmp_path):
    paths=_write_runs(tmp_path)
    with pytest.raises(ValueError,match="Duplicate"):
        aggregate(paths+[paths[0]],tmp_path/"summary.json")


def test_partial_aggregation_must_be_explicit_and_is_marked(tmp_path):
    paths=_write_runs(tmp_path)
    result=aggregate(paths[:3],tmp_path/"partial.json",require_complete=False)
    assert not result["complete"]
    assert result["macro"]["response"]["Pearson"]["n_valid"]==1
    assert result["macro"]["response"]["Pearson"]["variance"] is None


@pytest.mark.parametrize('field,value,message',[
    ('method','Different method','different methods'),
    ('environment','Different context','different test environments')])
def test_aggregate_rejects_mixed_experiments(tmp_path,field,value,message):
    paths=_write_runs(tmp_path)
    item=json.loads(paths[0].read_text());item[field]=value
    paths[0].write_text(json.dumps(item))
    with pytest.raises(ValueError,match=message):
        aggregate(paths,tmp_path/'invalid.json')


@pytest.mark.parametrize('field',['method','environment'])
def test_aggregate_requires_experiment_identity(tmp_path,field):
    paths=_write_runs(tmp_path)
    for path in paths:
        value=json.loads(path.read_text());value.pop(field)
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):aggregate(paths,tmp_path/'invalid.json')
