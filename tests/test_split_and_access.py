"""Frozen protocol identity and label-capability regression tests."""
import csv
import json
from pathlib import Path
import h5py
import numpy as np
import pytest
import torch
from cosit.data import CellStore
from cosit.io import read_json,sha256
from cosit.prediction import seal_predictions
from cosit.splits import CONTEXTS,FOLDS,make_fold,generate_splits
from cosit.core.metrics import internal_pds,distribution_metrics

ROOT=Path(__file__).resolve().parents[1]


def test_generated_five_fold_protocol_matches_all_released_metadata(tmp_path):
    targets=[r['target_stable_id'] for r in csv.DictReader((ROOT/'configs/axes/targets.tsv').open(),delimiter='\t')]
    generate_splits(targets,tmp_path/'splits',read_json(ROOT/'configs/j0.json'),ROOT/'configs/splits')
    for base in FOLDS:
        name=base['id']
        for file in ('FOLD.json','VALIDATION_PANEL.json'):
            assert read_json(tmp_path/'splits'/name/file)==read_json(ROOT/'configs/splits'/name/file)
        generated=read_json(tmp_path/'splits'/name/'TRAINING_TARGETS.json')
        released=read_json(ROOT/'configs/splits'/name/'TRAINING_TARGETS.json')
        assert generated['allowed_targets']==released['allowed_targets']
        assert generated['conditions']==released['conditions_per_epoch']


@pytest.fixture
def protected_store(tmp_path):
    genes=[f'ENSG{i:011d}' for i in range(1,9)];targets=genes[:4]
    fold=make_fold(FOLDS[0],targets,count=1)
    for env in CONTEXTS:
        with h5py.File(tmp_path/(env+'.h5'),'w') as h:
            h.attrs.update(schema='cosit_cells_v1',genes_json=json.dumps(genes),targets_json=json.dumps(targets),control_n_full=3)
            h.create_dataset('controls',data=np.ones((3,8),dtype='f4'))
            h.create_dataset('control_mean',data=np.ones(8,dtype='f4'))
            h.create_dataset('control_variance',data=np.zeros(8,dtype='f4'))
            h.create_dataset('control_detection',data=np.ones(8,dtype='f4'))
            h.create_dataset('control_ids',data=np.array(['a','b','c'],object),dtype=h5py.string_dtype())
            group=h.create_group('perturbations')
            for t in targets:group.create_dataset(t,data=np.ones((2,8),dtype='f4'))
    manifest={'contexts':{env:{'file':env+'.h5','sha256':sha256(tmp_path/(env+'.h5'))} for env in CONTEXTS}}
    (tmp_path/'MANIFEST.json').write_text(json.dumps(manifest))
    return tmp_path,fold


def test_training_store_rejects_internal_holdout_even_on_cache_hit(protected_store):
    path,fold=protected_store;store=CellStore(path,fold)
    env=fold['train'][0];held=fold['joint_holdout']['held_targets'][env][0]
    query=fold['train'][1];store.authorize_episode(query,(env,))
    store._p_cache[(env,held)]=np.ones((2,8),dtype='f4')
    with pytest.raises(PermissionError,match='Internal validation'):store.get_p(env,held,'source')
    store.authorize_episode(env,(query,))
    with pytest.raises(PermissionError,match='Internal validation'):store.get_p(env,held,'supervision')
    with pytest.raises(PermissionError,match='Held response'):store.get_p(fold['test'],store.targets[0],'source')
    assert not store.audit
    store.close()


def test_source_only_inference_cannot_become_gradient_supervision(protected_store):
    path,fold=protected_store;store=CellStore(path,fold,evaluation_source=True)
    env=fold['train'][0];held=fold['joint_holdout']['held_targets'][env][0]
    store.authorize_episode(fold['test'],(env,))
    with pytest.raises(PermissionError,match='no_grad'):store.get_p(env,held,'source')
    with torch.no_grad():assert store.get_p(env,held,'source').shape==(2,8)
    store.authorize_episode(env,(fold['train'][1],))
    with torch.no_grad(),pytest.raises(PermissionError,match='no_grad'):store.get_p(env,held,'supervision')
    store.close()


def test_evaluator_needs_seal_and_rechecks_changed_prediction_on_cache_hit(protected_store):
    path,fold=protected_store;store=CellStore(path,fold,role='evaluation')
    target=store.targets[0];env=fold['test']
    with pytest.raises(PermissionError,match='seal'):store.get_p(env,target,'test')
    checkpoint=path/'checkpoint.pt';checkpoint.write_bytes(b'trusted-local-checkpoint')
    prediction=path/'predictions.h5'
    with h5py.File(prediction,'w') as h:h.attrs['complete']=True
    seal=seal_predictions(checkpoint,[prediction],{env:[target]},path/'seal.json',purpose='test')
    store.unlock(seal)
    assert store.get_p(env,target,'test').shape==(2,8)
    with pytest.raises(PermissionError,match='outside'):store.get_p(env,store.targets[1],'test')
    with prediction.open('ab') as f:f.write(b'changed')
    with pytest.raises(PermissionError,match='changed'):store.get_p(env,target,'test')
    store.close()


def test_zero_predictor_pds_is_midrank_chance_not_zero_cosine():
    rng=np.random.default_rng(17);truth=rng.normal(size=(3,8))
    result=internal_pds(np.zeros_like(truth),truth,['g0','g1','g2'],[f'g{i}' for i in range(8)])
    assert result==[.5,.5,.5]


def test_signed_competitor_metric_requires_explicit_opt_in():
    x=torch.tensor([[-1.,0.],[1.,2.],[2.,3.]])
    y=torch.tensor([[0.,1.],[1.,2.],[2.,3.]])
    with pytest.raises(ValueError,match='nonnegative'):distribution_metrics(x,y,torch.ones(2))
    result=distribution_metrics(x,y,torch.ones(2),allow_signed=True)
    assert np.isfinite(result['marginal_W1_normalized'])
