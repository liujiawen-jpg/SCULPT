"""Synthetic numerical/access tests for the newly released comparators."""
from pathlib import Path
import json
import numpy as np
import h5py
import pytest
import torch

from Compared_methods.naive_baselines.definitions import source_mean, target_control, deterministic_support
from Compared_methods.naive_baselines.run import generate, evaluate, aggregate
from Compared_methods.state.native_api import make_batch, native_loss
from Compared_methods.state.sampling import bootstrap_controls, training_conditions, draw_training_batch
from cosit.data import CellStore
from cosit.io import sha256, read_json, write_json
from cosit.smoke import synthetic_inputs
from cosit.assets import FITTED_ASSET_FILES


def test_source_mean_absolute_expression_not_shift_or_cell_pool():
    source = [np.array([[2., 8.]]), np.full((9, 2), [8., 2.]), np.full((2, 2), [14., 11.])]
    result = source_mean(source)
    np.testing.assert_array_equal(result, [8., 7.])
    assert not np.allclose(result, np.concatenate(source).mean(0))
    assert result.ndim == 1
    with pytest.raises(ValueError): source_mean(source[:2])


def test_target_control_keeps_original_rows_and_exact_pooled_mean():
    x = np.arange(15, dtype=np.float32).reshape(5, 3)
    pooled = np.array([1., 2., 3.], dtype=np.float32)
    result = target_control(x, pooled)
    assert result['support'].shape == (512, 3)
    np.testing.assert_array_equal(result['analytic_mean'], pooled)
    assert result['analytic_support_mean_rms'] > 0
    assert all(any(np.array_equal(row, original) for original in x) for row in result['support'])
    assert not result['mean_matching_transform_applied']
    result['support'][0, 0] = -1
    assert x.min() == 0  # Returned rows cannot overwrite inputs.


def _toy_assets(directory, fold, genes, data):
    directory.mkdir(); (directory / 'statistics').mkdir()
    rng = np.random.default_rng(42)
    projection = rng.normal(size=(genes, 4)); projection /= np.linalg.norm(projection, axis=0)
    np.savez(directory / 'statistics/control_geometry.npz', scale=np.ones(genes, dtype='f4'),
        eval_projections=projection.astype('f4'), components=np.eye(genes, dtype='f4')[:4],
        pca_mean=np.zeros(genes, dtype='f4'), cluster_centers=np.zeros((2,4), dtype='f4'))
    np.savez(directory / 'statistics/response_scaling.npz', synthetic=True)
    np.savez(directory / 'connection_geometry.npz', synthetic=True)
    write_json(directory / 'BINDING.json', dict(fold=fold, data_manifest_sha256=sha256(Path(data)/"MANIFEST.json"), fitted_assets_sha256={
        p:sha256(directory/p) for p in FITTED_ASSET_FILES}))


def test_sealed_baseline_pipeline_and_no_held_response_reads(tmp_path):
    torch.set_num_threads(1)
    inputs = synthetic_inputs(tmp_path / 'inputs')
    out = tmp_path / 'output'
    paths = generate(inputs['data'], inputs['splits'], 'F1', out)
    audit = read_json(out / 'GENERATION_AUDIT.json')
    assert audit['query_perturbation_reads'] == 0 and audit['models_trained'] == 0
    assert audit['source_read_counts'] == {'source_conditions': 36}
    with h5py.File(paths[1]) as h:
        assert 'support' not in h
        assert h.attrs['response_mean_definition'] == 'equal_source_absolute_expression'
    with pytest.raises(FileExistsError): generate(inputs['data'], inputs['splits'], 'F1', out)
    fold = read_json(inputs['splits'] / 'F1/FOLD.json')
    assets = tmp_path / 'assets'; _toy_assets(assets, fold, 48, inputs["data"])
    outputs = evaluate(inputs['data'], out, assets)
    tc, sm = map(read_json, outputs)
    assert tc['seed'] is None and sm['seed'] is None
    assert tc['response']['PDS_delta'] == .5
    assert tc['distribution_available'] is True
    assert sm['distribution_available'] is False and sm['distribution'] == {}
    assert 'MMD2' in tc['distribution']
    assert tc['distribution_mean_may_differ']
    assert len(tc['per_condition_response']) == 12


def test_state_sampler_does_not_admit_internal_holdout(tmp_path):
    inputs = synthetic_inputs(tmp_path / 'inputs')
    fold = read_json(inputs['splits'] / 'F1/FOLD.json')
    store = CellStore(inputs['data'], fold)
    try:
        conditions = training_conditions(store)
        assert len(conditions) == 30  # 3 contexts * 9 legal P + 3 E groups.
        batch = draw_training_batch(store, conditions, [0, len(conditions)-1], np.random.default_rng(17))
        assert batch['ctrl_cell_emb'].shape == (2,512,48)
        assert batch['pert_emb'].shape == (2,512,13)
        assert batch['pert_emb'][1,0,-1] == 1
        e = fold['train'][0]; held = fold['joint_holdout']['held_targets'][e][0]
        with pytest.raises(PermissionError):
            draw_training_batch(store, [(e,held)], [0], np.random.default_rng(17))
    finally: store.close()


def test_state_native_loss_keeps_distribution_gradient_only():
    class FakeOfficial:
        confidence_token = None
        regularization = 0
        use_batch_token = False
        def _compute_distribution_loss(self, pred, target):
            return ((pred-target)**2).mean((1,2))
    x = torch.zeros(2,3,4)
    batch = make_batch(x, torch.tensor([0,2]), 3, target_cells=x)
    prediction = torch.ones_like(x, requires_grad=True)
    loss = native_loss(FakeOfficial(), batch, prediction)
    loss.backward()
    assert loss.item() == 1
    torch.testing.assert_close(prediction.grad, torch.full_like(prediction, 2/prediction.numel()))
    inference = make_batch(x, torch.tensor([0,2]), 3)
    assert 'pert_cell_emb' not in inference


def test_state_bootstrap_is_stable_whole_cell_selection():
    controls = np.arange(80,dtype='f4').reshape(20,4)
    a = bootstrap_controls(controls, seed=17, environment='context', target='gene')
    b = bootstrap_controls(controls, seed=17, environment='context', target='gene')
    np.testing.assert_array_equal(a,b)
    assert a.shape == (512,4)
    assert np.isin(a[:,0],controls[:,0]).all()


def test_baseline_aggregation_has_only_five_fold_values(tmp_path):
    from cosit.splits import FOLDS
    files=[]
    for i, f in enumerate(FOLDS):
        path=tmp_path/(f['id']+'.json');files.append(path)
        write_json(path,dict(method='source_mean',seed=None,fold=f['id'],environment=f['test'],
            response={'PDS_delta':i/10},distribution={}))
    r=aggregate(files,tmp_path/'summary.json')['macro']['response']['PDS_delta']
    assert r['mean']==.2 and r['n_valid']==5
    assert r['variance']==pytest.approx(.025)
    with pytest.raises(ValueError): aggregate(files[:-1]+[files[0]],tmp_path/'bad.json')


@pytest.mark.parametrize("which", ["data", "assets", "fold"])
def test_baseline_refuses_changed_data_or_asset_binding_before_label_read(tmp_path, monkeypatch, which):
    inputs = synthetic_inputs(tmp_path / 'inputs')
    out = tmp_path / 'output'
    generate(inputs['data'], inputs['splits'], 'F1', out)
    fold = read_json(inputs['splits'] / 'F1/FOLD.json')
    assets = tmp_path / 'assets'; _toy_assets(assets, fold, 48, inputs['data'])
    if which == 'data':
        # Same numerical axes and immutable observation files, changed manifest
        # identity: refusing this alone prevents data-source substitution.
        path = inputs['data'] / 'MANIFEST.json'
        manifest = read_json(path); manifest['unregistered_change'] = True
        write_json(path, manifest)
    else:
        path = assets / 'BINDING.json'; binding = read_json(path)
        if which == 'assets': binding['data_manifest_sha256'] = '0' * 64
        else: binding['fold']['train'] = list(reversed(binding['fold']['train']))
        write_json(path, binding)
    reads=[]
    def denied_label_read(*args, **kwargs):
        reads.append(True)
        raise AssertionError('No response labels may be touched before binding checks')
    monkeypatch.setattr(CellStore, 'get_p', denied_label_read)
    with pytest.raises(PermissionError, match='manifest|fold'):
        evaluate(inputs['data'], out, assets)
    assert not reads
    assert not (out / 'EVALUATED.json').exists()
