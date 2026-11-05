"""Fitted PCA, scales and connection coordinates cannot change on restore."""
import numpy as np
import pytest
from cosit.assets import FITTED_ASSET_FILES,prepare_assets,verify_fitted_assets
from cosit.io import sha256,write_json


@pytest.fixture
def fitted_assets(tmp_path):
    for relative in FITTED_ASSET_FILES:
        path=tmp_path/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez(path,coordinate=np.arange(8,dtype='f4'))
    binding={'fitted_assets_sha256':{name:sha256(tmp_path/name) for name in FITTED_ASSET_FILES}}
    write_json(tmp_path/'BINDING.json',binding)
    return tmp_path,binding


def test_untouched_fitted_assets_verify(fitted_assets):
    path,binding=fitted_assets
    assert verify_fitted_assets(path)==binding['fitted_assets_sha256']


@pytest.mark.parametrize('relative',FITTED_ASSET_FILES)
def test_changed_fitted_asset_refused_before_any_data_read(fitted_assets,relative):
    path,_=fitted_assets
    # Appending bytes leaves an NPZ readable; checksum verification must still
    # reject the modification instead of trusting a successful array load.
    with (path/relative).open('ab') as stream:stream.write(b'changed')
    with np.load(path/relative) as archive:assert archive['coordinate'].shape==(8,)
    with pytest.raises(ValueError,match='Fitted statistic asset changed'):
        prepare_assets('absent-data','absent-prior','absent-graph','absent-axis',{},path,'cpu',{},fit=False)


def test_incomplete_fitted_binding_is_not_accepted(fitted_assets):
    path,binding=fitted_assets
    binding['fitted_assets_sha256'].pop(FITTED_ASSET_FILES[0])
    with pytest.raises(ValueError,match='incomplete fitted asset binding'):
        verify_fitted_assets(path,binding)


def test_same_axis_replacement_data_is_rejected_before_scoring(fitted_assets, monkeypatch):
    """A valid new manifest cannot silently substitute different test values."""
    import h5py
    from cosit.smoke import synthetic_inputs
    from cosit.io import read_json
    import cosit.evaluation as evaluation

    directory, binding = fitted_assets
    inputs = synthetic_inputs(directory / 'synthetic_inputs')
    data = inputs['data']
    manifest = data / 'MANIFEST.json'
    binding['data_manifest_sha256'] = sha256(manifest)
    write_json(directory / 'BINDING.json', binding)
    fold = read_json(inputs['splits'] / 'F1/FOLD.json')
    path = data / (fold['test'] + '.h5')
    # Modify response values while retaining every gene, target and matrix axis.
    with h5py.File(path, 'r+') as h:
        target = next(iter(h['perturbations']))
        original_shape = h['perturbations'][target].shape
        h['perturbations'][target][0, 0] += 1.
        assert h['perturbations'][target].shape == original_shape
    updated = read_json(manifest)
    updated['contexts'][fold['test']]['sha256'] = sha256(path)
    write_json(manifest, updated)

    def forbidden_store(*args, **kwargs):
        pytest.fail('A mismatched dataset reached the label store')
    monkeypatch.setattr(evaluation, 'CellStore', forbidden_store)
    with pytest.raises(PermissionError, match='prepared data differs'):
        evaluation.evaluate(data, 'unopened-predictions.h5', 'unopened-checkpoint.pt',
            directory, directory / 'must_not_exist.json')
    assert not (directory / 'must_not_exist.json').exists()


def test_missing_dataset_binding_is_not_accepted_for_scoring(fitted_assets):
    from cosit.evaluation import evaluate
    directory, _ = fitted_assets
    # Missing binding must fail closed, before opening the supplied data path.
    with pytest.raises(PermissionError, match='prepared data differs'):
        evaluate('unopened-data', 'unopened-predictions.h5', 'unopened-checkpoint.pt',
            directory, directory / 'must_not_exist.json')
