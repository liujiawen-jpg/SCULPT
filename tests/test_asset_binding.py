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
