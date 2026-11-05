"""Scientific label/access checks for the additional portable preparation layer."""
from pathlib import Path
import importlib.util

import numpy as np
import pytest
from scipy.stats import mannwhitneyu

from Compared_methods.preprocess import bh, pie_labels, legal_rows, supervision
from cosit.data import CellStore
from cosit.io import read_json
from cosit.smoke import synthetic_inputs


def test_pie_labels_keep_full_control_mean_and_native_mwu_fc():
    x = np.array([[0, 0., .1], [0, 1., .2], [0, 2., .3], [0, 4., .4]], dtype='f4')
    e = np.array([[0, .1, .3], [0, .2, .1], [0, .3, .2], [0, .4, .1]], dtype='f4')
    pooled = np.array([0, .8, .25])
    r = pie_labels(x, e, pooled)
    np.testing.assert_allclose(r['delta_p'], x.astype('f8').mean(0)-pooled)
    expected = (np.expm1(x.astype('f8')).mean(0)+1e-9)/(np.expm1(e.astype('f8')).mean(0)+1e-9)
    np.testing.assert_allclose(r['fold_changes'][r['tested']], expected[r['tested']])
    raw = mannwhitneyu(x[:, 1:], e[:, 1:], axis=0, method='asymptotic', use_continuity=True).pvalue
    np.testing.assert_array_equal(r['fdr'][1:], bh(raw))
    assert r['fdr'][0] == 1 and not r['tested'][0] and np.isnan(r['lfc_true'][0])


def test_pie_all_tied_values_are_not_differential_expression():
    x = np.full((8, 3), .5, dtype='f4')
    r = pie_labels(x, x, np.full(3, .5))
    np.testing.assert_array_equal(r['fdr'], 1.)
    np.testing.assert_array_equal(r['delta_p'], 0.)
    np.testing.assert_array_equal(r['fold_changes'], 1.)
    with pytest.raises(ValueError): pie_labels(x-1, x, np.full(3, .5))


def test_preprocessing_checks_held_permissions_before_cached_response_access(tmp_path):
    p = synthetic_inputs(tmp_path/'inputs')
    f = read_json(p['splits']/'F1/FOLD.json')
    store = CellStore(p['data'], f, role='train')
    try:
        rows = legal_rows(store)
        assert len(rows) == 27
        e, target = rows[0]
        a = supervision(store, e, target)
        assert a.shape == (24, 48)
        assert store._episode is None
        held = f['joint_holdout']['held_targets'][e][0]
        before = dict(store.audit)
        with pytest.raises(PermissionError): supervision(store, e, held)
        assert dict(store.audit) == before
        assert store._episode is None
        with pytest.raises(PermissionError): supervision(store, f['test'], target)
    finally: store.close()


def _resource_script():
    path = Path(__file__).resolve().parents[1]/'scripts/download_resources.py'
    spec = importlib.util.spec_from_file_location('public_resources', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_resource_plan_is_pinned_and_hgnc_never_uses_latest_implicitly():
    mod = _resource_script()
    records = mod.plan(['tahoe', 'genept', 'string'])
    assert len(records) == 6
    assert all('/resolve/'+mod.TAHOE_REVISION+'/' in r['url'] for r in records if r['resource']=='tahoe')
    assert all(len(r['expected_sha256']) == 64 for r in records if r['resource']=='string')
    with pytest.raises(ValueError): mod.plan(['hgnc'])
    with pytest.raises(ValueError): mod.plan(['tahoe'], tahoe_revision='main')
