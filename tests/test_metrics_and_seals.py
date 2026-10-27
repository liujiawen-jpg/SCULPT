"""Independent numeric references and immutable inference receipts."""
import json
import h5py
import numpy as np
import pytest
from scipy.spatial.distance import cdist
from cosit.modern_metrics import energy_distance_normalized
from cosit.prediction import seal_predictions,verify_prediction_seal


def test_empirical_energy_matches_direct_reference():
    x=np.array([[0.,1.],[2.,3.],[1.,2.]])
    y=np.array([[2.,1.],[4.,3.]])
    scale=np.array([2.,.5])
    # Both populations fit under the cap. Use equal counts to retain all cells.
    y=np.vstack([y,[3.,2.]])
    a,b=x/scale,y/scale
    reference=(2*cdist(a,b).mean()-cdist(a,a).mean()-cdist(b,b).mean())/np.sqrt(2)
    got=energy_distance_normalized(x,y,scale,'condition',row_chunk=2,feature_chunk=1)
    assert got['Energy_distance_normalized']==pytest.approx(reference,abs=1e-12)
    reverse=energy_distance_normalized(y,x,scale,'condition')
    assert reverse['Energy_distance_normalized']==pytest.approx(reference,abs=1e-12)
    same=energy_distance_normalized(x,x,scale,'condition')
    assert same['Energy_distance_normalized']==pytest.approx(0.,abs=1e-12)


def test_scoring_requires_original_unchanged_seal(tmp_path):
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'checkpoint')
    prediction=tmp_path/'predictions.h5'
    with h5py.File(prediction,'w') as h:
        h.attrs['complete']=True;h.create_dataset('mean',data=[[1.]])
    seal=prediction.with_suffix('.seal.json');allowed={'context':['target']}
    with pytest.raises(PermissionError,match='Original'):
        verify_prediction_seal(checkpoint,[prediction],allowed,seal,purpose='test')
    seal_predictions(checkpoint,[prediction],allowed,seal,purpose='test')
    verify_prediction_seal(checkpoint,[prediction],allowed,seal,purpose='test')
    with pytest.raises(PermissionError,match='scope'):
        verify_prediction_seal(checkpoint,[prediction],{'context':['other']},seal,purpose='test')
    with h5py.File(prediction,'r+') as h:h['mean'][0,0]=2.
    with pytest.raises(PermissionError,match='changed'):
        verify_prediction_seal(checkpoint,[prediction],allowed,seal,purpose='test')
