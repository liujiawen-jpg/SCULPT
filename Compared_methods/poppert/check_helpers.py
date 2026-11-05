"""Native finite CPU checks, no training or benchmark data access."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from . import native_api as api
from .population_helpers import population_indices, condition_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream',required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    api.configure_upstream(args.upstream)
    registered = json.loads(Path(__file__).with_name('benchmark_config.json').read_text())
    assert api.configuration() == registered['configuration']
    indices = population_indices(256, condition_seed('example','target'))
    assert indices.shape == (5,128)
    np.testing.assert_array_equal(indices[0],np.arange(128))
    np.testing.assert_array_equal(indices[1],np.arange(128,256))
    assert np.array_equal(indices,population_indices(256,condition_seed('example','target')))
    assert population_indices(20,42).shape == (5,128)
    try:
        population_indices(3,42)
    except ValueError:
        pass
    else:
        raise AssertionError('Too few observations accepted')
    cfg = api.configuration()
    cfg.update(d_model=16,nhead=4,num_layers=1,dim_feedforward=32,copula_rank=4,
               n_samples_loss=16,evaluation_cells=32,bfloat16_transformer=False)
    model = api.create_model(16,17,cfg,device='cpu')
    assert model.cell_context_embedding is None
    params = torch.tensor([.3,.5,.5,.6,1.2,.3,.5]).repeat(2,12,1)
    gene = torch.arange(12,dtype=torch.long)
    labels = torch.tensor([12,13])
    factors = torch.ones(2,4)
    model.eval()
    with torch.no_grad():
        value, logs = api.loss(model,params,params,gene,labels,factors,factors,cfg)
    assert torch.isfinite(value)
    A,_ = torch.linalg.qr(torch.randn(12,4))
    psi = torch.ones(12)
    first = api.cloud(model,params,gene,labels,factors,A,psi,cfg,42)
    second = api.cloud(model,params,gene,labels,factors,A,psi,cfg,42)
    assert first.shape == (2,32,12)
    assert np.isfinite(first).all() and first.min()>=0 and first.max()<=14.9
    np.testing.assert_array_equal(first,second)
    print(json.dumps(dict(status='passed',training_updates=0,
        checks=['upstream_hashes','registered_config','population_sampling','small_native_CPU_forward',
                'native_objective_finite','native_copula_cloud_shape','official_export_range','same_seed_cloud_identity'],
        synthetic_native_loss=float(value))))


if __name__ == '__main__':
    main()
