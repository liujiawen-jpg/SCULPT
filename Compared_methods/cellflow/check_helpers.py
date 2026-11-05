"""Finite synthetic checks; no training, test labels, or benchmark files."""
import argparse
import json
from pathlib import Path
import numpy as np
from . import native_api as api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-smoke', action='store_true')
    args = parser.parse_args()
    registered = json.loads(Path(__file__).with_name('benchmark_config.json').read_text())
    assert api.configuration() == registered['configuration']
    condition = api.condition(2, np.arange(6), 3)
    assert condition['perturbation'].shape == (1,1,3)
    assert condition['context'].shape == (1,1,6)
    np.testing.assert_array_equal(condition['perturbation'][0,0], [0,0,1])
    rng = np.random.default_rng(7)
    controls = rng.normal(size=(8,6)).astype(np.float32)
    q, _ = np.linalg.qr(rng.normal(size=(6,4)))
    pca = dict(mean=np.arange(6,dtype=np.float32), components=q.T.astype(np.float32))
    class IdentitySolver:
        def predict(self, x, condition):
            return x
    result = api.predict(IdentitySolver(), controls, pca, condition)
    expected = ((controls-pca['mean']) @ pca['components'].T) @ pca['components']+pca['mean']
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
    assert result.shape == controls.shape and np.isfinite(result).all()
    assert api.generation_seed(17,'example','target') == api.generation_seed(17,'example','target')
    checks = ['registered_config', 'condition_axes', 'PCA_forward_inverse_no_extra_residual', 'finite_output', 'seed_reproducibility']
    if args.native_smoke:
        config = api.configuration()
        config['pca_dimension'] = 4
        solver, small = api.make_solver(17,3,6,config,smoke=True)
        assert small['hidden'] == [16]
        assert solver.vf_state.params is not None
        checks.append('official_network_CPU_initialization_no_training')
    print(json.dumps(dict(status='passed',checks=checks,training_updates=0)))


if __name__ == '__main__':
    main()
