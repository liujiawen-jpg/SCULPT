"""GPU-routing/RNG tests using mocks: never initialize or reserve a real GPU."""
import contextlib
import json
import random
import types
from unittest import mock
import numpy as np
import torch
from . import native_api as api


def _numpy_equal(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


def main():
    host_before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    gpu_states = {0: torch.tensor([101],dtype=torch.uint8),
                  2: torch.tensor([202],dtype=torch.uint8)}
    current = [0]
    calls = []
    @contextlib.contextmanager
    def device_context(index):
        old = current[0]
        current[0] = int(index)
        try:
            yield
        finally:
            current[0] = old
    def get_state(index):
        calls.append(('get',int(index)))
        return gpu_states[int(index)].clone()
    def set_state(state,index):
        calls.append(('restore',int(index)))
        gpu_states[int(index)] = state.clone()
    def selected_seed(seed):
        calls.append(('seed',current[0],int(seed)))
        gpu_states[current[0]] = torch.tensor([int(seed)%256],dtype=torch.uint8)
    def sample(*args,**kwargs):
        gpu_states[2] = gpu_states[2] + 1
        return torch.rand(1,3,4)
    class FakeModel:
        cell_context_embedding = None
        def __init__(self,**kwargs):
            self.kwargs = kwargs
        def to(self,device):
            self.device = device
            return self
        def eval(self):
            return self
    x = types.SimpleNamespace(device=torch.device('cuda:2'),is_cuda=True)
    cfg = api.configuration()
    cfg['evaluation_cells'] = 4
    try:
        with contextlib.ExitStack() as stack:
            patches = [
                mock.patch.object(torch.cuda,'current_device',side_effect=lambda:current[0]),
                mock.patch.object(torch.cuda,'device',side_effect=device_context),
                mock.patch.object(torch.cuda,'get_rng_state',side_effect=get_state),
                mock.patch.object(torch.cuda,'set_rng_state',side_effect=set_state),
                mock.patch.object(torch.cuda,'manual_seed',side_effect=selected_seed),
                mock.patch.object(torch.cuda,'manual_seed_all',side_effect=AssertionError('Global GPU seeding forbidden')),
                mock.patch.object(torch,'manual_seed',side_effect=AssertionError('Global torch seeding forbidden')),
                mock.patch.object(api,'forward',return_value={}),
                mock.patch.object(api,'sample_from_copula',side_effect=sample),
            ]
            for patch in patches:
                stack.enter_context(patch)
            old_gpu = {k:v.clone() for k,v in gpu_states.items()}
            old_cpu = torch.get_rng_state().clone()
            old_python = random.getstate()
            old_numpy = np.random.get_state()
            first = api.cloud(FakeModel(),x,None,None,None,None,None,cfg,42)
            second = api.cloud(FakeModel(),x,None,None,None,None,None,cfg,42)
            np.testing.assert_array_equal(first,second)
            assert torch.equal(torch.get_rng_state(),old_cpu)
            assert random.getstate() == old_python
            assert _numpy_equal(np.random.get_state(),old_numpy)
            assert all(torch.equal(gpu_states[k],v) for k,v in old_gpu.items())
            assert ('seed',2,42) in calls and all(c[1]==2 for c in calls)
            assert current[0] == 0
            with mock.patch.object(api,'sample_from_copula',side_effect=RuntimeError('synthetic failure')):
                try:
                    api.cloud(FakeModel(),x,None,None,None,None,None,cfg,43)
                except RuntimeError as exc:
                    assert str(exc) == 'synthetic failure'
                else:
                    raise AssertionError('Expected synthetic failure')
            assert torch.equal(torch.get_rng_state(),old_cpu)
            assert all(torch.equal(gpu_states[k],v) for k,v in old_gpu.items())
            with mock.patch.object(api,'DistributionTransformer',FakeModel), \
                 mock.patch.object(torch.cuda,'set_per_process_memory_fraction') as limit:
                model = api.create_model(16,17,cfg,device='cuda:2')
                limit.assert_called_once_with(cfg['gpu_memory_fraction'],device=2)
                assert model.device == torch.device('cuda:2')
                assert torch.equal(gpu_states[0],old_gpu[0])
                assert ('seed',2,17) in calls
            assert current[0] == 0
            # The historical one-visible-GPU execution resolves to cuda:0.
            assert api._cuda_device_index('cuda') == 0
            assert api._cuda_device_index('cuda:0') == 0
            assert api._cuda_device_index('cpu') is None
            with mock.patch.object(torch.cuda,'set_per_process_memory_fraction') as limit, \
                 mock.patch.object(api,'DistributionTransformer',FakeModel):
                api.create_model(16,17,cfg,device='cpu')
                limit.assert_not_called()
    finally:
        random.setstate(host_before[0])
        np.random.set_state(host_before[1])
        torch.set_rng_state(host_before[2])
    print(json.dumps(dict(status='passed',real_GPU_calls=0,training_updates=0,checks=[
        'explicit_cuda2_memory_limit','cuda2_only_seed_and_fork','other_GPU_state_untouched',
        'CPU_Python_NumPy_states_restored_after_cloud','repeat_cloud_identity',
        'exception_restores_RNG','current_device_restored','legacy_cuda0_resolution',
        'CPU_model_no_GPU_memory_operation'])))


if __name__ == '__main__':
    main()
