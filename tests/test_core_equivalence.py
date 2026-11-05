"""Scientific regression tests of the real J0 modules on synthetic inputs."""
import numpy as np
import pytest
import torch
from scipy import sparse

from cosit.core.batched_model import BatchedSharedPathModel
from cosit.core.shared_path_model import SharedPathModel
from cosit.core.geometry import geometry_from_arrays
from cosit.core.objective import objective_conditions
from cosit.core.runtime import statistics, positive_summary, sampling_variance
from cosit.core.rearrangement import invariants


@pytest.fixture
def scientific_case():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    rng = np.random.default_rng(17)
    q, s, g, n, p = 2, 2, 40, 16, 261
    priors = rng.normal(size=(g, p)).astype('f4')
    controls = rng.gamma(2, .4, (3, n, g)).astype('f4')
    controls[rng.random(controls.shape) < .3] = 0
    stats = np.stack([statistics(x) for x in controls])
    graph = sparse.csr_matrix(np.ones((g, g)) - np.eye(g))
    geometry = geometry_from_arrays(priors, stats, np.full(g, .3), graph, rank=4, ambient=8)
    ctrl = torch.as_tensor(controls[0]).expand(q, -1, -1).clone()
    target_stats = torch.as_tensor(stats[0]).expand(q, -1, -1).clone()
    source_stats = torch.as_tensor(stats[1:]).expand(q, -1, -1, -1).clone()
    positive = positive_summary(controls[0])
    batch = dict(observations=torch.randn(q,s,g,7)*.1, precision=torch.ones(q,s,g,7),
        source_controls=source_stats, target_control=target_stats,
        target_prior=torch.randn(q,p), source_joint=torch.randn(q,s,625)*.1,
        source_reliability=torch.full((q,s),.75), control_cells=ctrl,
        cell_features=torch.randn(q,n,48),
        positive_template=torch.as_tensor(positive['quantiles']).expand(q,-1,-1).clone(),
        positive_base_mean=torch.as_tensor(positive['positive_mean']).expand(q,-1).clone(),
        positive_missing=torch.zeros(q,g,dtype=torch.bool),
        target_control_mean=target_stats[:,:,0], target_control_detection=target_stats[:,:,2],
        source_control_means=source_stats[:,:,:,0], source_control_detections=source_stats[:,:,:,2],
        source_means=source_stats[:,:,:,0]+.01,
        source_detections=source_stats[:,:,:,2])
    model = BatchedSharedPathModel(geometry, width=16, state_rank=8).set_phase('J')
    return geometry, batch, model


def _condition(batch, index):
    return {k: v[index] for k,v in batch.items()}


def test_batched_cloud_matches_original_single_condition_path(scientific_case):
    geometry, batch, batched = scientific_case
    single = SharedPathModel(geometry, route='open_zero', width=16, state_rank=8)
    single.load_state_dict(batched.state_dict(), strict=True)
    batched.eval(); single.eval()
    with torch.no_grad():
        actual = batched.forward_conditions(batch)
        reference = [single(_condition(batch, i)) for i in range(2)]
    for key in ('rule','p','r','support','positive_quantiles'):
        expected = torch.stack([record[key] for record in reference])
        torch.testing.assert_close(actual[key], expected, atol=3e-5, rtol=3e-4)


def test_generated_cloud_retains_marginals_and_exact_hurdle_mean(scientific_case):
    _, batch, model = scientific_case
    model.eval()
    with torch.no_grad():
        out=model.forward_conditions(batch)
    assert torch.isfinite(out['support']).all()
    assert (out['support']>=0).all()
    for index in range(2):
        invariants(out['pre_reorder_support'][index],out['support'][index])
    torch.testing.assert_close(out['mean'],out['quantized_mean'],atol=3e-7,rtol=1e-6)
    assert torch.equal((out['support']>0).sum(1),out['positive_count'])


def test_joint_objective_backpropagates_into_all_three_prediction_modules(scientific_case):
    _, batch, model = scientific_case
    model.train()
    output=model.forward_conditions(batch)
    real=batch['control_cells']*1.2
    supervisions=[]
    for index in range(2):
        cells=real[index].numpy(); stats=statistics(cells); pos=positive_summary(cells)
        supervisions.append(dict(mean=torch.as_tensor(stats[:,0]),n_cells=len(cells),
            detection=torch.as_tensor(stats[:,2]),positive_mean=torch.as_tensor(pos['positive_mean']),
            positive_mask=torch.as_tensor(pos['positive_count']>0),
            count_positive=torch.as_tensor(pos['positive_count']),
            positive_quantiles=torch.as_tensor(pos['quantiles']),
            variance=torch.as_tensor(cells.var(0)),
            statistics=torch.as_tensor(stats-batch['target_control'][index].numpy())))
    projection=torch.randn(40,8)/np.sqrt(40)
    components=torch.linalg.qr(torch.randn(40,4)).Q.T
    loss, logs=objective_conditions(output,real,supervisions,batch['target_control_mean'],
        torch.ones(40),torch.ones(40),projection,components,
        own_genes=[0,1],control_cells=batch['control_cells'],real_effective_count=[16,16],
        source_batches=batch,source_readout_weight=.1)
    assert torch.isfinite(loss)
    loss.backward()
    groups={'backbone':['background.','gene.','joint_to_prior.'], 'shape':['shape.'], 'reorder':['reorder.']}
    for name,prefixes in groups.items():
        gradients=[p.grad for key,p in model.named_parameters() if any(key.startswith(x) for x in prefixes)]
        assert any(g is not None and torch.isfinite(g).all() and g.norm()>0 for g in gradients),name
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    torch.optim.AdamW(model.parameters(),lr=2e-4).step()


def test_precision_bootstrap_uses_analytic_mean_and_detection_uncertainty():
    x=np.array([[0.,1.],[2.,0.],[1.,4.],[0.,2.]],dtype='f4')
    variance=sampling_variance(x,seed=17)
    np.testing.assert_allclose(variance[:,0],x.var(0,ddof=1)/len(x),rtol=1e-6)
    rate=((x>0).sum(0)+.5)/(len(x)+1)
    np.testing.assert_allclose(variance[:,2],rate*(1-rate)/(len(x)+1),rtol=1e-6)


def test_implicit_connection_solution_and_gradients_match_dense_energy():
    """Compare matrix-free inference to an independently assembled SPD system.

    This checks numerical answers and implicit derivatives, not only agreement
    between two implementations that share the same matrix-free helper.
    """
    from cosit.core.batched_connection import BatchedConnectionInference
    torch.manual_seed(121)
    torch.set_num_threads(1)
    q, s, g, m, d = 2, 2, 4, 3, 2
    geometry = dict(
        bases=torch.randn(g, 3, d), edge_src=torch.tensor([0, 1, 2]),
        edge_dst=torch.tensor([1, 2, 3]), left=torch.randn(3, d, d),
        right=torch.randn(3, d, d), weight=torch.rand(3, d) + .2)
    solver = BatchedConnectionInference(geometry, strength=.15, ridge=.2,
        tolerance=1e-11, max_steps=100).double()
    a = torch.randn(q, s, g, m, d, dtype=torch.float64, requires_grad=True)
    y = torch.randn(q, s, g, m, dtype=torch.float64, requires_grad=True)
    precision = (torch.rand(q, s, g, m, dtype=torch.float64) + .2).requires_grad_()
    prior = torch.randn(q, g, d, dtype=torch.float64, requires_grad=True)
    actual = solver.forward_conditions(a, y, precision, prior)['rule']

    # Form a rectangular edge constraint matrix without calling graph helpers.
    k = torch.zeros(len(solver.edge_src) * d, g * d, dtype=torch.float64)
    for edge, (left_gene, right_gene) in enumerate(zip(solver.edge_src, solver.edge_dst)):
        k[edge*d:(edge+1)*d, left_gene*d:(left_gene+1)*d] = solver.left[edge]
        k[edge*d:(edge+1)*d, right_gene*d:(right_gene+1)*d] = -solver.right[edge]
    graph_hessian = k.T @ torch.diag(solver.weight.flatten()) @ k
    expected = []
    for condition in range(q):
        blocks, rhs = [], []
        for gene in range(g):
            operator = a[condition, :, gene].reshape(s*m, d)
            observations = y[condition, :, gene].reshape(-1)
            w = torch.diag(precision[condition, :, gene].reshape(-1))
            blocks.append(operator.T @ w @ operator + solver.ridge * torch.eye(d, dtype=torch.float64))
            rhs.append(operator.T @ w @ observations + solver.ridge * prior[condition, gene])
        hessian = torch.block_diag(*blocks) + solver.strength * graph_hessian
        expected.append(torch.linalg.solve(hessian, torch.cat(rhs)).reshape(g, d))
    expected = torch.stack(expected)
    torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-9)
    probe = torch.randn_like(actual)
    variables = (a, y, precision, prior)
    derivatives = torch.autograd.grad((actual * probe).sum(), variables, retain_graph=True)
    references = torch.autograd.grad((expected * probe).sum(), variables)
    for derivative, reference in zip(derivatives, references):
        torch.testing.assert_close(derivative, reference, rtol=3e-8, atol=3e-9)


def test_conditions_are_not_mixed_in_batched_predictions(scientific_case):
    """A different condition's source observations cannot alter this query."""
    _, batch, model = scientific_case
    model.eval()
    altered = {key: value.clone() for key, value in batch.items()}
    altered['observations'][1] += .35
    altered['target_prior'][1] *= 1.7
    with torch.no_grad():
        before = model.forward_conditions(batch)
        after = model.forward_conditions(altered)
    for key in ('rule', 'p', 'r', 'support', 'mean512_st'):
        torch.testing.assert_close(before[key][0], after[key][0], atol=0, rtol=0)
    assert not torch.equal(before['rule'][1], after['rule'][1])
