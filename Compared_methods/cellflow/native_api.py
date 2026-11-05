"""Preserved native CellFlow functions; not an end-to-end training controller."""
import functools
import hashlib
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from cellflow.networks import ConditionalVelocityField
from cellflow.solvers import OTFlowMatching
from cellflow._compat import ConstantNoiseFlow
from cellflow.utils import match_linear

def configuration():
    return dict(pca_dimension=100,condition_embedding_dim=256,condition_dropout=.9,
        perturbation_layers=[1024,1024],context_layers=[256,256],after_pool=[1024,1024],
        time_freqs=1024,time_encoder=[1024,1024,1024],hidden=[2048,2048,2048],
        decoder=[4096,4096,4096],batch_size=1024,learning_rate=5e-5,
        gradient_accumulation=20,ot_epsilon=.5,constant_noise=.5,
        ode_solver='Tsit5',ode_rtol=1e-5,ode_atol=1e-5,
        output_clip=False,min_iterations=50000,max_iterations=500000,
        validation_frequency=10000,patience=6,min_relative_improvement=.002)

def make_solver(seed, n_targets, n_genes, config=None, smoke=False):
    cfg=dict(configuration() if config is None else config)
    if smoke:
        cfg.update(condition_embedding_dim=8,condition_dropout=.1,perturbation_layers=[16],
            context_layers=[16],after_pool=[16],time_freqs=8,time_encoder=[16],hidden=[16],decoder=[16],
            gradient_accumulation=2)
    vf=ConditionalVelocityField(output_dim=cfg['pca_dimension'],max_combination_length=1,
        condition_mode='deterministic',regularization=0.,condition_embedding_dim=cfg['condition_embedding_dim'],
        covariates_not_pooled=[],pooling='attention_token',pooling_kwargs={},
        layers_before_pool={
            'perturbation':{'layer_type':'mlp','dims':cfg['perturbation_layers'],'dropout_rate':.5},
            'context':{'layer_type':'mlp','dims':cfg['context_layers'],'dropout_rate':0.}},
        layers_after_pool={'layer_type':'mlp','dims':cfg['after_pool'],'dropout_rate':0.},
        cond_output_dropout=cfg['condition_dropout'],condition_encoder_kwargs={},act_fn=nn.silu,
        time_freqs=cfg['time_freqs'],time_max_period=10000,time_encoder_dims=cfg['time_encoder'],
        time_encoder_dropout=0.,hidden_dims=cfg['hidden'],hidden_dropout=0.,
        conditioning='concatenation',conditioning_kwargs={},decoder_dims=cfg['decoder'],decoder_dropout=0.,
        layer_norm_before_concatenation=False,linear_projection_before_concatenation=False)
    solver=OTFlowMatching(vf=vf,probability_path=ConstantNoiseFlow(cfg['constant_noise']),
        match_fn=functools.partial(match_linear,epsilon=cfg['ot_epsilon'],tau_a=1.,tau_b=1.),
        optimizer=optax.MultiSteps(optax.adam(cfg['learning_rate']),cfg['gradient_accumulation']),
        conditions={'perturbation':jnp.zeros((1,1,n_targets),dtype=jnp.float32),
                    'context':jnp.zeros((1,1,n_genes),dtype=jnp.float32)},
        rng=jax.random.PRNGKey(seed))
    return solver,cfg

def condition(target_index, context_mean, n_targets):
    p=np.zeros((1,1,n_targets),dtype=np.float32);p[0,0,target_index]=1.
    c=np.asarray(context_mean,dtype=np.float32)[None,None,:]
    return {'perturbation':p,'context':c}

def predict(solver,controls,pca,condition_data):
    source=(np.asarray(controls,dtype=np.float32)-pca['mean'])@pca['components'].T
    projected=np.asarray(solver.predict(jnp.asarray(source),condition=condition_data),dtype=np.float32)
    # Exactly upstream reconstruct_pca: no clipping, residual retention, calibration or baseline addition.
    result=projected@pca['components']+pca['mean']
    if not np.isfinite(result).all(): raise FloatingPointError('CellFlow predicted nonfinite expression')
    return np.asarray(result,dtype=np.float32)

def generation_seed(seed,environment,target):
    return int.from_bytes(hashlib.sha256(f'CellFlow|{seed}|{environment}|{target}'.encode()).digest()[:4],'little')
