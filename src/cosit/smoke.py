"""Synthetic end-to-end exercise of the actual J0 model, not a substitute model."""
from __future__ import annotations
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from scipy import sparse
from .io import read_json,write_json,sha256
from .splits import CONTEXTS,generate_splits


def synthetic_inputs(output):
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Smoke output must be empty')
    output.mkdir(parents=True,exist_ok=True);data=output/'data';data.mkdir()
    rng=np.random.default_rng(71);genes=[f'G{i:04d}' for i in range(48)];targets=genes[:12]
    effect=rng.normal(0,.12,(12,48)).astype('f4');manifest={}
    for i,env in enumerate(CONTEXTS):
        path=data/(env+'.h5')
        control=rng.gamma(2,.25,(48,48)).astype('f4')+i*.03
        control[rng.random(control.shape)<.15]=0
        with h5py.File(path,'w') as h:
            h.attrs.update(schema='cosit_cells_v1',complete=True,genes_json=json.dumps(genes),
                targets_json=json.dumps(targets),control_n_full=len(control),units='synthetic normalized log1p')
            h.create_dataset('controls',data=control[:32]);h.create_dataset('control_mean',data=control.mean(0))
            h.create_dataset('control_variance',data=control.var(0));h.create_dataset('control_detection',data=(control>0).mean(0).astype('f4'))
            h.create_dataset('control_ids',data=np.asarray([f'{env}-ctrl-{j}' for j in range(32)],dtype=object),dtype=h5py.string_dtype())
            group=h.create_group('perturbations')
            for j,target in enumerate(targets):
                cells=np.maximum(0,control[rng.choice(len(control),24,replace=False)]+effect[j]*(1+i*.1))
                group.create_dataset(target,data=cells.astype('f4'))
        manifest[env]=dict(context=env,file=path.name,sha256=sha256(path))
    write_json(data/'MANIFEST.json',dict(schema='cosit_prepared_data_v1',contexts=manifest,synthetic=True))
    priors=output/'priors.npz';np.savez(priors,genes=np.asarray(genes),embedding=rng.normal(0,.1,(48,261)).astype('f4'))
    row=[];col=[]
    for i in range(48):
        for j in range(1,9):row.append(i);col.append((i+j)%48)
    graph=sparse.csr_matrix((np.ones(len(row)),(row,col)),shape=(48,48))
    sparse.save_npz(output/'graph.npz',graph.maximum(graph.T))
    (output/'graph_genes.tsv').write_text('gene_id\n'+'\n'.join(genes)+'\n')
    config=read_json(Path(__file__).with_name('defaults.json'))
    config.update(generated_cells_train=16,real_cells_train=16,marginal_genes=24,joint_projections=4,
        seeds=[17,29,43],targets=12,genes=48,cpu_threads=1,microbatch_per_environment=2,
        new_relation_candidates=4,validation_every_epochs=1)
    config['joint_holdout']['count_per_training_environment']=3
    config['joint_validation'].update(external_response_conditions=6,internal_response_conditions=3,
        external_distribution_conditions=2,internal_distribution_conditions=2)
    config['phases'][0].update(min_epochs=1,max_epochs=1,extended_max_epochs=1,min_optimizer_steps=0,
        patience=2,lr_patience=[1,2],post_lr_drop_epochs=0)
    write_json(output/'smoke_config.json',config)
    generate_splits(targets,output/'splits',config)
    return dict(root=output,data=data,priors=priors,graph=output/'graph.npz',graph_axis=output/'graph_genes.tsv',
                splits=output/'splits',config=output/'smoke_config.json')


def run_smoke(output):
    from .training import train,load_checkpoint
    from .assets import prepare_assets,evaluation_runtime
    from .core.batched_model import BatchedSharedPathModel
    from .prediction import generate,seal_predictions
    from .evaluation import evaluate
    inputs=synthetic_inputs(output);root=inputs['root'];run=root/'run'
    result=train(inputs['data'],inputs['priors'],inputs['graph'],inputs['graph_axis'],inputs['splits'],
                 'F1',17,run,inputs['config'],'cpu',smoke=True)
    if result['selected_checkpoint'] is None:raise AssertionError('Smoke validation did not select a checkpoint')
    checkpoint=Path(result['selected_checkpoint']);payload=load_checkpoint(checkpoint)
    p=payload['protocol'];fold=p['fold']
    assets,geometry,_=prepare_assets(inputs['data'],inputs['priors'],inputs['graph'],inputs['graph_axis'],
                                    fold,run/'assets','cpu',p,fit=False)
    infer=None
    try:
        model=BatchedSharedPathModel(geometry,route='open_zero',**p['model']);model.load_state_dict(payload['state_dict'])
        model.set_phase('J');model.eval();infer=evaluation_runtime(assets,geometry,root/'inference','cpu')
        # Explicitly synthetic: not a backdoor to the formal convergence gate.
        prediction=generate(model,infer,fold['test'],assets.target_names,root/'predictions.h5',batch_size=2)
        seal_predictions(checkpoint,[prediction],{fold['test']:assets.target_names},root/'predictions.seal.json',purpose='test')
        metrics=evaluate(inputs['data'],prediction,checkpoint,run/'assets',root/'metrics.json')
        summary=dict(status='passed',synthetic_only=True,formal_experiment=False,
            stages=['metadata_split','fresh_joint_training','four_panel_validation','checkpoint_selection',
                    'restore','512_cell_generation','seal','response_evaluation','distribution_evaluation'],
            conditions=metrics['conditions'],genes=metrics['genes'],response_metrics=list(metrics['response']),
            distribution_metrics=list(metrics['distribution']),checkpoint=str(checkpoint))
        write_json(root/'SMOKE_REPORT.json',summary)
        return summary
    finally:
        assets.train_store.close()
        if infer is not None:infer.assets.train_store.close()
