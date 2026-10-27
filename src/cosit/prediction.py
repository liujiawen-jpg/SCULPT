"""Inference receives controls and legal sources; query labels are separate."""
from __future__ import annotations
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from .io import sha256, write_json, read_json
from .core.batched_model import split_condition_outputs


@torch.no_grad()
def generate(model,runtime,environment,targets,path,*,distribution_targets=None,batch_size=4,n_cells=512):
    targets=list(targets)
    distribution_targets=targets if distribution_targets is None else list(distribution_targets)
    if len(set(targets))!=len(targets) or not set(distribution_targets)<=set(targets) or not targets:
        raise ValueError('Prediction requires unique nonempty legal axes')
    sources=tuple(e for e in runtime.training_environments if e!=environment)
    if not sources:raise PermissionError('No legal observed source for this query')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(f'Refusing to overwrite predictions: {path}')
    temporary=path.with_suffix(path.suffix+'.tmp')
    model.eval();g=len(runtime.genes)
    pos={t:i for i,t in enumerate(targets)};di={t:i for i,t in enumerate(distribution_targets)}
    with h5py.File(temporary,'w') as h:
        h.attrs.update(schema='cosit_predictions_v1',complete=False,environment=environment,
            genes_json=json.dumps(runtime.genes),targets_json=json.dumps(targets),
            distribution_targets_json=json.dumps(distribution_targets),
            source_environments_json=json.dumps(sources),generated_cells=n_cells,
            support_semantics='deterministic finite quantile cloud; not iid',
            old_prediction_anchor=False,method='CoSIT-J0')
        for name in ('mean','p','r'):
            h.create_dataset(name,(len(targets),g),dtype='f4')
        if di:
            for name,shape in (('support',(len(di),n_cells,g)),('positive_quantiles',(len(di),g,33)),
                               ('unit_mean_shape',(len(di),g,33)),('control_cells',(len(di),n_cells,g))):
                h.create_dataset(name,shape,dtype='f4',chunks=(1,*shape[1:]),compression='lzf')
            h.create_dataset('rank_penalty',(len(di),),dtype='f4')
        for cloud,axis in ((False,[t for t in targets if t not in di]),(True,distribution_targets)):
            for start in range(0,len(axis),batch_size):
                chosen=axis[start:start+batch_size]
                batches=runtime.batch_conditions([dict(query=environment,target=t,sources=sources) for t in chosen],
                                                  n_cells,'joint-fixed-validation' if len(targets)!=len(runtime.targets) else 'final-test')
                out=model.forward_conditions(batches,mean_only=not cloud)
                for t,batch,item in zip(chosen,batches,split_condition_outputs(out)):
                    array=lambda name:item[name].detach().cpu().numpy()
                    mean=array('mean512_st')
                    if not np.isfinite(mean).all() or (mean<0).any():raise RuntimeError('Invalid model mean')
                    for name in ('p','r'):h[name][pos[t]]=array(name)
                    h['mean'][pos[t]]=mean
                    if cloud:
                        y=array('support');before=array('pre_reorder_support')
                        if not np.isfinite(y).all() or (y<0).any():raise RuntimeError('Invalid generated cloud')
                        np.testing.assert_allclose(y.mean(0,dtype=np.float64),mean,rtol=3e-5,atol=3e-6)
                        np.testing.assert_array_equal(np.sort(y,axis=0),np.sort(before,axis=0))
                        for name in ('support','positive_quantiles','unit_mean_shape','rank_penalty'):
                            h[name][di[t]]=array(name)
                        h['control_cells'][di[t]]=batch['control_cells'].cpu().numpy()
        h.attrs['complete']=True
    temporary.replace(path)
    return path


def seal_predictions(checkpoint,predictions,allowed,path,*,purpose):
    """Seal every panel before permitting the first response-label read."""
    if purpose not in ('validation','test'):
        raise ValueError('Invalid evaluation purpose')
    files=[Path(checkpoint),*[Path(p) for p in predictions]]
    for prediction in predictions:
        with h5py.File(prediction,'r') as h:
            if not h.attrs.get('complete'):raise PermissionError('Incomplete prediction cannot be scored')
    record=dict(schema='cosit_prediction_seal_v1',status='frozen',purpose=purpose,
        allowed={k:list(v) for k,v in allowed.items()},
        files=[dict(path=str(p.resolve()),sha256=sha256(p)) for p in files],
        query_labels_used_in_prediction=False)
    write_json(path,record)
    return Path(path)


def verify_prediction_seal(checkpoint,predictions,allowed,path,*,purpose):
    """Verify the original inference receipt; scoring cannot silently reseal."""
    path=Path(path)
    if not path.is_file():
        raise PermissionError('Original inference seal is required before scoring')
    record=read_json(path)
    if (record.get('schema')!='cosit_prediction_seal_v1' or record.get('status')!='frozen'
            or record.get('purpose')!=purpose or record.get('query_labels_used_in_prediction') is not False
            or record.get('allowed')!={k:list(v) for k,v in allowed.items()}):
        raise PermissionError('Prediction seal has a different evaluation scope')
    expected=[Path(checkpoint).resolve(),*[Path(p).resolve() for p in predictions]]
    entries=record.get('files',[])
    if len(entries)!=len(expected) or [Path(r['path']).resolve() for r in entries]!=expected:
        raise PermissionError('Prediction seal does not bind these exact files')
    for item,source in zip(entries,expected):
        if sha256(source)!=item['sha256']:
            raise PermissionError('Prediction/checkpoint changed after sealing')
    return record
