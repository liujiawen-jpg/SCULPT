"""Sealed validation/test evaluation; these labels never enter model inputs."""
from __future__ import annotations
from pathlib import Path
import json
import math
import h5py
import numpy as np
import torch
from scipy.spatial.distance import pdist
from .data import CellStore
from .io import write_json,read_json,sha256
from .prediction import generate,seal_predictions,verify_prediction_seal
from .core.runtime import positive_summary,seed_int
from .core.objective import joint_response_objective_conditions,joint_distribution_objective_conditions
from .core.ranking import RankingPanels
from .core.metrics import response_metrics,distribution_metrics
from .modern_metrics import deg_ap,gaussian_mmd2_unbiased,energy_distance_normalized


def _tensor(value,device):
    return torch.as_tensor(np.array(value,copy=True),device=device)


def _finite_mean(values):
    values=[float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


@torch.no_grad()
def validate(model,runtime,assets,scales,checkpoint,panels,directory,protocol,epoch):
    if set(panels)!={assets.validation_environment,*assets.training_environments}:
        raise PermissionError('Selection needs external and three internal panels')
    if abs(sum(v['weight'] for v in panels.values())-1)>1e-10:
        raise ValueError('Validation weights must sum to one')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    paths={}
    for env,panel in panels.items():
        if env==assets.test_environment:raise PermissionError('Test context cannot select a checkpoint')
        paths[env]=generate(model,runtime,env,panel['response_targets'],directory/(env+'.h5'),
            distribution_targets=panel['distribution_targets'],batch_size=protocol['evaluation_condition_batch_size'])
    seal=seal_predictions(checkpoint,list(paths.values()),{e:p['response_targets'] for e,p in panels.items()},
                          directory/'SEAL.json',purpose='validation')
    store=CellStore(assets.train_store.root,assets.fold,role='evaluation')
    store.unlock(seal);records={};objective=0.
    device=runtime.device;gene_lookup={g:i for i,g in enumerate(runtime.genes)}
    try:
        for env,panel in panels.items():
            targets=panel['response_targets'];distribution=panel['distribution_targets']
            references=[]
            for target in targets:
                cells=store.get_p(env,target,'validation');count=len(cells)
                ref=dict(mean=cells.mean(0),variance=cells.var(0),n_cells=count)
                if target in set(distribution):
                    positive=positive_summary(cells);counts=positive['positive_count']
                    rng=np.random.default_rng(seed_int('joint-validation-real',env,target))
                    ids=rng.choice(count,64,replace=count<64)
                    ref.update(real=cells[ids],effective_count=len(np.unique(ids)),
                        positive_quantiles=positive['quantiles'],count_positive=counts,positive_mask=counts>0,
                        detection=(counts/count).astype('f4'),positive_mean=positive['positive_mean'])
                references.append(ref)
            truth=np.stack([r['mean'] for r in references]);control=assets.features.means[env]
            delta=truth-control;mask=runtime.retrieval_gene_mask.copy()
            signal=np.mean(delta[:,mask].astype('f8')**2,axis=1)
            noise=np.asarray([np.mean(r['variance'][mask].astype('f8')/r['n_cells']) for r in references])
            bank=dict(targets=targets,full_target_axis=runtime.targets,responses=_tensor(delta,device),
                mask=torch.as_tensor(mask,device=device),reliability=_tensor(signal/np.maximum(signal+noise,1e-12),device))
            ranking=RankingPanels(bank,runtime.genes,seed=17,epoch=0,
                panel_size=protocol['new_relation_candidates'],temperature=protocol['rank_temperature'],
                norm_floor=protocol['relation_norm_floor'])
            response_loss=0.;dist_loss=0.
            with h5py.File(paths[env],'r') as h:
                predicted=np.array(h['mean'])
                for start in range(0,len(targets),16):
                    selected=targets[start:start+16];n=len(selected)
                    loss,_=joint_response_objective_conditions(
                        dict(mean512_st=_tensor(predicted[start:start+n],device)),dict(mean=_tensor(truth[start:start+n],device)),
                        _tensor(np.broadcast_to(control,(n,len(control))),device),scales['response_scale'],scales['gene_scale'],
                        own_genes=[gene_lookup.get(t,-1) for t in selected],
                        relation=[ranking.relation_inputs(t) for t in selected],weights=protocol['loss_weights'])
                    response_loss+=float(loss)*n/len(targets)
                pos={t:i for i,t in enumerate(targets)}
                for j,target in enumerate(distribution):
                    i=pos[target];ref=references[i]
                    out={name:_tensor(h[name][j:j+1],device) for name in
                         ('support','positive_quantiles','unit_mean_shape','rank_penalty')}
                    out.update(p=_tensor(h['p'][i:i+1],device),r=_tensor(h['r'][i:i+1],device))
                    sup={k:_tensor(v,device) if isinstance(v,np.ndarray) else v
                         for k,v in ref.items() if k not in ('real','effective_count')}
                    gids=np.sort(np.random.default_rng(seed_int('joint-validation-genes',env,target)).choice(
                        len(runtime.genes),min(protocol['marginal_genes'],len(runtime.genes)),replace=False))
                    loss,_=joint_distribution_objective_conditions(out,_tensor(ref['real'][None],device),[sup],
                        scales['gene_scale'],scales['train_projections'][:,:protocol['joint_projections']],scales['components'],
                        own_genes=[gene_lookup.get(target,-1)],sampled_genes=torch.as_tensor(gids[None],device=device),
                        weights=protocol['loss_weights'],control_cells=_tensor(h['control_cells'][j:j+1],device),
                        real_effective_count=[ref['effective_count']],covariance_prior_strength=protocol['covariance_prior_strength'])
                    dist_loss+=float(loss)/len(distribution)
            value=response_loss+dist_loss
            if not math.isfinite(value):raise RuntimeError('Nonfinite validation objective')
            objective+=panel['weight']*value
            records[env]=dict(weight=panel['weight'],role=panel['role'],objective=value,
                response_objective=response_loss,distribution_objective=dist_loss,
                response_conditions=len(targets),distribution_conditions=len(distribution))
        result=dict(objective=objective,epoch=epoch,environments=records,
            selection='minimum fixed weighted output objective; exact ties earliest checkpoint',test_labels_read=0,
            all_predictions_sealed_before_labels=True,checkpoint_sha256=sha256(checkpoint))
        write_json(directory/'metrics.json',result)
        return result
    finally:store.close()


def _sanitize(value):
    if isinstance(value,dict):return {k:_sanitize(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_sanitize(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    return value


@torch.no_grad()
def evaluate(data,predictions,checkpoint,asset_directory,output,*,device='cpu',expected_fold=None):
    """Report modern response + distribution metrics without changing predictions.

    Distribution-less competitors remain distribution-less; no synthetic adapter
    is silently added. A prediction file is evaluated against its own ordered axis.
    """
    directory=Path(asset_directory)
    from .assets import verify_fitted_assets
    binding=read_json(directory/'BINDING.json')
    verify_fitted_assets(directory,binding)
    # Same gene/target names do not imply the same observations. Bind scoring
    # to the prepared dataset used to fit these scales and coordinates before
    # opening predictions, restoring a checkpoint, or authorizing any labels.
    expected_manifest=binding.get('data_manifest_sha256')
    if not isinstance(expected_manifest,str) or sha256(Path(data)/'MANIFEST.json')!=expected_manifest:
        raise PermissionError('Evaluation prepared data differs from the frozen asset binding')
    with h5py.File(predictions,'r') as h:
        if not h.attrs.get('complete'):raise PermissionError('Incomplete predictions')
        method=str(h.attrs.get('method','CoSIT-J0'))
        native=method!='CoSIT-J0'
        comparison_seed=int(h.attrs.get('seed',-1));comparison_fold=str(h.attrs.get('fold',''))
        env=str(h.attrs['environment']);targets=json.loads(h.attrs['targets_json']);genes=json.loads(h.attrs['genes_json'])
        prediction_mean=np.array(h['mean'])
        distribution_targets=json.loads(h.attrs.get('distribution_targets_json','[]'))
        has_cloud='support' in h
    if native:
        # A competitor's own checkpoint is not a CoSIT pickle. Its native
        # selection remains the method's responsibility; only the already
        # fixed fold, scaling and geometry are borrowed from the reference run.
        if comparison_seed not in (17,29,43):raise ValueError('Native export needs its actual initialization seed')
        protocol=read_json(directory.parent/'protocol.json')
        fold=read_json(directory/'BINDING.json')['fold']
        protocol=dict(protocol,fold=fold,seed=comparison_seed)
        if comparison_fold!=fold['id']:raise ValueError('Native prediction fold differs from evaluator geometry')
    else:
        from .training import load_checkpoint
        payload=load_checkpoint(checkpoint,'cpu');protocol=payload['protocol'];fold=protocol['fold']
        if read_json(directory/'BINDING.json')!=payload['asset_binding']:
            raise PermissionError('Checkpoint and evaluator training geometry are not bound')
    if expected_fold is not None and expected_fold!=fold['id']:raise ValueError('Wrong fold')
    if env!=fold['test']:raise PermissionError('Final evaluation requires the registered test environment')
    if not native and not payload['state']['controller']['converged'] and not protocol.get('smoke'):
        # The selected checkpoint predates stopping; SELECTION.json certifies
        # convergence of its complete run rather than modifying its weights.
        selection=read_json(Path(checkpoint).parent.parent/'SELECTION.json')
        if not selection.get('joint_converged') or selection.get('selected_checkpoint')!=str(Path(checkpoint)):
            raise PermissionError('Training convergence and validation selection must be certified')
    store=CellStore(data,fold,role='evaluation')
    if genes!=store.genes or targets!=store.targets:
        store.close();raise ValueError('Final test must preserve the complete frozen gene and target axes')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    seal_record=verify_prediction_seal(checkpoint,[predictions],{env:targets},
        Path(predictions).with_suffix('.seal.json'),purpose='test')
    seal=output.with_suffix('.seal.json')
    write_json(seal,seal_record)
    store.unlock(seal)
    try:
        control=store.get_e_mean(env);truth=[];ap=[]
        with np.load(directory/'statistics/control_geometry.npz',allow_pickle=False) as z:
            geometry={k:np.array(z[k]) for k in z.files}
        gene_scale=geometry['scale']
        # Kernel bandwidths depend only on legal training controls.
        train=np.concatenate([store.get_e(e)/gene_scale for e in fold['train']])
        selected=np.random.default_rng(17).choice(len(train),min(256,len(train)),replace=False)
        distances=pdist(train[selected].astype('f8'))
        base=max(float(np.median(distances[distances>0])) if (distances>0).any() else 1.,1e-6)
        bandwidths=[base*.5,base,base*2.]
        controls=store.get_e(env)
        for target in targets:
            real=store.get_p(env,target,'test');truth.append(real.mean(0))
            idx=genes.index(target) if target in genes else None
            score=deg_ap(real,controls,prediction_mean[len(truth)-1]-control,exclude_index=idx)
            ap.append(score)
        truth=np.stack(truth)
        response=response_metrics(prediction_mean-control,truth-control,targets,genes,compute_pds=True)
        # Historical raw MAE/MSE and legacy composite scores are intentionally
        # absent from the primary report. They remain inside core for old audits.
        response_keys=('PDS_delta','Pearson','Spearman','JAC20','JAC50','JAC100','signed_top100_coverage',
            'paired_raw_cosine','wrong_pair_raw_cosine','raw_cosine_margin','RMS_amplitude_ratio_macro',
            'direction_accuracy_nonzero_truth')
        response_summary={k:response['aggregate'].get(k) for k in response_keys}
        response_summary['DEG_AP']=_finite_mean([v.get('DEG_AP') for v in ap])
        per_response=[]
        for original,deg in zip(response['per_condition'],ap):
            keys=('target','PDS_delta','Pearson','Spearman','JAC20','JAC50','JAC100','signed_top100_coverage',
                  'paired_raw_cosine','direction_accuracy_nonzero_truth','RMS_amplitude_ratio')
            per_response.append(dict({k:original.get(k) for k in keys},**deg))
        distribution=[]
        if has_cloud:
            if distribution_targets!=targets:raise ValueError('Final distribution evaluation requires every test condition')
            with h5py.File(predictions,'r') as h:
                for i,target in enumerate(targets):
                    real=store.get_p(env,target,'test');cloud=np.array(h['support'][i])
                    np.testing.assert_allclose(cloud.mean(0,dtype=np.float64),prediction_mean[i],rtol=3e-5,atol=3e-6)
                    result=distribution_metrics(_tensor(cloud,device),_tensor(real,device),_tensor(gene_scale,device),
                        _tensor(geometry['eval_projections'],device),pca_components=_tensor(geometry['components'],device),
                        pca_offset=_tensor(control/gene_scale+geometry['pca_mean'],device),
                        cluster_centers=_tensor(geometry['cluster_centers'],device),allow_signed=native)
                    result['prediction_negative_fraction']=float(np.mean(cloud<0))
                    for key in ('absolute_mean_MSE','absolute_mean_MAE','absolute_mean_RMSE',
                                'legacy_distribution_composite_not_for_selection'):
                        result.pop(key,None)
                    result.update(gaussian_mmd2_unbiased(real,cloud,gene_scale,bandwidths,env+'|'+target))
                    result.update(energy_distance_normalized(real,cloud,gene_scale,env+'|'+target))
                    distribution.append(dict(target=target,**result))
        numeric=set().union(*(r.keys() for r in distribution)) if distribution else set()
        distribution_summary={k:_finite_mean([r.get(k) for r in distribution]) for k in numeric
            if k!='target' and all(r.get(k) is None or isinstance(r.get(k),(int,float,np.number)) for r in distribution)}
        result=dict(schema='cosit_metrics_v1',method=method,fold=fold['id'],seed=protocol['seed'],environment=env,
            conditions=len(targets),genes=len(genes),response=response_summary,distribution=distribution_summary,
            distribution_available=has_cloud,per_condition_response=per_response,per_condition_distribution=distribution,
            prediction_sha256=sha256(predictions),checkpoint_sha256=sha256(checkpoint),
            scope='retrospective internal normalized-log benchmark; not official VCC scores',
            mmd_bandwidths=bandwidths,primary_response_excludes_own_target=True,
            pds='raw response cosine candidate ranking; all panel target genes excluded; midrank ties',
            distribution_includes_all_genes=True)
        result=_sanitize(result);write_json(output,result)
        return result
    finally:store.close()
