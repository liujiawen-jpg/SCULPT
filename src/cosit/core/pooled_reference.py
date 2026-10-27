"""Derived RAM references; sealed empirical caches are never overwritten.

Mean and detection use the full eligible control roster. Standard deviation and
quantiles retain the archived 256-control shape sample and its bootstrap noise.
The precision columns for mean/detection are recomputed from sufficient moments,
not recovered by inverting clipped historical precision.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from types import MethodType
import numpy as np
import torch

CORE3 = ('K562_essential', 'HepG2', 'Orion_HCT116')
POOLED_COLUMNS = (0, 2)
SAMPLED_COLUMNS = (1, 3, 4, 5, 6)


def array_sha(x):
    x=np.ascontiguousarray(x)
    h=hashlib.sha256();h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
    return h.hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def _count(n, name):
    value=np.asarray(n)
    if value.ndim!=0 or not np.issubdtype(value.dtype,np.integer) or int(value)<2:
        raise ValueError(name+' must be an integer >=2')
    return int(value)


def pooled_controls(sample_stats, mean, detection):
    sample=np.asarray(sample_stats,dtype='f4');mean=np.asarray(mean,dtype='f4');detection=np.asarray(detection,dtype='f4')
    if sample.ndim!=2 or sample.shape[1]!=7 or mean.shape!=(len(sample),) or detection.shape!=mean.shape:
        raise ValueError('Control statistic axes differ')
    if not all(np.isfinite(x).all() for x in (sample,mean,detection)) or (sample<0).any() or (mean<0).any() or ((detection<0)|(detection>1)).any():
        raise ValueError('Invalid finite nonnegative control moments')
    out=sample.copy();out[:,0]=mean;out[:,2]=detection
    return out


def mean_detection_uncertainty(variance, detection, n):
    """Same analytic formulas as runtime.sampling_variance, from sufficient stats.

    Stored variance is population variance. Its unbiased sample mean variance is
    population_variance/(n-1). The empirical detection rate is half-count smoothed.
    """
    n=_count(n,'n_cells');v=np.asarray(variance,dtype=np.float64);d=np.asarray(detection,dtype=np.float64)
    if v.ndim!=1 or d.shape!=v.shape or not np.isfinite(v).all() or not np.isfinite(d).all() or (v<0).any() or ((d<0)|(d>1)).any():
        raise ValueError('Invalid variance or detection sufficient statistics')
    rate=(d*n+.5)/(n+1.)
    return v/(n-1.), rate*(1.-rate)/(n+1.)


def pooled_precision(old, p_variance, p_detection, n_p, e_variance, e_detection, n_e, stat_scale):
    old=np.asarray(old,dtype='f4');scale=np.asarray(stat_scale,dtype=np.float64)
    if old.ndim!=2 or old.shape[1]!=7 or scale.shape!=old.shape or not np.isfinite(old).all() or not np.isfinite(scale).all() or (scale<=0).any() or (old<.01-1e-7).any() or (old>100+1e-5).any():
        raise ValueError('Invalid precision axes/range/scale')
    pm,pd=mean_detection_uncertainty(p_variance,p_detection,n_p);em,ed=mean_detection_uncertainty(e_variance,e_detection,n_e)
    if pm.shape!=(len(old),) or em.shape!=pm.shape:raise ValueError('Precision sufficient-statistic axes differ')
    out=old.copy()
    for col,var in ((0,pm+em),(2,pd+ed)):
        out[:,col]=np.clip(1./(var/scale[:,col]**2+.01),.01,100).astype('f4')
    return out


def attach_source_readout_targets(runtime, batch):
    """Only already-selected, source-authorized observations; never query P."""
    sources=tuple(batch['sources']);target=batch['target'];query=batch['query']
    training_environments=tuple(getattr(runtime.assets,'training_environments',CORE3))
    if query in sources or not sources or not set(sources)<=set(training_environments):raise PermissionError('Invalid source readout scope')
    features=runtime.assets.features;j=features.tidx[target]
    means=[];detections=[];control_means=[];control_detections=[]
    for source in sources:
        runtime._source_authorize(source,target)
        block=features.summaries[source][0][j]
        means.append(block[:,7]);detections.append(block[:,9])
        control_means.append(features.means[source]);control_detections.append(features.pooled_detections[source])
    for name,values in (('source_means',means),('source_detections',detections),('source_control_means',control_means),('source_control_detections',control_detections)):
        value=np.stack(values).astype('f4')
        if value.shape!=(len(sources),len(runtime.genes)) or not np.isfinite(value).all():raise ValueError('Invalid source readout axis')
        batch[name]=torch.as_tensor(value,device=runtime.device).detach()
    return batch


def apply_pooled_reference(assets, geometry, scales, runtime, output_dir, *, geometry_builder=None, update_relation_noise=False):
    """Apply once before banks/models are created; return (new_geometry, audit).

    output_dir must be a fresh adapter directory. New connection geometry is
    independently bound there; inherited geometry remains unchanged. Both trial
    arms must use the same returned geometry and adapted runtime.
    """
    if update_relation_noise:
        raise ValueError('This paired trial fixes historical P-only ranking reliability; no reliability intervention authorized')
    if getattr(runtime,'_pooled_reference_applied',False):raise RuntimeError('Pooled reference already applied')
    output=Path(output_dir)
    if output.exists():raise FileExistsError('Derived adapter directory must be new')
    output.mkdir(parents=True)
    if runtime.assets is not assets or runtime.geometry is not geometry:raise ValueError('Runtime/asset geometry identity differs')
    f=assets.features;targets=list(runtime.targets);genes=list(runtime.genes);envs=list(f.controls)
    training_environments=tuple(getattr(assets,"training_environments",CORE3))
    if targets!=list(assets.target_names) or genes!=list(assets.gene_names) or f.tidx!={t:i for i,t in enumerate(targets)}:
        raise ValueError('Pooled adapter requires aligned target/gene axes')
    if set(training_environments)-set(envs):raise ValueError('Missing source control environments')
    if runtime._response_banks:raise RuntimeError('Apply pooled reference before building training response banks')
    scale=np.asarray(geometry['stat_scale']);scale_sha=array_sha(scale)
    old_scale_sha=array_sha(f.response_scale);old_tau=float(f.tau)
    old_binding=json.loads(json.dumps(assets.binding))
    audit=dict(schema='pooled_mean_detection_ram_reference_v1',reference_columns={'mean':0,'detection':2},
        sampled_shape_columns=list(SAMPLED_COLUMNS),response_scaling_changed=False,joint625_changed=False,
        old_sealed_cache_files_modified=False,task_weights_read=0,perturbation_file_reads=0,
        relation_reliability='unchanged P-only variance/n; omits control mean noise; no extra rank weighting intervention',
        precision='columns0,2 recomputed analytically from cached P moments plus full-roster E moments; other columns retain original sample-E bootstrap precision; floor0.01 standardized, clip[0.01,100]',
        source_precision_previously_included_sample_E_noise=True,
        controls={},conditions=0,old_asset_binding=old_binding)
    # Prepare and validate all control moments before mutating in-memory inputs.
    prepared={}
    for env in envs:
        n=_count(f.pooled_counts[env],env+' pooled controls')
        controls=pooled_controls(f.controls[env],f.means[env],f.pooled_detections[env])
        mean_detection_uncertainty(f.pooled_variances[env],f.pooled_detections[env],n)
        prepared[env]=controls
        audit['controls'][env]=dict(sample_control_sha256=array_sha(f.controls[env]),pooled_derived_control_sha256=array_sha(controls),full_roster_count=n,
            mean_reference_shift_rms=float(np.sqrt(np.mean((f.controls[env][:,0].astype(float)-controls[:,0])**2))),
            detection_reference_shift_rms=float(np.sqrt(np.mean((f.controls[env][:,2].astype(float)-controls[:,2])**2))))
    condition_audit=[]
    for env in training_environments:
        blocks=f.summaries[env][0]
        if blocks.shape!=(len(targets),len(genes),23):raise ValueError('Source summary axes changed')
        for j,target in enumerate(targets):
            sup=runtime._supervision_ram[(env,target)];stats=np.asarray(sup['full_stats']);n=_count(sup['n_cells'],'P cells')
            if stats.shape!=(len(genes),7) or not np.allclose(blocks[j,:,7:14],stats,rtol=1e-5,atol=1e-6):raise ValueError('Source and supervision moments disagree')
            old_precision,precision_n=runtime._precision_ram[(env,target)]
            if n!=int(precision_n):raise ValueError('Source precision count changed')
            new_precision=pooled_precision(old_precision,sup['variance'],sup['detection'],n,
                f.pooled_variances[env],f.pooled_detections[env],f.pooled_counts[env],scale)
            new_statistics=((stats-prepared[env])/scale).astype('f4')
            if not np.allclose(new_statistics[:,0]*scale[:,0],sup['response'],rtol=1e-5,atol=2e-7):raise ValueError('Mean statistic and response reference disagree')
            if not np.array_equal(new_precision[:,SAMPLED_COLUMNS],old_precision[:,SAMPLED_COLUMNS]):raise AssertionError('Sampled-shape precision altered')
            condition_audit.append(dict(environment=env,target=target,cells=n,old_precision_sha256=array_sha(old_precision),new_precision_sha256=array_sha(new_precision),
                old_statistics_sha256=array_sha(sup['statistics']),new_statistics_sha256=array_sha(new_statistics)))
            blocks[j,:,:7]=prepared[env];blocks[j,:,14:21]=stats-prepared[env]
            sup['statistics']=new_statistics
            runtime._precision_ram[(env,target)]=(new_precision,n)
    for env in envs:
        f.controls[env]=prepared[env]
        runtime._device['controls'][env]=torch.as_tensor(prepared[env].copy(),device=runtime.device,dtype=torch.float32)
    if geometry_builder is None:
        from .geometry import build_geometry
        geometry_builder=build_geometry
    new_geometry=geometry_builder(assets,output/'geometry',int(geometry['bases'].shape[2]),int(geometry['bases'].shape[1]))
    if array_sha(new_geometry['stat_scale'])!=scale_sha:raise ValueError('Unexpected scale change during control-only geometry rebuild')
    if array_sha(f.response_scale)!=old_scale_sha or float(f.tau)!=old_tau:raise AssertionError('Response scaling changed')
    geometry_path=output/'geometry/connection_geometry.npz';geometry_binding=output/'geometry/binding.json'
    if not geometry_path.is_file() or not geometry_binding.is_file():raise RuntimeError('New geometry needs independent persisted binding')
    runtime.geometry=new_geometry
    old_batch=runtime.batch
    def adapted_batch(self,*args,**kwargs):
        result=old_batch(*args,**kwargs)
        keys=('source_control_means','source_control_detections','source_means','source_detections')
        return result if all(key in result for key in keys) else attach_source_readout_targets(self,result)
    runtime.batch=MethodType(adapted_batch,runtime);runtime._pooled_reference_applied=True
    audit.update(conditions=len(condition_audit),geometry_path=str(geometry_path.resolve()),geometry_binding_path=str(geometry_binding.resolve()),
        geometry_sha256=hashlib.sha256(geometry_path.read_bytes()).hexdigest(),geometry_rebuilt_from_updated_controls=True,
        stat_scale_sha256=scale_sha,response_scale_sha256=old_scale_sha,condition_records=condition_audit,
        source_readout_fields=['source_control_means','source_control_detections','source_means','source_detections'],
        source_readout_scope='Only authorized selected sources in current batch; query P unavailable')
    audit_path=output/'POOLED_REFERENCE.json';_write(audit_path,audit)
    assets.binding=dict(schema='small_pooled_reference_derived_assets_v1',parent=old_binding,
        reference_audit_path=str(audit_path.resolve()),reference_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest())
    _write(output/'DERIVED_ASSET_BINDING.json',assets.binding)
    return new_geometry,audit
