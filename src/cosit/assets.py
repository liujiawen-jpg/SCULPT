"""Fresh fold-only statistics, general pretrained priors, and model inputs."""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import csv
import copy
import numpy as np
import torch
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from .data import CellStore
from .io import read_json, write_json, sha256, canonical_hash
from .splits import CONTEXTS
from .core.metrics import fixed_projections
from .core.pooled_reference import pooled_controls
from .core.runtime import seed_int, statistics, ConnectionRuntime
from .core.geometry import geometry_from_arrays


FITTED_ASSET_FILES = ('statistics/control_geometry.npz',
                      'statistics/response_scaling.npz',
                      'connection_geometry.npz')


def verify_fitted_assets(directory,binding=None):
    """Verify fitted numerical assets before restoration or evaluation.

    The binding is also stored inside each checkpoint, so changing both an NPZ
    and its local receipt cannot silently change a restored predictor.
    """
    directory=Path(directory)
    binding=read_json(directory/'BINDING.json') if binding is None else binding
    expected=binding.get('fitted_assets_sha256')
    if not isinstance(expected,dict) or set(expected)!=set(FITTED_ASSET_FILES):
        raise ValueError('Missing or incomplete fitted asset binding')
    for relative in FITTED_ASSET_FILES:
        path=directory/relative
        if not path.is_file() or sha256(path)!=expected[relative]:
            raise ValueError('Fitted statistic asset changed: '+relative)
    return expected


class SummaryTable:
    """Lazy bounded observational table; a cache hit never bypasses a role check."""
    def __init__(self,features,env,store):
        self.features,self.env,self.store=features,env,store
        self.shape=(len(store.targets),len(store.genes),23)
        self.dtype=np.dtype('f4')
        self.cache=OrderedDict()
        self.bytes=0
        self.budget=128 << 20

    def __getitem__(self,index):
        if isinstance(index,tuple):
            first,*tail=index
            return self[first][(slice(None),*tail)] if not np.isscalar(first) else self[first][tuple(tail)]
        if not np.isscalar(index):
            ids=np.arange(self.shape[0])[index]
            return np.stack([self[int(i)] for i in ids])
        j=int(index);target=self.store.targets[j]
        self.store._authorize(self.env,[target],'source')
        self.store._check_episode_access(self.env,'source')
        if j in self.cache:
            self.cache.move_to_end(j)
            return self.cache[j]
        x=self.store.get_p(self.env,target,'source');n=len(x)
        p=statistics(x);e=self.features.controls[self.env]
        result=np.column_stack((e,p,p-e,np.full(len(e),np.log1p(n)),np.full(len(e),n/(n+32)))).astype('f4')
        while self.cache and self.bytes+result.nbytes>self.budget:
            _,old=self.cache.popitem(last=False);self.bytes-=old.nbytes
        if result.nbytes<=self.budget:
            result.flags.writeable=False
            self.cache[j]=result;self.bytes+=result.nbytes
        return result


class FoldFeatures:
    def __init__(self,store,directory,*,fit=True,existing=None):
        self.store=store;self.root=Path(directory)
        self.root.mkdir(parents=True,exist_ok=True)
        self.targets,self.genes=list(store.targets),list(store.genes)
        self.tidx={t:i for i,t in enumerate(self.targets)}
        self.train_sources=store.training_environments
        self._all_e={e:store.get_e(e) for e in CONTEXTS}
        pooled={e:store.control_statistics(e) for e in CONTEXTS}
        self.means={e:pooled[e]['mean'] for e in CONTEXTS}
        self.pooled_detections={e:pooled[e]['detection'] for e in CONTEXTS}
        self.pooled_variances={e:pooled[e]['variance'] for e in CONTEXTS}
        self.pooled_counts={e:pooled[e]['n_cells'] for e in CONTEXTS}
        self.controls={e:pooled_controls(statistics(self._all_e[e]),self.means[e],self.pooled_detections[e]) for e in CONTEXTS}
        if existing is not None:
            self.geometry=existing.geometry
            self.response_scale=existing.response_scale
            self.tau=existing.tau
            self.strength_quartile=existing.strength_quartile
        elif fit:
            self._fit_geometry()
        else:
            with np.load(self.root/'control_geometry.npz',allow_pickle=False) as z:
                self.geometry={k:np.array(z[k]) for k in z.files}
            with np.load(self.root/'response_scaling.npz',allow_pickle=False) as z:
                self.response_scale=np.array(z['response_scale']);self.tau=float(z['tau'])
                self.strength_quartile=dict(zip(self.targets,z['quartile'].tolist()))
        self.summaries={e:(SummaryTable(self,e,store),None) for e in self.train_sources}
        self._prototypes={}
        for env,x in self._all_e.items():
            indices=np.resize(np.repeat(np.arange(len(x)),2),512)
            control=x[indices].astype('f4',copy=True)
            latent=self.project(control,env).astype('f4')
            uids=store.get_e_cell_ids(env);copies={};noise=[]
            for row in indices:
                uid=uids[int(row)];replica=copies.get(uid,0);copies[uid]=replica+1
                noise.append(np.random.default_rng(seed_int('scratch-prototypes',17,env,uid,replica)).normal(size=16))
            self._prototypes[env]=(control,latent,np.asarray(noise,dtype='f4'))
        if existing is None and fit:
            self._fit_response_scale()

    def _fit_geometry(self):
        x=np.concatenate([self._all_e[e] for e in self.train_sources])
        if min(x.shape)<32:
            raise ValueError('J0 requires at least 32 genes and 32 training control cells')
        scale=np.maximum(x.std(0,dtype=np.float64),.05).astype('f4')
        residual=np.concatenate([(self._all_e[e]-self.means[e])/scale for e in self.train_sources])
        pca=PCA(n_components=32,svd_solver='randomized',random_state=17)
        scores=pca.fit_transform(residual)
        km=KMeans(n_clusters=8,random_state=17,n_init=10).fit(scores)
        self.geometry=dict(scale=scale,components=pca.components_.astype('f4'),pca_mean=pca.mean_.astype('f4'),
            cluster_centers=km.cluster_centers_.astype('f4'),
            train_projections=fixed_projections(len(self.genes),32,17).numpy(),
            eval_projections=fixed_projections(len(self.genes),128,29).numpy())
        np.savez(self.root/'control_geometry.npz',**self.geometry)

    def _fit_response_scale(self):
        responses=[];square=np.zeros(len(self.targets),dtype='f8');counts=np.zeros(len(self.targets),dtype='i8')
        for env in self.train_sources:
            query=next(e for e in self.train_sources if e!=env)
            self.store.authorize_episode(query,(env,))
            for target in self.store.training_targets(env):
                j=self.tidx[target]
                response=self.summaries[env][0][j,:,7]-self.means[env]
                responses.append(response)
                square[j]+=np.square(response,dtype='f8').mean();counts[j]+=1
        self.store.clear_episode()
        if (counts==0).any():
            raise ValueError('A target has no lawful training evidence')
        response=np.stack(responses)
        global_scale=float(np.sqrt(np.square(response,dtype='f8').mean()))
        self.response_scale=np.maximum(np.sqrt(np.square(response,dtype='f8').mean(0)),max(.1*global_scale,.001)).astype('f4')
        self.tau=max(float(np.quantile(np.abs(response),.95)),.001)
        strength=np.sqrt(square/counts)
        order=sorted(range(len(self.targets)),key=lambda i:(strength[i],self.targets[i]))
        self.strength_quartile={self.targets[int(i)]:q for q,ids in enumerate(np.array_split(order,4),1) for i in ids}
        np.savez(self.root/'response_scaling.npz',response_scale=self.response_scale,tau=self.tau,
                 quartile=np.asarray([self.strength_quartile[t] for t in self.targets]))

    def project(self,x,env):
        g=self.geometry
        return ((x-self.means[env])/g['scale']-g['pca_mean'])@g['components'].T


def load_prior(path,axis):
    with np.load(path,allow_pickle=False) as z:
        names=list(z['genes'].astype(str));embedding=np.array(z['embedding'],dtype='f4')
    if len(set(names))!=len(names) or embedding.shape!=(len(names),261) or not np.isfinite(embedding).all():
        raise ValueError('Expected unique stable gene IDs and finite [genes,261] generic priors')
    lookup={g:i for i,g in enumerate(names)}
    result=np.zeros((len(axis),261),dtype='f4')
    for i,g in enumerate(axis):
        if g in lookup:result[i]=embedding[lookup[g]]
    return result


def align_graph(path,axis_path,genes):
    with Path(axis_path).open() as stream:
        reader=csv.DictReader(stream,delimiter='\t');rows=list(reader)
    nodes=[r['gene_id'] for r in rows]
    if len(set(nodes))!=len(nodes):raise ValueError('Duplicate graph gene IDs')
    full=sparse.load_npz(path).tocsr()
    if full.shape!=(len(nodes),len(nodes)) or not np.isfinite(full.data).all() or (full.data<0).any():
        raise ValueError('Invalid STRING adjacency or node axis')
    index={g:i for i,g in enumerate(nodes)}
    ids=np.asarray([index.get(g,-1) for g in genes]);valid=np.flatnonzero(ids>=0)
    sub=full[ids[valid]][:,ids[valid]].tocoo()
    graph=sparse.csr_matrix((sub.data,(valid[sub.row],valid[sub.col])),shape=(len(genes),len(genes)))
    return graph


def prepare_assets(data,priors,graph,graph_axis,fold,directory,device,protocol,*,fit=True):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    # Validate serialized statistics before FoldFeatures loads them. This
    # protects PCA, normalization and connection coordinates independently of
    # the immutable raw-cell and external-prior bindings below.
    recorded=None
    if not fit:
        recorded=read_json(directory/'BINDING.json')
        verify_fitted_assets(directory,recorded)
    store=CellStore(data,fold)
    features=FoldFeatures(store,directory/'statistics',fit=fit)
    gene_prior=load_prior(priors,store.genes);target_prior=load_prior(priors,store.targets)
    identities={}
    for env in CONTEXTS:
        path=Path(data)/(env+'.h5');stat=path.stat()
        identities[env]=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    binding=dict(schema='cosit_fold_assets_v1',fold=fold,files=[],prepared_file_identities=identities,
        data_manifest_sha256=sha256(Path(data)/'MANIFEST.json'),
        generic_priors_sha256=sha256(priors),graph_sha256=sha256(graph),graph_axis_sha256=sha256(graph_axis),
        target_axis_sha256=canonical_hash(store.targets),gene_axis_sha256=canonical_hash(store.genes),
        historical_task_weights=0,historical_prediction_inputs=0)
    # Bind only training P / public controls. The prepared manifest already
    # binds raw bytes once; runtime never opens validation/test P during fitting.
    assets=SimpleNamespace(train_store=store,features=features,gene_names=store.genes,target_names=store.targets,
        gene_prior=gene_prior,target_prior=target_prior,binding=binding,fold=fold,
        training_environments=tuple(fold['train']),environments=CONTEXTS,
        validation_environment=fold['validation'],test_environment=fold['test'],
        response_scale=features.response_scale,tau=features.tau,device=device,fold_cache=str(directory),
        canonical_fold_cache=str(directory),local_stager=None,joint_training=True,pooled_reference_applied=True)
    geometry_path=directory/'connection_geometry.npz'
    if fit:
        controls=np.stack([features.controls[e] for e in fold['train']])
        geometry=geometry_from_arrays(gene_prior,controls,features.geometry['scale'],
            align_graph(graph,graph_axis,store.genes),rank=protocol['local_rank'],ambient=protocol['ambient_dimension'])
        np.savez(geometry_path,**geometry)
        binding['fitted_assets_sha256']={relative:sha256(directory/relative) for relative in FITTED_ASSET_FILES}
        write_json(directory/'BINDING.json',binding)
    else:
        binding['fitted_assets_sha256']=recorded['fitted_assets_sha256']
        if recorded!=binding:
            raise ValueError('Prepared inputs differ from the training binding')
        with np.load(geometry_path,allow_pickle=False) as z:
            geometry={k:np.array(z[k]) for k in z.files}
    runtime=ConnectionRuntime(assets,geometry,directory/'training_runtime/source_precision',device,
        performance_settings=dict(pin_memory=False,non_blocking=False))
    return assets,geometry,runtime


def evaluation_runtime(assets,geometry,directory,device):
    store=CellStore(assets.train_store.root,assets.fold,evaluation_source=True)
    features=FoldFeatures(store,Path(directory)/'readonly_statistics',fit=False,existing=assets.features)
    view=copy.copy(assets);view.train_store=store;view.features=features;view.inference_sources_only=True
    # This capability can read training sources only under no_grad. Query
    # responses remain exclusive to a separately sealed evaluator store.
    return ConnectionRuntime(view,geometry,Path(directory)/'source_precision',device,
        performance_settings=dict(pin_memory=False,non_blocking=False))


def device_scales(assets,device):
    f=assets.features
    return {key:torch.as_tensor(value,device=device) for key,value in dict(
        response_scale=f.response_scale,gene_scale=f.geometry['scale'],components=f.geometry['components'],
        train_projections=f.geometry['train_projections'],eval_projections=f.geometry['eval_projections']).items()}
