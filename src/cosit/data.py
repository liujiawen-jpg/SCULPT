"""Role-checked access to immutable, normalized single-cell observations.

The gradient store cannot read held query conditions, even on cache hits.
Evaluation reads require a seal binding completed prediction and checkpoint bytes.
Python capabilities prevent accidental leakage; they are not a security sandbox.
"""
from __future__ import annotations
from collections import Counter, OrderedDict
from pathlib import Path
import json
import h5py
import numpy as np
from .io import sha256, read_json
from .splits import CONTEXTS


def verify_prepared_data(root):
    """Hash once, then reuse an identity-bound receipt for immutable files."""
    from .io import write_json
    root=Path(root);manifest_path=root/'MANIFEST.json';manifest=read_json(manifest_path)
    records=manifest['contexts']
    if isinstance(records,list):records={r['context']:r for r in records}
    if set(records)!=set(CONTEXTS):raise ValueError('Prepare all five contexts before training')
    identity={}
    for env in CONTEXTS:
        path=root/records[env]['file'];stat=path.stat()
        identity[env]=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,inode=stat.st_ino)
    expected=dict(manifest_sha256=sha256(manifest_path),identities=identity)
    receipt=root/'.cosit_integrity.json'
    if receipt.exists() and read_json(receipt)==expected:return expected
    for env in CONTEXTS:
        if sha256(root/records[env]['file'])!=records[env]['sha256']:
            raise ValueError('Prepared data differs from manifest: '+env)
    write_json(receipt,expected)
    return expected


class CellStore:
    def __init__(self, root, fold, *, role='train', evaluation_source=False):
        if role not in ('train','evaluation'):
            raise ValueError('Unknown cell-store role')
        self.root,self.fold,self.role=Path(root),fold,role
        verify_prepared_data(self.root)
        self.training_environments=tuple(fold['train'])
        self.evaluation_source=bool(evaluation_source)
        self._episode=None
        self.audit=Counter()
        self._handles={}
        self._seal=None
        self._checked_seals={}
        self._p_cache=OrderedDict()
        self._p_cache_bytes=0
        self._p_budget=256 << 20
        self._headers={}
        self._identities={}
        for env in CONTEXTS:
            path=self.root/(env+'.h5')
            with h5py.File(path,'r') as h:
                if h.attrs.get('schema')!='cosit_cells_v1':
                    raise ValueError(f'Unrecognized prepared-cell schema: {path}')
                genes=json.loads(h.attrs['genes_json']); targets=json.loads(h.attrs['targets_json'])
                if self._headers and (genes!=self.genes or targets!=self.targets):
                    raise ValueError('All contexts must have identical ordered gene/target axes')
                self.genes,self.targets=genes,targets
                if len(set(genes))!=len(genes) or len(set(targets))!=len(targets):
                    raise ValueError('Duplicate axis identifiers')
                if set(h['perturbations'])!=set(targets):
                    raise ValueError(f'Missing or extra prepared conditions in {env}')
                if h['controls'].shape[1]!=len(genes) or len(h['controls'])<2:
                    raise ValueError('Invalid control shape')
                self._headers[env]=dict(path=path,n_controls=int(h.attrs['control_n_full']))
            stat=path.stat()
            self._identities[env]=(stat.st_size,stat.st_mtime_ns,stat.st_ino)
        self.target_set=set(self.targets)
        self._held={e:set(v) for e,v in fold['joint_holdout']['held_targets'].items()}

    def _open(self,env):
        if env not in self._headers:
            raise PermissionError('Unknown environment')
        self._check_identity(env)
        if env not in self._handles:
            self._handles[env]=h5py.File(self._headers[env]['path'],'r')
        return self._handles[env]

    def _check_identity(self,env):
        stat=self._headers[env]['path'].stat()
        if (stat.st_size,stat.st_mtime_ns,stat.st_ino)!=self._identities[env]:
            raise PermissionError('Immutable prepared observations changed during execution')

    def training_targets(self,env):
        if env not in self.training_environments:
            raise PermissionError('Held environment is not a training source')
        return [t for t in self.targets if t not in self._held.get(env,set())]

    def authorize_episode(self,query,sources):
        sources=tuple(sources)
        if (self.role!='train' or query not in CONTEXTS or not sources or query in sources
                or len(set(sources))!=len(sources) or not set(sources)<=set(self.training_environments)):
            raise PermissionError('Invalid query/source capability')
        self._episode=(query,sources)
        return self

    def clear_episode(self):
        self._episode=None

    def _authorize(self,env,targets,purpose):
        if env not in CONTEXTS or not targets or len(set(targets))!=len(targets) or not set(targets)<=self.target_set:
            raise PermissionError('Invalid environment or target axis')
        self._check_identity(env)
        if self.role=='train':
            if env not in self.training_environments or purpose not in ('source','supervision'):
                raise PermissionError('Held response labels are forbidden to the training store')
            if self.evaluation_source:
                import torch
                if purpose!='source' or torch.is_grad_enabled():
                    raise PermissionError('Inference sources are available only under no_grad')
            elif self._held.get(env,set()).intersection(targets):
                raise PermissionError('Internal validation condition is forbidden in all gradient roles')
        else:
            if purpose not in ('validation','test','development','confirmation','reserved') or self._seal is None:
                raise PermissionError('Prediction seal required before evaluator label access')
            self._verify_seal()
            if not set(targets)<=set(self._seal['allowed'].get(env,[])):
                raise PermissionError('Labels are outside the current prediction seal')

    def _check_episode_access(self,env,purpose):
        if self.role!='train':
            return
        if self._episode is None:
            raise PermissionError('Bind a legal episode before any response read')
        query,sources=self._episode
        if purpose=='source' and env not in sources:
            raise PermissionError('Source response is not in the active episode')
        if purpose=='supervision' and env!=query:
            raise PermissionError('Supervision response is not the active query')

    def get_p(self,env,target,purpose='source'):
        self._authorize(env,[target],purpose)
        self._check_episode_access(env,purpose)
        key=(env,target)
        if key in self._p_cache:
            self._p_cache.move_to_end(key)
            return self._p_cache[key]
        x=np.asarray(self._open(env)['perturbations'][target],dtype='f4')
        if x.ndim!=2 or x.shape[1]!=len(self.genes) or len(x)<2 or not np.isfinite(x).all() or (x<0).any():
            raise ValueError('Invalid normalized response cells')
        self.audit[purpose+'_conditions']+=1
        if x.nbytes<=self._p_budget:
            while self._p_cache and self._p_cache_bytes+x.nbytes>self._p_budget:
                _,old=self._p_cache.popitem(last=False);self._p_cache_bytes-=old.nbytes
            x.flags.writeable=False
            self._p_cache[key]=x;self._p_cache_bytes+=x.nbytes
        return x

    def count_p(self,env,target,purpose='source'):
        self._authorize(env,[target],purpose);self._check_episode_access(env,purpose)
        return int(self._open(env)['perturbations'][target].shape[0])

    def get_e(self,env):
        return np.asarray(self._open(env)['controls'],dtype='f4')

    def get_e_mean(self,env):
        return np.asarray(self._open(env)['control_mean'],dtype='f4')

    def get_e_cell_ids(self,env):
        return list(self._open(env)['control_ids'].asstr()[:])

    def control_statistics(self,env):
        h=self._open(env)
        return dict(mean=np.array(h['control_mean']),variance=np.array(h['control_variance']),
                    detection=np.array(h['control_detection']),n_cells=int(h.attrs['control_n_full']))

    def unlock(self,seal_path):
        if self.role!='evaluation':
            raise PermissionError('A training store cannot unlock labels')
        seal=read_json(seal_path)
        if seal.get('schema')!='cosit_prediction_seal_v1' or seal.get('status')!='frozen':
            raise PermissionError('Invalid prediction seal')
        self._seal=seal
        self._verify_seal()

    def _verify_seal(self):
        for record in self._seal['files']:
            path=Path(record['path'])
            stat=path.stat();identity=(stat.st_size,stat.st_mtime_ns,stat.st_ino)
            key=(str(path),record['sha256'])
            if self._checked_seals.get(key)!=identity:
                if sha256(path)!=record['sha256']:
                    raise PermissionError('Prediction/checkpoint changed after sealing')
                self._checked_seals[key]=identity

    def close(self):
        for h in self._handles.values():h.close()
        self._handles.clear();self._p_cache.clear();self._p_cache_bytes=0
