"""Deterministic independent-condition batches and lossless frozen encodings.

A batch never merges biological distributions. Every record retains its query,
target, permitted sources, original episode identity and cell axis.
"""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .common import atomic, sha
from .runtime import seed_int
from .training_support import FrozenEncodingCache, detach_tree, transfer_frozen_records, load_verified_frozen_payload


def _source_rotation(query, environments, epoch, position):
    other = tuple(e for e in environments if e != query)
    if len(environments) != 3 or len(other) != 2:
        raise ValueError('Exactly three distinct training environments required')
    mode = (int(epoch) + int(position)) % 4
    return other if mode in (0, 2) else (other[0 if mode == 1 else 1],)


def condition_schedule(episodes, environments, *, seed, phase, epoch, batch_size,
                       validation_at=()):
    """Group original episodes by identical query/sources; split at eval boundaries.

    The original permutation defines immutable per-condition random identities.
    Full groups emit as they fill; residual groups emit in first-seen order.
    All original conditions occur exactly once, including final partial groups.
    """
    episodes = list(episodes); environments = tuple(environments)
    if not episodes or len(set(episodes)) != len(episodes) or batch_size < 1:
        raise ValueError('Unique episodes and positive batch size required')
    if len(environments) != 3 or len(set(environments)) != 3:
        raise ValueError('Three distinct training environments required')
    if any(env not in environments for env, _ in episodes):
        raise PermissionError('Episode query outside fold training environments')
    order = np.random.default_rng(seed_int('benchmark-episode', seed, phase, epoch)).permutation(len(episodes))
    pending = OrderedDict(); groups = []
    for original_offset, raw_position in enumerate(order):
        position = int(raw_position); env, target = episodes[position]
        sources = _source_rotation(env, environments, epoch, position)
        key = (env, sources)
        item = dict(position=position, original_offset=original_offset, query=env,
                    target=target, sources=sources)
        bucket = pending.setdefault(key, [])
        bucket.append(item)
        if len(bucket) == batch_size:
            groups.append(bucket[:]); pending[key] = []
    groups.extend(bucket[:] for bucket in pending.values() if bucket)
    lower = (int(epoch) - 1) * len(episodes); upper = lower + len(episodes)
    boundaries = sorted({int(v) for v in validation_at if lower < int(v) <= upper} | {upper})
    batches = []; consumed = lower
    for group in groups:
        remaining = list(group)
        while remaining:
            boundary = next(v for v in boundaries if v > consumed)
            take = min(len(remaining), boundary - consumed)
            items, remaining = remaining[:take], remaining[take:]
            consumed += take
            batches.append(dict(items=items, condition_count=take,
                phase_conditions_after=consumed, validation_due=consumed in set(validation_at)))
    flat = [x['position'] for batch in batches for x in batch['items']]
    if sorted(flat) != list(range(len(episodes))) or consumed != upper:
        raise AssertionError('Condition schedule lost or duplicated episodes')
    for batch in batches:
        identities = {(x['query'], tuple(x['sources'])) for x in batch['items']}
        if len(identities) != 1:
            raise AssertionError('Batch mixed source/query layouts')
    payload = dict(schema='condition_batches_v1', seed=int(seed), phase=phase, epoch=int(epoch),
        requested_batch_size=int(batch_size), episodes=len(episodes),
        original_order_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(), batches=batches)
    payload['schedule_sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def collate_condition_batches(batches):
    """Stack every tensor on Q, while keeping shared authorization metadata."""
    batches = list(batches)
    if not batches:
        raise ValueError('Empty condition batch')
    keys = set(batches[0])
    if any(set(item) != keys for item in batches):
        raise ValueError('Condition tensor schemas differ')
    if 'query' in keys and len({item['query'] for item in batches}) != 1:
        raise PermissionError('A condition batch must share one query environment')
    if 'sources' in keys:
        scopes = {tuple(item['sources']) for item in batches}
        if len(scopes) != 1 or ('query' in keys and batches[0]['query'] in next(iter(scopes))):
            raise PermissionError('A condition batch must share legal source environments')
    result = {'_private_condition_count': len(batches)}
    for key in keys:
        values = [item[key] for item in batches]
        if all(isinstance(v, torch.Tensor) for v in values):
            if any(v.shape != values[0].shape or v.dtype != values[0].dtype or v.device != values[0].device for v in values):
                raise ValueError('Condition tensor axes/dtypes/devices differ: ' + key)
            result[key] = torch.stack(values)
        elif any(isinstance(v, torch.Tensor) for v in values):
            raise TypeError('Tensor/non-tensor mixture: ' + key)
        elif key in ('query', 'sources', 'cache_format'):
            if any(v != values[0] for v in values):
                raise ValueError('Shared condition metadata changed: ' + key)
            result[key] = values[0]
        else:
            result[key] = values
    return result


class ConditionEncodingCache(FrozenEncodingCache):
    """Same cache contract, but all missing conditions use one batched solve."""
    def get_many(self, model, batches):
        from .batched_model import split_condition_outputs
        batches = list(batches)
        if not batches or model.phase not in ('B', 'C'):
            raise PermissionError('Frozen batched encodings require B/C conditions')
        if any(p.requires_grad for name, p in model.named_parameters()
               if not name.startswith(('shape.', 'reorder.'))):
            raise PermissionError('Upstream parameters must be frozen')
        if getattr(self, '_resident_authorize', None) is not None:
            for batch in batches:self._resident_authorize(batch)
        upstream_sha = self._verify_upstream(model)
        values = [None] * len(batches); missing = []
        for i, batch in enumerate(batches):
            key = hashlib.sha256(json.dumps([batch['query'], batch['target'], list(batch['sources'])]).encode()).hexdigest()
            path = self.directory / f'{key}.pt'
            if key in self.memory:
                self.hits += 1
                values[i] = self.memory.pop(key); self.memory[key] = values[i]
                if key in self.files:
                    self.files.move_to_end(key)
            elif key in getattr(self, '_resident_required_keys', ()) and getattr(self,'_resident_ready',False):
                raise RuntimeError('READY training encoding is missing from resident memory')
            elif path.exists():
                receipt_path = path.with_suffix('.json')
                if not receipt_path.exists():
                    raise PermissionError('Frozen encoding lacks byte receipt')
                receipt = json.loads(receipt_path.read_text())
                payload = load_verified_frozen_payload(path,receipt,key,upstream_sha,self.binding)
                self.hits += 1; values[i] = detach_tree(payload['encoded'], getattr(self,'_resident_device','cpu')); self.memory[key] = values[i]
                self.files.move_to_end(key)
            else:
                self.misses += 1; missing.append((i, key, path))
        if missing:
            with torch.no_grad():
                encoded = model.encode_conditions([batches[i] for i, _, _ in missing])
                outputs = split_condition_outputs(encoded)
            if len(outputs) != len(missing):
                raise RuntimeError('Batched encoding condition count changed')
            for (i, key, path), output in zip(missing, outputs):
                value = detach_tree(model.compress_encoded(output))
                temp = path.with_suffix('.tmp')
                torch.save(dict(binding=self.binding, upstream_sha256=upstream_sha, key=key, encoded=value), temp)
                temp.replace(path)
                atomic(path.with_suffix('.json'), dict(key=key, upstream_sha256=upstream_sha, sha256=sha(path)))
                size = path.stat().st_size
                self.files[key] = size; self.disk_bytes += size
                value = detach_tree(value, getattr(self,'_resident_device','cpu'))
                self.memory[key] = value; values[i] = value
            self._evict()
        while len(self.memory) > self.memory_entries:
            self.memory.popitem(last=False)
        devices={batch['control_cells'].device for batch in batches}
        if len(devices)!=1:raise ValueError('Frozen encoding devices differ')
        return transfer_frozen_records(values,next(iter(devices)))


    def enable_resident(self, device, *, max_bytes=3 << 30):
        """Current-seed compact encodings only; never shared across fold seeds."""
        from .training_support import ResidentEncodingMemory
        device=torch.device(device)
        if getattr(self,'_resident_device',None) is not None:
            if self._resident_device!=device or self.memory.max_bytes!=int(max_bytes):
                raise ValueError('Cannot change a live encoding residency contract')
            return self
        memory=ResidentEncodingMemory(max_bytes)
        for key,value in self.memory.items():memory[key]=detach_tree(value,device)
        self.memory=memory;self.memory_entries=1 << 30
        self._resident_device=device;self._resident_required_keys=set();self._resident_ready=False
        return self

    def prewarm_training(self, model, runtime, *, batch_size=32, n_cells=64):
        """Prepare exactly 3 query environments x 3 legal source sets x targets.

        The compressed upstream result is independent of sampled control cells;
        only later shape/reorder generation consumes that changing roster.
        """
        if getattr(self,'_resident_device',None) is None:
            raise RuntimeError('enable_resident is required before training prewarm')
        if batch_size<1:raise ValueError('Positive encoding preparation batch size required')
        def authorize(batch):
            query,target,sources=batch['query'],batch['target'],tuple(batch['sources'])
            runtime.assets.train_store.authorize_episode(query,sources)
            for source in sources:runtime._source_authorize(source,target)
        self._resident_authorize=authorize
        required=set()
        groups=[]
        for query in runtime.training_environments:
            others=tuple(e for e in runtime.training_environments if e!=query)
            for sources in (others,(others[0],),(others[1],)):
                records=[dict(query=query,target=t,sources=sources) for t in runtime.targets]
                groups.append(records)
                for record in records:
                    required.add(hashlib.sha256(json.dumps([query,record['target'],list(sources)]).encode()).hexdigest())
        if self._resident_ready:
            self._verify_upstream(model)
            if required!=self._resident_required_keys or not required<=set(self.memory):
                raise PermissionError('Resident training encoding scope changed')
            return self.audit()
        self._resident_required_keys=required
        self.memory.required=required
        for records in groups:
            for offset in range(0,len(records),batch_size):
                batches=runtime.batch_conditions(records[offset:offset+batch_size],n_cells=n_cells,
                    tag='resident-upstream-preparation')
                self.get_many(model,batches)
        if not required<=set(self.memory):raise RuntimeError('Encoding preparation did not retain all training conditions')
        self._resident_ready=True
        return self.audit()


def joint_condition_schedule(fold, targets, *, seed, epoch, per_environment=16):
    """Balanced effective batches; homogeneous microbatches preserve legal sources.

    Every allowed query condition occurs once. A held condition cannot occur as
    source either, including when it would otherwise belong to another query.
    """
    environments=tuple(fold['train']); targets=list(targets)
    held={env:set(fold['joint_holdout']['held_targets'][env]) for env in environments}
    if len(environments)!=3 or per_environment<1: raise ValueError('Three training environments required')
    orders={}
    for env in environments:
        allowed=[target for target in targets if target not in held[env]]
        rng=np.random.default_rng(seed_int('joint-environment-order',seed,epoch,env))
        orders[env]=[allowed[int(i)] for i in rng.permutation(len(allowed))]
    if len({len(v) for v in orders.values()})!=1: raise ValueError('Balanced holdouts require equal condition counts')
    # Consolidate identical source layouts before slicing each environment.
    # This preserves exactly one visit and 16 queries/environment/update while
    # reducing padded/source-shape microbatches from ~12 to normally three.
    ordered_records={}
    for env in environments:
        buckets=OrderedDict()
        for position,target in enumerate(orders[env]):
            candidates=tuple(e for e in environments if e!=env and target not in held[e])
            if not candidates: raise PermissionError('Query has no legal source response')
            rng=np.random.default_rng(seed_int('joint-source-subset',seed,epoch,env,target))
            if len(candidates)==2 and rng.random()<.5:
                candidates=(candidates[int(rng.integers(2))],)
            elif len(candidates)==2 and rng.random()<.5:
                candidates=candidates[::-1]
            record=dict(query=env,target=target,sources=candidates,position=position,original_offset=position)
            buckets.setdefault(candidates,[]).append(record)
        keys=list(buckets)
        rng=np.random.default_rng(seed_int('joint-source-layout-order',seed,epoch,env))
        keys=[keys[int(i)] for i in rng.permutation(len(keys))]
        ordered_records[env]=[record for key in keys for record in buckets[key]]
    batches=[]; total=0
    for start in range(0,len(next(iter(orders.values()))),per_environment):
        items=[]; groups=OrderedDict()
        for env in environments:
            for record in ordered_records[env][start:start+per_environment]:
                items.append(record); groups.setdefault((env,record['sources']),[]).append(record)
        total+=len(items)
        batches.append(dict(items=items,microbatches=[dict(items=group) for group in groups.values()],
            condition_count=len(items),phase_conditions_after=(epoch-1)*sum(map(len,orders.values()))+total))
    payload=dict(schema='joint_balanced_condition_batches_v1',seed=seed,epoch=epoch,
        episodes=total,requested_batch_size=3*per_environment,
        grouping='randomized source-layout buckets within each environment, then balanced slices',batches=batches)
    payload['schedule_sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    return payload
