"""One/two/three-source evidence without pair-mixture or response-weighted gating."""
from __future__ import annotations

import hashlib
import contextlib
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
import json
from numbers import Integral
from pathlib import Path
import sys

import numpy as np
import torch

CORE3 = ('K562_essential', 'HepG2', 'Orion_HCT116')
HELD = ('RPE1', 'Jurkat')


@dataclass(frozen=True)
class RuntimePerformanceSettings:
    """CPU payload budgets and transfer policy; no scientific cache inputs."""
    source_moment_cache_bytes: int = 768 << 20
    supervision_moment_cache_bytes: int = 1024 << 20
    precision_cache_bytes: int = 256 << 20
    pin_memory: bool = True
    non_blocking: bool = True

    def __post_init__(self):
        for name in ('source_moment_cache_bytes', 'supervision_moment_cache_bytes', 'precision_cache_bytes'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(name + ' must be a nonnegative integer byte budget')
            object.__setattr__(self, name, int(value))
        if not isinstance(self.pin_memory, bool) or not isinstance(self.non_blocking, bool):
            raise ValueError('Transfer settings must be bools')


def _cpu_payload_bytes(value):
    """Conservative retained-payload accounting, including container metadata.

    Views are charged for their entire backing allocation so a small view cannot
    conceal a large retained CPU buffer. Shared buffers are counted once within
    an entry, and may be overcounted across entries to keep eviction conservative.
    """
    seen = set()
    def size(item):
        if id(item) in seen: return 0
        seen.add(id(item))
        if isinstance(item, np.ndarray):
            # sys.getsizeof already includes owned numpy data; a view retains
            # its base and is charged for that allocation plus view metadata.
            return sys.getsizeof(item) + (size(item.base) if item.base is not None else 0)
        if isinstance(item, torch.Tensor):
            if item.device.type != 'cpu': raise ValueError('Observation memory caches must remain on CPU')
            return sys.getsizeof(item) + item.untyped_storage().nbytes()
        if isinstance(item, dict):
            return sys.getsizeof(item) + sum(size(k) + size(v) for k, v in item.items())
        if isinstance(item, (tuple, list)):
            return sys.getsizeof(item) + sum(size(v) for v in item)
        return sys.getsizeof(item)
    return size(value)


class CPUByteLRU(OrderedDict):
    """LRU observations with a hard accounted-byte limit, including zero budget."""
    def __init__(self, max_bytes):
        super().__init__()
        self.max_bytes = int(max_bytes)
        if self.max_bytes < 0: raise ValueError('Negative cache byte budget')
        self.current_bytes = 0
        self.entry_bytes = {}
        self.evictions = self.oversized = 0

    def __setitem__(self, key, value):
        cost = _cpu_payload_bytes((key, value)) + 128  # LRU node/accounting overhead
        if key in self: self.__delitem__(key)
        if cost > self.max_bytes:
            self.oversized += 1
            return
        while self and self.current_bytes + cost > self.max_bytes:
            self.popitem(last=False)
            self.evictions += 1
        super().__setitem__(key, value)
        self.entry_bytes[key] = cost
        self.current_bytes += cost

    def __delitem__(self, key):
        super().__delitem__(key)
        self.current_bytes -= self.entry_bytes.pop(key)

    def popitem(self, last=True):
        key, value = super().popitem(last=last)
        self.current_bytes -= self.entry_bytes.pop(key)
        return key, value

    def pop(self, key, *default):
        if key not in self:
            if default: return default[0]
            raise KeyError(key)
        value = super().__getitem__(key)
        self.__delitem__(key)
        return value

    def clear(self):
        super().clear()
        self.entry_bytes.clear()
        self.current_bytes = 0

    def audit(self):
        return dict(entries=len(self), bytes=self.current_bytes, max_bytes=self.max_bytes,
                    evictions=self.evictions, oversized=self.oversized)


def seed_int(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'little')


def tensor(x, device):
    return torch.as_tensor(np.array(x, copy=True), device=device, dtype=torch.float32)


def statistics(x):
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('At least two finite observed cells required')
    return np.column_stack((x.mean(0), x.std(0), (x > 0).mean(0),
                            np.quantile(x, [.25, .5, .75, .9], axis=0).T)).astype('f4')


def sampling_variance(x, seed, replicates=12):
    """Within-condition cell bootstrap; it is not cross-environment disagreement."""
    rng = np.random.default_rng(seed)
    draws = np.stack([statistics(x[rng.integers(len(x), size=len(x))]) for _ in range(replicates)])
    var = draws.var(0, ddof=1)
    # Analytic mean/detection uncertainty avoids an accidental zero bootstrap
    # variance for rare detections and known binomial boundaries.
    var[:, 0] = x.var(0, ddof=1) / len(x)
    rate = ((x > 0).sum(0) + .5) / (len(x) + 1)
    var[:, 2] = rate * (1 - rate) / (len(x) + 1)
    return var.astype('f4')


def validate_sources(query, sources, training_environments=CORE3, environments=None):
    sources = tuple(sources)
    training_environments = tuple(training_environments)
    environments = tuple(environments or (*CORE3, *HELD))
    if query not in environments or not sources or len(set(sources)) != len(sources):
        raise PermissionError('Unknown query, repeated or empty sources')
    if query in sources or not set(sources) <= set(training_environments):
        raise PermissionError('Query response and held responses cannot be source evidence')
    return tuple(s for s in training_environments if s in sources)



QUANTILE_GRID = np.linspace(0., 1., 33, dtype=np.float64)


def positive_summary(x):
    """Exact per-gene positive empirical quantiles; absent genes remain masked."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Positive summary needs finite nonnegative multiple-cell data')
    positive = x > 0
    count = positive.sum(0).astype(np.int64)
    ordered = np.sort(np.where(positive, x, np.inf), axis=0)
    position = np.maximum(count - 1, 0)[:, None] * QUANTILE_GRID[None]
    lo, hi = np.floor(position).astype(np.int64), np.ceil(position).astype(np.int64)
    columns = np.arange(x.shape[1])[:, None]
    lower, upper = ordered[lo, columns], ordered[hi, columns]
    absent = count == 0
    lower[absent] = 0.; upper[absent] = 0.
    fraction = position - lo
    quantiles = ((1 - fraction) * lower + fraction * upper).astype('f4')
    mean = (x.sum(0, dtype=np.float64) / np.maximum(count, 1)).astype('f4')
    return dict(quantiles=quantiles, positive_mean=mean, positive_count=count)


def _array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def _save_npz(path, **values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream: np.savez(stream, **values)
    temporary.replace(path)


def _save_json(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(values, indent=2, sort_keys=True) + '\n'); temporary.replace(path)


def _file_identity(path):
    path = Path(path).resolve(); st = path.stat()
    return dict(path=str(path), bytes=st.st_size, mtime_ns=st.st_mtime_ns,
                ctime_ns=st.st_ctime_ns, inode=st.st_ino, device=st.st_dev)


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): digest.update(block)
    return digest.hexdigest()


def _integer_count(value):
    value = np.asarray(value)
    if value.ndim != 0 or not np.issubdtype(value.dtype, np.integer) or int(value) < 2:
        raise ValueError('Cell count must be an integer scalar at least two')
    return int(value)


def _validate_positive(quantiles, means, counts, n_cells, genes):
    quantiles, means, counts = np.asarray(quantiles), np.asarray(means), np.asarray(counts)
    if quantiles.shape != (genes, 33) or means.shape != (genes,) or counts.shape != (genes,):
        raise ValueError('Invalid positive-statistic axes')
    if not np.issubdtype(counts.dtype, np.integer) or (counts < 0).any() or (counts > n_cells).any():
        raise ValueError('Invalid positive cell counts')
    if not np.isfinite(quantiles).all() or not np.isfinite(means).all() or (quantiles < 0).any() or (means < 0).any():
        raise ValueError('Invalid positive values')
    if (np.diff(quantiles, axis=1) < -1e-6).any(): raise ValueError('Positive quantiles must be ordered')
    absent = counts == 0; present = ~absent
    if (quantiles[absent] != 0).any() or (means[absent] != 0).any():
        raise ValueError('Absent positive distribution must be zero and masked')
    if (quantiles[present] <= 0).any() or (means[present] <= 0).any():
        raise ValueError('Observed positive distribution must be strictly positive')
    tolerance = 1e-5 * np.maximum(1., means)
    if ((means[present] < quantiles[present, 0] - tolerance[present]) |
        (means[present] > quantiles[present, -1] + tolerance[present])).any():
        raise ValueError('Positive mean outside empirical support')


class ConnectionRuntime:
    """Separate source-only forward data and training-only supervision APIs."""
    training_environments = CORE3
    environments = (*CORE3, *HELD)
    def __init__(self, assets, geometry, cache_dir, device, *, performance_settings=None):
        self.assets, self.geometry, self.device = assets, geometry, device
        self.performance_settings = (performance_settings if isinstance(performance_settings, RuntimePerformanceSettings)
            else RuntimePerformanceSettings(**(performance_settings or {})))
        self._transfer_audit = dict(prepared_conditions=0, packed_batches=0,
            device_transfer_groups=0, device_transfer_bytes=0, pinned_transfer_groups=0,
            pin_memory_failures=0)
        self.training_environments = tuple(getattr(assets, "training_environments", CORE3))
        self.environments = tuple(getattr(assets, "environments", (*CORE3, *HELD)))
        self.genes, self.targets = list(assets.gene_names), list(assets.target_names)
        self.cache = Path(cache_dir); self.cache.mkdir(parents=True, exist_ok=True)
        self.control_variance = {}
        self.source_cache = CPUByteLRU(self.performance_settings.precision_cache_bytes)
        self._sealed_files = {}
        self._bound_inputs = []
        stager = getattr(assets, 'local_stager', None)
        self._condition_origin = Path(assets.canonical_fold_cache)/'runtime' if stager is not None else None
        for record in assets.binding.get('files', []):
            if stager is not None:
                # Local snapshots have already been proven byte-identical to
                # the canonical binding; protect that fixed snapshot in RAM.
                path = stager.stage_record(record)
                current = _file_identity(path)
            else:
                current = _file_identity(record['path'])
                expected = record.get('identity')
                if expected is not None and current != expected:
                    raise ValueError('Bound input file changed before runtime construction')
                if expected is None and _file_sha(record['path']) != record['sha256']:
                    raise ValueError('Bound input byte hash differs before runtime construction')
            self._bound_inputs.append(current)
        axis_sha = hashlib.sha256('\0'.join(self.genes).encode()).hexdigest()
        binding_sha = hashlib.sha256(json.dumps(assets.binding, sort_keys=True).encode()).hexdigest()
        definition = dict(schema="sealed_condition_precision_v2", gene_axis_sha256=axis_sha, bootstrap_replicates=12, seed=17,
                          asset_binding_sha256=binding_sha,
                          stat_scale_sha256=_array_sha(geometry['stat_scale']),
                          within_condition_cells_only=True, environment_disagreement_used=False,
                          variance_floor_standardized=.01,
                          sampling_design='cell bootstrap; does not model batch/guide random effects')
        binding = self.cache / 'definition.json'
        if binding.exists() and json.loads(binding.read_text()) != definition:
            raise ValueError('Observation precision cache definition changed')
        _save_json(binding, definition)
        self.precision_definition_sha = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
        self.moment_root = self.cache.parent / 'hurdle_observations'
        self.moment_root.mkdir(parents=True, exist_ok=True)
        moment_definition = dict(schema='hurdle_observations_v4_sealed_positive_v2', asset_binding_sha256=binding_sha,
            genes=self.genes, targets=self.targets, environments=list(self.training_environments),
            pca={key:_array_sha(assets.features.geometry[key]) for key in ('components','scale','pca_mean')},
            control_inputs={env:dict(statistics=_array_sha(assets.features.controls[env]),
                mean=_array_sha(assets.features.means[env]),
                detection=_array_sha(assets.features.pooled_detections[env])) for env in self.environments},
            stat_scale_sha256=_array_sha(geometry['stat_scale']), quantile_grid=QUANTILE_GRID.tolist(),
            source_joint='Pmean32/Pstd32/Estd32/deltaCovUpper528/logN1',
            supervision='full cached authorized cell pool, before any per-epoch loss subsampling',
            positive_fallback='target E; positive-count-weighted source quantile barycenter; training-fold E gene; training-fold E global')
        moment_binding = self.moment_root / 'definition.json'
        if moment_binding.exists() and json.loads(moment_binding.read_text()) != moment_definition:
            raise ValueError('Hurdle observation cache definition changed')
        _save_json(moment_binding, moment_definition)
        self.moment_definition_sha = hashlib.sha256(json.dumps(moment_definition, sort_keys=True).encode()).hexdigest()
        self._source_moments = CPUByteLRU(self.performance_settings.source_moment_cache_bytes)
        self._supervision_moments = CPUByteLRU(self.performance_settings.supervision_moment_cache_bytes)
        self._response_banks = {}
        self._source_control_joint = {}
        self._positive_control, self._positive_domain = {}, None
        self._device = dict(target_prior=tensor(assets.target_prior, device), controls={}, means={}, detections={}, prototypes={})
        for env in self.environments:
            self._device['controls'][env] = tensor(assets.features.controls[env], device)
            self._device['means'][env] = tensor(assets.features.means[env], device)
            detection = np.asarray(assets.features.pooled_detections[env])
            if detection.shape != (len(self.genes),) or not np.isfinite(detection).all() or ((detection < 0) | (detection > 1)).any():
                raise ValueError('Invalid pooled control detection vector')
            self._device['detections'][env] = tensor(detection, device)
            control, latent, noise = assets.features._prototypes[env]
            self._device['prototypes'][env] = (tensor(control, device),
                tensor(np.concatenate((latent,noise),axis=1), device))
        self._upper = np.triu_indices(32)
        # A single mask excludes every training-target gene, even when a
        # retrieval bank happens to contain only a subset of perturbations.
        self.retrieval_gene_mask = np.asarray([g not in set(self.targets) for g in self.genes], dtype=bool)
        if not self.retrieval_gene_mask.any(): raise ValueError('Retrieval mask removed every output gene')
        from .observation_reuse import configure_observation_reuse
        configure_observation_reuse(self)

    def _guard_bound_inputs(self):
        owner = getattr(self, '_resident_owner', None)
        if owner is not None and owner.ready and not owner.preparing:
            owner.check_runtime(self)
            return  # Independent verified RAM snapshot; no backing-file dependency.
        # No expression arrays or repeated multi-GB hashing: the byte-bound
        # immutable input snapshot is protected by full filesystem identities.
        for expected in self._bound_inputs:
            if _file_identity(expected['path']) != expected:
                raise ValueError('Bound input changed during this runtime: ' + expected['path'])

    def _condition_path(self, purpose, env, target):
        path = self.cache / env / f'{target}.npz' if purpose == 'precision' else self.moment_root / purpose / env / f'{target}.npz'
        origin = getattr(self, '_condition_origin', None)
        if origin is not None and not path.exists():
            source = origin / 'source_precision' / env / path.name if purpose == 'precision' else origin / 'hurdle_observations' / purpose / env / path.name
            if source.exists():
                self.assets.local_stager.stage_condition(source,path,self._condition_identity(purpose,env,target))
        if not path.exists():
            from .observation_reuse import try_reuse_condition
            try_reuse_condition(self,path,purpose,env,target)
        return path

    def _condition_identity(self, purpose, env, target):
        return dict(schema='sealed_condition_v2', purpose=purpose, environment=env, target=target,
            definition_sha256=self.precision_definition_sha if purpose == 'precision' else self.moment_definition_sha)

    def _verify_condition(self, path, purpose, env, target):
        path = Path(path); receipt = path.with_suffix('.receipt.json')
        expected = self._condition_identity(purpose, env, target)
        stamp = (_file_identity(path), _file_identity(receipt))
        previous = self._sealed_files.get(str(path))
        if previous is not None:
            if previous['stamp'] != stamp or previous['identity'] != expected:
                raise ValueError('Sealed condition cache changed during runtime')
            return previous['receipt']
        record = json.loads(receipt.read_text())
        if record.get('identity') != expected or record.get('bytes') != stamp[0]['bytes']:
            raise ValueError('Condition cache identity or byte count mismatch')
        if record.get('sha256') != _file_sha(path):
            raise ValueError('Condition cache byte fingerprint mismatch')
        if stamp != (_file_identity(path), _file_identity(receipt)):
            raise ValueError('Condition cache changed during verification')
        self._sealed_files[str(path)] = dict(stamp=stamp, identity=expected, receipt=record)
        return record

    def _save_condition(self, path, purpose, env, target, values):
        identity = self._condition_identity(purpose, env, target)
        _save_npz(path, **values, condition_identity_json=np.asarray(json.dumps(identity, sort_keys=True)))
        receipt = dict(identity=identity, bytes=Path(path).stat().st_size, sha256=_file_sha(path),
            arrays={k:dict(shape=list(np.asarray(v).shape), dtype=str(np.asarray(v).dtype)) for k,v in values.items()})
        _save_json(Path(path).with_suffix('.receipt.json'), receipt)
        self._sealed_files.pop(str(path), None)
        self._verify_condition(path, purpose, env, target)

    def _load_condition(self, path, purpose, env, target, fields=None):
        record = self._verify_condition(path, purpose, env, target)
        with np.load(path, allow_pickle=False) as f:
            declared = set(record.get('arrays', {}))
            if set(f.files) != declared | {'condition_identity_json'}:
                raise ValueError('Condition archive field layout differs from receipt')
            selected = declared if fields is None else set(fields)
            if not selected <= declared:
                raise ValueError('Requested condition field absent from receipt')
            result = {k:np.array(f[k]) for k in selected | {'condition_identity_json'}}
        raw_identity = result.pop('condition_identity_json', None)
        if raw_identity is None or json.loads(str(raw_identity)) != self._condition_identity(purpose, env, target):
            raise ValueError('Condition payload identity mismatch')
        layout = {k:dict(shape=list(v.shape), dtype=str(v.dtype)) for k,v in result.items()}
        expected_layout = {k:record['arrays'][k] for k in result}
        if layout != expected_layout: raise ValueError('Condition array layout differs from receipt')
        return result

    @staticmethod
    def _remember(cache, key, value):
        cache[key] = value
        if key in cache: cache.move_to_end(key)
        return value

    def _source_authorize(self, env, target):
        if getattr(self.assets, 'inference_sources_only', False) and torch.is_grad_enabled():
            raise PermissionError('Evaluation-only source capability requires no_grad')
        if env not in self.training_environments or target not in self.targets:
            raise PermissionError('Source condition outside training environment/target axes')
        if getattr(self.assets.train_store, 'role', None) != 'train':
            raise PermissionError('Training store required for source evidence')
        self._guard_bound_inputs()
        self.assets.train_store._authorize(env, [target], 'source')
        self.assets.train_store._check_episode_access(env, 'source')
        if getattr(self.assets.train_store, '_episode', None) is None:
            raise PermissionError('Bind a source episode before opening cached source observations')

    @contextlib.contextmanager
    def _training_supervision_scope(self, env, target):
        if env not in self.training_environments or target not in self.targets:
            raise PermissionError('Supervision API exposes training environments/targets only')
        self._guard_bound_inputs()
        store = self.assets.train_store
        if getattr(store, 'role', None) != 'train': raise PermissionError('Training store required')
        previous = getattr(store, '_episode', None)
        store.authorize_episode(env, tuple(e for e in self.training_environments if e != env))
        try:
            store._authorize(env, [target], 'supervision')
            store._check_episode_access(env, 'supervision')
            yield
        finally:
            if previous is None: store.clear_episode()
            else: store.authorize_episode(*previous)

    def _source_condition(self, env, target):
        self._source_authorize(env, target)  # includes cache-hit checks
        key = env, target
        inference_owner = getattr(self, '_inference_training_owner', None)
        if inference_owner is not None and key in inference_owner.source:
            if torch.is_grad_enabled(): raise PermissionError('Evaluation source evidence cannot enter a gradient-enabled forward')
            return inference_owner.source[key]
        owner = getattr(self, '_resident_owner', None)
        if owner is not None and owner.ready and not owner.preparing:
            owner.check_runtime(self)
            return owner.source[key]
        path = self._condition_path('source',env,target)
        if key in self._source_moments:
            self._verify_condition(path, 'source', env, target)
            self._source_moments.move_to_end(key); return self._source_moments[key]
        if path.exists():
            result = self._load_condition(path, 'source', env, target)
            self.assets.train_store.audit['cached_source_moment_conditions'] += 1
        else:
            x = self.assets.train_store.get_p(env, target, 'source')
            if len(x) < 2: raise ValueError('At least two source cells required')
            z = self.assets.features.project(x, env)
            if z.shape[1] != 32: raise ValueError('Source joint requires 32 control PCA directions')
            centered = z - z.mean(0,keepdims=True)
            covariance = centered.T @ centered / len(z)
            if env not in self._source_control_joint:
                ze = self.assets.features.project(self.assets.features._all_e[env], env)
                ze_centered = ze - ze.mean(0,keepdims=True)
                self._source_control_joint[env] = ze.std(0), ze_centered.T @ ze_centered / len(ze)
            e_std, e_covariance = self._source_control_joint[env]
            joint = np.r_[z.mean(0),z.std(0),e_std,(covariance-e_covariance)[self._upper],np.log1p(len(x))].astype('f4')
            positive = positive_summary(x)
            result = dict(joint=joint, positive_quantiles=positive['quantiles'],
                positive_mean=positive['positive_mean'], positive_count=positive['positive_count'], n_cells=np.int64(len(x)))
        required = {'joint','positive_quantiles','positive_mean','positive_count','n_cells'}
        if set(result) != required or result['joint'].shape != (625,) or not np.isfinite(result['joint']).all():
            raise ValueError('Invalid cached source statistic axes')
        count = _integer_count(result['n_cells'])
        if (result['joint'][32:96] < 0).any() or not np.isclose(result['joint'][-1], np.log1p(count), rtol=1e-6, atol=1e-6):
            raise ValueError('Source joint moments or count are inconsistent')
        _validate_positive(result['positive_quantiles'], result['positive_mean'], result['positive_count'], count, len(self.genes))
        if not path.exists(): self._save_condition(path, 'source', env, target, result)
        return self._remember(self._source_moments,key,result)

    def _positive_reference(self, env, source_moments):
        if env not in self._positive_control:
            self._positive_control[env] = positive_summary(self.assets.features._all_e[env])
        if self._positive_domain is None:
            training_e = np.concatenate([self.assets.features._all_e[e] for e in self.training_environments])
            domain = positive_summary(training_e)
            values = training_e[training_e > 0]
            if not len(values): raise ValueError('No positive expression in legal training controls')
            domain['global_quantiles'] = np.quantile(values,QUANTILE_GRID).astype('f4')
            domain['global_mean'] = float(values.mean(dtype=np.float64))
            self._positive_domain = domain
        control, domain = self._positive_control[env], self._positive_domain
        template, base_mean = control['quantiles'].copy(), control['positive_mean'].copy()
        missing = control['positive_count'] == 0
        level = np.zeros(len(self.genes),np.int64)
        # Only absent target-control genes can use a source fallback. Keep the
        # original float64 weighting and source reduction order, but never form
        # a redundant S x all-G x 33 temporary for observed control genes.
        missing_ids = np.flatnonzero(missing)
        source_counts = np.stack([r['positive_count'][missing_ids] for r in source_moments]).astype('f8')
        total = source_counts.sum(0)
        supported_ids = missing_ids[total > 0]
        weights = source_counts / np.maximum(total,1)[None]
        source_q = np.sum(np.stack([r['positive_quantiles'][missing_ids] for r in source_moments]) * weights[:,:,None],axis=0)
        source_mean = np.sum(np.stack([r['positive_mean'][missing_ids] for r in source_moments]) * weights,axis=0)
        template[supported_ids] = source_q[total > 0]; base_mean[supported_ids] = source_mean[total > 0]; level[supported_ids]=1
        absent = missing.copy(); absent[supported_ids] = False
        supported_domain = absent & (domain['positive_count'] > 0)
        template[supported_domain] = domain['quantiles'][supported_domain]
        base_mean[supported_domain] = domain['positive_mean'][supported_domain]; level[supported_domain]=2
        global_fallback = absent & ~supported_domain
        template[global_fallback] = domain['global_quantiles']; base_mean[global_fallback]=domain['global_mean']; level[global_fallback]=3
        if not np.isfinite(template).all() or (template <= 0).any() or not np.isfinite(base_mean).all() or (base_mean <= 0).any():
            raise ValueError('Positive templates must be finite, strictly positive and explicitly supported/fallback')
        return template, base_mean, missing, level

    def training_supervision(self, env, target, phase=None):
        """Phase-specific legal labels, with no unused quantile copies in A/C.

        Persisted sufficient statistics and their binding remain unchanged.
        Selective NPZ fields are decoded only after full byte/receipt integrity
        validation. Cache capacity is global across phases, never per phase.
        """
        if phase not in (None, 'A', 'B', 'C', 'J', 'BANK'):
            raise ValueError('Unknown supervision phase')
        with self._training_supervision_scope(env,target):
            owner = getattr(self, '_resident_owner', None)
            if owner is not None and owner.ready and not owner.preparing:
                return owner.supervision_record(self, env, target, phase)
            if getattr(self, '_supervision_phase', phase) != phase:
                self._supervision_moments.clear()
            self._supervision_phase = phase
            key = (env,target) if phase is None else (env,target,phase)
            path = self._condition_path('supervision',env,target)
            if key in self._supervision_moments:
                self._verify_condition(path, 'supervision', env, target)
                self.assets.train_store.audit['cached_supervision_moment_conditions'] += 1
                self._supervision_moments.move_to_end(key)
                return self._supervision_moments[key]
            fields={'full_stats','n_cells'}
            if phase != 'C':fields.add('variance')
            if phase not in ('C','BANK'):fields.add('count_positive')
            if phase in (None,'B','J'):fields.add('positive_quantiles')
            if path.exists():
                raw = self._load_condition(path,'supervision',env,target,fields=fields)
                self.assets.train_store.audit['cached_supervision_moment_conditions'] += 1
            else:
                full=self.assets.train_store.get_p(env,target,'supervision')
                if len(full)<2:raise ValueError('At least two training supervision cells required')
                positive=positive_summary(full)
                complete=dict(full_stats=statistics(full),variance=full.var(axis=0),
                    count_positive=positive['positive_count'],positive_quantiles=positive['quantiles'],n_cells=np.int64(len(full)))
                self._save_condition(path,'supervision',env,target,complete)
                raw={k:complete[k] for k in fields}
            if set(raw)!=fields:raise ValueError('Incomplete phase supervision payload')
            count=_integer_count(raw['n_cells']);stats=np.asarray(raw['full_stats'],dtype='f4')
            if stats.shape!=(len(self.genes),7) or not np.isfinite(stats).all() or (stats<0).any():
                raise ValueError('Invalid training supervision statistic axes')
            if (np.diff(stats[:,3:],axis=1)<-1e-6).any():raise ValueError('Unordered supervision quantiles')
            mean=stats[:,0].copy()  # Do not retain the full backing statistic array.
            result=dict(mean=mean,n_cells=count)
            if phase=='C':return self._remember(self._supervision_moments,key,result)
            variance=np.asarray(raw['variance'],dtype='f4')
            if variance.shape!=mean.shape or not np.isfinite(variance).all() or (variance<0).any():
                raise ValueError('Invalid training supervision variance')
            if not np.allclose(stats[:,1]**2,variance,rtol=1e-5,atol=1e-7):
                raise ValueError('Supervision std and variance differ')
            if phase=='BANK':
                result=dict(response=(mean-self.assets.features.means[env]).astype('f4'),mean_noise=(variance/count).astype('f4'),n_cells=count)
                return self._remember(self._supervision_moments,key,result)
            counts=np.asarray(raw['count_positive'])
            if counts.shape!=mean.shape or not np.issubdtype(counts.dtype,np.integer) or (counts<0).any() or (counts>count).any():
                raise ValueError('Invalid positive supervision counts')
            detection=counts.astype('f4')/count;positive=counts>0
            if not np.allclose(stats[:,2],detection,rtol=1e-6,atol=1e-7):raise ValueError('Detection differs from full-pool counts')
            if (stats[~positive]!=0).any() or (variance[~positive]!=0).any():raise ValueError('Inconsistent empirical support')
            positive_mean=np.divide(mean,detection,out=np.zeros_like(mean),where=positive)
            result.update(count_positive=counts,positive_mask=positive)
            if phase in (None,'A','J'):
                result.update(detection=detection,positive_mean=positive_mean,
                    statistics=((stats-self.assets.features.controls[env])/self.geometry['stat_scale']).astype('f4'))
            if phase in (None,'B','J'):
                quantiles=np.asarray(raw['positive_quantiles'],dtype='f4')
                _validate_positive(quantiles,positive_mean,counts,count,len(self.genes))
                result.update(positive_quantiles=quantiles,variance=variance)
            if phase is None:
                result.update(full_stats=stats,response=(mean-self.assets.features.means[env]).astype('f4'),mean_noise=(variance/count).astype('f4'))
            return self._remember(self._supervision_moments,key,result)

    def release_training_response_banks(self):
        """Drop A-only truth banks and supervision RAM before distribution training."""
        released=len(self._response_banks)
        self._response_banks.clear();self._supervision_moments.clear()
        self._supervision_phase=None
        return dict(released_response_banks=released,supervision_cache_cleared=True)

    def training_response_bank(self, env, targets=None):
        """Training-only CPU bank; fixed all-training-target exclusion mask."""
        if env not in self.training_environments: raise PermissionError('Held environments cannot supply training retrieval labels')
        targets=list((self.assets.train_store.training_targets(env) if hasattr(self.assets.train_store, 'training_targets') else self.targets) if targets is None else targets)
        if not targets or len(set(targets)) != len(targets) or not set(targets)<=set(self.targets):
            raise PermissionError('Retrieval targets must be unique training targets')
        key=env,tuple(targets)
        # Even an in-memory bank hit rechecks the training label capability.
        for target in targets:
            with self._training_supervision_scope(env,target): pass
        if key in self._response_banks: return self._response_banks[key]
        moments=[self.training_supervision(env,target,phase='BANK') for target in targets]
        responses=np.stack([m['response'] for m in moments]).astype('f4')
        noise=np.stack([m['mean_noise'] for m in moments]).astype('f4')
        # This mask is fixed from the complete immutable target metadata axis
        # at construction, even when this environment/panel holds out targets.
        # Candidate-bank membership must not change discrimination geometry.
        mask=self.retrieval_gene_mask.copy()
        signal=np.mean(responses[:,mask].astype('f8')**2,axis=1)
        noise_energy=np.mean(noise[:,mask].astype('f8'),axis=1)
        reliability=(signal/np.maximum(signal+noise_energy,1e-12)).astype('f4')
        result=dict(targets=targets,responses=responses,mask=mask,reliability=reliability,
                    counts=np.asarray([m['n_cells'] for m in moments],np.int64),
                    mean_noise_energy=noise_energy.astype('f4'))
        if getattr(self.assets,'joint_training',False) or getattr(self.assets,'fold',{}).get('joint_holdout'):
            result['full_target_axis']=list(self.targets)
        self._response_banks[key]=result
        return result

    def precision(self, env, target):
        # The same role, episode and source capability protects disk/memory hits.
        self._source_authorize(env, target)
        key = env, target
        inference_owner = getattr(self, '_inference_training_owner', None)
        if inference_owner is not None and key in inference_owner.precision:
            if torch.is_grad_enabled(): raise PermissionError('Evaluation source evidence cannot enter a gradient-enabled forward')
            return inference_owner.precision[key]
        owner = getattr(self, '_resident_owner', None)
        if owner is not None and owner.ready and not owner.preparing:
            owner.check_runtime(self)
            return owner.precision[key]
        path = self._condition_path('precision',env,target)
        if key in self.source_cache:
            self._verify_condition(path, 'precision', env, target)
            self.source_cache.move_to_end(key)
            return self.source_cache[key]
        if path.exists():
            raw = self._load_condition(path, 'precision', env, target)
        else:
            if env not in self.control_variance:
                self.control_variance[env] = sampling_variance(
                    self.assets.features._all_e[env], seed_int('E-noise', env))
            source = self.assets.train_store.get_p(env, target, 'source')
            count = len(source)
            var = sampling_variance(source, seed_int('P-noise', env, target)) + self.control_variance[env]
            var /= self.geometry['stat_scale'] ** 2
            raw = dict(precision=np.clip(1 / (var + .01), .01, 100).astype('f4'), count=np.int64(count))
            if getattr(self.assets, 'pooled_reference_applied', False):
                from .pooled_reference import pooled_precision
                f = self.assets.features
                raw['precision'] = pooled_precision(raw['precision'], source.var(0), (source > 0).sum(0).astype('f4') / count, count,
                    f.pooled_variances[env], f.pooled_detections[env], f.pooled_counts[env], self.geometry['stat_scale'])
        if set(raw) != {'precision','count'}: raise ValueError('Invalid precision payload')
        precision, count = np.asarray(raw['precision']), _integer_count(raw['count'])
        if precision.shape != (len(self.genes), 7) or not np.isfinite(precision).all() or (precision < .01-1e-7).any() or (precision > 100+1e-5).any():
            raise ValueError('Invalid observation precision cache')
        if not path.exists(): self._save_condition(path, 'precision', env, target, raw)
        return self._remember(self.source_cache, key, (precision, count))

    def _prepare_condition(self, env, target, sources=None, n_cells=64, tag='evaluation'):
        """Synchronous legal CPU observations plus already-resident constants.

        No supervision is read here. Mutable store capabilities are checked and
        used in this calling thread; batch preparation never starts workers.
        """
        if target not in self.targets or n_cells < 2:
            raise ValueError('Unknown target or fewer than two generated cells')
        sources = validate_sources(env, tuple(s for s in self.training_environments if s != env) if sources is None else sources, self.training_environments, self.environments)
        self.assets.train_store.authorize_episode(env,sources)
        # Reject the complete request before indexing any numerical source table.
        for source in sources:
            self._source_authorize(source, target)
        j = self.assets.features.tidx[target]
        owner = getattr(self, '_resident_owner', None)
        resident = owner is not None and owner.ready
        observations,precisions,controls,counts,source_moments = [],[],[],[],[]
        for source in sources:
            block=np.asarray(self.assets.features.summaries[source][0][j])
            observations.append(owner.gpu_source[source]['observations'][j] if resident else block[:,14:21]/self.geometry['stat_scale'])
            precision,count=self.precision(source,target)
            moment=self._source_condition(source,target)
            if int(moment['n_cells']) != count: raise ValueError('Source precision and joint statistic cell counts differ')
            precisions.append(owner.gpu_source[source]['precision'][j] if resident else precision); counts.append(count); source_moments.append(moment)
            controls.append(self._device['controls'][source])
        template,positive_mean,missing,fallback_level=self._positive_reference(env,source_moments)
        control, _ = self._device['prototypes'][env]
        rng=np.random.default_rng(seed_int('whole-cell-roster',17,tag,env,target))
        indices=np.sort(rng.choice(len(control),n_cells,replace=n_cells>len(control)))
        result = dict(observations=torch.stack(observations) if resident else tensor(np.stack(observations),'cpu'),
            precision=torch.stack(precisions) if resident else tensor(np.stack(precisions),'cpu'), source_controls=torch.stack(controls),
            target_control=self._device['controls'][env], target_prior=self._device['target_prior'][j],
            _private_cell_indices=torch.as_tensor(indices),
            target_control_mean=self._device['means'][env],
            target_control_detection=self._device['detections'][env],
            source_joint=torch.stack([owner.gpu_source[s]['joint'][j] for s in sources]) if resident else tensor(np.stack([r['joint'] for r in source_moments]),'cpu'),
            source_reliability=tensor(np.asarray(counts)/(np.asarray(counts)+32.),'cpu'),
            positive_template=tensor(template,'cpu'), positive_base_mean=tensor(positive_mean,'cpu'),
            positive_missing=torch.as_tensor(missing),
            positive_fallback_level=torch.as_tensor(fallback_level),
            sources=sources,query=env,target=target,counts=counts)
        if (getattr(self.assets, 'pooled_reference_applied', False) or
                getattr(self, '_pooled_reference_applied', False)):
            # These are observed, selected source labels for the source readout
            # loss. The query P pool is never consulted by the input API.
            fields = {k: [] for k in ('source_means', 'source_detections',
                'source_control_means', 'source_control_detections')}
            for source in sources:
                self._source_authorize(source, target)
                block = self.assets.features.summaries[source][0][j]
                fields['source_means'].append(block[:, 7])
                fields['source_detections'].append(block[:, 9])
                fields['source_control_means'].append(self.assets.features.means[source])
                fields['source_control_detections'].append(self.assets.features.pooled_detections[source])
            for key, values in fields.items():
                value = np.stack(values).astype('f4')
                if value.shape != (len(sources), len(self.genes)) or not np.isfinite(value).all():
                    raise ValueError('Invalid source readout axis')
                result[key] = torch.as_tensor(value).detach()
        self._transfer_audit['prepared_conditions'] += 1
        return result

    def _transfer_fields(self, values, device=None):
        """Transfer each already-packed CPU field once; preserve static tensors."""
        destination = self._device['target_prior'].device if device is None else torch.device(device)
        result = dict(values)
        for key, value in values.items():
            if not isinstance(value, torch.Tensor) or value.device == destination: continue
            if value.device.type != 'cpu':
                raise ValueError('Batch transfers require CPU tensors or resident destination constants')
            pinned = False
            if self.performance_settings.pin_memory and destination.type == 'cuda' and torch.cuda.is_available():
                try:
                    if not value.is_pinned(): value = value.pin_memory()
                    pinned = value.is_pinned()
                except RuntimeError:
                    self._transfer_audit['pin_memory_failures'] += 1
            result[key] = value.to(destination, non_blocking=self.performance_settings.non_blocking and pinned)
            self._transfer_audit['device_transfer_groups'] += 1
            self._transfer_audit['device_transfer_bytes'] += value.numel() * value.element_size()
            self._transfer_audit['pinned_transfer_groups'] += int(pinned)
        return result

    def _materialize_roster(self, values):
        result = dict(values)
        indices = result.pop('_private_cell_indices')
        control, cell_features = self._device['prototypes'][result['query']]
        result['cell_features'], result['control_cells'] = cell_features[indices], control[indices]
        return result

    @staticmethod
    def _split_conditions(values):
        count = values['_private_condition_count']
        result = [{} for _ in range(count)]
        for key, value in values.items():
            if key == '_private_condition_count': continue
            for i, item in enumerate(result):
                item[key] = (value if key in ('query', 'sources', 'cache_format') else value[i])
        return result

    def batch(self, env, target, sources=None, n_cells=64, tag='evaluation'):
        """Compatible single-condition input API with the original random roster."""
        return self._materialize_roster(self._transfer_fields(
            self._prepare_condition(env, target, sources, n_cells, tag)))

    def batch_conditions(self, records, n_cells=64, tag='evaluation', *, packed=False):
        """Pack authorized inputs on CPU, then transfer once per dynamic field.

        Records contain query, target and optional sources/tag/n_cells. A batch
        must share a query, selected sources, and tensor axes. Static controls
        and priors remain resident; packed roster indices select those controls
        in one device operation. ``packed=False`` returns scalar-schema views.
        """
        from .condition_batching import collate_condition_batches
        records = list(records)
        if not records: raise ValueError('Empty condition batch')
        # Reject an invalid complete layout before opening any observations.
        # This validation does not grant capabilities; every preparation still
        # enters its own episode and performs all role/receipt/cache-hit checks.
        layouts, cell_counts = set(), set()
        for record in records:
            query, target = record['query'], record['target']
            count = record.get('n_cells', n_cells)
            if target not in self.targets or count < 2:
                raise ValueError('Unknown target or fewer than two generated cells')
            sources = record.get('sources')
            sources = tuple(e for e in self.training_environments if e != query) if sources is None else sources
            sources = validate_sources(query, sources, self.training_environments, self.environments)
            layouts.add((query, sources)); cell_counts.add(count)
        if len(layouts) != 1: raise PermissionError('A condition batch must share one legal query/source layout')
        if len(cell_counts) != 1: raise ValueError('Condition cell axes differ')
        prepared = [self._prepare_condition(record['query'], record['target'], record.get('sources'),
            record.get('n_cells', n_cells), record.get('tag', tag)) for record in records]
        result = self._materialize_roster(self._transfer_fields(collate_condition_batches(prepared)))
        self._transfer_audit['packed_batches'] += 1
        return result if packed else self._split_conditions(result)

    def training_supervision_conditions(self, records, *, device=None, packed=False, phase=None):
        """Separate training-only labels: prepare CPU, optionally transfer in bulk.

        Each condition enters and leaves its own synchronous supervision scope.
        Held query labels and evaluation stores remain forbidden on cache hits.
        With no device, all returned label tensors stay on CPU.
        """
        from .condition_batching import collate_condition_batches
        records = list(records)
        if not records: raise ValueError('Empty supervision batch')
        labels = []
        owner = getattr(self, '_resident_owner', None)
        use_device_table = (owner is not None and owner.ready and owner.phase == phase
            and device is not None and torch.device(device) == owner.device)
        if use_device_table:
            # Authorization is mandatory for every record, including table hits.
            for record in records:
                env, target = record['query'], record['target']
                with self._training_supervision_scope(env,target):
                    owner.check_runtime(self)
                    j = owner._target_index[target]
                    item = {key: value[j] for key,value in owner.gpu_supervision[env].items()}
                    item['n_cells'] = int(owner.supervision_raw[env,target]['n_cells'])
                    labels.append(item)
            result = collate_condition_batches(labels)
            return result if packed else self._split_conditions(result)
        for record in records:
            raw = self.training_supervision(record['query'], record['target'],phase=phase)
            labels.append({key: torch.as_tensor(value)
                if isinstance(value, np.ndarray) else value for key, value in raw.items()})
        result = collate_condition_batches(labels)
        if device is not None: result = self._transfer_fields(result, device)
        return result if packed else self._split_conditions(result)

    def transfer_arrays(self, values, *, device=None):
        """Stack homogeneous CPU arrays/tensors and make one device transfer."""
        values = list(values)
        if not values: raise ValueError('Empty tensor group')
        tensors = [torch.as_tensor(value) for value in values]
        destination = torch.device(self.device if device is None else device)
        if any(value.device.type != 'cpu' and value.device != destination for value in tensors):
            raise ValueError('Tensor groups must be CPU or resident on the destination')
        if len({value.device for value in tensors}) != 1:
            raise ValueError('Tensor group devices differ')
        if any(value.shape != tensors[0].shape or value.dtype != tensors[0].dtype for value in tensors):
            raise ValueError('Tensor group axes/dtypes differ')
        return self._transfer_fields({'values': torch.stack(tensors)}, device)['values']

    def prewarm_resident(self, phase, raw_cells=False, **kwargs):
        """Portable runtime uses bounded CPU caches and grouped device transfers.

        The old server-specific resident-memory orchestration is intentionally
        excluded. Scientific observations, bootstrap seeds and solver inputs
        are unchanged; the caller can cache raw arrays through its data store.
        """
        return dict(enabled=False, reason='portable_bounded_cache', phase=phase)

    def training_real_cells(self, records, *, seed, epoch, n_cells=64):
        """Original logical NumPy draws over an authorized portable data store."""
        records = list(records)
        if not records or isinstance(n_cells, bool) or int(n_cells) != n_cells or n_cells < 2:
            raise ValueError('Nonempty real-cell batch and at least two cells required')
        # Validate every capability before reading any response values.
        for record in records:
            with self._training_supervision_scope(record['query'], record['target']):
                pass
        selected, counts = [], []
        for record in records:
            env, target = record['query'], record['target']
            with self._training_supervision_scope(env, target):
                full = self.assets.train_store.get_p(env, target, 'supervision')
                rng = np.random.default_rng(seed_int('benchmark-supervision', seed, epoch, env, target))
                ids = rng.choice(len(full), int(n_cells), replace=len(full) < int(n_cells))
                selected.append(tensor(full[ids], self.device))
                counts.append(int(len(np.unique(ids))))
        return torch.stack(selected), counts

    def performance_audit(self):
        """Accounted CPU cache retention and observed grouped transfers."""
        return dict(observation_data_reuse=getattr(self,'_observation_reuse_counts',{}), settings=asdict(self.performance_settings), source_moments=self._source_moments.audit(),
            supervision_moments=self._supervision_moments.audit(), precision=self.source_cache.audit(),
            transfers=dict(self._transfer_audit),
            resident=(self._resident_owner.report() if getattr(self,'_resident_owner',None) is not None else None))


def fork_seed_runtime(shared_runtime, seed_assets):
    """Fresh capabilities and mutable caches; scientific input definitions match."""
    import copy
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    if digest(seed_assets.binding) != digest(shared_runtime.assets.binding):
        raise PermissionError('Cannot share observations across different asset bindings')
    if seed_assets.train_store is shared_runtime.assets.train_store:
        raise PermissionError('Each seed requires a newly constructed training capability')
    result = copy.copy(shared_runtime)
    result.assets = seed_assets
    result._transfer_audit = {key: 0 for key in shared_runtime._transfer_audit}
    result._response_banks = {}
    result._supervision_phase = None
    result._source_moments = CPUByteLRU(result.performance_settings.source_moment_cache_bytes)
    result._supervision_moments = CPUByteLRU(result.performance_settings.supervision_moment_cache_bytes)
    result.source_cache = CPUByteLRU(result.performance_settings.precision_cache_bytes)
    result._resident_owner = None
    return result


def clone_evaluation_runtime(runtime, cache_dir, *, evaluation_assets=None):
    """Construct an independent no-gradient source capability for evaluation.

    The portable caller explicitly supplies its source-only asset namespace.
    Query response labels never become part of the inference runtime.
    """
    if evaluation_assets is None:
        raise ValueError('Supply an explicit source-only evaluation_assets namespace')
    if not getattr(evaluation_assets, 'inference_sources_only', False):
        raise PermissionError('Evaluation assets must be source-only')
    return ConnectionRuntime(evaluation_assets, runtime.geometry,
        Path(cache_dir) / 'source_precision', runtime.device,
        performance_settings=runtime.performance_settings)
