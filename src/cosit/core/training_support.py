"""Bounded, checkpoint-bound frozen encodings and explicit stopping policies."""
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import torch
from .common import atomic, sha


def detach_tree(value, device='cpu'):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {k: detach_tree(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(detach_tree(v, device) for v in value)
    if isinstance(value, list):
        return [detach_tree(v, device) for v in value]
    return value


def load_verified_frozen_payload(path, receipt, key, upstream_sha, binding):
    """Hash and deserialize the same local bytes; avoid a second disk pass."""
    if receipt.get('key') != key or receipt.get('upstream_sha256') != upstream_sha:
        raise PermissionError('Frozen encoding identity changed')
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != receipt.get('sha256'):
        raise PermissionError('Frozen encoding bytes changed')
    payload = torch.load(io.BytesIO(content), map_location='cpu', weights_only=True)
    if payload['binding'] != binding or payload.get('upstream_sha256') != upstream_sha or payload.get('key') != key:
        raise PermissionError('Frozen encoding binding mismatch')
    return payload


class FrozenEncodingCache:
    """Only current-run, float32 compact encodings. No learned cache across runs."""
    def __init__(self, directory, binding, max_bytes=8 << 30, memory_entries=8, memory_max_bytes=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.binding = dict(binding)
        self.namespace = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        self.max_bytes, self.memory_entries = int(max_bytes), int(memory_entries)
        if memory_max_bytes is not None:
            from .runtime import CPUByteLRU
            self.memory = CPUByteLRU(memory_max_bytes)
            self.memory_entries = 1 << 30  # Byte budget controls explicit performance profiles.
        else:
            self.memory = OrderedDict()
        self.files = OrderedDict()
        self.disk_bytes = 0
        self.hits = self.misses = self.evictions = 0
        self._model_stamp = self._upstream_sha = None
        # Never silently consume encodings whose parent model or data differ.
        self.directory = self.directory / self.namespace
        self.directory.mkdir(exist_ok=True)
        for p in sorted(self.directory.glob('*.pt'), key=lambda x: x.stat().st_mtime_ns):
            self.files[p.stem] = p.stat().st_size
            self.disk_bytes += p.stat().st_size
        self._evict()

    def _verify_upstream(self, model):
        # Normal optimizer/load_state_dict writes increment tensor versions.
        # Re-hash only when the model object or one of these versions changes.
        values = [(n,p) for n,p in model.named_parameters()
                  if not n.startswith(('shape.','reorder.'))]
        values += [('buffer:'+n,b) for n,b in model.named_buffers()
                   if n != '_phase_code' and not n.startswith(('shape.','reorder.'))]
        # Equal tensor values can have different meanings under another route
        # or solver configuration; include those semantics in the binding.
        inference = getattr(model, 'inference', None)
        semantics = dict(model_class=type(model).__module__ + '.' + type(model).__qualname__,
                         route=getattr(model, 'route', None),
                         solver={name: getattr(inference, name, None)
                                 for name in ('strength', 'ridge', 'tolerance', 'max_steps')})
        semantic_json = json.dumps(semantics, sort_keys=True)
        stamp = (id(model), semantic_json, tuple((n,id(v),v._version) for n,v in values))
        if stamp != self._model_stamp:
            digest=hashlib.sha256()
            digest.update(semantic_json.encode())
            for name,value in sorted(values):
                digest.update(name.encode())
                digest.update(str((tuple(value.shape),value.dtype)).encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            actual=digest.hexdigest()
            receipt=self.directory/'UPSTREAM.json'
            if receipt.exists():
                record=json.loads(receipt.read_text())
                if record.get('binding')!=self.binding or record.get('upstream_sha256')!=actual:
                    raise PermissionError('Frozen cache backbone differs from its bound model')
            else:
                atomic(receipt,dict(binding=self.binding,upstream_sha256=actual,semantics=semantics))
            self._upstream_sha=actual
            self._model_stamp=stamp
        return self._upstream_sha

    def _evict(self):
        while self.disk_bytes > self.max_bytes and self.files:
            key, size = self.files.popitem(last=False)
            (self.directory / f'{key}.pt').unlink(missing_ok=True)
            (self.directory / f'{key}.json').unlink(missing_ok=True)
            self.disk_bytes -= size
            self.memory.pop(key, None)
            self.evictions += 1

    def get(self, model, batch, query, target, sources):
        if model.phase not in ('B', 'C'):
            raise PermissionError('Frozen encodings cannot be used to train phase A')
        # A live upstream parameter invalidates the mathematical cache contract.
        if any(p.requires_grad for n, p in model.named_parameters()
               if not n.startswith(('shape.', 'reorder.'))):
            raise PermissionError('Backbone must be frozen before encoding reuse')
        upstream_sha = self._verify_upstream(model)
        key = hashlib.sha256(json.dumps([query, target, list(sources)]).encode()).hexdigest()
        path = self.directory / f'{key}.pt'
        if key in self.memory:
            self.hits += 1
            value = self.memory.pop(key)
            self.memory[key] = value
            if key in self.files:
                self.files.move_to_end(key)
        elif path.exists():
            receipt_path=path.with_suffix('.json')
            if not receipt_path.exists():
                raise PermissionError('Frozen encoding has no byte receipt')
            receipt=json.loads(receipt_path.read_text())
            payload = load_verified_frozen_payload(path,receipt,key,upstream_sha,self.binding)
            self.hits += 1
            value = payload['encoded']
            self.memory[key] = value
            self.files.move_to_end(key)
        else:
            self.misses += 1
            with torch.no_grad():
                value = detach_tree(model.compress_encoded(model.encode(batch)))
            tmp = path.with_suffix('.tmp')
            torch.save(dict(binding=self.binding, upstream_sha256=upstream_sha, key=key, encoded=value), tmp)
            tmp.replace(path)
            atomic(path.with_suffix('.json'),dict(key=key,upstream_sha256=upstream_sha,sha256=sha(path)))
            size = path.stat().st_size
            self.files[key] = size
            self.disk_bytes += size
            self.memory[key] = value
            self._evict()
        while len(self.memory) > self.memory_entries:
            self.memory.popitem(last=False)
        return detach_tree(value, batch['control_cells'].device)

    def audit(self):
        return dict(binding=self.binding, hits=self.hits, misses=self.misses,
                    evictions=self.evictions, disk_bytes=self.disk_bytes,
                    max_bytes=self.max_bytes, memory=(self.memory.audit() if hasattr(self.memory,'audit') else {'entries':len(self.memory),'entry_limit':self.memory_entries}),
                    format='lossless float32 compact rule and marginals')


class EarlyStopping:
    def __init__(self, min_epochs, patience, min_delta):
        self.min_epochs, self.patience, self.min_delta = int(min_epochs), int(patience), float(min_delta)
        self.best = None
        self.bad_epochs = 0

    def update(self, epoch, score):
        # Undefined/unqualified candidates cannot establish a successful plateau.
        if score is None:
            if self.best is not None:
                self.bad_epochs += 1
        elif self.best is None or self.best - float(score) >= self.min_delta:
            self.best, self.bad_epochs = float(score), 0
        else:
            self.bad_epochs += 1
        return self.best is not None and epoch >= self.min_epochs and self.bad_epochs >= self.patience


def frozen_fingerprint(model):
    h = hashlib.sha256()
    entries = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    entries += [('buffer:'+n, b) for n, b in model.named_buffers()]
    for name, tensor in sorted(entries):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def transfer_frozen_records(records, device):
    """Pack same-layout compact records on CPU and copy once per tensor field."""
    records=list(records)
    if not records: return []
    device=torch.device(device)
    keys=set(records[0])
    if any(set(x)!=keys for x in records): raise ValueError('Frozen record fields differ')
    out=[{} for _ in records]
    for key in keys:
        values=[x[key] for x in records]
        if all(isinstance(x,torch.Tensor) for x in values):
            if any(x.shape!=values[0].shape or x.dtype!=values[0].dtype for x in values):
                raise ValueError('Frozen tensor layouts differ')
            packed=torch.stack(values)
            if device.type=='cuda' and packed.device.type=='cpu':
                packed=packed.pin_memory().to(device,non_blocking=True)
            else: packed=packed.to(device)
            for i,value in enumerate(packed.unbind(0)):out[i][key]=value
        elif any(isinstance(x,torch.Tensor) for x in values):
            raise ValueError('Mixed frozen tensor fields')
        else:
            for i,value in enumerate(values):out[i][key]=value
    return out


class ResidentEncodingMemory(OrderedDict):
    """Bounded device cache; required training keys cannot be evicted after READY."""
    def __init__(self, max_bytes):
        super().__init__()
        self.max_bytes=int(max_bytes);self.current_bytes=0;self.entry_bytes={};self.required=set()
        if self.max_bytes<=0:raise ValueError('Positive encoding byte budget required')
        self.evictions=0
    def __setitem__(self,key,value):
        from .memory import payload_bytes
        cost=payload_bytes(value)+2048
        if cost>self.max_bytes:raise MemoryError('Single compact encoding exceeds resident budget')
        if key in self:self.__delitem__(key)
        while self.current_bytes+cost>self.max_bytes:
            victim=next((k for k in self if k not in self.required),None)
            if victim is None:raise MemoryError('Resident encoding budget cannot retain the complete training key set')
            self.__delitem__(victim);self.evictions+=1
        super().__setitem__(key,value);self.entry_bytes[key]=cost;self.current_bytes+=cost
    def __delitem__(self,key):
        super().__delitem__(key);self.current_bytes-=self.entry_bytes.pop(key)
    def pop(self,key,*default):
        if key not in self:
            if default:return default[0]
            raise KeyError(key)
        value=super().__getitem__(key);self.__delitem__(key);return value
    def popitem(self,last=True):
        key=next(reversed(self)) if last else next(iter(self));return key,self.pop(key)
    def clear(self):
        super().clear();self.entry_bytes.clear();self.current_bytes=0;self.required.clear()
    def audit(self):
        return dict(entries=len(self),bytes=self.current_bytes,max_bytes=self.max_bytes,
            required_keys=len(self.required),evictions=self.evictions,device_resident=True)
