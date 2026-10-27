import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, Path): return str(x)
    if isinstance(x, torch.Tensor): return clean(x.detach().cpu().numpy())
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, (np.floating, float)): return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.bool_): return bool(x)
    return x


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda: f.read(8 << 20), b''): h.update(x)
    return h.hexdigest()


def event(run, kind, **kw):
    record = dict(kw)
    record.update(event=kind, time_unix=time.time())
    record = clean(record)
    with (Path(run) / 'events.jsonl').open('a') as f: f.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)


def status(run, phase, **kw):
    record = dict(status='running', phase=phase, pid=os.getpid(), run_dir=str(run), time_unix=time.time(), **kw)
    atomic(Path(run) / 'STATUS.json', record)
    event(run, 'status', phase=phase, **kw)
