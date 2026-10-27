"""Storage accounting extracted from the original J0 runtime (no I/O)."""
import numpy as np
import torch

def payload_bytes(value):
    """Unique retained array/tensor storage, including NumPy view bases."""
    seen, storages = set(), set()
    def size(x):
        if id(x) in seen: return 0
        seen.add(id(x))
        if isinstance(x, np.ndarray):
            base = x
            while isinstance(base.base, np.ndarray): base = base.base
            key = ('numpy', id(base))
            if key in storages: return 0
            storages.add(key)
            return int(base.nbytes)
        if isinstance(x, torch.Tensor):
            s = x.untyped_storage(); key = (str(x.device), s.data_ptr())
            if key in storages: return 0
            storages.add(key)
            return s.nbytes()
        if isinstance(x, dict): return sum(size(v) for v in x.values())
        if isinstance(x, (tuple, list)): return sum(size(v) for v in x)
        return 0
    return size(value)


