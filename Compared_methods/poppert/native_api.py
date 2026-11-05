"""Native PopPert helpers; configure_upstream() binds separately obtained code.

No external source, predictions, checkpoints or automatic jobs are bundled.
"""
import hashlib
import importlib
import random
import sys
from pathlib import Path
import numpy as np
import torch

UPSTREAM_COMMIT = "248fc3e1428de96335bef1d0271d30ca1e4d5565"
SOURCE_SHA256 = {
    "model.py": "10a58b51dcd537a7b6d72cae0fb88d8e8f2407b3849cd4cbbcde637f678782a3",
    "sampling.py": "339279ea0a5689ec6ef96a7cef86fb905209aacdcb344120f9b852754eb5a4dc",
    "gmm_utils.py": "114b98447c6bf11f7428d43491772b930894e4bef6f8b14511a3a29ccc07abc4",
}
DistributionTransformer = None
compute_loss = None
sample_from_copula = None


def configure_upstream(directory):
    """Bind pinned official modules and fail on generic module name collisions."""
    global DistributionTransformer, compute_loss, sample_from_copula
    directory = Path(directory).expanduser().resolve()
    for filename, digest in SOURCE_SHA256.items():
        path = directory / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Pinned PopPert native source differs: " + filename)
        existing = sys.modules.get(path.stem)
        if existing is not None and Path(getattr(existing, "__file__", "")).resolve() != path:
            raise RuntimeError("Conflicting module; use an isolated process: " + path.stem)
    sys.path.insert(0, str(directory))
    model = importlib.import_module("model")
    sampling = importlib.import_module("sampling")
    DistributionTransformer = model.DistributionTransformer
    compute_loss = sampling.compute_loss
    sample_from_copula = sampling.sample_from_copula
    return importlib.import_module("gmm_utils")

def configuration():
    return dict(d_model=256,nhead=4,num_layers=4,dim_feedforward=1024,dropout=.1,n_components=2,
        copula_rank=32,copula_lambda=.2,n_cell_types=0,perturb_mode='gene',lr=1e-4,
        weight_decay=1e-4,gradient_clip=1.,microbatch_size=16,effective_batch_size=16,gpu_memory_fraction=.28,
        max_epochs=100,minimum_epochs=20,patience=15,validation_every_epochs=1,
        n_samples_loss=256,evaluation_cells=512,truncated=True,bfloat16_transformer=True,
        prediction_clip_min=0.,prediction_clip_max=14.9,optional_pretrained_embeddings=False,
        selector='fixed_validation_Rdelta_plus_CRPS_plus_0.25_centeredSW1',
        lr_scheduler='ReduceLROnPlateau_factor0.5_patience3')

def _cuda_device_index(device):
    device = torch.device(device)
    if device.type != 'cuda':
        return None
    return torch.cuda.current_device() if device.index is None else device.index


def _seed_torch_device(seed, device):
    # torch.manual_seed also seeds every CUDA device. Seed CPU explicitly,
    # then only the device that this model/input actually uses.
    torch.random.default_generator.manual_seed(seed)
    index = _cuda_device_index(device)
    if index is not None:
        with torch.cuda.device(index):
            torch.cuda.manual_seed(seed)


def seed_all(seed, device='cuda'):
    """Initialize this run's host RNGs and only its requested GPU generator."""
    random.seed(seed)
    np.random.seed(seed)
    _seed_torch_device(seed, device)

def create_model(vocab_size,seed,cfg=None,device='cuda'):
    if DistributionTransformer is None:
        raise RuntimeError('Call configure_upstream(path_to_pinned_checkout) first')
    cfg=configuration() if cfg is None else cfg
    device = torch.device(device)
    seed_all(seed, device=device)
    index = _cuda_device_index(device)
    if index is not None:
        torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'], device=index)
    model=DistributionTransformer(vocab_size=vocab_size,n_components=cfg['n_components'],
      d_model=cfg['d_model'],nhead=cfg['nhead'],num_layers=cfg['num_layers'],
      dim_feedforward=cfg['dim_feedforward'],dropout=cfg['dropout'],perturb_mode='gene',
      n_cell_types=0,copula_rank=cfg['copula_rank']).to(device)
    assert model.cell_context_embedding is None
    return model

def forward(model,x_ctrl,gene_indices,label,s_ctrl,cfg):
    device_type=x_ctrl.device.type
    with torch.autocast(device_type=device_type,dtype=torch.bfloat16,
                        enabled=cfg['bfloat16_transformer'] and device_type=='cuda'):
        output=model(x_ctrl,gene_indices,perturb_indices=label,s_ctrl=s_ctrl,cell_context_id=None)
    return {k:v.float() for k,v in output.items()}

def loss(model,x_ctrl,y_target,gene_indices,label,s_ctrl,s_target,cfg):
    preds=forward(model,x_ctrl,gene_indices,label,s_ctrl,cfg)
    value,logs=compute_loss(preds,y_target.float(),x_ctrl.float(),n_samples=cfg['n_samples_loss'],
        s_target=s_target.float(),lam_dep=cfg['copula_lambda'],truncated=cfg['truncated'])
    if not torch.isfinite(value):raise FloatingPointError('Nonfinite native PopPert objective')
    return value,logs

@torch.no_grad()
def cloud(model,x_ctrl,gene_indices,label,s_ctrl,A,psi,cfg,seed):
    model.eval()
    index = _cuda_device_index(x_ctrl.device)
    devices = [] if index is None else [index]
    with torch.random.fork_rng(devices=devices):
        _seed_torch_device(seed, x_ctrl.device)
        preds=forward(model,x_ctrl,gene_indices,label,s_ctrl,cfg)
        samples=sample_from_copula(preds,A,n_samples=cfg['evaluation_cells'],psi=psi,
                                  truncated=cfg['truncated'])
        # Match the official inference.py expression-domain export exactly.
        samples=samples.clamp(cfg['prediction_clip_min'],cfg['prediction_clip_max'])
        result=samples.permute(0,2,1).contiguous().cpu().numpy()
    if not np.isfinite(result).all() or (result<0).any():raise FloatingPointError('Invalid native cloud')
    return result

def generation_seed(seed,environment,target):
    return int.from_bytes(hashlib.sha256(f'PopPert|{seed}|{environment}|{target}'.encode()).digest()[:4],'little')
