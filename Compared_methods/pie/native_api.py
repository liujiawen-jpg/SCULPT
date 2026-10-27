"""Construct the official full PIE configuration; no task/data model substitute."""
from __future__ import annotations

from pathlib import Path


def make_config(*, preprocessed_dirs, split_dir, source_dirs, gene_text_dir,
                run_dir, seed=17):
    import pie
    from omegaconf import OmegaConf
    from pie.config import TrainConfig
    template = Path(pie.__file__).resolve().parent / "configs" / "train.yaml"
    if not template.is_file():
        raise FileNotFoundError("Install the pinned official PIE source with its config templates")
    raw = OmegaConf.to_container(OmegaConf.load(template),resolve=False)
    raw.pop("defaults")
    raw.update(vars={},experiment_name=f"PIE_J0/seed_{seed}",seed=int(seed),
        run_dir=str(run_dir),overwrite=False,resume=False)
    raw["data"].update(preprocessed_dirs=preprocessed_dirs,dataset_weights={"J0":1.0},
        split_dir=str(split_dir),source_dirs=source_dirs,gene_text_dir=str(gene_text_dir),
        aliases_path=None,delta_p={"bin_width_fold_change":1.1},
        evidence={"seed":42,"chunk":256,"lfc_clip_percentile":95.0},
        batch_size=16,num_workers=0)
    raw["trainer"].update(devices=1,num_nodes=1,max_steps=5000,accumulate_grad_batches=8)
    raw["logger"].update(enabled=False)
    return TrainConfig.model_validate(raw)


def make_model(config, fitted_data_stats, source_dims, gene_queries):
    """Keep full token sequences and native missing-source masks in the loader."""
    from pie.model import PieModel
    return PieModel(config.model,fitted_data_stats,source_dims,gene_queries)


def make_optimizer(model, config):
    from pie.train import build_optimizer
    return build_optimizer(model.parameters(),config.optimizer,config.scheduler,config.trainer.max_steps)


def native_losses(model, batch, config, fitted_data_stats):
    """Return all four official loss terms, with no added PDS surrogate."""
    from pie.model import compute_losses
    output = model(batch)
    return compute_losses(output,batch,config.model,fitted_data_stats.delta_p,temperatures=False)
