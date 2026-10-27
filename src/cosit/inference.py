"""Restore one selected independent J0 predictor; no historical prediction inputs."""
from pathlib import Path
import torch
from .assets import prepare_assets,evaluation_runtime
from .training import load_checkpoint,seed_all
from .core.batched_model import BatchedSharedPathModel
from .prediction import generate,seal_predictions
from .io import read_json,write_json


def predict_run(data,priors,graph,graph_axis,run,output,device='cuda:0',batch_size=4):
    run=Path(run);selection=read_json(run/'SELECTION.json')
    if not selection.get('joint_converged'):
        raise PermissionError('Test prediction is disabled until training convergence is confirmed')
    checkpoint=Path(selection['selected_checkpoint']);payload=load_checkpoint(checkpoint,'cpu')
    protocol=payload['protocol'];fold=protocol['fold'];seed_all(protocol['seed'])
    assets,geometry,training_runtime=prepare_assets(data,priors,graph,graph_axis,fold,run/'assets',device,protocol,fit=False)
    infer=None
    try:
        if assets.binding!=payload['asset_binding']:raise ValueError('Prediction inputs do not match training')
        model=BatchedSharedPathModel(geometry,route='open_zero',**protocol['model']).to(device)
        model.load_state_dict(payload['state_dict'],strict=True);model.set_phase('J');model.eval()
        infer=evaluation_runtime(assets,geometry,run/'final_inference_sources',device)
        path=generate(model,infer,fold['test'],assets.target_names,output,batch_size=batch_size,
                      n_cells=protocol['generated_cells_eval'])
        seal=seal_predictions(checkpoint,[path],{fold['test']:assets.target_names},Path(output).with_suffix('.seal.json'),purpose='test')
        return dict(status='completed',predictions=str(path),seal=str(seal),checkpoint=str(checkpoint),
                    query_response_labels_read=0,conditions=len(assets.target_names),generated_cells=512)
    finally:
        assets.train_store.close()
        if infer is not None:infer.assets.train_store.close()
