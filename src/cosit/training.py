"""From-scratch, balanced J0 joint training with held-condition selection."""
from __future__ import annotations
from collections import OrderedDict
import copy
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from .assets import prepare_assets,evaluation_runtime,device_scales
from .data import CellStore
from .io import read_json,write_json,append_json,sha256,canonical_hash
from .core.batched_model import BatchedSharedPathModel
from .core.condition_batching import joint_condition_schedule
from .core.objective import objective_conditions
from .core.ranking import RankingPanels
from .core.runtime import seed_int
from .stopping import JointPlateauController


def seed_all(seed,threads=4):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.set_num_threads(threads)


def _rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path,model,optimizer,state,protocol,binding):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    torch.save(dict(schema='cosit_checkpoint_v1',state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
                    state=copy.deepcopy(state),protocol=protocol,asset_binding=binding,rng=_rng()),tmp)
    tmp.replace(path)
    write_json(path.with_suffix('.sha256.json'),dict(sha256=sha256(path),bytes=path.stat().st_size))


def load_checkpoint(path,device='cpu'):
    path=Path(path)
    if sha256(path)!=read_json(path.with_suffix('.sha256.json'))['sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    # Checkpoints are trusted local training artifacts, never untrusted downloads.
    payload=torch.load(path,map_location=device,weights_only=False)
    if payload.get('schema')!='cosit_checkpoint_v1':raise ValueError('Invalid checkpoint schema')
    return payload


def micro_loss(model,runtime,scales,panels,records,epoch,protocol):
    batch=runtime.batch_conditions(records,protocol['generated_cells_train'],f'train-epoch-{epoch}')
    output=model.forward_conditions(batch)
    labels=runtime.training_supervision_conditions(records,device=runtime.device,phase='J')
    real,counts=runtime.training_real_cells(records,seed=protocol['seed'],epoch=epoch,n_cells=protocol['real_cells_train'])
    gids=[]
    for record in records:
        serial=(epoch-1)*protocol['conditions_per_epoch']+record['original_offset']+1
        selected=np.random.default_rng(seed_int('gene-sample',protocol['seed'],serial)).choice(
            len(runtime.genes),min(protocol['marginal_genes'],len(runtime.genes)),replace=False)
        gids.append(torch.as_tensor(np.sort(selected),device=runtime.device))
    weights=dict(protocol['loss_weights'])
    warmup=min(1.,max(protocol['distribution_weight_warmup_minimum'],epoch/protocol['distribution_weight_warmup_epochs']))
    for key in ('joint_marginal','joint_structure'):weights[key]*=warmup
    return objective_conditions(output,list(real.unbind(0)),labels,
        torch.stack([b['target_control_mean'] for b in batch]),scales['response_scale'],scales['gene_scale'],
        scales['train_projections'][:,:protocol['joint_projections']],scales['components'],
        own_genes=[runtime.genes.index(r['target']) if r['target'] in runtime.genes else -1 for r in records],
        sampled_genes=gids,weights=weights,
        relation=[panels[r['query']].relation_inputs(r['target']) for r in records],relation_multiplier=[1.]*len(records),
        control_cells=torch.stack([b['control_cells'] for b in batch]),real_effective_count=counts,
        covariance_prior_strength=protocol['covariance_prior_strength'],source_batches=batch,
        source_readout_weight=protocol['source_readout_weight'],
        source_readout_detection_weight=protocol['source_readout_detection_weight'])


def train(data,priors,graph,graph_axis,splits,fold_id,seed,output,config,device='cuda:0',*,resume=False,smoke=False):
    protocol=read_json(config);fold=read_json(Path(splits)/fold_id/'FOLD.json')
    panels_meta=read_json(Path(splits)/fold_id/'VALIDATION_PANEL.json')
    seed=int(seed)
    if seed not in protocol['seeds'] and not smoke:raise ValueError('Unregistered initialization seed')
    run=Path(output);run.mkdir(parents=True,exist_ok=True)
    if (run/'last.pt').exists() and not resume:raise FileExistsError('Existing run: use --resume or a new output directory')
    if (run/'COMPLETED.json').exists():return read_json(run/'COMPLETED.json')
    protocol=copy.deepcopy(protocol)
    protocol.update(seed=seed,fold=fold,smoke=bool(smoke))
    if (run/'protocol.json').exists() and read_json(run/'protocol.json')!=protocol:
        raise ValueError('Resume may not alter the scientific protocol')
    write_json(run/'protocol.json',protocol)
    seed_all(seed,protocol.get('cpu_threads',4))
    assets,geometry,runtime=prepare_assets(data,priors,graph,graph_axis,fold,run/'assets',device,protocol,
                                         fit=not (run/'assets/BINDING.json').exists())
    infer=None
    try:
        protocol['conditions_per_epoch']=sum(len(assets.train_store.training_targets(e)) for e in fold['train'])
        scales=device_scales(assets,device)
        seed_all(seed,protocol.get('cpu_threads',4))
        model=BatchedSharedPathModel(geometry,route='open_zero',**protocol['model']).to(device)
        model.set_phase('J')
        if not all(p.requires_grad for p in model.parameters()):raise RuntimeError('Unexpected frozen task weights')
        spec=protocol['phases'][0]
        optimizer=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=protocol['weight_decay'])
        controller=JointPlateauController(min_epochs=spec['min_epochs'],patience=spec['patience'],
            relative_delta=spec['min_delta'],reductions=tuple(spec['lr_patience']),
            min_optimizer_steps=spec['min_optimizer_steps'],post_lr_drop_epochs=spec['post_lr_drop_epochs'])
        state=dict(epoch=1,batch_index=0,conditions_seen=0,optimizer_steps=0,lr_multiplier=1.,
                   history=[],controller=controller.state_dict(),best_objective=None,best_epoch=None)
        if resume:
            payload=load_checkpoint(run/'last.pt',device)
            if payload['asset_binding']!=assets.binding:raise ValueError('Resume inputs changed')
            model.load_state_dict(payload['state_dict'],strict=True);model.set_phase('J')
            optimizer.load_state_dict(payload['optimizer']);state=payload['state']
            controller=JointPlateauController.from_state(state['controller']);_restore_rng(payload['rng'])
        else:
            write_json(run/'INITIALIZATION.json',dict(seed=seed,task_weights_fresh=True,
                historical_task_weights_loaded=False,source_mean_output_anchor=False,
                parameters=sum(p.numel() for p in model.parameters())))
        banks={e:runtime.training_response_bank(e) for e in fold['train']}
        # Move fixed training-only candidate banks once, not once per epoch.
        for bank in banks.values():
            for key in ('responses','reliability','mask'):
                bank[key]=torch.as_tensor(bank[key],device=device)
        infer=evaluation_runtime(assets,geometry,run/'evaluation_sources',device)
        from .evaluation import validate
        converged=controller.converged
        cap=int(spec['extended_max_epochs'])
        for epoch in range(state['epoch'],cap+1):
            if converged:break
            model.train();started=time.monotonic()
            rank_panels={e:RankingPanels(banks[e],runtime.genes,seed=seed,epoch=epoch,
                panel_size=protocol['new_relation_candidates'],temperature=protocol['rank_temperature'],
                norm_floor=protocol['relation_norm_floor']) for e in fold['train']}
            schedule=joint_condition_schedule(fold,runtime.targets,seed=seed,epoch=epoch,
                per_environment=protocol.get('microbatch_per_environment',16))
            if state['batch_index'] and state.get('schedule_hash')!=schedule['schedule_sha256']:
                raise ValueError('Resume schedule changed')
            state['schedule_hash']=schedule['schedule_sha256']
            epoch_loss=0.;seen=0
            for bi in range(state['batch_index'],len(schedule['batches'])):
                records=schedule['batches'][bi]['items'];groups=OrderedDict()
                for record in records:groups.setdefault((record['query'],tuple(record['sources'])),[]).append(record)
                optimizer.zero_grad(set_to_none=True)
                loss_value=0.;terms_value={}
                micro_limit=int(protocol.get('execution',{}).get('microbatch_size',16))
                if micro_limit<1:raise ValueError('Microbatch limit must be positive')
                micros=[group[start:start+micro_limit] for group in groups.values()
                        for start in range(0,len(group),micro_limit)]
                for records_micro in micros:
                    # Runtime's independent condition batching is preserved.
                    loss,terms=micro_loss(model,runtime,scales,rank_panels,records_micro,epoch,protocol)
                    weight=len(records_micro)/len(records);(loss*weight).backward()
                    loss_value+=float(loss.detach())*weight
                    for k,v in terms.items():terms_value[k]=terms_value.get(k,0.)+float(v)*weight
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['gradient_clip'])
                if not math.isfinite(loss_value) or not bool(torch.isfinite(grad)):
                    raise RuntimeError('Nonfinite loss/gradient; optimizer was not stepped')
                warmup=min(1.,(state['conditions_seen']+len(records))/max(1,protocol['warmup_updates']))
                for group in optimizer.param_groups:
                    group['lr']=spec['learning_rate']*state['lr_multiplier']*warmup
                optimizer.step();state['conditions_seen']+=len(records);state['optimizer_steps']+=1
                state.update(epoch=epoch,batch_index=bi+1)
                epoch_loss+=loss_value*len(records);seen+=len(records)
                if state['optimizer_steps']==1 or state['optimizer_steps']%20==0:
                    append_json(run/'training.jsonl',dict(epoch=epoch,optimizer_steps=state['optimizer_steps'],
                        conditions_seen=state['conditions_seen'],loss=loss_value,terms=terms_value,
                        lr=optimizer.param_groups[0]['lr'],seconds=time.monotonic()-started))
                if state['optimizer_steps']%100==0:
                    state['controller']=controller.state_dict()
                    save_checkpoint(run/'last.pt',model,optimizer,state,protocol,assets.binding)
            state.update(epoch=epoch+1,batch_index=0,schedule_hash=None)
            if epoch%protocol['validation_every_epochs']==0:
                checkpoint=run/'checkpoints'/f'epoch_{epoch:03d}.pt'
                save_checkpoint(checkpoint,model,optimizer,state,protocol,assets.binding)
                rng=_rng()
                try:
                    metrics=validate(model,infer,assets,scales,checkpoint,panels_meta,
                                     run/'validation'/f'epoch_{epoch:03d}',protocol,epoch)
                finally:_restore_rng(rng);model.train()
                score=float(metrics['objective'])
                state['history'].append(dict(epoch=epoch,objective=score,checkpoint=str(checkpoint)))
                if state['best_objective'] is None or score<state['best_objective']:
                    state.update(best_objective=score,best_epoch=epoch,best_checkpoint=str(checkpoint))
                decision=controller.observe(epoch,{'joint_objective':score},state['optimizer_steps'])
                if decision['reduce_lr']:state['lr_multiplier']*=.5
                converged=decision['converged']
                write_json(run/'VALIDATION_HISTORY.json',dict(candidates=state['history'],stopping=controller.state_dict(),
                    selected_epoch=state['best_epoch'],selected_objective=state['best_objective']))
            state['controller']=controller.state_dict()
            save_checkpoint(run/'last.pt',model,optimizer,state,protocol,assets.binding)
            write_json(run/'STATUS.json',dict(status='converged' if converged else 'training',epoch=epoch,
                optimizer_steps=state['optimizer_steps'],selected_epoch=state['best_epoch'],test_labels_read=0))
        result=dict(status='completed' if converged else 'needs_decision',joint_converged=bool(converged),
            fold=fold_id,seed=seed,selected_epoch=state['best_epoch'],selected_checkpoint=state.get('best_checkpoint'),
            selected_objective=state['best_objective'],test_labels_read=0,smoke=bool(smoke))
        write_json(run/'SELECTION.json',result)
        if converged:write_json(run/'COMPLETED.json',result)
        return result
    finally:
        assets.train_store.close()
        if infer is not None:infer.assets.train_store.close()
