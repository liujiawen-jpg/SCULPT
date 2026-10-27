#!/usr/bin/env python3
"""Run the registered five-context, three-initialization benchmark sequentially.

No background monitor or GPU auto-kill is installed. An interrupted run resumes
only its own checkpoint. Other projects and processes are never touched.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from cosit.io import read_json


def execute(args):
    print('+', ' '.join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data','priors','graph','graph-axis','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/j0.json'))
    p.add_argument('--frozen-splits',type=Path,default=Path('configs/splits'))
    p.add_argument('--folds',nargs='+',choices=['F1','F2','F3','F4','F5'],default=['F1','F2','F3','F4','F5'])
    p.add_argument('--seeds',nargs='+',type=int,choices=[17,29,43],default=[17,29,43])
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--evaluation-device',default='cpu')
    p.add_argument('--train-only',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    base=[sys.executable,'-m','cosit'];splits=args.output/'splits'
    execute(base+['audit-data','--data',args.data])
    execute(base+['split','--data',args.data,'--config',args.config,'--output',splits,'--verify-frozen',args.frozen_splits])
    metrics=[]
    for fold in args.folds:
        for seed in args.seeds:
            run=args.output/fold/f'seed_{seed}'
            command=base+['train','--data',args.data,'--priors',args.priors,'--graph',args.graph,
                '--graph-axis',args.graph_axis,'--splits',splits,'--fold',fold,'--seed',seed,
                '--output',run,'--config',args.config,'--device',args.device]
            if (run/'last.pt').exists():command+=['--resume']
            execute(command)
            selection=read_json(run/'SELECTION.json')
            if not selection.get('joint_converged'):
                raise RuntimeError(f'{fold}/seed_{seed}: hard epoch cap is not convergence; test remains sealed')
            if args.train_only:continue
            prediction=run/'test_predictions.h5'
            if not prediction.exists():
                execute(base+['predict','--data',args.data,'--priors',args.priors,'--graph',args.graph,
                    '--graph-axis',args.graph_axis,'--run',run,'--output',prediction,'--device',args.device])
            result=run/'test_metrics.json'
            if not result.exists():
                execute(base+['evaluate','--data',args.data,'--predictions',prediction,
                    '--checkpoint',selection['selected_checkpoint'],'--assets',run/'assets',
                    '--output',result,'--device',args.evaluation_device])
            metrics.append(result)
    if metrics:
        command=base+['aggregate','--metrics',*metrics,'--output',args.output/'summary.json']
        if len(metrics)!=15:command+=['--allow-incomplete']
        execute(command)


if __name__=='__main__':main()
