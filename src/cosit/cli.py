"""Public command-line interface. Importing CoSIT never starts work."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import json
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def parser():
    p=argparse.ArgumentParser(prog='cosit',description='CoSIT J0 reproducible cross-context perturbation prediction')
    sub=p.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare',help='Normalize raw public datasets into immutable cell banks')
    prepare.add_argument('--config',type=Path,default=Path('configs/datasets.yaml'))
    prepare.add_argument('--raw-root',type=Path,required=True)
    prepare.add_argument('--output',type=Path,required=True)
    prepare.add_argument('--contexts',nargs='+')
    prepare.add_argument('--overwrite',action='store_true')
    audit=sub.add_parser('audit-data',help='Check prepared files and released coordinate axes')
    audit.add_argument('--data',type=Path,required=True)
    audit.add_argument('--axes',type=Path,default=Path('configs/axes'))
    split=sub.add_parser('split',help='Generate fixed metadata-only train/validation/test splits')
    split.add_argument('--data',type=Path,required=True)
    split.add_argument('--config',type=Path,default=Path('configs/j0.json'))
    split.add_argument('--output',type=Path,required=True)
    split.add_argument('--verify-frozen',type=Path)
    train=sub.add_parser('train',help='Train every J0 task component jointly from scratch')
    for flag in ('data','priors','graph','graph-axis','splits','output'):
        train.add_argument('--'+flag,type=Path,required=True)
    train.add_argument('--config',type=Path,default=Path('configs/j0.json'))
    train.add_argument('--fold',choices=['F1','F2','F3','F4','F5'],required=True)
    train.add_argument('--seed',type=int,default=17)
    train.add_argument('--device',default='cuda:0')
    train.add_argument('--resume',action='store_true')
    train.add_argument('--smoke',action='store_true',help='Mark synthetic/development run; never a benchmark result')
    pred=sub.add_parser('predict',help='Generate and seal all test clouds using the validation-selected checkpoint')
    for flag in ('data','priors','graph','graph-axis','run','output'):
        pred.add_argument('--'+flag,type=Path,required=True)
    pred.add_argument('--device',default='cuda:0')
    pred.add_argument('--batch-size',type=int,default=4)
    eva=sub.add_parser('evaluate',help='Score frozen test mean/distribution predictions')
    for flag in ('data','predictions','checkpoint','assets','output'):
        eva.add_argument('--'+flag,type=Path,required=True)
    eva.add_argument('--device',default='cpu')
    agg=sub.add_parser('aggregate',help='Report within-context seeds, then five-context mean and variance')
    agg.add_argument('--metrics',type=Path,nargs='+',required=True)
    agg.add_argument('--output',type=Path,required=True)
    agg.add_argument('--allow-incomplete',action='store_true')
    smoke=sub.add_parser('smoke-test',help='Run real model/loss/inference/evaluation on synthetic cells')
    smoke.add_argument('--output',type=Path,required=True)
    return p


def audit_data(data,axes):
    import csv
    import h5py
    from .io import read_json,sha256
    from .splits import CONTEXTS
    manifest=read_json(data/'MANIFEST.json')
    records=manifest.get('contexts',manifest.get('datasets',[]))
    if isinstance(records,dict):records=list(records.values())
    by_file={r.get('file',r.get('path','')):r for r in records}
    genes=[r['gene_id'] for r in csv.DictReader((axes/'primary_genes.tsv').open(),delimiter='\t')]
    targets=[r['target_stable_id'] for r in csv.DictReader((axes/'targets.tsv').open(),delimiter='\t')]
    for env in CONTEXTS:
        path=data/(env+'.h5');record=by_file.get(path.name)
        if record is None or record.get('sha256')!=sha256(path):raise ValueError('Prepared file is not byte-bound: '+env)
        with h5py.File(path,'r') as h:
            if json.loads(h.attrs['genes_json'])!=genes or json.loads(h.attrs['targets_json'])!=targets:
                raise ValueError('Prepared data does not follow the published coordinate axes')
    return dict(status='passed',contexts=5,genes=len(genes),targets=len(targets))


def main(argv=None):
    args=parser().parse_args(argv)
    if args.command=='prepare':
        from .preprocessing import prepare_data
        result=prepare_data(args.config,args.raw_root,args.output,contexts=args.contexts,overwrite=args.overwrite)
    elif args.command=='audit-data':result=audit_data(args.data,args.axes)
    elif args.command=='split':
        import h5py
        from .splits import generate_splits
        from .io import read_json
        with h5py.File(args.data/'K562_essential.h5','r') as h:targets=json.loads(h.attrs['targets_json'])
        result=generate_splits(targets,args.output,read_json(args.config),args.verify_frozen)
    elif args.command=='train':
        from .training import train
        result=train(args.data,args.priors,args.graph,args.graph_axis,args.splits,args.fold,args.seed,args.output,args.config,
                     args.device,resume=args.resume,smoke=args.smoke)
    elif args.command=='predict':
        from .inference import predict_run
        result=predict_run(args.data,args.priors,args.graph,args.graph_axis,args.run,args.output,args.device,args.batch_size)
    elif args.command=='evaluate':
        from .evaluation import evaluate
        result=evaluate(args.data,args.predictions,args.checkpoint,args.assets,args.output,device=args.device)
        result={k:v for k,v in result.items() if not k.startswith('per_condition')}
    elif args.command=='aggregate':
        from .aggregate import aggregate
        result=aggregate(args.metrics,args.output,require_complete=not args.allow_incomplete)
    elif args.command=='smoke-test':
        from .smoke import run_smoke
        result=run_smoke(args.output)
    from .io import json_default
    print(json.dumps(result,indent=2,default=json_default,allow_nan=False))


if __name__=='__main__':main()
