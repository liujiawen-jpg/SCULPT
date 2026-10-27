"""Equal context weighting after averaging initialization replicates."""
from pathlib import Path
import numpy as np
from .io import read_json,write_json
from .splits import FOLDS


def _summary(values):
    valid=[float(v) for v in values if v is not None and np.isfinite(v)]
    return dict(mean=float(np.mean(valid)) if valid else None,
                variance=float(np.var(valid,ddof=1)) if len(valid)>1 else None,
                std=float(np.std(valid,ddof=1)) if len(valid)>1 else None,n_valid=len(valid))


def aggregate(paths,output,*,require_complete=True):
    records=[read_json(p) for p in paths];keys=[(r['fold'],int(r['seed'])) for r in records]
    if not records:raise ValueError('No evaluation results supplied')
    methods={r.get('method') for r in records}
    if len(methods)!=1 or any(not isinstance(m,str) or not m.strip() for m in methods):
        raise ValueError('Cannot aggregate different methods or missing method identities')
    if len(set(keys))!=len(keys):raise ValueError('Duplicate fold/seed result')
    expected={(f'F{i}',s) for i in range(1,6) for s in (17,29,43)}
    if require_complete and set(keys)!=expected:
        raise ValueError('Primary table requires all five folds × three seeds')
    groups={f:[r for r in records if r['fold']==f] for f in sorted({r['fold'] for r in records})}
    if any(len({r.get('environment') for r in rows})!=1 for rows in groups.values()):
        raise ValueError('A fold cannot contain different test environments')
    registered={f['id']:f['test'] for f in FOLDS}
    if any(r.get('environment')!=registered.get(r['fold']) or r['fold'] not in registered for r in records):
        raise ValueError('Each fold must identify its registered test environment')
    result=dict(schema='cosit_summary_v1',aggregation='within-fold seed mean, then equal context mean; ddof=1',
                method=next(iter(methods)),complete=set(keys)==expected,folds={},macro={})
    for family in ('response','distribution'):
        metrics=set().union(*(r.get(family,{}) for r in records))
        for fold,rows in groups.items():
            result['folds'].setdefault(fold,{})['environment']=registered[fold]
            result['folds'].setdefault(fold,{})[family]={m:_summary([r.get(family,{}).get(m) for r in rows]) for m in sorted(metrics)}
        result['macro'][family]={m:_summary([result['folds'][f][family][m]['mean'] for f in groups]) for m in sorted(metrics)}
    write_json(output,result)
    return result
