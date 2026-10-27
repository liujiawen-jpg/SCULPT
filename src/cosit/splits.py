"""The published split is generated from identifiers, never response values."""
from __future__ import annotations
import hashlib
from pathlib import Path
from .io import write_json, read_json, canonical_hash

CONTEXTS = ('K562_essential', 'RPE1', 'HepG2', 'Jurkat', 'Orion_HCT116')
FOLDS = [
    dict(id='F1', train=['K562_essential','HepG2','Orion_HCT116'], validation='RPE1', test='Jurkat'),
    dict(id='F2', train=['HepG2','Jurkat','Orion_HCT116'], validation='K562_essential', test='RPE1'),
    dict(id='F3', train=['RPE1','Jurkat','Orion_HCT116'], validation='HepG2', test='K562_essential'),
    dict(id='F4', train=['K562_essential','RPE1','Jurkat'], validation='Orion_HCT116', test='HepG2'),
    dict(id='F5', train=['K562_essential','RPE1','HepG2'], validation='Jurkat', test='Orion_HCT116'),
]


def fixed_targets(targets, count, fold, purpose):
    values=list(targets)
    chosen=sorted(range(len(values)),key=lambda i:(
        hashlib.sha256(f'{fold}|{purpose}|{values[i]}'.encode()).digest(),values[i]))[:min(count,len(values))]
    # Preserve the supplied axis, including hash-ordered internal holdouts.
    return [values[i] for i in sorted(chosen)]


def make_fold(base, targets, count=153, seed=17):
    targets = list(targets)
    if len(set(targets)) != len(targets) or 3 * count > len(targets) or count < 1:
        raise ValueError('Need unique targets and three disjoint nonempty holdouts')
    fold = dict(base)
    order = sorted(targets, key=lambda t: hashlib.sha256(
        f'joint-condition-holdout-v1|{seed}|{fold["id"]}|{t}'.encode()).hexdigest())
    fold['joint_holdout'] = dict(schema='joint_condition_holdout_v1', seed=seed,
        count_per_environment=count,
        held_targets={e:order[i*count:(i+1)*count] for i,e in enumerate(fold['train'])},
        selection_uses_response_values=False,
        forbidden_training_roles=['query','source','supervision','negative_bank','fitted_statistics'],
        evaluation_source_use_permitted=True)
    return fold


def validation_panels(fold, targets, config):
    panels = {fold['validation']: dict(environment=fold['validation'],role='external',weight=.5,
        response_targets=fixed_targets(targets,config['external_response_conditions'],
                                      fold['id'],'joint_external_response'))}
    for env in fold['train']:
        panels[env] = dict(environment=env,role='internal',weight=1/6,
            response_targets=fixed_targets(fold['joint_holdout']['held_targets'][env],
                config['internal_response_conditions'],fold['id'],'joint_internal_response_'+env))
    for env,panel in panels.items():
        n=config['external_distribution_conditions' if panel['role']=='external' else 'internal_distribution_conditions']
        panel['distribution_targets']=fixed_targets(panel['response_targets'],n,fold['id'],'joint_distribution_'+env)
    return panels


def generate_splits(targets, output, protocol, frozen=None):
    output=Path(output)
    for base in FOLDS:
        fold=make_fold(base,targets,protocol['joint_holdout']['count_per_training_environment'],
                       protocol['joint_holdout']['seed'])
        panels=validation_panels(fold,targets,protocol['joint_validation'])
        allowed={e:[t for t in targets if t not in set(fold['joint_holdout']['held_targets'][e])]
                 for e in fold['train']}
        if frozen:
            reference=Path(frozen)/fold['id']
            if read_json(reference/'FOLD.json') != fold or read_json(reference/'VALIDATION_PANEL.json') != panels:
                raise ValueError('Generated split differs from the released benchmark')
        root=output/fold['id']
        write_json(root/'FOLD.json',fold)
        write_json(root/'VALIDATION_PANEL.json',panels)
        write_json(root/'TRAINING_TARGETS.json',dict(allowed_targets=allowed,
            conditions=sum(map(len,allowed.values())),response_values_used=False))
    write_json(output/'MANIFEST.json',dict(schema='cosit_splits_v1',target_axis_sha256=canonical_hash(list(targets)),
        initialization_seeds=protocol['seeds'],splits_depend_on_initialization_seed=False,
        response_values_read=0,folds=[f['id'] for f in FOLDS]))
    return output
