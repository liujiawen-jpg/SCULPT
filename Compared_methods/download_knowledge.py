"""Download pinned public comparator knowledge files into user-selected storage.

This does not download perturbation labels, train models, or modify the upstream
checkouts. PIE knowledge can be large: use --dry-run to inspect the plan first.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

PIE_REVISION = 'fb624a5158a7f37dfa13a015cdacadb985549d26'
PIE_REPOSITORY = 'arcinstitute/PIE_sources'
TXPERT_COMMIT = '08d82eea86746b044cf7531f4ec8c5f60e1cb73f'
TXPERT_GRAPHS = [
    ('go/go_top_50.csv', 'ad469c852ba9b8b5489749c7987467687aa566598fc0706fd7232aa27edce27b'),
    ('string/v11.5.parquet', '55d312f8d6186078eb00a7caf108ef731cd27aa08a5b48de59cd0333f0206404'),
]


def txpert_graphs(directory, dry_run):
    base = 'https://raw.githubusercontent.com/valence-labs/TxPert/'+TXPERT_COMMIT+'/data/graphs/'
    records = [dict(file=n, url=base+n, sha256=h) for n, h in TXPERT_GRAPHS]
    if dry_run:
        print(json.dumps(records, indent=2)); return
    import requests
    for item in records:
        path = directory / item['file']
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Different existing graph: '+item['file'])
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix+'.partial')
        h = hashlib.sha256()
        with requests.get(item['url'], stream=True, timeout=(30, 180)) as r:
            r.raise_for_status()
            with tmp.open('wb') as f:
                for b in r.iter_content(8<<20):
                    if b: f.write(b); h.update(b)
        if h.hexdigest() != item['sha256']:
            raise ValueError('Official graph bytes differ from the registered benchmark: '+item['file'])
        tmp.replace(path)
    (directory/'GRAPH_SOURCE.json').write_text(json.dumps(dict(revision=TXPERT_COMMIT, files=records), indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('method', choices=['pie', 'txpert'])
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--include', nargs='+', help='Optional upstream glob patterns; missing required sources must not be replaced by zero/random features')
    a = p.parse_args()
    if a.method == 'txpert':
        txpert_graphs(a.directory, a.dry_run); return
    value = dict(repository=PIE_REPOSITORY, repo_type='dataset', revision=PIE_REVISION,
                 local_dir=str(a.directory), allow_patterns=a.include,
                 required_sources=['esm2','ncbi_text','string_space','depmap_gene_effect',
                                   'context_text','perturbation_text','gene_text'])
    if a.dry_run:
        print(json.dumps(value, indent=2)); return
    from huggingface_hub import snapshot_download
    directory = snapshot_download(repo_id=PIE_REPOSITORY, repo_type='dataset', revision=PIE_REVISION,
                                  local_dir=a.directory, allow_patterns=a.include)
    value['downloaded_directory'] = str(directory)
    (a.directory/'KNOWLEDGE_SOURCE.json').write_text(json.dumps(value, indent=2)+'\n')


if __name__ == '__main__': main()
