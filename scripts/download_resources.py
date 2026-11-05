#!/usr/bin/env python3
"""Download public generic resources; never fetch task-trained checkpoints.

--dry-run performs no HTTP requests. Existing files must match their recorded
hash when available. The actual URLs and hashes are recorded beside downloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


STRING_BASE = 'https://stringdb-downloads.org/download/'
TAHOE_REVISION = 'd218a580b9c2500ae9dfc8367a398545e6f017a8'
RECORDS = {
 'tahoe': [
  ('model.safetensors', 'https://huggingface.co/tahoebio/Tahoe-x1/resolve/main/1b-model/model.safetensors',
   '21f2afc7ee5ec88cadddd6743188aea590b89306c8f2538e137bf9dcacfc0260'),
  ('vocab.json', 'https://huggingface.co/tahoebio/Tahoe-x1/resolve/main/1b-model/vocab.json', None)],
 'genept': [('GenePT_emebdding_v2.zip',
   'https://zenodo.org/records/10833191/files/GenePT_emebdding_v2.zip?download=1', None)],
 'string': [
  ('9606.protein.links.v12.0.txt.gz', STRING_BASE+'protein.links.v12.0/9606.protein.links.v12.0.txt.gz',
   '3e22f32572211aa341d5b4bd08d30c32e693e294603202120936872f87719d4f'),
  ('9606.protein.info.v12.0.txt.gz', STRING_BASE+'protein.info.v12.0/9606.protein.info.v12.0.txt.gz',
   '144de4b0d98c6a7dfde6ddc2591cf88657f27b989eadff4f501450c3ed1f0f1c'),
  ('9606.protein.aliases.v12.0.txt.gz', STRING_BASE+'protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz',
   'b65f730b993ed0c1bd72edf4565d3d425db42861101b29699704810e8f125680')],
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def plan(resources, *, tahoe_revision=TAHOE_REVISION, hgnc_url=None, hgnc_sha256=None):
    result = []
    for resource in resources:
        if resource == 'hgnc':
            if not hgnc_url or not hgnc_sha256:
                raise ValueError('Pin HGNC with both --hgnc-url and --hgnc-sha256; an unpinned latest mapping is not substituted')
            if not hgnc_url.startswith('https://') or not re.fullmatch('[0-9a-f]{64}', hgnc_sha256):
                raise ValueError('HGNC needs HTTPS and a lowercase SHA256')
            result.append(dict(resource=resource, filename='hgnc_complete_set.txt', url=hgnc_url, expected_sha256=hgnc_sha256))
            continue
        for name, url, sha in RECORDS[resource]:
            if resource == 'tahoe' and tahoe_revision:
                if not re.fullmatch('[0-9a-f]{40}', tahoe_revision):
                    raise ValueError('Use an immutable 40-character Tahoe commit')
                url = url.replace('/resolve/main/', '/resolve/'+tahoe_revision+'/')
            result.append(dict(resource=resource, filename=name, url=url, expected_sha256=sha))
    return result


def fetch(record, directory):
    import requests
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory / record['filename']
    receipt = directory / (record['filename']+'.download.json')
    if path.exists():
        actual = digest(path)
        expected = record['expected_sha256']
        if expected is not None and actual != expected:
            raise ValueError('Existing public resource has different bytes: '+path.name)
        if expected is None:
            if not receipt.exists() or json.loads(receipt.read_text()).get('actual_sha256') != actual:
                raise ValueError('Unbound existing resource; provide a fresh download directory: '+path.name)
        return path
    partial = path.with_name(path.name+'.partial')
    with requests.get(record['url'], stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        final_url = response.url
        headers = {k: response.headers[k] for k in ('ETag','X-Repo-Commit','Content-Length') if k in response.headers}
        with partial.open('wb') as f:
            for block in response.iter_content(8 << 20):
                if block: f.write(block)
    actual = digest(partial)
    if record['expected_sha256'] and actual != record['expected_sha256']:
        raise ValueError('Resource bytes differ from the registered release; the .partial file was not published: '+path.name)
    partial.replace(path)
    # Signed redirect URLs can contain transient tokens; do not store them.
    value = dict(record, actual_sha256=actual, bytes=path.stat().st_size, provider_headers=headers,
                 final_host=final_url.split('/')[2], task_checkpoint=False)
    receipt.write_text(json.dumps(value, indent=2)+'\n')
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resources', nargs='+', choices=[*RECORDS, 'hgnc'], default=['tahoe','genept','string'])
    p.add_argument('--output', type=Path, default=Path('data/external'))
    p.add_argument('--tahoe-revision', default=TAHOE_REVISION)
    p.add_argument('--hgnc-url'); p.add_argument('--hgnc-sha256')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    records = plan(a.resources, tahoe_revision=a.tahoe_revision, hgnc_url=a.hgnc_url, hgnc_sha256=a.hgnc_sha256)
    if a.dry_run:
        print(json.dumps(records, indent=2)); return
    if 'tahoe' in a.resources and not a.tahoe_revision:
        p.error('For a real Tahoe download, set --tahoe-revision to an immutable commit; see docs/PRIORS.md')
    for r in records: print(fetch(r, a.output))


if __name__ == '__main__': main()
