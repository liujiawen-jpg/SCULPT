#!/usr/bin/env python3
"""Build a deterministic anonymous ZIP from a clean, verified source tree."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import stat
import zipfile
from check_anonymity import audit


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source,output):
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Refusing to replace an existing archive')
    if source in output.parents:raise ValueError('Write archive outside the source tree')
    result=audit(source)
    if result['issues']:raise RuntimeError(f"Source-tree anonymity checks failed: {result['issues']}")
    files=sorted(p for p in source.rglob('*') if p.is_file())
    manifest=source/'SOURCE_SHA256SUMS'
    text=''.join(f'{digest(p)}  {p.relative_to(source).as_posix()}\n' for p in files if p!=manifest)
    manifest.write_text(text)
    files=sorted(p for p in source.rglob('*') if p.is_file())
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in files:
            info=zipfile.ZipInfo('SCULPT/'+path.relative_to(source).as_posix(),date_time=(1980,1,1,0,0,0))
            info.create_system=3
            mode=0o755 if path.suffix=='.sh' else 0o644
            info.external_attr=(stat.S_IFREG|mode)<<16
            info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,path.read_bytes())
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for path in files:
            assert archive.read('SCULPT/'+path.relative_to(source).as_posix())==path.read_bytes()
    checksum=output.with_suffix(output.suffix+'.sha256')
    checksum.write_text(f'{digest(output)}  {output.name}\n')
    return len(files),digest(output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    n,sha=build(a.source,a.output);print(f'files={n}\nsha256={sha}\narchive={a.output}')

if __name__=='__main__':main()
