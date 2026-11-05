#!/usr/bin/env python3
"""Check a source tree for submission-identity and generated-artifact leaks.

This is a conservative packaging check, not a guarantee about a hosting account.
Public dataset and third-party software attribution must be retained.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import io
import tokenize

FORBIDDEN_DIRS = {'.git', '__pycache__', '.pytest_cache', '.ruff_cache', '.venv',
                  'wandb', '.ipynb_checkpoints', '.idea', '.vscode'}
FORBIDDEN_SUFFIXES = {'.pyc','.pyo','.pt','.pth','.ckpt','.safetensors','.h5','.h5ad',
                      '.parquet','.npz','.npy','.pickle','.pkl','.log','.ipynb'}
PATTERNS = {
    'personal_email': re.compile(r'(?<![\w.])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),
    'server_path': re.compile(r'/(?:root|home|mnt|Users)/[A-Za-z0-9_.-]+(?:[/\\][^\s\"\x27<>]*)?'),
    'windows_user_path': re.compile(r'[A-Za-z]:[\\/]+Users[\\/]+[^\s\"\x27]+'),
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'embedded_credentials': re.compile(r'https?://[^/\s:@]+:[^/@\s]+@'),
    'personal_name': re.compile(r'Jia' + r'wen|Liu\s+Jia' + r'wen',re.I),
    'private_workspace_id': re.compile(r'(?<![a-z])l' + r'jw(?![a-z])',re.I),
    'private_launch_id': re.compile(r'20\d{6}T\d{6}Z'),
    'ipv4_address': re.compile(r'(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\d.])'),
    'service_token': re.compile(r'(?<![A-Za-z0-9])(?:h' + r'f_|gh' + r'p_|github_' + r'pat_)[A-Za-z0-9_]{20,}'),
}
# Exact public domains/identities used for external attribution are reviewed in
# registry.json and the data/prior catalogs. They are not this paper's authors.
PUBLIC_GITHUB_OWNERS = {'valence-labs','theislab','arcinstitute','yongzhuangliulab',
                        'whd1125','tahoebio','yiqunchen','scverse','pytorch'}


def audit(root):
    root=Path(root).resolve();issues=[];files=0;third_party_links=set()
    for path in sorted(root.rglob('*')):
        relative=path.relative_to(root)
        if path.is_symlink():issues.append({'file':str(relative),'type':'symlink'});continue
        if path.is_dir():
            if path.name in FORBIDDEN_DIRS:issues.append({'file':str(relative),'type':'generated_or_identity_directory'})
            continue
        files+=1
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name.startswith('.env'):
            issues.append({'file':str(relative),'type':'binary_runtime_or_secret_artifact'})
        if path.stat().st_size>8*1024*1024:issues.append({'file':str(relative),'type':'oversized_source_file'})
        try:text=path.read_text(encoding='utf-8-sig')
        except UnicodeError:
            issues.append({'file':str(relative),'type':'unexpected_binary'});continue
        # This checker necessarily contains its own pattern definitions.
        if relative.as_posix()=='scripts/check_anonymity.py':continue
        email_text=text
        if path.suffix=='.py':
            # Emails in Python matter in literals/comments; matrix products
            # such as x@np.asarray are executable operators, not contacts.
            lines=text.splitlines(keepends=True); masked=[list(' '*len(line)) for line in lines]
            try:
                for tok in tokenize.generate_tokens(io.StringIO(text).readline):
                    if tok.type not in (tokenize.STRING, tokenize.COMMENT):continue
                    (r0,c0),(r1,c1)=tok.start,tok.end
                    for row in range(r0-1,r1):
                        lo=c0 if row==r0-1 else 0; hi=c1 if row==r1-1 else len(lines[row])
                        masked[row][lo:hi]=list(lines[row][lo:hi])
                email_text=''.join(''.join(line) for line in masked)
            except tokenize.TokenError:pass
        for kind,pattern in PATTERNS.items():
            for match in pattern.finditer(email_text if kind=='personal_email' else text):
                # Do not expose any matched personal data in the report itself.
                issues.append({'file':str(relative),'line':text.count('\n',0,match.start())+1,'type':kind})
        for match in re.finditer(r'https://github\.com/([^/\s)]+)',text):
            owner=match.group(1).lower()
            if owner not in PUBLIC_GITHUB_OWNERS:
                issues.append({'file':str(relative),'line':text.count('\n',0,match.start())+1,'type':'unreviewed_github_owner'})
            else:third_party_links.add(owner)
    return {'status':'passed' if not issues else 'failed','files_scanned':files,
            'issues':issues,'reviewed_external_github_owners':sorted(third_party_links),
            'scope':'source-tree checks; no Git history, host-account or external-service anonymity claim'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',nargs='?',default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    result=audit(a.root);print(json.dumps(result,indent=2));raise SystemExit(bool(result['issues']))

if __name__=='__main__':main()
