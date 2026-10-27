#!/usr/bin/env python3
"""Prepare public J0 STRING assets and static priors without training a model."""
import argparse
import csv
from pathlib import Path
from cosit.priors import build_string_graph, build_prior


def read_axis(path):
    with Path(path).open() as stream:
        reader=csv.DictReader(stream,delimiter='\t')
        key=next((name for name in ('gene_id','target_stable_id','target_id') if name in reader.fieldnames),None)
        if key is None:raise ValueError('Axis TSV needs gene_id or target_stable_id column')
        return [row[key] for row in reader]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    graph=commands.add_parser('string',help='Map proteins uniquely and build STRING v12 score>=700 graph')
    for name in ('hgnc','links','info','aliases','out'):graph.add_argument('--'+name,type=Path,required=True)
    graph.add_argument('--genes',type=Path);graph.add_argument('--targets',type=Path)
    prior=commands.add_parser('prior',help='Create the original five-block 261-dimensional J0 gene prior')
    for name in ('genes','targets','tahoe-weights','tahoe-vocab','genept','hgnc','graph-directory','out'):
        prior.add_argument('--'+name,type=Path,required=True)
    prior.add_argument('--frozen-genept-features',type=Path)
    prior.add_argument('--trust-genept-pickle',action='store_true')
    args=parser.parse_args();kw=vars(args);command=kw.pop('command')
    genes=kw.pop('genes',None);targets=kw.pop('targets',None)
    axis=list(dict.fromkeys((read_axis(genes) if genes else [])+(read_axis(targets) if targets else [])))
    if command=='string':path=build_string_graph(**kw,include_genes=axis)
    else:path=build_prior(axis,**kw)
    print(path)


if __name__=='__main__':main()
