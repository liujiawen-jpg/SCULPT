import csv
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from safetensors.torch import save_file
import torch
from cosit.priors import build_string_graph, build_prior, l2, resolve_candidates


def write_tsv(path,fields,rows):
    with path.open('w') as f:
        writer=csv.writer(f,delimiter='\t');writer.writerow(fields);writer.writerows(rows)


def test_identifier_conflicts_are_rejected():
    assert resolve_candidates({'A'},False,{'A'})==('accepted_stable','A')
    assert resolve_candidates({'A'},False,{'B'})==('symbol_conflict',None)
    assert resolve_candidates({'A','B'},False,set())==('conflict',None)
    assert resolve_candidates(set(),True,{'A'})==('ambiguous',None)
    assert resolve_candidates(set(),False,{'A'})==('accepted_symbol_fallback','A')


def test_string_and_five_block_prior_match_explicit_recipe(tmp_path):
    genes=[f'ENSG{i:011d}' for i in range(1,5)]
    hgnc=tmp_path/'hgnc.tsv'
    write_tsv(hgnc,['ensembl_gene_id','symbol','status','hgnc_id','entrez_id'],
              [(g,f'G{i}','Approved',f'HGNC:{i}',str(i)) for i,g in enumerate(genes,1)])
    info=tmp_path/'info.tsv';aliases=tmp_path/'aliases.tsv';links=tmp_path/'links.txt'
    write_tsv(info,['#string_protein_id','preferred_name'],[(f'P{i}',f'G{i}') for i in range(1,5)])
    write_tsv(aliases,['#string_protein_id','alias','source'],[(f'P{i}',g,'Ensembl_gene') for i,g in enumerate(genes,1)])
    links.write_text('protein1 protein2 combined_score\nP1 P2 700\nP2 P1 900\nP2 P3 800\nP3 P4 699\nP1 P1 1000\n')
    graph_dir=build_string_graph(hgnc,links,info,aliases,tmp_path/'string')
    a=sparse.load_npz(graph_dir/'graphs/KS/A.npz')
    assert a.nnz==4
    assert a[0,1]==.9 and a[1,2]==.8 and a[2,3]==0
    p=sparse.load_npz(graph_dir/'graphs/KS/P.npz')
    np.testing.assert_allclose(np.asarray(p.sum(1)).ravel(),[1,1,1,0])
    rng=np.random.default_rng(9)
    vectors=rng.normal(size=(2,1536)).astype('f4')
    genept=tmp_path/'genept.npz';np.savez(genept,symbols=np.array(['G1','G2']),embedding=vectors)
    tahoe=rng.normal(size=(2,2048)).astype('f4');weights=tmp_path/'model.safetensors'
    save_file({'model.gene_encoder.embedding.weight':torch.as_tensor(tahoe)},str(weights))
    vocab=tmp_path/'vocab.json';vocab.write_text(json.dumps({genes[0]:0,genes[2]:1}))
    prior=build_prior(genes,tahoe_weights=weights,tahoe_vocab=vocab,genept=genept,
                      hgnc=hgnc,graph_directory=graph_dir,out=tmp_path/'prior')
    with np.load(prior) as f:actual=f['embedding'];assert list(f['genes'])==genes
    blocks=[]
    t=np.zeros((4,2048),dtype='f4');t[[0,2]]=tahoe;blocks.append(t)
    f=np.zeros((4,1536),dtype='f4');f[:2]=vectors;blocks.append(f)
    f0=l2(f)[0];h1=(p@f0.astype('f8')).astype('f4');h2=(p@h1.astype('f8')).astype('f4')
    blocks.extend([f0,h1,h2]);rng=np.random.default_rng(17)
    expected=np.zeros((4,256),dtype='f4');flags=[]
    for block in blocks:
        matrix=rng.standard_normal((block.shape[1],256),dtype=np.float32)/np.sqrt(block.shape[1]*5)
        unit,valid=l2(block);expected+=unit@matrix;flags.append(valid.astype('f4'))
    expected=np.column_stack((expected,*flags))
    np.testing.assert_allclose(actual,expected,atol=0,rtol=0)
    assert (actual[3]==0).all()
    assert actual[2,257]==0 and actual[2,259]==1  # missing direct GenePT; graph neighbor available
