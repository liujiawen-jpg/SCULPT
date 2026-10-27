"""Response-free J0 priors: Tahoe, GenePT and two STRING propagation steps.

The recipe is the original seed-17 five-block random projection. No task
weights, expression responses, fitted output predictor or WaveGC is used.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import csv
import gzip
import hashlib
import json
import pickle
import re
import numpy as np
from scipy import sparse

GENEPT_SHA256 = 'fd297510ddd3040744033fde0b0f2cf15a40ac8b2fd2fb02f10667295e55c862'
TAHOE_SHA256 = '21f2afc7ee5ec88cadddd6743188aea590b89306c8f2538e137bf9dcacfc0260'
STABLE_TAGS = {'Ensembl_HGNC_ensembl_gene_id','Ensembl_gene','Ensembl_HGNC_hgnc_id','Ensembl_HGNC_entrez_id'}
SYMBOL_TAG = 'Ensembl_HGNC_symbol'


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):digest.update(chunk)
    return digest.hexdigest()


def stable(value):
    value=re.sub(r'\.\d+$','',str(value).strip())
    return value if re.fullmatch(r'ENSG\d{11}',value) else None


def _rows(path):
    opening=gzip.open if str(path).endswith('.gz') else open
    with opening(path,'rt',encoding='utf8') as stream:
        reader=csv.DictReader(stream,delimiter='\t')
        reader.fieldnames=[name.removeprefix('#') for name in reader.fieldnames]
        yield from reader


def _write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def hgnc_maps(path):
    symbols=defaultdict(set);hgnc=defaultdict(set);entrez=defaultdict(set);by_symbol=defaultdict(set)
    for row in _rows(path):
        gene=stable(row.get('ensembl_gene_id',''))
        symbol=row.get('symbol','')
        if row.get('status')!='Approved' or gene is None:continue
        symbols[gene].add(symbol);by_symbol[symbol].add(gene)
        if row.get('hgnc_id'):hgnc[row['hgnc_id']].add(gene)
        if row.get('entrez_id'):entrez[row['entrez_id']].add(gene)
    return symbols,hgnc,entrez,by_symbol


def resolve_candidates(stable_candidates, ambiguous, symbols):
    """Reject conflicting identities; no fuzzy matching or arbitrary tie break."""
    stable_candidates,symbols=set(stable_candidates),set(symbols)
    if ambiguous:return 'ambiguous',None
    if len(stable_candidates)>1:return 'conflict',None
    if len(stable_candidates)==1:
        gene=next(iter(stable_candidates))
        if symbols-{gene}:return 'symbol_conflict',None
        return 'accepted_stable',gene
    if len(symbols)==1:return 'accepted_symbol_fallback',next(iter(symbols))
    return ('symbol_ambiguous' if symbols else 'unresolved'),None


def build_string_graph(hgnc,links,info,aliases,out,include_genes=()):
    """Build the pinned human STRING v12 gene graph (combined_score >=700).

    Writes the original graph layout under ``out``. Duplicate protein edges
    collapsing to the same gene pair retain their maximum integer score.
    """
    out=Path(out);(out/'source').mkdir(parents=True,exist_ok=True)
    (out/'graphs/KS').mkdir(parents=True,exist_ok=True)
    symbols,hgnc_ids,entrez_ids,by_symbol=hgnc_maps(hgnc)
    ambiguous_genes={gene for gene,value in symbols.items() if len(value)!=1}
    include=set(map(str,include_genes))
    if any(stable(g)!=g for g in include):raise ValueError('include_genes must be canonical stable ENSG identifiers')
    nodes=sorted((set(symbols)-ambiguous_genes)|include)
    lookup={gene:i for i,gene in enumerate(nodes)}
    names={row['string_protein_id']:row['preferred_name'] for row in _rows(info)}
    evidence=defaultdict(lambda:dict(stable=set(),symbols=set(),ambiguous=False))
    for row in _rows(aliases):
        tags=set(row['source'].split())&(STABLE_TAGS|{SYMBOL_TAG})
        if not tags:continue
        record=evidence[row['string_protein_id']];alias=row['alias']
        for tag in tags:
            if tag==SYMBOL_TAG:
                record['symbols'].update(by_symbol.get(alias,set()));continue
            if tag in {'Ensembl_HGNC_ensembl_gene_id','Ensembl_gene'}:
                gene=stable(alias)
                candidates={gene} if gene in symbols and gene not in ambiguous_genes else set()
                record['ambiguous']|=gene in ambiguous_genes
            else:
                candidates=(hgnc_ids if tag=='Ensembl_HGNC_hgnc_id' else entrez_ids).get(alias,set())
                record['ambiguous']|=len(candidates)>1
            if len(candidates)==1:record['stable'].update(candidates)
    mapping={};counts=Counter();audit=[]
    for protein in sorted(names):
        record=evidence[protein]
        candidates=record['symbols']|by_symbol.get(names[protein],set())
        status,gene=resolve_candidates(record['stable'],record['ambiguous'],candidates)
        if gene and gene not in lookup:status,gene='unresolved_node',None
        mapping[protein]=gene;counts[status]+=1
        audit.append((protein,gene or '',status))
    edge_scores={};input_rows=0
    opening=gzip.open if str(links).endswith('.gz') else open
    with opening(links,'rt',encoding='utf8') as stream:
        if stream.readline().split()!=['protein1','protein2','combined_score']:
            raise ValueError('Expected STRING links protein1 protein2 combined_score')
        for line in stream:
            a,b,score=line.split();score=int(score);input_rows+=1
            if not 0<=score<=1000:raise ValueError('STRING score outside [0,1000]')
            if score<700:continue
            ga,gb=mapping.get(a),mapping.get(b)
            if ga is None or gb is None or ga==gb:continue
            pair=tuple(sorted((lookup[ga],lookup[gb])))
            edge_scores[pair]=max(score,edge_scores.get(pair,0))
    if not edge_scores:raise ValueError('STRING graph has no accepted gene edges')
    endpoints=np.asarray(sorted(edge_scores),dtype='i4')
    scores=np.asarray([edge_scores[tuple(edge)] for edge in endpoints],dtype='u2')
    i,j=endpoints.T;w=scores.astype('f8')/1000
    adjacency=sparse.csr_matrix((np.r_[w,w],(np.r_[i,j],np.r_[j,i])),shape=(len(nodes),len(nodes)))
    degree=np.asarray(adjacency.sum(1)).ravel()
    transition=sparse.diags(np.divide(1.,degree,out=np.zeros_like(degree),where=degree>0))@adjacency
    sparse.save_npz(out/'graphs/KS/A.npz',adjacency)
    sparse.save_npz(out/'graphs/KS/P.npz',transition.tocsr())
    np.savez(out/'graphs/KS/edges.npz',edges=endpoints,scores=scores)
    with (out/'source/node_axis.tsv').open('w') as stream:
        writer=csv.writer(stream,delimiter='\t');writer.writerow(['node_index','gene_id'])
        writer.writerows(enumerate(nodes))
    with (out/'source/protein_gene_mapping.tsv').open('w') as stream:
        writer=csv.writer(stream,delimiter='\t');writer.writerow(['protein','gene_id','status']);writer.writerows(audit)
    _write_json(out/'STRING_PROVENANCE.json',dict(schema='cosit_string_v12_700_v1',
        inputs={name:file_sha(path) for name,path in dict(hgnc=hgnc,links=links,info=info,aliases=aliases).items()},
        nodes=len(nodes),edges=len(endpoints),input_protein_links=input_rows,score_threshold=700,
        mapping_counts=dict(counts),directed=False,self_loops=False,duplicate_rule='maximum integer score',
        transition='row normalized A; isolates zero; no self loops or teleportation',expression_labels_read=False))
    return out


def l2(x):
    """Original float32 block, float64 norms and zero-row availability rule."""
    x=np.asarray(x,np.float32);norm=np.linalg.norm(x.astype(np.float64),axis=1)
    y=np.zeros_like(x);valid=norm>1e-12;y[valid]=x[valid]/norm[valid,None]
    return y,valid


def load_genept(path,*,trust_pickle=False):
    path=Path(path)
    if path.suffix=='.npz':
        with np.load(path,allow_pickle=False) as z:
            names=list(z['symbols'].astype(str));values=np.asarray(z['embedding'],dtype='f4')
        if values.shape!=(len(names),1536) or len(set(names))!=len(names):raise ValueError('GenePT NPZ requires unique symbols and [N,1536] embedding')
        result=dict(zip(names,values))
    else:
        if not trust_pickle and file_sha(path)!=GENEPT_SHA256:
            raise ValueError('GenePT pickle is not the verified official v2 file; use NPZ or explicit trust_pickle=True')
        with path.open('rb') as stream:result=pickle.load(stream)
    if not isinstance(result,dict):raise ValueError('Expected a GenePT symbol dictionary')
    return result


def build_prior(genes,*,tahoe_weights,tahoe_vocab,genept,hgnc,graph_directory,out,
                frozen_genept_features=None,trust_genept_pickle=False):
    """Create ``prior.npz`` containing ordered genes and J0 [G,261] embedding.

    ``genes`` should be output genes UNION perturbation targets. The graph must
    cover the complete approved-gene universe, not just benchmark output genes,
    because off-panel public neighbors contribute to STRING propagation.
    """
    from safetensors import safe_open
    genes=list(map(str,genes))
    if not genes or len(set(genes))!=len(genes) or any(stable(g)!=g for g in genes):
        raise ValueError('A unique canonical stable ENSG gene axis is required')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);graph_directory=Path(graph_directory)
    with (graph_directory/'source/node_axis.tsv').open() as stream:
        nodes=[row['gene_id'] for row in csv.DictReader(stream,delimiter='\t')]
    transition=sparse.load_npz(graph_directory/'graphs/KS/P.npz').tocsr()
    if transition.shape!=(len(nodes),len(nodes)):raise ValueError('STRING graph axis mismatch')
    symbols,*_=hgnc_maps(hgnc);db=load_genept(genept,trust_pickle=trust_genept_pickle)
    def direct(axis):
        result=np.zeros((len(axis),1536),dtype='f4')
        for i,gene in enumerate(axis):
            candidates=symbols.get(gene,set())
            if len(candidates)!=1:continue
            value=db.get(next(iter(candidates)))
            if value is None:continue
            value=np.asarray(value,dtype='f4')
            if value.shape==(1536,) and np.isfinite(value).all() and np.linalg.norm(value.astype('f8'))>1e-12:result[i]=value
        return result
    f0,_=l2(direct(nodes))
    # Optional external frozen generic mapping preserves historical target rows
    # when their mapping version differs; it never contains fitted task weights.
    if frozen_genept_features is not None:
        with np.load(frozen_genept_features,allow_pickle=False) as z:
            axis=list(z['target_ids'].astype(str));vectors=np.asarray(z['l2'],dtype='f4');missing=np.asarray(z['missing_mask'],bool)
        if vectors.shape!=(len(axis),1536) or missing.shape!=(len(axis),) or not np.isfinite(vectors).all():raise ValueError('Invalid frozen generic GenePT rows')
        ni={g:i for i,g in enumerate(nodes)}
        for gene,value,absent in zip(axis,vectors,missing):
            if gene in ni:f0[ni[gene]]=0 if absent else value
    projection=np.zeros((len(genes),256),dtype='f4');masks=np.zeros((len(genes),5),dtype='u1')
    rng=np.random.default_rng(17)
    matrices=[rng.standard_normal((dim,256),dtype=np.float32)/np.sqrt(dim*5) for dim in (2048,1536,1536,1536,1536)]
    def add(values,indices,block):
        for start in range(0,len(indices),256):
            ids=np.asarray(indices[start:start+256],dtype='i8')
            unit,valid=l2(values[start:start+256]);projection[ids]+=unit@matrices[block];masks[ids,block]=valid
    vocab=json.loads(Path(tahoe_vocab).read_text())
    with safe_open(str(tahoe_weights),framework='pt',device='cpu') as stream:
        key='model.gene_encoder.embedding.weight'
        table=stream.get_slice(key)
        if len(table.get_shape())!=2 or table.get_shape()[1]!=2048:raise ValueError('J0 requires the Tahoe-x1 1B 2048-dimensional gene embedding')
        selected=[(i,int(vocab[g])) for i,g in enumerate(genes) if g in vocab]
        for start in range(0,len(selected),256):
            part=selected[start:start+256]
            values=np.stack([table[row].float().numpy() for _,row in part])
            add(values,[i for i,_ in part],0)
    add(direct(genes),list(range(len(genes))),1)
    del db
    node_lookup={g:i for i,g in enumerate(nodes)}
    selected=[(node_lookup[g],i) for i,g in enumerate(genes) if g in node_lookup]
    node_ids=np.asarray([i for i,_ in selected],dtype='i8');out_ids=[i for _,i in selected]
    add(f0[node_ids],out_ids,2)
    # The archived H1 was stored as float32, and the original prior builder
    # computed H2 by multiplying P into that stored H1 (not unrounded H1).
    h1=np.lib.format.open_memmap(out/'H1_all.npy',mode='w+',dtype='f4',shape=f0.shape)
    h2=np.lib.format.open_memmap(out/'H2_all.npy',mode='w+',dtype='f4',shape=f0.shape)
    for start in range(0,1536,192):
        stop=min(start+192,1536)
        h1[:,start:stop]=(transition@np.asarray(f0[:,start:stop],dtype='f8')).astype('f4')
        h2[:,start:stop]=(transition@np.asarray(h1[:,start:stop],dtype='f8')).astype('f4')
    h1.flush();h2.flush();add(h1[node_ids],out_ids,3);add(h2[node_ids],out_ids,4)
    embedding=np.column_stack((projection,masks.astype('f4'))).astype('f4')
    if not np.isfinite(embedding).all():raise ValueError('Nonfinite generic gene prior')
    path=out/'prior.npz';np.savez_compressed(path,genes=np.asarray(genes),embedding=embedding)
    _write_json(out/'PRIOR_PROVENANCE.json',dict(schema='cosit_j0_five_block_prior_v1',
        seed=17,dimension=261,projection_dimension=256,block_dimensions=[2048,1536,1536,1536,1536],
        blocks=['Tahoe1B','GenePT','STRING_F0','STRING_PF0','STRING_P2F0'],
        projection='sum(rowL2(block) @ normal(dim,256)/sqrt(dim*5)); append five availability flags',
        inputs={name:file_sha(value) for name,value in dict(tahoe_weights=tahoe_weights,tahoe_vocab=tahoe_vocab,genept=genept,hgnc=hgnc,
            string_transition=graph_directory/'graphs/KS/P.npz',string_axis=graph_directory/'source/node_axis.tsv').items()},
        tahoe_matches_original_checkpoint=file_sha(tahoe_weights)==TAHOE_SHA256,
        genept_matches_original_file=file_sha(genept)==GENEPT_SHA256,
        gene_count=len(genes),available=masks.sum(0).astype(int).tolist(),all_missing=int((masks.sum(1)==0).sum()),
        frozen_genept_features_sha256=file_sha(frozen_genept_features) if frozen_genept_features else None,
        task_weights_read=0,expression_labels_read=False))
    return path
