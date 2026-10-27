"""Exact marginal-preserving permutations with a deliberately biased STE gradient.

No dependency on differentiable-sort extensions. The smooth CDF is a training
proxy only: emitted values are always the original column multisets.
"""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def ranks(x):
    order=torch.argsort(x,dim=0,stable=True)
    rank=torch.empty_like(order)
    return rank.scatter_(0,order,torch.arange(len(x),device=x.device)[:,None].expand_as(order))


def hard_reorder(sorted_values,score):
    if sorted_values.shape!=score.shape or score.ndim!=2 or len(score)<2:
        raise ValueError('Matching nonempty cell/gene matrices with at least two cells required')
    return sorted_values.gather(0,ranks(score))


def smooth_proxy(values,score,temperature,anchors,anchor_values=None,rank_values=None):
    """Endpoint-covered quantile proxy; only its score derivative is used.

    A subsampled sigmoid CDF cannot reach its top quantiles: with A anchors,
    q_max <= 1-1/(2*A). At N=512,A=32 this hides the last seven positive
    order statistics completely. Detached endpoint calibration removes that
    finite-anchor blind spot without changing the exact hard permutation.
    """
    n=len(score)
    ix=torch.linspace(0,n-1,min(anchors,n),device=score.device).round().long()
    anchor=score.detach().sort(0).values[ix] if anchor_values is None else anchor_values
    q=torch.sigmoid((score[:,None,:]-anchor[None,:,:])/temperature).mean(1)
    qmin=q.min(0,keepdim=True).values.detach()
    qmax=q.max(0,keepdim=True).values.detach()
    span=qmax-qmin
    endpoint_margin=1e-3
    regular=endpoint_margin+(n-1-2*endpoint_margin)*(q-qmin)/span.clamp_min(1e-4)
    # For exact/near ties use the same stable order as the hard branch and
    # a bounded straight-through rank slope. No divide by a tiny CDF span.
    rank=ranks(score) if rank_values is None else rank_values
    fallback=endpoint_margin+(n-1-2*endpoint_margin)*rank.to(score.dtype)/(n-1)
    fallback=fallback+(score-score.detach())*(n-1)
    position=torch.where(span>=1e-4,regular,fallback).clamp(0,n-1)
    lo=position.floor().long();hi=(lo+1).clamp(max=n-1)
    fraction=position-lo
    return values.gather(0,lo)*(1-fraction)+values.gather(0,hi)*fraction


class ReorderHead(nn.Module):
    def __init__(self,cell_dim=934,gene_dim=560,width=128,rank=32,max_shift=.2,
                 temperature=.025,anchors=64,gene_chunk=128,joint_dim=0):
        super().__init__()
        if min(cell_dim,gene_dim,width,rank,anchors,gene_chunk)<1 or not 0<max_shift<=.5 or temperature<=0:
            raise ValueError('Invalid head configuration')
        self.rank=rank;self.max_shift=max_shift;self.temperature=temperature
        self.anchors=anchors;self.gene_chunk=gene_chunk
        self.joint_dim=int(joint_dim)
        if self.joint_dim<0:raise ValueError('Invalid joint feature dimension')
        if self.joint_dim:
            # Separately trainable phase-C projection of frozen source joint
            # evidence; its parameters remain under the reorder namespace.
            self.joint=nn.Sequential(nn.LayerNorm(self.joint_dim),nn.Linear(self.joint_dim,width),
                                     nn.GELU(),nn.Linear(width,rank))
            gene_dim+=rank
        self.cell=nn.Sequential(nn.LayerNorm(cell_dim),nn.Linear(cell_dim,width),nn.GELU(),nn.Linear(width,rank))
        self.gene=nn.Sequential(nn.LayerNorm(gene_dim),nn.Linear(gene_dim,width),nn.GELU(),nn.Linear(width,rank))
        # Small random initial scores give both branches a first-step gradient.
        nn.init.normal_(self.gene[-1].weight, std=.02)
        nn.init.normal_(self.gene[-1].bias, std=.002)

    def forward(self,y,c,g,strength=1.,surrogate=None,joint=None,confidence=None):
        if not 0<=strength<=1:raise ValueError('Strength outside [0,1]')
        if y.ndim!=2 or len(y)<2 or c.shape[0]!=len(y) or g.shape[0]!=y.shape[1]:
            raise ValueError('Cell/gene axes mismatch')
        if not all(bool(torch.isfinite(x).all()) for x in (y,c,g)) or bool((y<0).any()):
            raise ValueError('Nonfinite or negative input')
        if surrogate is None:surrogate=self.training and torch.is_grad_enabled()
        input_ranks=ranks(y)
        base=(input_ranks.to(y.dtype)+.5)/len(y)
        values=y.sort(0).values  # hard permutations retain the generator gradient
        if self.joint_dim:
            if joint is None or joint.ndim!=2 or joint.shape[1]!=self.joint_dim or joint.shape[0] not in (1,len(g)):
                raise ValueError('Missing or misaligned source joint evidence')
            embedded=self.joint(joint)
            if confidence is not None:
                if confidence.shape!=(len(joint),):raise ValueError('Invalid joint confidence shape')
                embedded=embedded*confidence[:,None]
            g=torch.cat((g,embedded.expand(len(g),-1)),-1)
        latent=self.cell(c);loading=self.gene(g)
        raw=(latent@loading.T)/math.sqrt(self.rank)
        # Remove score shifts that cannot change the hard column ordering.
        raw=raw-raw.mean(0,keepdim=True)
        shift=self.max_shift*strength*torch.tanh(raw)
        score=base+shift
        outputs=[];proxy_error=[];score_ranks=[]
        rows=torch.arange(len(y),device=y.device)[:,None]
        anchor_ix=torch.linspace(0,len(y)-1,min(self.anchors,len(y)),device=y.device).round().long()
        for start in range(0,y.shape[1],self.gene_chunk):
            v=values[:,start:start+self.gene_chunk];s=score[:,start:start+self.gene_chunk]
            order=torch.argsort(s,dim=0,stable=True)
            rank=torch.empty_like(order).scatter_(0,order,rows.expand_as(order))
            score_ranks.append(rank)
            hard=v.gather(0,rank)
            if surrogate:
                # Exact gather supplies the generator gradient once.
                # Detached values in the proxy supply only a score gradient.
                # Reuse the stable hard ordering for detached proxy anchors.
                # Equal scores have identical anchor values, so the proxy's
                # original tie behavior and derivative are unchanged.
                anchor_values=s.detach().gather(0,order)[anchor_ix]
                soft=checkpoint(lambda a,b,d,r:smooth_proxy(a,b,self.temperature,self.anchors,d,r),
                                v.detach(),s,anchor_values,rank,use_reentrant=False)
                outputs.append(hard+(soft-soft.detach()))
                proxy_error.append((soft.detach()-hard.detach()).abs().mean())
            else:outputs.append(hard)
        out=torch.cat(outputs,1)
        displacement=(torch.cat(score_ranks,1)-input_ranks).abs().to(y.dtype)/len(y)
        return out,{'rank_penalty':(shift/self.max_shift).square().mean(),
                    'mean_normalized_rank_displacement':displacement.mean().detach(),
                    'changed_value_fraction':(out.detach()!=y).float().mean(),
                    'soft_hard_MAE':torch.stack(proxy_error).mean() if proxy_error else y.new_zeros(())}


def invariants(original,new):
    if original.shape!=new.shape or not torch.equal(original.sort(0).values,new.sort(0).values):
        raise RuntimeError('Column multiset changed')
    if not torch.equal((original==0).sum(0),(new==0).sum(0)):raise RuntimeError('Detection changed')
    mean_error=float((original.double().mean(0)-new.double().mean(0)).abs().max())
    if mean_error>1e-10:raise RuntimeError('Mean changed')
    return {'exact_column_multisets':True,'exact_zero_counts':True,'mean_max_abs':mean_error}


def covariance(y,scale,components):
    z=(y/scale)@components.T;z=z-z.mean(0,keepdim=True)
    return z.T@z/len(z)


def cov_errors(a,b):
    d=(a-b).abs();mask=~torch.eye(len(d),device=d.device,dtype=torch.bool)
    return d.diagonal().mean(),d[mask].mean()
