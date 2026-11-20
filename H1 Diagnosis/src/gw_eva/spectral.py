"""Bounded-memory merge/truncate SVD, including an uncentered GW target.

The SVD of each merge is exact, but repeated rank truncation is an
approximation to the SVD of all historical rows. No random approximation.
"""
from __future__ import annotations
import torch


class StreamingSVD:
    def __init__(self,d,max_rank=16,centered=False,chunk_rows=64,tolerance=.99):
        self.d=d;self.max_rank=min(max_rank,d);self.centered=centered
        self.chunk_rows=chunk_rows;self.tolerance=tolerance;self.n=0
        self.mean=torch.zeros(d,dtype=torch.float64)
        self.components=torch.empty(0,d);self.singular=torch.empty(0)
        self.energy=0.;self.discarded=0.;self.history=[]

    @torch.no_grad()
    def update(self,rows):
        rows=rows.detach().to(device='cpu',dtype=torch.float32)
        if rows.ndim!=2 or rows.shape[1]!=self.d:raise ValueError('Invalid row shape')
        if not torch.isfinite(rows).all():raise ValueError('Nonfinite SVD input')
        for part in rows.split(self.chunk_rows):
            if not len(part):continue
            count=len(part);mean=part.double().mean(0)
            if self.centered:
                incoming=part-mean.float();new_energy=float(incoming.square().sum())
                if self.n:
                    correction=(self.mean-mean)*(self.n*count/(self.n+count))**.5
                    incoming=torch.cat([incoming,correction.float()[None]],0)
                    new_energy+=float(correction.square().sum())
            else:
                incoming=part;new_energy=float(part.square().sum())
            self.mean=(self.mean*self.n+mean*count)/(self.n+count)
            self.n+=count;self.energy+=new_energy
            merged=torch.cat([self.singular[:,None]*self.components,incoming],0)
            old=self.components
            _,s,v=torch.linalg.svd(merged,full_matrices=False)
            kept=min(self.max_rank,len(s));self.discarded+=float(s[kept:].square().sum())
            self.singular=s[:kept].clone();self.components=v[:kept].clone()
            # Deterministic sign convention, irrelevant to projector geometry.
            indices=self.components.abs().argmax(1)
            signs=self.components[torch.arange(kept),indices].sign();signs[signs==0]=1
            self.components.mul_(signs[:,None])
            overlap=None;minimum_cosine=None
            if old.shape==self.components.shape:
                cosines=torch.linalg.svdvals(old@self.components.T).clamp(0,1)
                overlap=float(cosines.square().mean());minimum_cosine=float(cosines.min())
            self.history.append({'samples':self.n,'subspace_overlap_previous':overlap,
                 'minimum_principal_cosine_previous':minimum_cosine,
                 'converged':minimum_cosine is not None and minimum_cosine>=self.tolerance})
        return self

    def state(self):
        # Keep 16 candidate axes even if the measured target is rank deficient.
        # Nullspace completion is deterministic, label-free, and has zero score.
        v=self.components.clone();s=self.singular.clone()
        if len(v)<self.max_rank:
            for j in range(self.d):
                axis=torch.zeros(self.d);axis[j]=1
                for _ in range(2):
                    if len(v):axis-=v.T@(v@axis)
                norm=axis.norm()
                if norm>1e-5:
                    v=torch.cat([v,(axis/norm)[None]]);s=torch.cat([s,torch.zeros(1)])
                if len(v)==self.max_rank:break
        scores=s.double().square()/self.energy if self.energy>0 else torch.zeros(len(s),dtype=torch.float64)
        return {'components':v,'singular_values':s,'scores':scores,'samples':self.n,
                'energy':self.energy,'discarded_merge_energy':self.discarded,
                'retained_energy_ratio':float(scores.sum()),'centered':self.centered,
                'history':self.history,'converged':bool(self.history and self.history[-1]['converged']),
                'algorithm':'exact skinny merge SVD with repeated max16 truncation; approximate global SVD',
                'nullspace_completion':len(self.components)<self.max_rank}


def compare(v,w):
    """Both inputs have orthonormal rows and the SAME rank."""
    if v.shape!=w.shape:raise ValueError('Matched dimensions/ranks required')
    cosines=torch.linalg.svdvals(v.double()@w.double().T).clamp(0,1)
    rank=len(v);overlap=float(cosines.square().sum()/rank)
    return {'overlap':overlap,'principal_angles_degrees':torch.rad2deg(torch.acos(cosines)).tolist(),
            'projection_frobenius':float(torch.sqrt((2*rank-2*cosines.square().sum()).clamp_min(0))),
            'orthogonality_error_v':float((v@v.T-torch.eye(rank)).abs().max()),
            'orthogonality_error_w':float((w@w.T-torch.eye(rank)).abs().max())}
