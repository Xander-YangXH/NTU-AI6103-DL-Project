"""PEFT-style count budget and exact integer prefix-constrained cost budget."""
import math
import functools
import numpy as np


def count_allocate(scores,budget=576):
    # PEFT EVA: descending global ranking of layer-normalized spectral scores.
    candidates=[]
    for name,values in sorted(scores.items()):
        values=np.asarray(values,dtype=float)
        if not np.isfinite(values).all() or (values<0).any() or (np.diff(values)>1e-10).any():raise ValueError('Invalid ordered scores')
        candidates.extend((float(value),name,j) for j,value in enumerate(values))
    if budget>len(candidates):raise ValueError('Insufficient candidates')
    candidates.sort(key=lambda x:(-x[0],x[1],x[2]))
    ranks={name:0 for name in scores}
    for value,name,j in candidates[:budget]:ranks[name]+=1
    return ranks


def cost_allocate(scores,costs,budget=1327104):
    """Multiple-choice knapsack: option r is the prefix sum of r scores.

    Exact global optimum for the supplied finite-precision candidate scores,
    with deterministic tie preference for higher utilized budget.
    """
    names=sorted(scores);unit=functools.reduce(math.gcd,[int(costs[n]) for n in names])
    limit=budget//unit;dp=np.full(limit+1,-np.inf);dp[0]=0
    back=[]
    for name in names:
        values=np.asarray(scores[name],dtype=float)
        if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid scores')
        prefixes=np.r_[0,np.cumsum(values)];step=int(costs[name])//unit
        new=np.full_like(dp,-np.inf);choices=np.full(limit+1,-1,dtype=np.int16)
        for rank,score in enumerate(prefixes):
            shift=rank*step
            if shift>limit:break
            candidate=dp[:limit+1-shift]+score;dest=new[shift:]
            improve=candidate>dest
            dest[improve]=candidate[improve];choices[shift:][improve]=rank
        dp=new;back.append(choices)
    best=float(np.max(dp));used=int(np.flatnonzero(dp==best)[-1]);remaining=used;ranks={}
    for i in range(len(names)-1,-1,-1):
        name=names[i];rank=int(back[i][remaining]);assert rank>=0
        ranks[name]=rank;remaining-=rank*(int(costs[name])//unit)
    assert remaining==0 and sum(costs[n]*ranks[n] for n in names)<=budget
    return ranks,{'algorithm':'exact multiple-choice prefix knapsack dynamic programming',
            'globally_optimal_for_supplied_scores':True,'score':best,'cost_unit':unit,
            'budget':budget,'used_parameters':used*unit,'fraction':used*unit/budget}
