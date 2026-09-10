from __future__ import annotations
import numpy as np

def paired_bootstrap(a,b,n_bootstrap=5000,seed=2026,statistic=np.mean):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if a.shape!=b.shape or a.ndim!=1:raise ValueError('a and b must be paired 1-D arrays of equal length')
    rng=np.random.default_rng(seed); n=len(a); obs=float(statistic(a)-statistic(b)); vals=np.empty(n_bootstrap)
    for k in range(n_bootstrap):
        idx=rng.integers(0,n,n); vals[k]=statistic(a[idx])-statistic(b[idx])
    lo,hi=np.percentile(vals,[2.5,97.5]); return {'effect':obs,'ci95':[float(lo),float(hi)],'n_bootstrap':n_bootstrap,'seed':seed}

def hierarchical_seed_study_bootstrap(a,b,n_bootstrap=5000,seed=2026):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if a.shape!=b.shape or a.ndim!=2:raise ValueError('Expected paired arrays shaped [seed, study]')
    rng=np.random.default_rng(seed); S,N=a.shape; vals=np.empty(n_bootstrap); obs=float(np.mean(a-b))
    for k in range(n_bootstrap):
        sidx=rng.integers(0,S,S); diffs=[]
        for s in sidx:
            iidx=rng.integers(0,N,N); diffs.append((a[s,iidx]-b[s,iidx]).mean())
        vals[k]=np.mean(diffs)
    lo,hi=np.percentile(vals,[2.5,97.5]); return {'effect':obs,'ci95':[float(lo),float(hi)],'n_bootstrap':n_bootstrap,'seed':seed}
