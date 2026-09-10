from __future__ import annotations
import numpy as np

def paired_bootstrap(a,b,metric_fn,n_bootstrap=5000,seed=2026):
    if len(a)!=len(b): raise ValueError('paired arrays must have same length')
    n=len(a); rng=np.random.default_rng(seed); diffs=[]
    base=metric_fn(a)-metric_fn(b)
    for _ in range(n_bootstrap):
        idx=rng.integers(0,n,n)
        aa=[a[i] for i in idx]; bb=[b[i] for i in idx]
        diffs.append(metric_fn(aa)-metric_fn(bb))
    lo,hi=np.percentile(diffs,[2.5,97.5])
    return {'effect':float(base),'ci95':[float(lo),float(hi)],'n_bootstrap':n_bootstrap}

def hierarchical_seed_study_bootstrap(seed_arrays_a,seed_arrays_b,metric_fn,n_bootstrap=5000,seed=2026):
    rng=np.random.default_rng(seed); S=len(seed_arrays_a); diffs=[]
    for _ in range(n_bootstrap):
        sidx=rng.integers(0,S,S); xa=[]; xb=[]
        for s in sidx:
            n=len(seed_arrays_a[s]); idx=rng.integers(0,n,n)
            xa.extend([seed_arrays_a[s][i] for i in idx]); xb.extend([seed_arrays_b[s][i] for i in idx])
        diffs.append(metric_fn(xa)-metric_fn(xb))
    return {'effect':float(np.mean(diffs)),'ci95':[float(x) for x in np.percentile(diffs,[2.5,97.5])],'n_bootstrap':n_bootstrap}
