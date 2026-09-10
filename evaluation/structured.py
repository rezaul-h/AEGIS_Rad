from __future__ import annotations
from dataclasses import dataclass, asdict
import re

@dataclass
class StructuredFinding:
    name: str
    state: str = 'present'
    region: str | None = None
    confidence: float | None = None

def normalize_finding(x):
    if isinstance(x,StructuredFinding): return x
    return StructuredFinding(**x)

def positives(items): return [normalize_finding(x) for x in items if normalize_finding(x).state=='present']

def region_compatible(a,b):
    if a is None or b is None: return True
    return a==b

def ufr(pred,ref):
    pp=positives(pred); rr=positives(ref)
    if not pp: return 0.0
    unsupported=0
    for p in pp:
        ok=any(p.name==r.name and region_compatible(p.region,r.region) for r in rr)
        unsupported += int(not ok)
    return unsupported/len(pp)

def supported_recall(pred,ref):
    pp=positives(pred); rr=positives(ref)
    if not rr: return 1.0
    hit=sum(any(p.name==r.name and region_compatible(p.region,r.region) for p in pp) for r in rr)
    return hit/len(rr)

def grounding_f1(pred,ref):
    P={(x.name,x.region) for x in positives(pred) if x.region is not None}
    R={(x.name,x.region) for x in positives(ref) if x.region is not None}
    if not P and not R: return 1.0
    if not P or not R: return 0.0
    tp=len(P&R); pr=tp/len(P); rc=tp/len(R)
    return 2*pr*rc/(pr+rc) if pr+rc else 0.0

def critical_omission(pred,ref,critical):
    critical=set(critical); rr=[x for x in positives(ref) if x.name in critical]
    if not rr: return None
    pp=positives(pred)
    return float(any(not any(p.name==r.name and region_compatible(p.region,r.region) for p in pp) for r in rr))

def ece(conf,correct,n_bins=10):
    if not conf:return 0.0
    import numpy as np
    conf=np.asarray(conf,float); correct=np.asarray(correct,float)
    edges=np.linspace(0,1,n_bins+1); val=0.0
    for i in range(n_bins):
        m=(conf>=edges[i]) & (conf < edges[i+1] if i<n_bins-1 else conf<=edges[i+1])
        if m.any(): val += m.mean()*abs(correct[m].mean()-conf[m].mean())
    return float(val)
