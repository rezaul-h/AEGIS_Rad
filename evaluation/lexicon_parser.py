"""Lightweight smoke-test parser. Do not use as a substitute for the frozen CheXbert/RadGraph evaluation stack."""
from __future__ import annotations
import re
from .structured import StructuredFinding

NEG=re.compile(r'\b(no|without|absent|negative for)\b',re.I)
UNC=re.compile(r'\b(possible|may|might|could|question of|suggestive)\b',re.I)

def parse(text,ontology):
    out=[]; low=text.lower()
    for f in ontology.findings:
        terms=[f.name,*f.aliases]
        for term in terms:
            m=re.search(r'\b'+re.escape(term.lower())+r'\b',low)
            if not m: continue
            ctx=low[max(0,m.start()-40):m.end()+20]
            state='absent' if NEG.search(ctx) else ('uncertain' if UNC.search(ctx) else 'present')
            out.append(StructuredFinding(f.name,state,None,None)); break
    return out
