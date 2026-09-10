from __future__ import annotations
import numpy as np
from .structured import ufr,supported_recall,grounding_f1,critical_omission

def aggregate_structured(pred_rows,ref_rows,critical_findings=()):
    assert len(pred_rows)==len(ref_rows)
    vals={'ufr':[],'supported_recall':[],'grounding_f1':[],'cor':[]}
    for p,r in zip(pred_rows,ref_rows):
        pf=p.get('findings',p); rf=r.get('findings',r)
        vals['ufr'].append(ufr(pf,rf)); vals['supported_recall'].append(supported_recall(pf,rf)); vals['grounding_f1'].append(grounding_f1(pf,rf))
        c=critical_omission(pf,rf,critical_findings)
        if c is not None: vals['cor'].append(c)
    return {k:(float(np.mean(v)) if v else None) for k,v in vals.items()}

def binary_f1(y_true,y_pred):
    y_true=np.asarray(y_true).astype(int); y_pred=np.asarray(y_pred).astype(int)
    tp=((y_true==1)&(y_pred==1)).sum(); fp=((y_true==0)&(y_pred==1)).sum(); fn=((y_true==1)&(y_pred==0)).sum()
    p=tp/max(tp+fp,1); r=tp/max(tp+fn,1)
    return 2*p*r/max(p+r,1e-12)

def clinical_macro_f1(pred_label_matrix,ref_label_matrix):
    P=np.asarray(pred_label_matrix); R=np.asarray(ref_label_matrix)
    return float(np.mean([binary_f1(R[:,j],P[:,j]) for j in range(R.shape[1])]))
