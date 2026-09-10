from __future__ import annotations
import numpy as np

def sweep_threshold(rows,thresholds=None,tau_con=.55,tau_suppress=.35):
    if thresholds is None: thresholds=np.round(np.linspace(.30,.85,56),3)
    out=[]
    for tau in thresholds:
        supported=ref_pos=gen_pos=unsupported=0
        for row in rows:
            for c in row['candidates']:
                state='absent' if c['contradiction']>tau_con or c['support']<tau_suppress else ('present' if c['support']>=tau else 'uncertain')
                if state=='present':
                    gen_pos+=1; supported+=int(c.get('reference_positive',False)); unsupported+=int(not c.get('reference_positive',False))
                ref_pos+=int(c.get('reference_positive',False))
        out.append({'tau_e':float(tau),'ufr':unsupported/max(gen_pos,1),'supported_recall':supported/max(ref_pos,1),'positive_count':gen_pos/max(len(rows),1)})
    return out
