"""Create an IU X-Ray/Open-I manifest from a local metadata CSV.
Expected input columns: study_id, image_path, view, report. Splitting by UID/hash is dataset-specific;
this script provides deterministic grouped splitting but should be replaced by the exact frozen split table for reproduction.
"""
import argparse,pandas as pd,numpy as np

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--metadata',required=True); ap.add_argument('--output',required=True); ap.add_argument('--seed',type=int,default=2026); a=ap.parse_args()
    df=pd.read_csv(a.metadata); rng=np.random.default_rng(a.seed); studies=np.array(sorted(df.study_id.astype(str).unique())); rng.shuffle(studies)
    n=len(studies); ntr=round(n*.70); nv=round(n*.10); mapping={s:('train' if i<ntr else 'val' if i<ntr+nv else 'test') for i,s in enumerate(studies)}
    rows=[]
    for sid,g in df.groupby(df.study_id.astype(str)):
        frontal=g[g.view.str.upper().isin(['PA','AP','FRONTAL'])]; lateral=g[g.view.str.upper().isin(['LATERAL','LL'])]
        if frontal.empty:continue
        rows.append({'study_id':sid,'split':mapping[sid],'frontal_path':frontal.iloc[0].image_path,'lateral_path':'' if lateral.empty else lateral.iloc[0].image_path,'report':frontal.iloc[0].report,'findings_json':'{}','region_targets_json':'{}','support_targets_json':'{}'})
    pd.DataFrame(rows).to_csv(a.output,index=False)
    print('Wrote',len(rows),'rows. For manuscript reproduction, use the exact frozen UID/perceptual-hash split rather than this generic deterministic split.')
if __name__=='__main__':main()
