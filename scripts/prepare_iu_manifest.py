"""Build the IU X-Ray/Open-I manifest using a pre-frozen split table.
The manuscript uses 2,769/395/791 train/val/test reports after UID/perceptual-hash grouping.
"""
import argparse,pandas as pd
EXPECTED={'train':2769,'val':395,'test':791}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--metadata',required=True,help='study_id,image_path,view,report'); ap.add_argument('--split-table',required=True,help='frozen study_id,split table produced by UID/perceptual-hash grouping'); ap.add_argument('--output',required=True); ap.add_argument('--enforce-manuscript-counts',action='store_true'); a=ap.parse_args()
    df=pd.read_csv(a.metadata); split=pd.read_csv(a.split_table,dtype={'study_id':str}); df['study_id']=df.study_id.astype(str); split['study_id']=split.study_id.astype(str)
    rows=[]
    mapping=dict(zip(split.study_id,split.split))
    for sid,g in df.groupby('study_id'):
        if sid not in mapping:continue
        frontal=g[g.view.astype(str).str.upper().isin(['PA','AP','FRONTAL'])]; lateral=g[g.view.astype(str).str.upper().isin(['LATERAL','LL'])]
        if frontal.empty:continue
        rows.append({'study_id':sid,'split':mapping[sid],'frontal_path':frontal.iloc[0].image_path,'lateral_path':'' if lateral.empty else lateral.iloc[0].image_path,'report':frontal.iloc[0].report,'findings_json':'{}','region_targets_json':'{}','support_targets_json':'{}'})
    out=pd.DataFrame(rows)
    if a.enforce_manuscript_counts:
        got=out.split.value_counts().to_dict()
        if any(got.get(k,0)!=v for k,v in EXPECTED.items()):raise SystemExit(f'Split-count mismatch: expected {EXPECTED}, got {got}')
    out.to_csv(a.output,index=False); print('Wrote',len(out),'rows using the supplied frozen split table.')
if __name__=='__main__':main()
