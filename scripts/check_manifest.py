import argparse,json,pandas as pd
from pathlib import Path
ALLOWED_STATES={'present','absent','uncertain','unmentioned'}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True); ap.add_argument('--skip-path-check',action='store_true'); a=ap.parse_args()
    df=pd.read_csv(a.manifest); req=['study_id','split','frontal_path','report','findings_json','region_targets_json','support_targets_json']
    missing=[c for c in req if c not in df.columns]
    if missing:raise SystemExit(f'Missing required columns: {missing}')
    bad=[]
    for i,r in df.iterrows():
        if str(r['split']) not in {'train','val','test'}:bad.append((i,'split',r['split']))
        if not a.skip_path_check and not Path(str(r.frontal_path)).exists():bad.append((i,'frontal_path',r.frontal_path))
        parsed={}
        for c in ['findings_json','region_targets_json','support_targets_json']:
            try:parsed[c]=json.loads(r[c]) if pd.notna(r[c]) and str(r[c]).strip() else {}
            except Exception as e:bad.append((i,c,str(e))); parsed[c]={}
        if not isinstance(parsed['findings_json'],dict):bad.append((i,'findings_json','must be JSON object'))
        else:
            for finding,state in parsed['findings_json'].items():
                if str(state).lower() not in ALLOWED_STATES:bad.append((i,'findings_json',f'{finding}: invalid state {state!r}'))
    print('rows',len(df),'splits',df.split.value_counts().to_dict(),'issues',len(bad))
    for x in bad[:50]:print(x)
    if bad:raise SystemExit(2)
if __name__=='__main__':main()
