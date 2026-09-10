import argparse,json,pandas as pd
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True); a=ap.parse_args()
    df=pd.read_csv(a.manifest)
    req=['study_id','split','frontal_path','report','findings_json','region_targets_json','support_targets_json']
    missing=[c for c in req if c not in df.columns]
    if missing: raise SystemExit(f'Missing required columns: {missing}')
    bad=[]
    for i,r in df.iterrows():
        if not Path(str(r.frontal_path)).exists(): bad.append((i,'frontal_path',r.frontal_path))
        for c in ['findings_json','region_targets_json','support_targets_json']:
            try: json.loads(r[c]) if pd.notna(r[c]) and str(r[c]).strip() else {}
            except Exception as e: bad.append((i,c,str(e)))
    print('rows',len(df),'splits',df.split.value_counts().to_dict(),'issues',len(bad))
    for x in bad[:50]: print(x)
    if bad: raise SystemExit(2)
if __name__=='__main__':main()
