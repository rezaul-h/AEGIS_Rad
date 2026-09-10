"""Build a study-level MIMIC-CXR-JPG manifest from local PhysioNet files.
No patient data are downloaded or redistributed.
"""
import argparse,pandas as pd
from pathlib import Path
EXPECTED={'train':222758,'validate':1808,'test':3269,'val':1808}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--metadata',required=True); ap.add_argument('--split',required=True); ap.add_argument('--reports',required=True); ap.add_argument('--image-root',required=True); ap.add_argument('--output',required=True); ap.add_argument('--enforce-manuscript-counts',action='store_true'); a=ap.parse_args()
    meta=pd.read_csv(a.metadata); spl=pd.read_csv(a.split); rep=pd.read_csv(a.reports)
    df=meta.merge(spl,on=['dicom_id','study_id','subject_id'],how='inner').merge(rep,on='study_id',how='inner'); rows=[]
    for sid,g in df.groupby('study_id'):
        frontal=g[g.ViewPosition.isin(['PA','AP'])]; lateral=g[g.ViewPosition.isin(['LATERAL','LL'])]
        if frontal.empty:continue
        fr=frontal.iloc[0]; la=None if lateral.empty else lateral.iloc[0]
        def img_path(r):return str(Path(a.image_root)/str(r['subject_id'])/str(r['study_id'])/(str(r['dicom_id'])+'.jpg'))
        split='val' if str(fr['split'])=='validate' else str(fr['split'])
        rows.append({'study_id':sid,'split':split,'frontal_path':img_path(fr),'lateral_path':'' if la is None else img_path(la),'report':fr['report'],'findings_json':'{}','region_targets_json':'{}','support_targets_json':'{}'})
    out=pd.DataFrame(rows)
    if a.enforce_manuscript_counts:
        got=out.split.value_counts().to_dict(); exp={'train':222758,'val':1808,'test':3269}
        if any(got.get(k,0)!=v for k,v in exp.items()):raise SystemExit(f'Split-count mismatch: expected {exp}, got {got}')
    out.to_csv(a.output,index=False); print('Wrote',len(out),'studies. Populate structured JSON fields only with the frozen parser/mapping outputs.')
if __name__=='__main__':main()
