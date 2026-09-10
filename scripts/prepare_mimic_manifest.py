"""Create a study-level MIMIC-CXR manifest from locally downloaded PhysioNet metadata.
No dataset is downloaded by this script.
"""
import argparse,json,pandas as pd
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--metadata',required=True,help='MIMIC image metadata CSV'); ap.add_argument('--split',required=True,help='official split CSV'); ap.add_argument('--reports',required=True,help='CSV with study_id,report'); ap.add_argument('--image-root',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    meta=pd.read_csv(a.metadata); spl=pd.read_csv(a.split); rep=pd.read_csv(a.reports)
    df=meta.merge(spl,on=['dicom_id','study_id','subject_id'],how='inner').merge(rep,on='study_id',how='inner')
    rows=[]
    for sid,g in df.groupby('study_id'):
        frontal=g[g.ViewPosition.isin(['PA','AP'])]; lateral=g[g.ViewPosition.isin(['LATERAL','LL'])]
        if frontal.empty:continue
        fr=frontal.iloc[0]; la=None if lateral.empty else lateral.iloc[0]
        def img_path(r): return str(Path(a.image_root)/str(r['subject_id'])/str(r['study_id'])/(str(r['dicom_id'])+'.jpg'))
        rows.append({'study_id':sid,'split':fr['split'],'frontal_path':img_path(fr),'lateral_path':'' if la is None else img_path(la),'report':fr['report'],'findings_json':'{}','region_targets_json':'{}','support_targets_json':'{}'})
    pd.DataFrame(rows).to_csv(a.output,index=False)
    print('Wrote',len(rows),'studies. Fill structured JSON columns with the frozen parser outputs used in your protocol.')
if __name__=='__main__':main()
