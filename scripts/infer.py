import argparse,json,torch
from functools import partial
from torch.utils.data import DataLoader
from common import setup,load_checkpoint
from aegis_rad.data.dataset import ReportDataset,collate_reports
from aegis_rad.generation import generate_controlled_report

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--tokenizer',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--split',default='test'); ap.add_argument('--output',required=True); a=ap.parse_args()
    cfg,tok,model,ontology=setup(a.config,a.tokenizer); cfg.data.manifest=a.manifest; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device); load_checkpoint(model,a.checkpoint,device); model.eval(); ds=ReportDataset(a.manifest,a.split,tok,ontology,cfg.data,False)
    dl=DataLoader(ds,batch_size=1,shuffle=False,collate_fn=partial(collate_reports,pad_id=tok.pad_id)); rows=[]
    for b in dl:
        b={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in b.items()}; rows.append({'study_id':b['study_id'][0],**generate_controlled_report(model,b,tok,ontology,cfg)})
    with open(a.output,'w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
if __name__=='__main__':main()
