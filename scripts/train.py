import argparse,torch
from functools import partial
from torch.utils.data import DataLoader
from common import setup
from aegis_rad.data.dataset import ReportDataset,collate_reports
from aegis_rad.training import Trainer
from aegis_rad.utils.seed import seed_everything

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--tokenizer',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--seed',type=int,default=42); a=ap.parse_args()
    seed_everything(a.seed); cfg,tok,model,ontology=setup(a.config,a.tokenizer); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train=ReportDataset(cfg.data.manifest,'train',tok,ontology,cfg.data,True); val=ReportDataset(cfg.data.manifest,'val',tok,ontology,cfg.data,False)
    coll=partial(collate_reports,pad_id=tok.pad_id)
    tr=DataLoader(train,batch_size=cfg.training.batch_size,shuffle=True,num_workers=cfg.training.num_workers,pin_memory=True,collate_fn=coll)
    va=DataLoader(val,batch_size=cfg.training.batch_size,shuffle=False,num_workers=cfg.training.num_workers,pin_memory=True,collate_fn=coll)
    Trainer(model,cfg,tok,device,a.output_dir).fit(tr,va)
if __name__=='__main__':main()
