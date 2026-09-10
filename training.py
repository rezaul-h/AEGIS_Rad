from __future__ import annotations
from pathlib import Path
import math, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .losses import AEGISLoss

class Trainer:
    def __init__(self,model,cfg,tokenizer,device,output_dir):
        self.model=model.to(device); self.cfg=cfg; self.tok=tokenizer; self.device=device
        self.output=Path(output_dir); self.output.mkdir(parents=True,exist_ok=True)
        self.optim=torch.optim.AdamW(model.parameters(),lr=cfg.training.lr,weight_decay=cfg.training.weight_decay)
        self.scaler=torch.amp.GradScaler('cuda',enabled=cfg.training.amp and device.type=='cuda')
        self.criterion=AEGISLoss(cfg,tokenizer.pad_id)
        self.best=float('inf'); self.bad=0
    def _move(self,b):
        return {k:(v.to(self.device) if torch.is_tensor(v) else v) for k,v in b.items()}
    def _lr(self,epoch):
        if epoch<self.cfg.training.warmup_epochs: return self.cfg.training.lr*(epoch+1)/self.cfg.training.warmup_epochs
        t=(epoch-self.cfg.training.warmup_epochs)/max(1,self.cfg.training.epochs-self.cfg.training.warmup_epochs)
        return self.cfg.training.lr*0.5*(1+math.cos(math.pi*t))
    def run_epoch(self,loader,train=True):
        self.model.train(train); sums={}; n=0
        for b in tqdm(loader,leave=False):
            b=self._move(b); inp=b['report_ids'][:,:-1]; mask=~b['report_mask'][:,:-1]
            with torch.set_grad_enabled(train), torch.autocast(device_type=self.device.type,enabled=self.cfg.training.amp and self.device.type=='cuda'):
                out=self.model(b['frontal'],b['lateral'],b['lateral_missing'],inp,mask)
                losses=self.criterion(out,b)
            if train:
                self.optim.zero_grad(set_to_none=True)
                self.scaler.scale(losses['loss']).backward()
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.cfg.training.grad_clip)
                self.scaler.step(self.optim); self.scaler.update()
            bs=b['frontal'].shape[0]; n+=bs
            for k,v in losses.items(): sums[k]=sums.get(k,0)+float(v.detach())*bs
        return {k:v/max(n,1) for k,v in sums.items()}
    def fit(self,train_loader,val_loader):
        history=[]
        for epoch in range(self.cfg.training.epochs):
            lr=self._lr(epoch)
            for g in self.optim.param_groups:g['lr']=lr
            tr=self.run_epoch(train_loader,True); va=self.run_epoch(val_loader,False)
            row={'epoch':epoch+1,'lr':lr,'train':tr,'val':va}; history.append(row)
            torch.save({'model':self.model.state_dict(),'config':self.cfg.__dict__,'epoch':epoch+1},self.output/'last.pt')
            if va['loss']<self.best:
                self.best=va['loss']; self.bad=0
                torch.save({'model':self.model.state_dict(),'epoch':epoch+1},self.output/'best.pt')
            else: self.bad+=1
            print(row)
            if self.bad>=self.cfg.training.early_stopping_patience: break
        import json
        (self.output/'history.json').write_text(json.dumps(history,indent=2))
