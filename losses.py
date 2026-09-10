from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn

class AsymmetricLoss(nn.Module):
    def __init__(self,gamma_neg=4,gamma_pos=1,eps=1e-8):
        super().__init__(); self.gn=gamma_neg; self.gp=gamma_pos; self.eps=eps
    def forward(self,logits,targets,mask=None):
        p=torch.sigmoid(logits); y=targets.float()
        pos=y*torch.log(p.clamp_min(self.eps))*((1-p)**self.gp)
        neg=(1-y)*torch.log((1-p).clamp_min(self.eps))*(p**self.gn)
        loss=-(pos+neg)
        if mask is not None: loss=loss*mask.float(); return loss.sum()/mask.sum().clamp_min(1)
        return loss.mean()

def label_smoothed_ce(logits,targets,pad_id,label_smoothing=.1):
    return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1),ignore_index=pad_id,label_smoothing=label_smoothing)

def grounding_loss(finding_tokens,region_tokens,region_targets,temperature=.07):
    f=F.normalize(finding_tokens,dim=-1); r=F.normalize(region_tokens,dim=-1)
    sim=torch.einsum('bfh,brh->bfr',f,r)/temperature
    valid=region_targets>=0
    if valid.any():
        ce=F.cross_entropy(sim[valid],region_targets[valid])
    else: ce=sim.sum()*0
    return ce, sim

def contradiction_loss(state_logits,state_targets,contradiction_scores=None):
    valid=state_targets>=0
    ce=F.cross_entropy(state_logits[valid],state_targets[valid]) if valid.any() else state_logits.sum()*0
    reg=contradiction_scores.mean() if contradiction_scores is not None else ce*0
    return ce+0.05*reg

def unsupported_finding_loss(state_logits,support_scores):
    p_present=torch.softmax(state_logits,-1)[...,1]
    return (p_present*(1-support_scores)).mean()

def brier_loss(prob,target):
    valid=torch.isfinite(target)
    return ((prob[valid]-target[valid])**2).mean() if valid.any() else prob.sum()*0

class AEGISLoss(nn.Module):
    def __init__(self,cfg,pad_id):
        super().__init__(); self.cfg=cfg; self.pad_id=pad_id; self.asl=AsymmetricLoss()
    def forward(self,out,batch):
        logits=out['token_logits'][:,:-1]
        tgt=batch['report_ids'][:,1:]
        l_gen=label_smoothed_ce(logits,tgt,self.pad_id,self.cfg.training.label_smoothing)
        states=batch['finding_states']
        known=states>=0; y=(states==1).float()
        l_clin=self.asl(out['clin_logits'],y,known)
        l_grd,_=grounding_loss(out['finding_tokens'],out['anatomy_tokens'],batch['region_targets'],self.cfg.loss.grounding_contrastive_temperature)
        l_con=contradiction_loss(out['state_logits'],states,out['contradiction_scores'])
        l_hall=unsupported_finding_loss(out['state_logits'],out['support_scores'])
        l_cal=brier_loss(out['support_scores'],batch['support_targets'])
        w=self.cfg.loss
        total=w.gen*l_gen+w.clin*l_clin+w.grd*l_grd+w.con*l_con+w.hall*l_hall+w.cal*l_cal
        return {'loss':total,'gen':l_gen,'clin':l_clin,'grd':l_grd,'con':l_con,'hall':l_hall,'cal':l_cal}
