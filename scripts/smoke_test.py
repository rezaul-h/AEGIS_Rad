import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT.parent))
import torch
from aegis_rad.config import load_config
from aegis_rad.factory import build_model
from aegis_rad.models.controller import PRESENT,ABSENT,UNCERTAIN,UNMENTIONED,EMIT,HEDGE,SUPPRESS

def main():
    cfg=load_config(ROOT/'configs/base.yaml'); cfg.model.hidden_dim=96; cfg.model.decoder_heads=8; cfg.model.graph_heads=8
    cfg.model.decoder_layers=2; cfg.model.ff_dim=192; cfg.model.vocab_size=128; cfg.model.image_size=64; cfg.data.image_size=64
    model,ont=build_model(cfg,ROOT); B=2; x=torch.randn(B,1,64,64); ids=torch.randint(0,128,(B,16)); miss=torch.tensor([False,True])
    out=model(x,x,miss,ids)
    assert out['state_logits'].shape==(B,len(ont.findings),4)
    states=torch.tensor([PRESENT,PRESENT,UNCERTAIN,ABSENT,UNMENTIONED])
    support=torch.tensor([.8,.5,.5,.8,.8]); con=torch.tensor([.1,.1,.1,.1,.1])
    actions=model.controller.actions_from_scores(states,support,con).tolist()
    assert actions==[EMIT,HEDGE,HEDGE,EMIT,SUPPRESS]
    print('PASS',{'token_logits':tuple(out['token_logits'].shape),'state_logits':tuple(out['state_logits'].shape),'actions':actions})
if __name__=='__main__':main()
