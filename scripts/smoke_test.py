import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import torch
from aegis_rad.config import load_config
from aegis_rad.factory import build_model

def main():
    cfg=load_config(ROOT/'configs/base.yaml'); cfg.model.hidden_dim=96; cfg.model.decoder_heads=8; cfg.model.graph_heads=8; cfg.model.decoder_layers=2; cfg.model.ff_dim=192; cfg.model.vocab_size=128; cfg.model.image_size=64; cfg.data.image_size=64
    model,ont=build_model(cfg,ROOT)
    B=2; x=torch.randn(B,1,64,64); ids=torch.randint(0,128,(B,16)); miss=torch.tensor([False,True])
    out=model(x,x,miss,ids)
    print('token_logits',tuple(out['token_logits'].shape),'state_logits',tuple(out['state_logits'].shape),'support',tuple(out['support_scores'].shape))
if __name__=='__main__':main()
