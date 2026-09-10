from pathlib import Path
import torch
from aegis_rad.config import load_config
from aegis_rad.factory import build_model

ROOT=Path(__file__).resolve().parents[1]
def test_forward_small():
    cfg=load_config(ROOT/'configs/base.yaml')
    cfg.model.hidden_dim=64; cfg.model.decoder_heads=8; cfg.model.graph_heads=8; cfg.model.decoder_layers=1; cfg.model.ff_dim=128; cfg.model.vocab_size=100; cfg.model.image_size=64
    m,o=build_model(cfg,ROOT)
    x=torch.randn(1,1,64,64); ids=torch.randint(0,100,(1,12))
    out=m(x,x,torch.tensor([False]),ids)
    assert out['token_logits'].shape==(1,12,100)
    assert out['state_logits'].shape[1]==len(o.findings)
