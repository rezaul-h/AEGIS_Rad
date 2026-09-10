from __future__ import annotations
import torch
from torch import nn

class ConvVisionEncoder(nn.Module):
    """Dependency-light fallback used only by unit/smoke tests, not manuscript runs."""
    def __init__(self,in_channels=1,hidden_dim=768):
        super().__init__(); dims=[64,128,256,hidden_dim]; layers=[]; c=in_channels
        for d in dims:
            layers += [nn.Conv2d(c,d,3,2,1),nn.GroupNorm(min(32,d),d),nn.GELU()]; c=d
        self.net=nn.Sequential(*layers); self.cls=nn.Parameter(torch.zeros(1,1,hidden_dim)); nn.init.normal_(self.cls,std=.02)
    def forward(self,x):
        z=self.net(x); patches=z.flatten(2).transpose(1,2); cls=patches.mean(1,keepdim=True)+self.cls
        return torch.cat([cls,patches],1)

class HFVisionEncoder(nn.Module):
    def __init__(self,model_name='microsoft/rad-dino',revision=None,hidden_dim=768,in_channels=1):
        super().__init__()
        try: from transformers import AutoModel
        except Exception as e: raise ImportError('Install transformers to use encoder_backend=hf') from e
        self.model=AutoModel.from_pretrained(model_name,revision=revision,trust_remote_code=True)
        source_dim=getattr(self.model.config,'hidden_size',hidden_dim)
        self.proj=nn.Identity() if source_dim==hidden_dim else nn.Linear(source_dim,hidden_dim)
    def forward(self,x):
        if x.shape[1]==1:x=x.repeat(1,3,1,1)
        out=self.model(pixel_values=x); tokens=getattr(out,'last_hidden_state',None)
        if tokens is None: raise RuntimeError('Vision model did not return last_hidden_state')
        return self.proj(tokens)

def build_vision_encoder(cfg):
    if cfg.encoder_backend.lower()=='hf': return HFVisionEncoder(cfg.encoder_name,cfg.encoder_revision,cfg.hidden_dim,cfg.in_channels)
    if cfg.encoder_backend.lower()=='conv': return ConvVisionEncoder(cfg.in_channels,cfg.hidden_dim)
    raise ValueError(f'Unknown encoder_backend={cfg.encoder_backend!r}')
