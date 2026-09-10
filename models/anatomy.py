from __future__ import annotations
import torch
from torch import nn

class AnatomyQueryExtractor(nn.Module):
    def __init__(self, num_regions=24, num_findings=14, hidden_dim=768, heads=12, dropout=0.15):
        super().__init__()
        self.num_regions=num_regions
        self.queries=nn.Parameter(torch.randn(1,num_regions,hidden_dim)*0.02)
        self.cross_attn=nn.MultiheadAttention(hidden_dim,heads,dropout=dropout,batch_first=True)
        self.region_embed=nn.Embedding(num_regions,hidden_dim//8)
        self.abnormality=nn.Linear(hidden_dim,num_findings)
        self.uncertainty=nn.Linear(hidden_dim,1)
        self.token_proj=nn.Linear(hidden_dim+hidden_dim//8+2+num_findings+1,hidden_dim)
        self.norm=nn.LayerNorm(hidden_dim)
    def forward(self, study_tokens, region_coords):
        B=study_tokens.shape[0]
        q=self.queries.expand(B,-1,-1)
        appearance,_=self.cross_attn(q,study_tokens,study_tokens,need_weights=False)
        rid=torch.arange(self.num_regions,device=study_tokens.device)
        rembed=self.region_embed(rid)[None].expand(B,-1,-1)
        coords=region_coords.to(study_tokens.device)[None].expand(B,-1,-1)
        abn=self.abnormality(appearance)
        unc=torch.sigmoid(self.uncertainty(appearance))
        token=self.token_proj(torch.cat([appearance,rembed,coords,abn,unc],-1))
        return self.norm(token), abn, unc.squeeze(-1)
