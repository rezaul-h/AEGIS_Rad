from __future__ import annotations
import torch
from torch import nn

class MultiViewFusion(nn.Module):
    def __init__(self, hidden_dim=768, heads=12, dropout=0.15):
        super().__init__()
        self.view_embedding=nn.Embedding(2,hidden_dim)
        self.missing_view=nn.Parameter(torch.randn(1,1,hidden_dim)*0.02)
        layer=nn.TransformerEncoderLayer(hidden_dim,heads,hidden_dim*4,dropout,batch_first=True,norm_first=True)
        self.fuser=nn.TransformerEncoder(layer,num_layers=1)
        self.norm=nn.LayerNorm(hidden_dim)
    def forward(self, frontal_tokens, lateral_tokens, lateral_missing):
        B=frontal_tokens.shape[0]
        f=frontal_tokens+self.view_embedding.weight[0][None,None,:]
        l=lateral_tokens+self.view_embedding.weight[1][None,None,:]
        if lateral_missing.any():
            repl=self.missing_view.expand(B,l.shape[1],-1)+self.view_embedding.weight[1][None,None,:]
            l=torch.where(lateral_missing[:,None,None],repl,l)
        x=torch.cat([f,l],1)
        return self.norm(self.fuser(x))
