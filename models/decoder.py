from __future__ import annotations
import torch
from torch import nn

class ConfidenceGatedDecoder(nn.Module):
    def __init__(self,vocab_size,hidden_dim=768,layers=6,heads=12,ff_dim=3072,dropout=.15,max_tokens=256):
        super().__init__(); self.hidden_dim=hidden_dim
        self.embed=nn.Embedding(vocab_size,hidden_dim); self.pos=nn.Embedding(max_tokens,hidden_dim)
        layer=nn.TransformerDecoderLayer(hidden_dim,heads,ff_dim,dropout,batch_first=True,norm_first=True)
        self.decoder=nn.TransformerDecoder(layer,layers)
        self.anatomy_attn=nn.MultiheadAttention(hidden_dim,heads,dropout=dropout,batch_first=True)
        self.gate=nn.Linear(hidden_dim*3,hidden_dim); self.context_proj=nn.Linear(hidden_dim*2,hidden_dim)
        self.lm_head=nn.Linear(hidden_dim,vocab_size,bias=False); self.lm_head.weight=self.embed.weight
        self.norm=nn.LayerNorm(hidden_dim)
    @staticmethod
    def causal_mask(n,device): return torch.triu(torch.full((n,n),float('-inf'),device=device),diagonal=1)
    def forward(self,input_ids,study_tokens,anatomy_tokens,pad_mask=None,anatomy_uncertainty=None):
        B,T=input_ids.shape; pos=torch.arange(T,device=input_ids.device)[None]
        tgt=self.embed(input_ids)+self.pos(pos)
        h=self.decoder(tgt,study_tokens,tgt_mask=self.causal_mask(T,input_ids.device),tgt_key_padding_mask=pad_mask)
        c_v=study_tokens.mean(1)[:,None,:].expand(-1,T,-1)
        reliability=None
        attn_tokens=anatomy_tokens
        if anatomy_uncertainty is not None:
            reliability=(1-anatomy_uncertainty).clamp(0,1)
            attn_tokens=anatomy_tokens*reliability.unsqueeze(-1)
        c_a,w=self.anatomy_attn(h,attn_tokens,attn_tokens,need_weights=True,average_attn_weights=True)
        g=torch.sigmoid(self.gate(torch.cat([h,c_v,c_a],-1)))
        if reliability is not None:
            attended_rel=torch.einsum('btr,br->bt',w,reliability).unsqueeze(-1)
            g=g*attended_rel
        c=g*c_a+(1-g)*c_v
        z=self.norm(h+self.context_proj(torch.cat([h,c],-1)))
        return self.lm_head(z), {'gate':g,'global_context':c_v,'anatomy_context':c_a,'hidden':h,
                                'fused_context':c,'anatomy_attention':w,'anatomy_reliability':reliability}
