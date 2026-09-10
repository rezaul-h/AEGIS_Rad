from __future__ import annotations
import math
import torch
from torch import nn
import yaml
from pathlib import Path

RELATIONS=['adjacency','laterality','containment','view_consistency','finding_location']

class RelationMessageLayer(nn.Module):
    """Multi-head relation-aware graph attention over anatomy and finding nodes."""
    def __init__(self, hidden_dim=768, num_relations=len(RELATIONS), heads=8, dropout=0.15):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError('hidden_dim must be divisible by graph heads')
        self.heads=heads; self.d=hidden_dim//heads; self.hidden_dim=hidden_dim
        self.q=nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.k=nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.v=nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.rel_k=nn.Embedding(num_relations,hidden_dim)
        self.rel_v=nn.Embedding(num_relations,hidden_dim)
        self.out=nn.Linear(hidden_dim,hidden_dim)
        self.norm=nn.LayerNorm(hidden_dim); self.drop=nn.Dropout(dropout)

    def forward(self,x,edge_index,edge_type):
        if edge_index.numel()==0: return x
        B,N,H=x.shape; s,d=edge_index[0],edge_index[1]; E=s.numel()
        q=self.q(x).view(B,N,self.heads,self.d)
        k=self.k(x).view(B,N,self.heads,self.d)
        v=self.v(x).view(B,N,self.heads,self.d)
        rk=self.rel_k(edge_type).view(E,self.heads,self.d)
        rv=self.rel_v(edge_type).view(E,self.heads,self.d)
        logits=(q[:,d]*(k[:,s]+rk[None])).sum(-1)/math.sqrt(self.d)
        alpha=torch.zeros_like(logits)
        for node in range(N):
            m=(d==node)
            if m.any(): alpha[:,m,:]=torch.softmax(logits[:,m,:],dim=1)
        msg=(v[:,s]+rv[None])*alpha[...,None]
        agg=torch.zeros((B,N,self.heads,self.d),device=x.device,dtype=x.dtype)
        for e in range(E): agg[:,d[e]] += msg[:,e]
        agg=agg.reshape(B,N,H)
        return self.norm(x+self.drop(self.out(agg)))

class FindingAnatomyGraph(nn.Module):
    def __init__(self,num_regions=24,num_findings=14,hidden_dim=768,layers=2,heads=8,dropout=0.15):
        super().__init__(); self.num_regions=num_regions; self.num_findings=num_findings
        self.finding_embed=nn.Parameter(torch.randn(1,num_findings,hidden_dim)*0.02)
        self.layers=nn.ModuleList([RelationMessageLayer(hidden_dim,len(RELATIONS),heads,dropout) for _ in range(layers)])
        self.norm=nn.LayerNorm(hidden_dim)
    def forward(self,region_tokens,edge_index,edge_type):
        B=region_tokens.shape[0]
        x=torch.cat([region_tokens,self.finding_embed.expand(B,-1,-1)],1)
        for layer in self.layers: x=layer(x,edge_index,edge_type)
        return self.norm(x[:,:self.num_regions]), self.norm(x[:,self.num_regions:])

def load_graph_edges(path, ontology):
    data=yaml.safe_load(Path(path).read_text()) or {}; edges=[]; types=[]
    rel2id={r:i for i,r in enumerate(RELATIONS)}; R=len(ontology.regions)
    for e in data.get('edges',[]):
        st=e['source_type']; dt=e['target_type']; rel=e['relation']
        s=ontology.region_to_id[e['source']] if st=='region' else R+ontology.finding_to_id[e['source']]
        d=ontology.region_to_id[e['target']] if dt=='region' else R+ontology.finding_to_id[e['target']]
        rid=rel2id[rel]; edges.append((s,d)); types.append(rid)
        if e.get('bidirectional',True): edges.append((d,s)); types.append(rid)
    if not edges: return torch.empty((2,0),dtype=torch.long),torch.empty((0,),dtype=torch.long)
    return torch.tensor(edges,dtype=torch.long).t().contiguous(),torch.tensor(types,dtype=torch.long)
