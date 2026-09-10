from __future__ import annotations
import torch
from torch import nn
import yaml
from pathlib import Path

RELATIONS=['adjacency','laterality','containment','view_consistency','finding_location']

class RelationMessageLayer(nn.Module):
    """Pure-PyTorch relation-aware message passing; avoids a torch-geometric dependency."""
    def __init__(self, hidden_dim=768, num_relations=len(RELATIONS), dropout=0.15):
        super().__init__()
        self.rel=nn.Embedding(num_relations,hidden_dim)
        self.src=nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.dst=nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.msg=nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.score=nn.Linear(hidden_dim,1,bias=False)
        self.out=nn.Linear(hidden_dim,hidden_dim)
        self.norm=nn.LayerNorm(hidden_dim)
        self.drop=nn.Dropout(dropout)
    def forward(self,x,edge_index,edge_type):
        if edge_index.numel()==0: return x
        s,d=edge_index[0],edge_index[1]
        rel=self.rel(edge_type)
        src=self.src(x[:,s])+rel[None]
        dst=self.dst(x[:,d])
        logits=self.score(torch.tanh(src+dst)).squeeze(-1)
        B,N,H=x.shape
        alpha=torch.zeros_like(logits)
        for node in range(N):
            m=(d==node)
            if m.any(): alpha[:,m]=torch.softmax(logits[:,m],dim=-1)
        messages=self.msg(x[:,s]+rel[None])*alpha[...,None]
        agg=torch.zeros_like(x)
        for e in range(s.numel()): agg[:,d[e]] += messages[:,e]
        return self.norm(x+self.drop(self.out(agg)))

class FindingAnatomyGraph(nn.Module):
    def __init__(self,num_regions=24,num_findings=14,hidden_dim=768,layers=2,dropout=0.15):
        super().__init__()
        self.num_regions=num_regions; self.num_findings=num_findings
        self.finding_embed=nn.Parameter(torch.randn(1,num_findings,hidden_dim)*0.02)
        self.layers=nn.ModuleList([RelationMessageLayer(hidden_dim,len(RELATIONS),dropout) for _ in range(layers)])
        self.norm=nn.LayerNorm(hidden_dim)
    def forward(self,region_tokens,edge_index,edge_type):
        B=region_tokens.shape[0]
        x=torch.cat([region_tokens,self.finding_embed.expand(B,-1,-1)],1)
        for layer in self.layers: x=layer(x,edge_index,edge_type)
        return self.norm(x[:,:self.num_regions]), self.norm(x[:,self.num_regions:])

def load_graph_edges(path, ontology):
    data=yaml.safe_load(Path(path).read_text()) or {}
    edges=[]; types=[]
    rel2id={r:i for i,r in enumerate(RELATIONS)}
    R=len(ontology.regions)
    for e in data.get('edges',[]):
        st=e['source_type']; dt=e['target_type']; rel=e['relation']
        if st=='region': s=ontology.region_to_id[e['source']]
        else: s=R+ontology.finding_to_id[e['source']]
        if dt=='region': d=ontology.region_to_id[e['target']]
        else: d=R+ontology.finding_to_id[e['target']]
        rid=rel2id[rel]
        edges.append((s,d)); types.append(rid)
        if e.get('bidirectional',True): edges.append((d,s)); types.append(rid)
    if not edges:
        return torch.empty((2,0),dtype=torch.long), torch.empty((0,),dtype=torch.long)
    return torch.tensor(edges,dtype=torch.long).t().contiguous(), torch.tensor(types,dtype=torch.long)
