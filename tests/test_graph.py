import torch
from aegis_rad.models.graph import FindingAnatomyGraph

def test_graph_shapes():
    g=FindingAnatomyGraph(4,3,32,2,.0)
    x=torch.randn(2,4,32)
    edge=torch.tensor([[0,1,4],[1,2,0]])
    typ=torch.tensor([0,1,4])
    r,f=g(x,edge,typ)
    assert r.shape==(2,4,32) and f.shape==(2,3,32)
