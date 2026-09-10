import torch
from aegis_rad.models.graph import RelationMessageLayer

def test_multihead_relation_attention_shape():
    layer=RelationMessageLayer(hidden_dim=32,num_relations=5,heads=4,dropout=0.0)
    x=torch.randn(2,5,32); edge_index=torch.tensor([[0,1,2],[1,2,3]]); edge_type=torch.tensor([0,1,4])
    y=layer(x,edge_index,edge_type); assert y.shape==x.shape
