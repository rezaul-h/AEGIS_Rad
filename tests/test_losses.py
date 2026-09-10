import torch
from aegis_rad.losses import grounding_loss,unsupported_finding_loss,brier_loss,contradiction_loss

def test_losses_support_four_states():
    f=torch.randn(2,3,8); r=torch.randn(2,4,8); region_logits=torch.randn(2,3,4); target=torch.tensor([[0,1,-100],[2,3,0]])
    l,_=grounding_loss(f,r,region_logits,target); assert torch.isfinite(l)
    states=torch.randn(2,3,4); support=torch.rand(2,3); assert torch.isfinite(unsupported_finding_loss(states,support))
    state_targets=torch.tensor([[0,1,2],[3,1,0]]); assert torch.isfinite(contradiction_loss(states,state_targets,torch.rand(2,3)))
    t=torch.tensor([[1.,0.,float('nan')],[1.,0.,1.]]); assert torch.isfinite(brier_loss(support,t))
