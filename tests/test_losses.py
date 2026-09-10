import torch
from aegis_rad.losses import grounding_loss, unsupported_finding_loss, brier_loss

def test_losses_finite():
    f=torch.randn(2,3,8); r=torch.randn(2,4,8); target=torch.tensor([[0,1,-100],[2,3,0]])
    l,_=grounding_loss(f,r,target)
    assert torch.isfinite(l)
    states=torch.randn(2,3,3); support=torch.rand(2,3)
    assert torch.isfinite(unsupported_finding_loss(states,support))
    t=torch.tensor([[1.0,0.0,float('nan')],[1.0,0.0,1.0]])
    assert torch.isfinite(brier_loss(support,t))
