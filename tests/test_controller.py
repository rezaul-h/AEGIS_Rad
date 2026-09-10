import torch
from aegis_rad.models.controller import FindingStateController,PRESENT,UNCERTAIN,ABSENT

def test_threshold_states():
    c=FindingStateController(hidden_dim=8,num_findings=3,num_regions=2,tau_e=.62,tau_con=.55,tau_suppress=.35)
    s=torch.tensor([.8,.5,.2,.8]); r=torch.tensor([.1,.1,.1,.8])
    out=c.decisions_from_scores(s,r).tolist()
    assert out==[PRESENT,UNCERTAIN,ABSENT,ABSENT]
