import torch
from aegis_rad.models.controller import *

def test_four_semantic_states_three_actions_and_absent_not_suppress():
    c=FindingStateController(hidden_dim=8,num_findings=3,num_regions=2,num_semantic_states=4,tau_e=.62,tau_con=.55,tau_suppress=.35)
    states=torch.tensor([PRESENT,PRESENT,UNCERTAIN,ABSENT,UNMENTIONED,PRESENT])
    support=torch.tensor([.8,.5,.5,.8,.8,.2]); risk=torch.tensor([.1,.1,.1,.1,.1,.1])
    assert c.actions_from_scores(states,support,risk).tolist()==[EMIT,HEDGE,HEDGE,EMIT,SUPPRESS,SUPPRESS]
    assert ABSENT!=SUPPRESS

def test_high_contradiction_suppresses_without_changing_semantic_state():
    c=FindingStateController(hidden_dim=8,num_findings=1,num_regions=1,num_semantic_states=4)
    states=torch.tensor([PRESENT]); action=c.actions_from_scores(states,torch.tensor([.9]),torch.tensor([.9]))
    assert states.item()==PRESENT and action.item()==SUPPRESS
