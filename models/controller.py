from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn

PRESENT=1; UNCERTAIN=2; ABSENT=0

@dataclass
class FindingDecision:
    finding: str
    state: str
    support: float
    contradiction_risk: float
    region: str | None = None

class FindingStateController(nn.Module):
    def __init__(self, hidden_dim=768, num_findings=14, num_regions=24, tau_e=.62, tau_con=.55, tau_suppress=.35):
        super().__init__()
        self.tau_e=tau_e; self.tau_con=tau_con; self.tau_suppress=tau_suppress
        self.state_head=nn.Linear(hidden_dim,3)
        self.support_head=nn.Linear(hidden_dim,1)
        self.contradiction_head=nn.Linear(hidden_dim,1)
        self.region_head=nn.Linear(hidden_dim,num_regions)
    def forward(self,finding_tokens):
        return {
            'state_logits':self.state_head(finding_tokens),
            'support_scores':torch.sigmoid(self.support_head(finding_tokens)).squeeze(-1),
            'contradiction_scores':torch.sigmoid(self.contradiction_head(finding_tokens)).squeeze(-1),
            'region_logits':self.region_head(finding_tokens),
        }
    def decisions_from_scores(self,support,contradiction):
        state=torch.full_like(support,UNCERTAIN,dtype=torch.long)
        state[support>=self.tau_e]=PRESENT
        state[support<self.tau_suppress]=ABSENT
        state[contradiction>self.tau_con]=ABSENT
        return state

class FindingStateMemory:
    def __init__(self): self.memory={}
    def conflicts(self,finding,region,state):
        key=(finding,region)
        prev=self.memory.get(key)
        return prev is not None and prev!=state and {prev,state}=={'present','absent'}
    def commit(self,finding,region,state): self.memory[(finding,region)]=state
