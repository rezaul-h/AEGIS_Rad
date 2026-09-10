from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn

ABSENT = 0
PRESENT = 1
UNCERTAIN = 2
UNMENTIONED = 3
STATE_NAMES = {ABSENT: "absent", PRESENT: "present", UNCERTAIN: "uncertain", UNMENTIONED: "unmentioned"}
STATE_TO_ID = {v: k for k, v in STATE_NAMES.items()}

EMIT = 0
HEDGE = 1
SUPPRESS = 2
ACTION_NAMES = {EMIT: "emit", HEDGE: "hedge", SUPPRESS: "suppress"}
ACTION_TO_ID = {v: k for k, v in ACTION_NAMES.items()}

@dataclass
class FindingDecision:
    finding: str
    semantic_state: str
    action: str
    support: float
    contradiction_risk: float
    region: str | None = None

class FindingStateController(nn.Module):
    """Predict four report-semantic states and apply a separate three-action policy.

    Semantic states: PRESENT, ABSENT, UNCERTAIN, UNMENTIONED.
    Decoder actions: EMIT, HEDGE, SUPPRESS.

    ABSENT is never used as a synonym for SUPPRESS. A supported absent state may
    be verbalized as an explicit negative statement; suppression leaves an
    unsupported proposal unmentioned.
    """
    def __init__(self, hidden_dim=768, num_findings=14, num_regions=24,
                 num_semantic_states=4, tau_e=.62, tau_con=.55, tau_suppress=.35):
        super().__init__()
        if num_semantic_states != 4:
            raise ValueError("AEGIS-Rad manuscript contract requires four semantic states")
        self.tau_e = tau_e
        self.tau_con = tau_con
        self.tau_suppress = tau_suppress
        self.state_head = nn.Linear(hidden_dim, 4)
        self.support_head = nn.Linear(hidden_dim, 1)
        self.contradiction_head = nn.Linear(hidden_dim, 1)
        self.region_head = nn.Linear(hidden_dim, num_regions)

    def forward(self, finding_tokens):
        return {
            "state_logits": self.state_head(finding_tokens),
            "support_scores": torch.sigmoid(self.support_head(finding_tokens)).squeeze(-1),
            "contradiction_scores": torch.sigmoid(self.contradiction_head(finding_tokens)).squeeze(-1),
            "region_logits": self.region_head(finding_tokens),
        }

    @staticmethod
    def semantic_states_from_logits(state_logits: torch.Tensor) -> torch.Tensor:
        return state_logits.argmax(dim=-1)

    def actions_from_scores(self, semantic_states: torch.Tensor, support: torch.Tensor,
                            contradiction: torch.Tensor) -> torch.Tensor:
        """Deterministic validation-threshold policy used at inference.

        The policy operates *after* semantic-state prediction. Contradiction or
        very weak evidence suppresses a proposal. PRESENT with intermediate
        evidence is hedged. UNCERTAIN is hedged when sufficiently supported.
        ABSENT is emitted only when negative evidence is sufficiently supported.
        UNMENTIONED is suppressed by definition.
        """
        actions = torch.full_like(semantic_states, EMIT, dtype=torch.long)

        hard_suppress = (contradiction > self.tau_con) | (support < self.tau_suppress)
        actions[hard_suppress] = SUPPRESS
        actions[semantic_states == UNMENTIONED] = SUPPRESS

        present = semantic_states == PRESENT
        mid_present = present & (~hard_suppress) & (support < self.tau_e)
        actions[mid_present] = HEDGE

        uncertain = semantic_states == UNCERTAIN
        actions[uncertain & (~hard_suppress)] = HEDGE

        absent = semantic_states == ABSENT
        actions[absent & (~hard_suppress)] = EMIT
        return actions

class FindingStateMemory:
    def __init__(self):
        self.memory = {}

    def conflicts(self, finding: str, region: str | None, semantic_state: str) -> bool:
        key = (finding, region)
        prev = self.memory.get(key)
        return prev is not None and prev != semantic_state and {prev, semantic_state} == {"present", "absent"}

    def commit(self, finding: str, region: str | None, semantic_state: str) -> None:
        self.memory[(finding, region)] = semantic_state
