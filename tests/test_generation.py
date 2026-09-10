from aegis_rad.generation import proposed_semantic_state
from aegis_rad.models.controller import PRESENT,ABSENT,UNCERTAIN

def test_sentence_semantics():
    assert proposed_semantic_state('Right pleural effusion is present.')==PRESENT
    assert proposed_semantic_state('No pleural effusion.')==ABSENT
    assert proposed_semantic_state('Possible right basilar atelectasis.')==UNCERTAIN
