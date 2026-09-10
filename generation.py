from __future__ import annotations
import re
import torch
import torch.nn.functional as F
from .models.controller import (
    ABSENT,PRESENT,UNCERTAIN,UNMENTIONED,STATE_NAMES,
    EMIT,HEDGE,SUPPRESS,ACTION_NAMES,FindingStateMemory,
)

@torch.no_grad()
def beam_search(model,enc,tokenizer,beam_size=3,max_tokens=256,device=None):
    device=device or enc['study_tokens'].device
    beams=[([tokenizer.bos_id],0.0)]
    for _ in range(max_tokens-1):
        cand=[]
        for seq,score in beams:
            if seq[-1]==tokenizer.eos_id:
                cand.append((seq,score)); continue
            ids=torch.tensor([seq],device=device)
            logits,_=model.decoder(ids,enc['study_tokens'],enc['anatomy_tokens'],anatomy_uncertainty=enc.get('anatomy_uncertainty'))
            lp=F.log_softmax(logits[0,-1],-1); vals,idx=torch.topk(lp,beam_size)
            for v,j in zip(vals.tolist(),idx.tolist()): cand.append((seq+[j],score+v))
        cand.sort(key=lambda x:x[1]/(len(x[0])**0.7),reverse=True); beams=cand[:beam_size]
        if all(s[-1]==tokenizer.eos_id for s,_ in beams): break
    return beams[0][0]

class FindingLexicon:
    def __init__(self,ontology): self.ontology=ontology
    def detect(self,text):
        low=text.lower(); found=[]
        for f in self.ontology.findings:
            terms=(f.name,)+f.aliases
            if any(re.search(r'\b'+re.escape(t.lower())+r'\b',low) for t in terms): found.append(f)
        return found

def proposed_semantic_state(sentence: str) -> int:
    s=sentence.lower()
    uncertainty=[r'\bpossible\b',r'\bpossibly\b',r'\bmay\b',r'\bmight\b',r'\bcould\b',r'\bcannot exclude\b',r'\bsuspicious for\b',r'\bquestion of\b',r'\blikely\b']
    negative=[r'\bno\b',r'\bwithout\b',r'\bnegative for\b',r'\babsence of\b',r'\bfree of\b',r'\bno evidence of\b']
    if any(re.search(p,s) for p in uncertainty): return UNCERTAIN
    if any(re.search(p,s) for p in negative): return ABSENT
    return PRESENT

def hedge_sentence(sentence):
    s=sentence.strip()
    if not s:return s
    if re.match(r'(?i)^(possible|possibly|may|could|question of|suggestive|cannot exclude)',s): return s
    return 'Possible '+s[0].lower()+s[1:]

@torch.no_grad()
def generate_controlled_report(model,batch,tokenizer,ontology,cfg):
    enc=model.encode(batch['frontal'],batch['lateral'],batch['lateral_missing'])
    if batch['frontal'].shape[0]!=1: raise ValueError('generate_controlled_report currently expects batch size 1')
    ids=beam_search(model,enc,tokenizer,cfg.controller.beam_size,cfg.model.max_report_tokens)
    raw=tokenizer.decode(ids)
    support=enc['support_scores'][0].cpu(); con=enc['contradiction_scores'][0].cpu()
    semantic=model.controller.semantic_states_from_logits(enc['state_logits'])[0].cpu()
    base_actions=model.controller.actions_from_scores(semantic,support,con).cpu()
    regions=enc['region_logits'][0].argmax(-1).cpu()
    lex=FindingLexicon(ontology); memory=FindingStateMemory(); kept=[]; audit=[]
    sentences=[x.strip() for x in re.split(r'(?<=[.!?])\s+',raw) if x.strip()]
    for sent in sentences:
        fs=lex.detect(sent)
        if not fs:
            kept.append(sent); continue
        proposal=proposed_semantic_state(sent); action=EMIT; reasons=[]
        for f in fs:
            pred=int(semantic[f.id]); sp=float(support[f.id]); cr=float(con[f.id]); act=int(base_actions[f.id])
            region=ontology.regions[int(regions[f.id])].name if len(ontology.regions) else None

            if {proposal,pred}=={PRESENT,ABSENT}: act=SUPPRESS

            if proposal==ABSENT and pred!=ABSENT: act=SUPPRESS

            if proposal==PRESENT and pred==UNMENTIONED: act=SUPPRESS
            if memory.conflicts(f.name,region,STATE_NAMES.get(proposal,'unmentioned')): act=SUPPRESS
            action=max(action,act)
            reasons.append({'finding':f.name,'predicted_state':STATE_NAMES[pred],
                            'proposed_state':STATE_NAMES[proposal],'action':ACTION_NAMES[act],
                            'support':sp,'contradiction_risk':cr,'region':region})
        if action==EMIT:
            kept.append(sent)
            for r in reasons: memory.commit(r['finding'],r['region'],r['proposed_state'])
        elif action==HEDGE:
            kept.append(hedge_sentence(sent))
            for r in reasons: memory.commit(r['finding'],r['region'],'uncertain')
        audit.append({'sentence':sent,'action':ACTION_NAMES[action],'findings':reasons})
    return {'raw_report':raw,'report':' '.join(kept),'audit':audit,
            'support_scores':support.tolist(),'contradiction_scores':con.tolist(),
            'semantic_states':[STATE_NAMES[int(x)] for x in semantic.tolist()],
            'actions':[ACTION_NAMES[int(x)] for x in base_actions.tolist()],
            'region_ids':regions.tolist()}
