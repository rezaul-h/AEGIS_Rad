from __future__ import annotations
import math, re
import torch
import torch.nn.functional as F
from .models.controller import PRESENT, UNCERTAIN, ABSENT, FindingStateMemory

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
            logits,_=model.decoder(ids,enc['study_tokens'],enc['anatomy_tokens'])
            lp=F.log_softmax(logits[0,-1],-1)
            vals,idx=torch.topk(lp,beam_size)
            for v,j in zip(vals.tolist(),idx.tolist()): cand.append((seq+[j],score+v))
        cand.sort(key=lambda x:x[1]/(len(x[0])**0.7),reverse=True)
        beams=cand[:beam_size]
        if all(s[-1]==tokenizer.eos_id for s,_ in beams): break
    return beams[0][0]

class FindingLexicon:
    def __init__(self,ontology):
        self.ontology=ontology
    def detect(self,text):
        low=text.lower(); found=[]
        for f in self.ontology.findings:
            terms=(f.name,)+f.aliases
            if any(re.search(r'\b'+re.escape(t.lower())+r'\b',low) for t in terms): found.append(f)
        return found

def hedge_sentence(sentence):
    s=sentence.strip()
    if not s: return s
    if re.match(r'(?i)^(possible|may|could|question of|suggestive)',s): return s
    return 'Possible ' + s[0].lower()+s[1:]

@torch.no_grad()
def generate_controlled_report(model,batch,tokenizer,ontology,cfg):
    enc=model.encode(batch['frontal'],batch['lateral'],batch['lateral_missing'])
    if batch['frontal'].shape[0]!=1:
        raise ValueError('generate_controlled_report currently expects batch size 1')
    ids=beam_search(model,enc,tokenizer,cfg.controller.beam_size,cfg.model.max_report_tokens)
    raw=tokenizer.decode(ids)
    support=enc['support_scores'][0].cpu(); con=enc['contradiction_scores'][0].cpu()
    states=model.controller.decisions_from_scores(support,con).cpu()
    lex=FindingLexicon(ontology); memory=FindingStateMemory(); kept=[]; audit=[]
    sentences=[x.strip() for x in re.split(r'(?<=[.!?])\s+',raw) if x.strip()]
    for sent in sentences:
        fs=lex.detect(sent)
        if not fs:
            kept.append(sent); continue
        action='emit'; reasons=[]
        for f in fs:
            st=int(states[f.id]); sp=float(support[f.id]); cr=float(con[f.id])
            name={PRESENT:'present',UNCERTAIN:'uncertain',ABSENT:'absent'}[st]
            if st==ABSENT: action='suppress'; reasons.append((f.name,name,sp,cr))
            elif st==UNCERTAIN and action!='suppress': action='hedge'; reasons.append((f.name,name,sp,cr))
        if action=='emit': kept.append(sent)
        elif action=='hedge': kept.append(hedge_sentence(sent))
        audit.append({'sentence':sent,'action':action,'findings':reasons})
    return {'raw_report':raw,'report':' '.join(kept),'audit':audit,
            'support_scores':support.tolist(),'contradiction_scores':con.tolist(),'states':states.tolist()}
