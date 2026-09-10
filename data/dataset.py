from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from .preprocessing import ChestXrayPreprocessor

STATE_TO_ID={"absent":0,"present":1,"uncertain":2,"unknown":-100}

class ReportDataset(Dataset):
    def __init__(self, manifest: str|Path, split: str, tokenizer, ontology, cfg, training=False):
        df=pd.read_csv(manifest)
        self.df=df[df['split'].astype(str)==split].reset_index(drop=True)
        self.tokenizer=tokenizer
        self.ontology=ontology
        self.cfg=cfg
        self.transform=ChestXrayPreprocessor(
            image_size=cfg.image_size,
            percentile_low=cfg.percentile_low,
            percentile_high=cfg.percentile_high,
            training=training,
            rotation_deg=cfg.rotation_deg,
            translate_frac=cfg.translate_frac,
            contrast_jitter=cfg.contrast_jitter,
        )

    def __len__(self): return len(self.df)

    @staticmethod
    def _loads(x):
        if x is None or (isinstance(x,float) and pd.isna(x)) or x=='': return {}
        if isinstance(x,dict): return x
        return json.loads(x)

    def __getitem__(self, idx):
        r=self.df.iloc[idx]
        frontal=self.transform(Image.open(r['frontal_path']))
        lateral_missing=('lateral_path' not in r or pd.isna(r.get('lateral_path')) or str(r.get('lateral_path','')).strip()=='')
        if lateral_missing:
            lateral=torch.zeros_like(frontal)
        else:
            lateral=self.transform(Image.open(r['lateral_path']))
        ids=self.tokenizer.encode(str(r['report']), self.cfg.max_report_tokens)
        finding_states=self._loads(r.get('findings_json','{}'))
        region_targets=self._loads(r.get('region_targets_json','{}'))
        support_targets=self._loads(r.get('support_targets_json','{}'))
        F=len(self.ontology.findings)
        state=torch.full((F,), -100, dtype=torch.long)
        region=torch.full((F,), -100, dtype=torch.long)
        support=torch.full((F,), float('nan'), dtype=torch.float32)
        for f in self.ontology.findings:
            if f.name in finding_states:
                state[f.id]=STATE_TO_ID.get(str(finding_states[f.name]).lower(), -100)
            rt=region_targets.get(f.name)
            if isinstance(rt,list) and rt: rt=rt[0]
            if rt in self.ontology.region_to_id:
                region[f.id]=self.ontology.region_to_id[rt]
            if f.name in support_targets:
                support[f.id]=float(support_targets[f.name])
        return {
            'study_id': str(r['study_id']), 'frontal':frontal, 'lateral':lateral,
            'lateral_missing':torch.tensor(lateral_missing,dtype=torch.bool),
            'report_ids':torch.tensor(ids,dtype=torch.long), 'finding_states':state,
            'region_targets':region, 'support_targets':support,
            'report_text':str(r['report'])
        }

def collate_reports(batch, pad_id: int):
    maxlen=max(x['report_ids'].numel() for x in batch)
    B=len(batch)
    ids=torch.full((B,maxlen),pad_id,dtype=torch.long)
    mask=torch.zeros((B,maxlen),dtype=torch.bool)
    for i,x in enumerate(batch):
        n=x['report_ids'].numel(); ids[i,:n]=x['report_ids']; mask[i,:n]=True
    return {
        'study_id':[x['study_id'] for x in batch],
        'frontal':torch.stack([x['frontal'] for x in batch]),
        'lateral':torch.stack([x['lateral'] for x in batch]),
        'lateral_missing':torch.stack([x['lateral_missing'] for x in batch]),
        'report_ids':ids, 'report_mask':mask,
        'finding_states':torch.stack([x['finding_states'] for x in batch]),
        'region_targets':torch.stack([x['region_targets'] for x in batch]),
        'support_targets':torch.stack([x['support_targets'] for x in batch]),
        'report_text':[x['report_text'] for x in batch],
    }
