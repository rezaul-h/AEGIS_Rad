from __future__ import annotations
import torch
from torch import nn
from .vision import build_vision_encoder
from .multiview import MultiViewFusion
from .anatomy import AnatomyQueryExtractor
from .graph import FindingAnatomyGraph
from .controller import FindingStateController
from .decoder import ConfidenceGatedDecoder

class AEGISRad(nn.Module):
    def __init__(self,cfg,ontology,edge_index,edge_type,region_coords):
        super().__init__()
        m=cfg.model
        self.cfg=cfg; self.ontology=ontology
        self.encoder=build_vision_encoder(m)
        self.multiview=MultiViewFusion(m.hidden_dim,m.decoder_heads,m.dropout)
        self.anatomy=AnatomyQueryExtractor(m.anatomy_slots,m.num_findings,m.hidden_dim,m.decoder_heads,m.dropout)
        self.graph=FindingAnatomyGraph(m.anatomy_slots,m.num_findings,m.hidden_dim,m.graph_layers,m.dropout)
        self.decoder=ConfidenceGatedDecoder(m.vocab_size,m.hidden_dim,m.decoder_layers,m.decoder_heads,m.ff_dim,m.dropout,m.max_report_tokens)
        self.controller=FindingStateController(m.hidden_dim,m.num_findings,m.anatomy_slots,cfg.controller.tau_e,cfg.controller.tau_con,cfg.controller.tau_suppress)
        self.register_buffer('edge_index',edge_index,persistent=False)
        self.register_buffer('edge_type',edge_type,persistent=False)
        self.register_buffer('region_coords',region_coords,persistent=False)
        self.clin_head=nn.Linear(m.hidden_dim,m.num_findings)
    def encode(self,frontal,lateral,lateral_missing):
        f=self.encoder(frontal); l=self.encoder(lateral)
        study=self.multiview(f,l,lateral_missing)
        anatomy,abn,unc=self.anatomy(study,self.region_coords)
        anatomy_refined,finding_tokens=self.graph(anatomy,self.edge_index,self.edge_type)
        controller=self.controller(finding_tokens)
        pooled=finding_tokens.mean(1)
        clin_logits=self.clin_head(pooled)
        return {'study_tokens':study,'anatomy_tokens':anatomy_refined,'finding_tokens':finding_tokens,
                'anatomy_abnormality_logits':abn,'anatomy_uncertainty':unc,'clin_logits':clin_logits,**controller}
    def forward(self,frontal,lateral,lateral_missing,input_ids,pad_mask=None):
        enc=self.encode(frontal,lateral,lateral_missing)
        logits,dec=self.decoder(input_ids,enc['study_tokens'],enc['anatomy_tokens'],pad_mask)
        return {'token_logits':logits,**enc,'decoder_aux':dec}
