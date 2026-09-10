from __future__ import annotations
from pathlib import Path
import torch
from .ontology import Ontology
from .models.graph import load_graph_edges
from .models.aegis_rad import AEGISRad

def resolve(base,path):
    p=Path(path); return p if p.is_absolute() else Path(base)/p

def build_model(cfg,project_root='.'):
    ontology=Ontology.from_yaml(resolve(project_root,cfg.regions_path),resolve(project_root,cfg.findings_path))
    cfg.model.anatomy_slots=len(ontology.regions); cfg.model.num_findings=len(ontology.findings)
    if cfg.model.anatomy_slots!=24: raise ValueError(f'Manuscript contract requires 24 anatomy slots, got {cfg.model.anatomy_slots}')
    if cfg.model.num_findings!=14: raise ValueError(f'Manuscript contract requires 14 candidate findings, got {cfg.model.num_findings}')
    edge_index,edge_type=load_graph_edges(resolve(project_root,cfg.graph_path),ontology)
    coords=torch.tensor([r.xy for r in ontology.regions],dtype=torch.float32)
    return AEGISRad(cfg,ontology,edge_index,edge_type,coords),ontology
