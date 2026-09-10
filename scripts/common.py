from pathlib import Path
import torch
from aegis_rad.config import load_config
from aegis_rad.data.tokenizer import SentencePieceTokenizer
from aegis_rad.factory import build_model

def setup(config_path,tokenizer_path):
    root=Path(__file__).resolve().parents[1]
    cfg=load_config(config_path); tok=SentencePieceTokenizer(tokenizer_path); model,ontology=build_model(cfg,root)
    return cfg,tok,model,ontology

def load_checkpoint(model,path,device='cpu'):
    obj=torch.load(path,map_location=device); model.load_state_dict(obj['model'] if isinstance(obj,dict) and 'model' in obj else obj); return obj
