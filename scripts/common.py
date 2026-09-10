from pathlib import Path
import sys, torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from aegis_rad.config import load_config
from aegis_rad.factory import build_model
from aegis_rad.data.tokenizer import SentencePieceTokenizer

def setup(config_path,tokenizer_path):
    cfg=load_config(config_path)
    tok=SentencePieceTokenizer(tokenizer_path)
    cfg.model.vocab_size=tok.vocab_size
    model,ontology=build_model(cfg,ROOT)
    return cfg,tok,model,ontology

def load_checkpoint(model,path,device):
    ck=torch.load(path,map_location=device)
    model.load_state_dict(ck['model'],strict=True)
    return ck
