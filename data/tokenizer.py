from __future__ import annotations
import sentencepiece as spm
from pathlib import Path

class SentencePieceTokenizer:
    def __init__(self, model_path: str | Path):
        self.sp=spm.SentencePieceProcessor(model_file=str(model_path))
        self.pad_id=self.sp.pad_id()
        self.bos_id=self.sp.bos_id()
        self.eos_id=self.sp.eos_id()
        self.unk_id=self.sp.unk_id()
        self.vocab_size=self.sp.get_piece_size()

    def encode(self, text: str, max_length: int=256) -> list[int]:
        ids=[self.bos_id]+self.sp.encode(text, out_type=int)+[self.eos_id]
        return ids[:max_length-1]+([self.eos_id] if len(ids)>max_length else [])

    def decode(self, ids) -> str:
        clean=[int(i) for i in ids if int(i)>=0 and int(i) not in {self.pad_id,self.bos_id,self.eos_id}]
        return self.sp.decode(clean)
