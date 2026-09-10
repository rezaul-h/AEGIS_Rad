from __future__ import annotations
import sentencepiece as spm
class SentencePieceTokenizer:
    def __init__(self,path):
        self.sp=spm.SentencePieceProcessor(model_file=str(path)); self.pad_id=self.sp.pad_id(); self.bos_id=self.sp.bos_id(); self.eos_id=self.sp.eos_id()
    def encode(self,text,max_tokens=256):
        ids=[self.bos_id]+self.sp.encode(text,out_type=int)[:max_tokens-2]+[self.eos_id]; return ids
    def decode(self,ids):
        ids=[int(x) for x in ids if int(x) not in {self.pad_id,self.bos_id,self.eos_id}]; return self.sp.decode(ids)
