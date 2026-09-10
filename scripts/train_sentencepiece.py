import argparse, pandas as pd, sentencepiece as spm
from pathlib import Path

def normalize_report(s):
    import re
    s=str(s).replace('\n',' ')
    s=re.sub(r'\[\*\*.*?\*\*\]','<DEID>',s)
    return re.sub(r'\s+',' ',s).strip()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--vocab-size',type=int,default=12000); a=ap.parse_args()
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.manifest); text=out/'train_reports.txt'
    with text.open('w') as f:
        for s in df[df.split.astype(str)=='train'].report:
            z=normalize_report(s)
            if z:f.write(z+'\n')
    spm.SentencePieceTrainer.train(input=str(text),model_prefix=str(out/'aegis'),vocab_size=a.vocab_size,model_type='bpe',character_coverage=1.0,pad_id=0,unk_id=1,bos_id=2,eos_id=3,user_defined_symbols=['<DEID>'])
if __name__=='__main__':main()
