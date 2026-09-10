import argparse,tempfile
from pathlib import Path
import pandas as pd
import sentencepiece as spm

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--vocab-size',type=int,default=12000); a=ap.parse_args()
    df=pd.read_csv(a.manifest); train=df[df.split.astype(str)=='train']; out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',encoding='utf-8',delete=False) as f:
        for x in train.report.astype(str):f.write(x.replace('\n',' ')+'\n')
        corpus=f.name
    spm.SentencePieceTrainer.train(input=corpus,model_prefix=str(out/'aegis'),vocab_size=a.vocab_size,model_type='unigram',pad_id=0,unk_id=1,bos_id=2,eos_id=3,character_coverage=1.0)
    Path(corpus).unlink(missing_ok=True)
if __name__=='__main__':main()
