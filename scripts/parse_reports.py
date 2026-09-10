import argparse
from pathlib import Path
from common import setup
from aegis_rad.utils.io import read_jsonl,write_jsonl
from aegis_rad.evaluation.lexicon_parser import parse

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True,help='JSONL with report/text field')
    ap.add_argument('--config',required=True)
    ap.add_argument('--tokenizer',required=True,help='Used only to construct the project ontology/model config')
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    _,_,_,ontology=setup(a.config,a.tokenizer)
    rows=[]
    for row in read_jsonl(a.input):
        text=row.get('report',row.get('text',''))
        rows.append({'study_id':row.get('study_id'),'findings':[x.__dict__ for x in parse(text,ontology)]})
    write_jsonl(a.output,rows)
    print('WARNING: this lightweight lexicon parser is for smoke testing only; use the frozen CheXbert/RadGraph stack for manuscript-equivalent metrics.')
if __name__=='__main__':main()
