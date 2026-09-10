import argparse,json
import pandas as pd
from aegis_rad.evaluation.bootstrap import paired_bootstrap

def read_jsonl(path):return pd.read_json(path,lines=True)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--a',required=True); ap.add_argument('--b',required=True); ap.add_argument('--metrics',nargs='+',required=True); ap.add_argument('--n-bootstrap',type=int,default=5000); ap.add_argument('--seed',type=int,default=2026); a=ap.parse_args()
    A=read_jsonl(a.a); B=read_jsonl(a.b); M=A.merge(B,on='study_id',suffixes=('_a','_b')); out={}
    for m in a.metrics:out[m]=paired_bootstrap(M[f'{m}_a'],M[f'{m}_b'],a.n_bootstrap,a.seed)
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
