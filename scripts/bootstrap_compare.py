import argparse,json,numpy as np
from pathlib import Path
from aegis_rad.utils.io import read_jsonl
from aegis_rad.evaluation.bootstrap import paired_bootstrap

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--a',required=True); ap.add_argument('--b',required=True); ap.add_argument('--metrics',nargs='+',required=True); ap.add_argument('--n-bootstrap',type=int,default=5000); ap.add_argument('--seed',type=int,default=2026); ap.add_argument('--output'); args=ap.parse_args()
    A=read_jsonl(args.a); B=read_jsonl(args.b); result={}
    for m in args.metrics:
        aa=[x[m] for x in A]; bb=[x[m] for x in B]
        result[m]=paired_bootstrap(aa,bb,lambda x:float(np.mean(x)),args.n_bootstrap,args.seed)
    txt=json.dumps(result,indent=2); print(txt)
    if args.output:Path(args.output).write_text(txt)
if __name__=='__main__':main()
