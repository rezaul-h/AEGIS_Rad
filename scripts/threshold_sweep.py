import argparse,pandas as pd
from aegis_rad.utils.io import read_jsonl
from aegis_rad.evaluation.threshold import sweep_threshold

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--candidate-scores',required=True); ap.add_argument('--output',required=True); ap.add_argument('--tau-con',type=float,default=.55); ap.add_argument('--tau-suppress',type=float,default=.35); a=ap.parse_args()
    pd.DataFrame(sweep_threshold(read_jsonl(a.candidate_scores),tau_con=a.tau_con,tau_suppress=a.tau_suppress)).to_csv(a.output,index=False)
if __name__=='__main__':main()
