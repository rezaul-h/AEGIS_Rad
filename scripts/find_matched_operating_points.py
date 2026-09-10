import argparse,pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--sweep',required=True); ap.add_argument('--target-ufr',type=float); ap.add_argument('--target-recall',type=float); a=ap.parse_args()
    df=pd.read_csv(a.sweep)
    if a.target_ufr is not None:
        r=df.iloc[(df.ufr-a.target_ufr).abs().argsort()[:1]]; print('matched UFR\n',r.to_string(index=False))
    if a.target_recall is not None:
        r=df.iloc[(df.supported_recall-a.target_recall).abs().argsort()[:1]]; print('matched recall\n',r.to_string(index=False))
if __name__=='__main__':main()
