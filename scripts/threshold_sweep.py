import argparse,pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--candidate-scores',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    df=pd.read_json(a.candidate_scores,lines=True); rows=[]
    for tau in [x/100 for x in range(30,81)]:
        emit=df.support>=tau
        ufr=(~df.supported & emit).sum()/max(1,emit.sum())
        recall=(df.supported & emit).sum()/max(1,df.supported.sum())
        rows.append({'tau_e':tau,'ufr':ufr,'supported_recall':recall,'coverage':emit.mean()})
    pd.DataFrame(rows).to_csv(a.output,index=False)
if __name__=='__main__':main()
