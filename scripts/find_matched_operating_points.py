import argparse,pandas as pd

def nearest(df,column,target):
    row=df.iloc[(df[column]-target).abs().argsort()[:1]].iloc[0]; return row.to_dict()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--sweep',required=True); ap.add_argument('--target-ufr',type=float,required=True); ap.add_argument('--target-recall',type=float,required=True); a=ap.parse_args(); df=pd.read_csv(a.sweep)
    print('matched_ufr',nearest(df,'ufr',a.target_ufr)); print('matched_recall',nearest(df,'supported_recall',a.target_recall))
if __name__=='__main__':main()
