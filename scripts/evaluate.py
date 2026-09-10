import argparse,json
from pathlib import Path
from common import setup
from aegis_rad.utils.io import read_jsonl
from aegis_rad.evaluation.metrics import aggregate_structured
from aegis_rad.evaluation.text_metrics import bleu4,meteor

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--predictions',required=True); ap.add_argument('--references',required=True); ap.add_argument('--structured-pred'); ap.add_argument('--structured-ref'); ap.add_argument('--config',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    from aegis_rad.config import load_config
    cfg=load_config(a.config)
    pred=read_jsonl(a.predictions); ref=read_jsonl(a.references)
    ptext=[x.get('report',x.get('text','')) for x in pred]; rtext=[x.get('report',x.get('text','')) for x in ref]
    out={'bleu4':bleu4(ptext,rtext),'meteor':meteor(ptext,rtext)}
    if a.structured_pred and a.structured_ref:
        out.update(aggregate_structured(read_jsonl(a.structured_pred),read_jsonl(a.structured_ref),cfg.eval.critical_findings))
    Path(a.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
