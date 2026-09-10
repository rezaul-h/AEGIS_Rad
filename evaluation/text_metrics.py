from __future__ import annotations

def bleu4(preds,refs):
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    hypotheses=[p.split() for p in preds]; references=[[r.split()] for r in refs]
    return float(corpus_bleu(references,hypotheses,weights=(.25,.25,.25,.25),smoothing_function=SmoothingFunction().method1))

def rouge_l(preds,refs):
    try:
        from rouge_score import rouge_scorer
    except ImportError as e:
        raise ImportError("Install `pip install -e '.[text_metrics]'` for ROUGE-L") from e
    s=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True)
    return sum(s.score(r,p)['rougeL'].fmeasure for p,r in zip(preds,refs))/max(len(preds),1)

def meteor(preds,refs):
    from nltk.translate.meteor_score import meteor_score
    return sum(meteor_score([r.split()],p.split()) for p,r in zip(preds,refs))/max(len(preds),1)
