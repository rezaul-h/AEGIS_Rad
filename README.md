# AEGIS-Rad

This repository implements the architecture, training objective, inference controller, and evaluation protocol for AEGIS-Rad manuscript.

## What is implemented

- multi-view chest-radiograph encoder with explicit missing-view handling;
- 24 fixed anatomical query slots with image-dependent token values;
- typed finding--anatomy relation graph with relation-aware message passing;
- confidence-controlled global/regional fusion;
- transformer report decoder;
- mutually exclusive finding-state controller: **PRESENT / UNCERTAIN / ABSENT**;
- six losses: generation, clinical labels, grounding, contradiction, unsupported-finding control, and calibration;
- validation-only threshold sweeps;
- beam-search report generation (beam size 3 by default);
- reference-conditioned UFR, supported-positive recall, Grounding-F1, COR, ECE;
- paired study-level bootstrap confidence intervals;
- five-seed training support;
- MIMIC-CXR and IU X-Ray manifest preparation templates;
- SentencePiece tokenizer training with 12,000 vocabulary and 256-token maximum length.

## Hyperparameters encoded by default

| Item | Setting |
|---|---|
| input resolution | 384 x 384 |
| anatomy slots | 24 |
| graph layers | 2 |
| decoder | 6 layers, hidden 768, 12 heads |
| batch size | 16 |
| optimizer | AdamW |
| learning rate | 1e-4 |
| weight decay | 1e-4 |
| schedule | cosine |
| epochs | 120 |
| dropout | 0.15 |
| gradient clipping | 1.0 |
| beam size | 3 |
| support threshold `tau_e` | 0.62 |
| contradiction threshold `tau_con` | 0.55 |
| loss weights `(gen, clin, grd, con, hall, cal)` | `(1.0, .35, .25, .15, .30, .05)` |
| seeds | 42, 123, 2024, 3407, 9999 |
| precision | AMP |
| manuscript train hardware | RTX 4090 24 GB |

## Installation

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -e .
# For a Hugging Face RAD-DINO checkpoint:
pip install -e '.[rad_dino]'
```

## Data manifest format

The training code does not redistribute MIMIC-CXR or IU X-Ray images. Create a CSV with one row per study/report:

```text
study_id,split,frontal_path,lateral_path,report,findings_json,region_targets_json,support_targets_json
```

- `lateral_path` may be empty.
- `findings_json`: mapping from finding name to state (`present`, `uncertain`, `absent`, or `unknown`).
- `region_targets_json`: mapping from finding name to one or more 24-slot region names.
- `support_targets_json`: optional binary/calibration target per finding.


## Tokenizer

```bash
python scripts/train_sentencepiece.py \
  --manifest data/mimic_manifest.csv \
  --output-dir artifacts/tokenizer \
  --vocab-size 12000
```

## Training

```bash
python scripts/train.py \
  --config configs/mimic.yaml \
  --tokenizer artifacts/tokenizer/aegis.model \
  --output-dir runs/mimic_seed42 \
  --seed 42
```

Run the five manuscript seeds with:

```bash
python scripts/run_five_seeds.py \
  --config configs/mimic.yaml \
  --tokenizer artifacts/tokenizer/aegis.model \
  --output-root runs/mimic
```

## Inference

```bash
python scripts/infer.py \
  --config configs/mimic.yaml \
  --checkpoint runs/mimic_seed42/best.pt \
  --tokenizer artifacts/tokenizer/aegis.model \
  --manifest data/mimic_manifest.csv \
  --split test \
  --output predictions.jsonl
```

## Evaluation

For manuscript, first run the same frozen CheXbert/RadGraph/parser stack on generated and reference reports, then save normalized structures as JSONL. The repository includes a lightweight lexicon parser for smoke tests only; it is **not** a substitute for the frozen manuscript evaluation stack.

```bash
python scripts/evaluate.py \
  --predictions predictions.jsonl \
  --references references.jsonl \
  --structured-pred pred_structured.jsonl \
  --structured-ref ref_structured.jsonl \
  --config configs/mimic.yaml \
  --output metrics.json
```

Paired bootstrap:

```bash
python scripts/bootstrap_compare.py \
  --a aegis_per_study.jsonl \
  --b dart_per_study.jsonl \
  --metrics ufr supported_recall grounding_f1 c_f1 \
  --n-bootstrap 5000
```

Threshold sweep:

```bash
python scripts/threshold_sweep.py \
  --candidate-scores validation_candidate_scores.jsonl \
  --output threshold_sweep.csv
```

## Smoke test

```bash
pytest -q
python scripts/smoke_test.py
```

The smoke test uses randomly generated tensors and validates the shapes, losses, and controller logic without any patient data.
