# AEGIS-Rad: A chest-radiograph report-generation framework. 

## Method overview

For each candidate finding, AEGIS-Rad maintains a four-way report-semantic state

`present`, `absent`, `uncertain`, `unmentioned`

and applies a separate three-way decoder action

`emit`, `hedge`, `suppress`.

Clinical `absent` is a semantic state and is not equivalent to `suppress`. A supported negative finding may be emitted as an explicit negative statement, whereas an inadequately supported positive proposal is suppressed and remains `unmentioned`.

The implementation contains:

- shared RAD-DINO visual encoding for manuscript experiments;
- explicit frontal/lateral missing-view handling;
- 24 fixed thoracic anatomical query slots;
- 14 candidate findings;
- two multi-head relation-aware finding–anatomy graph layers;
- confidence-controlled global/regional fusion with anatomy-uncertainty attenuation;
- a 6-layer Transformer report decoder with hidden size 768 and 12 attention heads;
- four semantic states and three independent realization actions;
- six training objectives for report generation, clinical labels, grounding, contradiction control, unsupported-positive control, and calibration;
- validation-only operating-point selection;
- beam-search inference with beam size 3;
- five-seed experiment support;
- study-level paired bootstrap evaluation with 5,000 resamples.

## Repository structure

```text
AEGIS_Rad/
├── configs/                 Experiment, ontology, and graph configuration
├── data/                    Dataset loading, preprocessing, and tokenization
├── evaluation/              Structured evaluation and bootstrap utilities
├── models/                  AEGIS-Rad model components
├── scripts/                 Training, inference, evaluation, and data utilities
├── tests/                   Model and manuscript-contract tests
├── config.py                Typed experiment configuration
├── factory.py               Model and ontology construction
├── generation.py            Beam search and selective report control
├── losses.py                Multi-objective training losses
├── ontology.py              Finding and anatomical ontology loader
├── training.py              Training loop and checkpointing
├── pyproject.toml           Package metadata
└── requirements.txt         Runtime dependencies
```

## Installation

Python 3.10 or later is recommended.

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

`configs/base.yaml` uses the lightweight convolutional encoder for local unit tests and smoke tests without downloading external weights. The configurations, `configs/mimic.yaml` and `configs/iu.yaml`, select the Hugging Face RAD-DINO backend with `microsoft/rad-dino`.

## Data preparation

Each manifest contains one row per radiographic study/report with the following schema:

```text
study_id,split,frontal_path,lateral_path,report,findings_json,region_targets_json,support_targets_json
```

`findings_json` uses the four report-semantic states. A finding that is not asserted in the manuscript report is `unmentioned`, not `absent`.

Example:

```json
{
  "Cardiomegaly": "present",
  "Pleural Effusion": "absent",
  "Atelectasis": "uncertain"
}
```

For MIMIC-CXR-JPG, use the official patient-disjoint split and keep the test partition isolated from model selection, early stopping, threshold selection, and loss-weight selection. IU X-Ray/Open-I should be evaluated with the frozen grouped split used for the experiment rather than generating a new random split.

Validate a prepared manifest with:

```bash
python scripts/check_manifest.py --manifest data/mimic_manifest.csv
```

## Manuscript configuration

| Parameter | Setting |
|---|---|
| Input resolution | 384 × 384 |
| Anatomical query slots | 24 |
| Candidate findings | 14 |
| Graph layers / heads | 2 / 8 |
| Decoder | 6 layers, hidden 768, 12 heads |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | 1 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| Learning-rate schedule | Cosine with warm-up |
| Maximum training horizon | 120 epochs |
| Dropout | 0.15 |
| Gradient clipping | 1.0 |
| Beam size | 3 |
| Support threshold, `tau_e` | 0.62 |
| Contradiction threshold, `tau_con` | 0.55 |
| Loss weights `(gen, clin, grd, con, hall, cal)` | `(1.0, 0.35, 0.25, 0.15, 0.30, 0.05)` |
| Training seeds | 42, 123, 2024, 3407, 9999 |
| Precision | Automatic mixed precision |
| Reported training hardware | NVIDIA RTX 4090 24 GB |

## Tokenizer

The report decoder uses a SentencePiece vocabulary of 12,000 tokens and a maximum report length of 256 tokens.

```bash
python scripts/train_sentencepiece.py \
  --manifest data/mimic_manifest.csv \
  --output-dir artifacts/tokenizer \
  --vocab-size 12000
```

## Training

A single MIMIC-CXR run can be launched with:

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

Training uses AdamW, warm-up followed by cosine decay, automatic mixed precision, validation-based early stopping, and gradient clipping according to the manuscript configuration.

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

Inference records the predicted semantic state and decoder action separately. Support and contradiction thresholds are selected using validation data and remain frozen during test evaluation.

## Evaluation protocol

The primary manuscript endpoints are:

- clinical efficacy F1 (`C-F1`);
- reference-conditioned unsupported-finding rate (`UFR`);
- supported-positive recall;
- report-derived finding–anatomy Grounding-F1.

Secondary diagnostics include contradiction rate, calibration error, critical omission rate, report length, and positive findings per study.

Direct contemporary comparisons must use the same frozen test partition, CheXbert implementation, RadGraph implementation, report parser, finding–region mapping, UFR definition, and metric implementation for every compared method. The lightweight structured utilities in this repository support format validation and reproducibility checks; they are not intended to replace the frozen CheXbert/RadGraph evaluation stack used for manuscript reporting.

Paired comparisons use 5,000 study-level bootstrap resamples. When predictions from multiple training seeds are combined, uncertainty can be estimated hierarchically across seeds and studies. Where formal hypothesis tests are reported across the four prespecified primary endpoints, Holm adjustment is applied.

Matched-UFR and matched-supported-recall operating points are selected on validation data and then frozen before evaluation on the test set.

Example bootstrap comparison:

```bash
python scripts/bootstrap_compare.py \
  --a aegis_per_study.jsonl \
  --b comparator_per_study.jsonl \
  --metrics ufr supported_recall grounding_f1 c_f1 \
  --n-bootstrap 5000
```

Example threshold sweep:

```bash
python scripts/threshold_sweep.py \
  --candidate-scores validation_candidate_scores.jsonl \
  --output threshold_sweep.csv
```

## Reproducibility record

For each experiment, retain the configuration file, seed, software environment, CUDA/PyTorch versions, GPU model, validation history, selected checkpoint, frozen operating thresholds, generated reports, and per-study evaluation outputs. For reproduced contemporary methods, retain implementation provenance, checkpoint or retraining status, decoding configuration, and the invocation of the common frozen evaluation stack.

Capture the local software environment with:

```bash
python scripts/capture_environment.py --output environment.json
```

The model configuration exposes `encoder_revision` so a specific RAD-DINO repository revision can be pinned when reproducing an archived experiment. Leaving it unset loads the configured published model identifier through Hugging Face Transformers.

## Verification

Run the automated verification suite with:

```bash
pytest -q
python scripts/smoke_test.py
```

The tests verify the implementation, including 24 anatomical regions, 14 candidate findings, four semantic states, three decoder actions, five training seeds, manuscript loss weights and thresholds, RAD-DINO selection for manuscript configurations, and separation of `ABSENT` from `SUPPRESS`.

## Contact 

Rezaul Haue: rezaulh603@gmail.com 

Abdullah Al Sakib  sakibabdulla685@gmail.com
## License

This repository is distributed under the terms provided in `LICENSE`.
