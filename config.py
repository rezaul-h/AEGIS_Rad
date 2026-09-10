from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

@dataclass
class ModelConfig:
    image_size: int = 384
    in_channels: int = 1
    encoder_backend: str = "conv"
    encoder_name: str = "microsoft/rad-dino"
    encoder_revision: str | None = None
    hidden_dim: int = 768
    anatomy_slots: int = 24
    graph_layers: int = 2
    graph_heads: int = 8
    decoder_layers: int = 6
    decoder_heads: int = 12
    ff_dim: int = 3072
    dropout: float = 0.15
    max_report_tokens: int = 256
    vocab_size: int = 12000
    num_findings: int = 14
    num_semantic_states: int = 4

@dataclass
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 120
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    amp: bool = True
    num_workers: int = 4
    warmup_epochs: int = 5
    label_smoothing: float = 0.1
    early_stopping_patience: int = 15
    seeds: list[int] = field(default_factory=lambda: [42, 123, 2024, 3407, 9999])

@dataclass
class ControllerConfig:
    tau_e: float = 0.62
    tau_con: float = 0.55
    tau_suppress: float = 0.35
    beam_size: int = 3
    max_sentences: int = 12
    max_sentence_tokens: int = 48

@dataclass
class LossConfig:
    gen: float = 1.0
    clin: float = 0.35
    grd: float = 0.25
    con: float = 0.15
    hall: float = 0.30
    cal: float = 0.05
    grounding_contrastive_temperature: float = 0.07
    grounding_assignment_weight: float = 1.0
    grounding_contrastive_weight: float = 1.0

@dataclass
class DataConfig:
    manifest: str = "data/manifest.csv"
    image_size: int = 384
    max_report_tokens: int = 256
    percentile_low: float = 0.5
    percentile_high: float = 99.5
    rotation_deg: float = 5.0
    translate_frac: float = 0.03
    contrast_jitter: float = 0.10
    no_horizontal_flip: bool = True

@dataclass
class EvalConfig:
    n_bootstrap: int = 5000
    bootstrap_seed: int = 2026
    critical_findings: list[str] = field(default_factory=list)
    primary_metrics: list[str] = field(default_factory=lambda: ["ufr", "supported_recall", "c_f1", "grounding_f1"])

@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    regions_path: str = "configs/regions.yaml"
    findings_path: str = "configs/findings.yaml"
    graph_path: str = "configs/graph.yaml"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            model=ModelConfig(**d.get("model", {})),
            training=TrainingConfig(**d.get("training", {})),
            controller=ControllerConfig(**d.get("controller", {})),
            loss=LossConfig(**d.get("loss", {})),
            data=DataConfig(**d.get("data", {})),
            eval=EvalConfig(**d.get("eval", {})),
            regions_path=d.get("regions_path", "configs/regions.yaml"),
            findings_path=d.get("findings_path", "configs/findings.yaml"),
            graph_path=d.get("graph_path", "configs/graph.yaml"),
        )

def _deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out

def load_config(path: str | Path, base_path: str | Path | None = None) -> ExperimentConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    if base_path is None and "base" in data:
        base_path = path.parent / data.pop("base")
    if base_path is not None:
        base = yaml.safe_load(Path(base_path).read_text()) or {}
        data = _deep_update(base, data)
    return ExperimentConfig.from_dict(data)
