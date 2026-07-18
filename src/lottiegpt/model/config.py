from __future__ import annotations

from dataclasses import dataclass

import yaml


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 512
    n_layer: int = 10
    n_head: int = 8
    ffn_dim: int = 2048
    max_seq_len: int = 8192
    dropout: float = 0.0
    tie_embeddings: bool = True
    rope: bool = True
    rope_theta: float = 10000.0

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)
