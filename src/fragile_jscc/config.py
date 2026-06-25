from dataclasses import dataclass
import json
from pathlib import Path
from typing import List


@dataclass
class ExperimentConfig:
    experiment_id: str
    seed: int
    dataset: str
    data_root: str
    num_classes: int
    image_size: int
    latent_channels: int
    channel_group_size: int
    granularity: str
    snr_db: List[float]
    oracle_mc_samples: int
    topk_ratios: List[float]
    batch_size: int
    num_workers: int
    max_samples: int
    classifier_checkpoint: str
    jscc_checkpoint: str
    ranking_seed: int = 2007
    heldout_seed: int = 3007
    corruption_seed: int = 4007
    bootstrap_seed: int = 5007
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        return cls(**values)
