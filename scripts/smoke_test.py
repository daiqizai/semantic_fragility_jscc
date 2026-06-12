#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fragile_jscc.evaluation import evaluate_topk_corruption
from fragile_jscc.models import ConvDeepJSCC, TinySemanticClassifier
from fragile_jscc.scoring import all_ranking_scores
from fragile_jscc.semantic import freeze
from fragile_jscc.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_seed(7)
    device = torch.device(args.device)
    jscc = ConvDeepJSCC(latent_channels=8).to(device)
    classifier = TinySemanticClassifier(num_classes=10).to(device)
    freeze(jscc)
    freeze(classifier)

    images = torch.rand(2, 3, 32, 32, device=device)
    labels = classifier(images).argmax(dim=-1)
    scores = all_ranking_scores(
        jscc=jscc,
        classifier=classifier,
        images=images,
        snr_db=5.0,
        granularity="channel",
        channel_group_size=2,
        oracle_mc_samples=2,
    )

    results = {}
    for name, method_scores in scores.items():
        results[name] = {
            "score_shape": list(method_scores.shape),
            "topk": evaluate_topk_corruption(
                jscc=jscc,
                classifier=classifier,
                images=images,
                labels=labels,
                scores=method_scores,
                snr_db=5.0,
                topk_ratio=0.5,
                granularity="channel",
                channel_group_size=2,
            ),
        }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

