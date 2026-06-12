#!/usr/bin/env python3
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fragile_jscc.config import ExperimentConfig
from fragile_jscc.data import cifar_loaders
from fragile_jscc.evaluation import evaluate_topk_corruption, spearman_correlation
from fragile_jscc.experiment import ExperimentRun
from fragile_jscc.models import CifarResNet18, ConvDeepJSCC, load_model_state
from fragile_jscc.scoring import all_ranking_scores, oracle_fragility_scores
from fragile_jscc.semantic import freeze
from fragile_jscc.utils import set_seed


def average_metrics(sums, count):
    return {key: value / count for key, value in sums.items()}


def evaluate(cfg, run: ExperimentRun, device: torch.device) -> None:
    set_seed(cfg.seed)

    jscc = ConvDeepJSCC(cfg.latent_channels)
    classifier = CifarResNet18(cfg.num_classes)
    for model, checkpoint in [
        (jscc, cfg.jscc_checkpoint),
        (classifier, cfg.classifier_checkpoint),
    ]:
        path = ROOT / checkpoint
        if not path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: {path}. Run the matching training script first."
            )
        missing, unexpected = load_model_state(model, str(path))
        if missing or unexpected:
            print(
                f"checkpoint warning for {path.name}: "
                f"missing={missing}, unexpected={unexpected}"
            )
    jscc.to(device)
    classifier.to(device)
    freeze(jscc)
    freeze(classifier)

    _, test_loader = cifar_loaders(
        cfg.dataset,
        cfg.data_root,
        cfg.batch_size,
        cfg.num_workers,
        download=False,
    )
    results = {}
    for snr_db in cfg.snr_db:
        aggregate = {}
        seen = 0
        progress = tqdm(test_loader, desc=f"ranking snr={snr_db:g}")
        for images, labels in progress:
            remaining = cfg.max_samples - seen
            if remaining <= 0:
                break
            images = images[:remaining].to(device)
            labels = labels[:remaining].to(device)
            scores = all_ranking_scores(
                jscc,
                classifier,
                images,
                snr_db,
                cfg.granularity,
                cfg.channel_group_size,
                cfg.oracle_mc_samples,
            )
            with torch.no_grad():
                z = jscc.encode_for_channel(images)
                validation_base_noise = jscc.channel.sample_noise(z, snr_db)
                validation_alternate_noise = jscc.channel.sample_noise(z, snr_db)
            heldout_singleton_effect = oracle_fragility_scores(
                jscc,
                classifier,
                images,
                snr_db,
                cfg.granularity,
                cfg.channel_group_size,
                max(1, cfg.oracle_mc_samples),
            )
            for method, method_scores in scores.items():
                rank_correlation = spearman_correlation(
                    method_scores, heldout_singleton_effect
                )
                for ratio in cfg.topk_ratios:
                    key = f"{method}/topk={ratio:g}"
                    metrics = evaluate_topk_corruption(
                        jscc,
                        classifier,
                        images,
                        labels,
                        method_scores,
                        snr_db,
                        ratio,
                        cfg.granularity,
                        cfg.channel_group_size,
                        base_noise=validation_base_noise,
                        alternate_noise=validation_alternate_noise,
                    )
                    metrics["spearman_to_heldout_fragility"] = rank_correlation
                    bucket = aggregate.setdefault(
                        key, {name: 0.0 for name in metrics}
                    )
                    for name, value in metrics.items():
                        bucket[name] += value * images.shape[0]
            seen += images.shape[0]
        results[str(snr_db)] = {
            key: average_metrics(metrics, seen)
            for key, metrics in aggregate.items()
        }

    output_path = run.output_dir / "ranking_results.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    for snr_db, methods in results.items():
        for method_key, metrics in methods.items():
            run.append_metrics(
                {"snr_db": float(snr_db), "method": method_key, **metrics}
            )
    print(f"saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-S2-002_ranking.json"
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config_path = ROOT / args.config
    cfg = ExperimentConfig.load(str(config_path))
    effective_config = asdict(cfg)
    effective_config["runtime_device"] = args.device
    with ExperimentRun(ROOT, config_path, effective_config) as run:
        evaluate(cfg, run, torch.device(args.device))


if __name__ == "__main__":
    main()
