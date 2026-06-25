#!/usr/bin/env python3
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fragile_jscc.config import ExperimentConfig
from fragile_jscc.data import cifar_loaders
from fragile_jscc.evaluation import (
    bootstrap_mean_ci,
    deletion_auc,
    evaluate_topk_corruption_samples,
    rank_correlations,
)
from fragile_jscc.experiment import ExperimentRun
from fragile_jscc.groups import num_groups
from fragile_jscc.models import CifarResNet18, ConvDeepJSCC, load_model_state
from fragile_jscc.scoring import all_ranking_scores, oracle_fragility_scores
from fragile_jscc.semantic import freeze
from fragile_jscc.utils import set_seed


CURVE_METRICS = (
    "baseline_accuracy",
    "corrupted_accuracy",
    "accuracy_drop",
    "prediction_consistency",
    "semantic_failure_rate",
    "mean_semantic_kl",
)
ADVANTAGE_METRICS = (
    "accuracy_drop",
    "semantic_failure_rate",
    "mean_semantic_kl",
    "prediction_inconsistency",
)


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def summarize(
    values: torch.Tensor,
    cfg: ExperimentConfig,
    generator: torch.Generator,
):
    return bootstrap_mean_ci(
        values,
        cfg.bootstrap_samples,
        cfg.confidence_level,
        generator,
    )


def mean_summary(values: torch.Tensor):
    finite = values[torch.isfinite(values)]
    return {
        "mean": finite.float().mean().item() if finite.numel() else float("nan"),
        "num_samples": int(finite.numel()),
    }


def append_metric_row(
    run: ExperimentRun,
    snr_db: float,
    method: str,
    metric_type: str,
    metric_name: str,
    values,
    **extra,
) -> None:
    run.append_metrics(
        {
            "snr_db": snr_db,
            "method": method,
            "metric_type": metric_type,
            "metric_name": metric_name,
            **extra,
            **values,
        }
    )


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
    results = {
        "metadata": {
            "bootstrap_samples": cfg.bootstrap_samples,
            "confidence_level": cfg.confidence_level,
            "corruption_seed": cfg.corruption_seed,
            "deletion_curve": "all group counts from 0 through G",
            "heldout_seed": cfg.heldout_seed,
            "max_samples": cfg.max_samples,
            "oracle_mc_samples": cfg.oracle_mc_samples,
            "ranking_seed": cfg.ranking_seed,
            "topk_ratios": cfg.topk_ratios,
        },
        "snr_results": {},
    }
    per_sample_results = {}
    for snr_index, snr_db in enumerate(cfg.snr_db):
        seed_offset = snr_index * 100000
        ranking_generator = make_generator(
            device, cfg.ranking_seed + seed_offset
        )
        heldout_generator = make_generator(
            device, cfg.heldout_seed + seed_offset
        )
        corruption_generator = make_generator(
            device, cfg.corruption_seed + seed_offset
        )
        bootstrap_generator = torch.Generator()
        bootstrap_generator.manual_seed(cfg.bootstrap_seed + seed_offset)

        aggregate = {}
        seen = 0
        group_count = None
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
                ranking_generator,
            )
            with torch.no_grad():
                z = jscc.encode_for_channel(images)
                current_groups = num_groups(
                    tuple(z.shape),
                    cfg.granularity,
                    cfg.channel_group_size,
                )
                if group_count is None:
                    group_count = current_groups
                elif group_count != current_groups:
                    raise RuntimeError("Group count changed between batches")
                validation_base_noise = jscc.channel.sample_noise(
                    z, snr_db, corruption_generator
                )
                validation_alternate_noise = jscc.channel.sample_noise(
                    z, snr_db, corruption_generator
                )
            heldout_singleton_effect = oracle_fragility_scores(
                jscc,
                classifier,
                images,
                snr_db,
                cfg.granularity,
                cfg.channel_group_size,
                max(1, cfg.oracle_mc_samples),
                heldout_generator,
            )
            for method, method_scores in scores.items():
                method_bucket = aggregate.setdefault(
                    method,
                    {
                        "rank": {"spearman": [], "kendall": []},
                        "curve": {
                            k: {name: [] for name in CURVE_METRICS}
                            for k in range(1, current_groups + 1)
                        },
                    },
                )
                correlations = rank_correlations(
                    method_scores, heldout_singleton_effect
                )
                for name, values in correlations.items():
                    method_bucket["rank"][name].append(values)
                for k in range(1, current_groups + 1):
                    metrics = evaluate_topk_corruption_samples(
                        jscc,
                        classifier,
                        images,
                        labels,
                        method_scores,
                        snr_db,
                        k / current_groups,
                        cfg.granularity,
                        cfg.channel_group_size,
                        base_noise=validation_base_noise,
                        alternate_noise=validation_alternate_noise,
                        topk_count=k,
                    )
                    for name in CURVE_METRICS:
                        method_bucket["curve"][k][name].append(
                            metrics[name].detach().cpu()
                        )
            seen += images.shape[0]
        if seen == 0 or group_count is None:
            raise RuntimeError("Ranking evaluation processed no samples")

        snr_summary = {
            "num_samples": seen,
            "num_groups": group_count,
            "methods": {},
        }
        snr_per_sample = {}
        fractions = [0.0] + [
            k / group_count for k in range(1, group_count + 1)
        ]
        for method, method_bucket in aggregate.items():
            rank_values = {
                name: torch.cat(chunks)
                for name, chunks in method_bucket["rank"].items()
            }
            curve_values = {
                k: {
                    name: torch.cat(chunks)
                    for name, chunks in metrics.items()
                }
                for k, metrics in method_bucket["curve"].items()
            }
            rank_summary = {
                name: summarize(values, cfg, bootstrap_generator)
                for name, values in rank_values.items()
            }
            curve_summary = {
                str(k): {
                    "k": k,
                    "selected_fraction": k / group_count,
                    "metrics": {
                        name: mean_summary(values)
                        for name, values in metrics.items()
                    },
                }
                for k, metrics in curve_values.items()
            }

            zero = torch.zeros(seen)
            auc_curves = {
                "accuracy_drop": torch.stack(
                    [zero]
                    + [
                        curve_values[k]["accuracy_drop"]
                        for k in range(1, group_count + 1)
                    ],
                    dim=1,
                ),
                "semantic_failure_rate": torch.stack(
                    [zero]
                    + [
                        curve_values[k]["semantic_failure_rate"]
                        for k in range(1, group_count + 1)
                    ],
                    dim=1,
                ),
                "semantic_kl": torch.stack(
                    [zero]
                    + [
                        curve_values[k]["mean_semantic_kl"]
                        for k in range(1, group_count + 1)
                    ],
                    dim=1,
                ),
                "prediction_inconsistency": torch.stack(
                    [zero]
                    + [
                        1.0
                        - curve_values[k]["prediction_consistency"]
                        for k in range(1, group_count + 1)
                    ],
                    dim=1,
                ),
            }
            auc_values = {
                name: deletion_auc(curve, fractions)
                for name, curve in auc_curves.items()
            }
            auc_summary = {
                name: summarize(values, cfg, bootstrap_generator)
                for name, values in auc_values.items()
            }
            topk_summary = {}
            for ratio in cfg.topk_ratios:
                k = max(
                    1,
                    min(group_count, int(math.ceil(group_count * ratio))),
                )
                topk_summary[f"{ratio:g}"] = {
                    "requested_ratio": ratio,
                    "k": k,
                    "selected_fraction": k / group_count,
                    "metrics": {
                        name: summarize(
                            values, cfg, bootstrap_generator
                        )
                        for name, values in curve_values[k].items()
                    },
                }

            snr_summary["methods"][method] = {
                "rank_correlation_to_heldout_singleton": rank_summary,
                "deletion_auc": auc_summary,
                "deletion_curve": curve_summary,
                "topk": topk_summary,
            }
            snr_per_sample[method] = {
                "rank_correlation_to_heldout_singleton": rank_values,
                "deletion_auc": auc_values,
                "deletion_curve": curve_values,
            }

            for name, values in rank_summary.items():
                append_metric_row(
                    run,
                    snr_db,
                    method,
                    "rank_correlation",
                    name,
                    values,
                )
            for name, values in auc_summary.items():
                append_metric_row(
                    run,
                    snr_db,
                    method,
                    "deletion_auc",
                    name,
                    values,
                )
            for ratio, entry in topk_summary.items():
                for name, values in entry["metrics"].items():
                    append_metric_row(
                        run,
                        snr_db,
                        method,
                        "topk",
                        name,
                        values,
                        requested_ratio=float(ratio),
                        k=entry["k"],
                        selected_fraction=entry["selected_fraction"],
                    )
        reference_method = "semantic_fragility"
        if reference_method not in snr_per_sample:
            raise RuntimeError("Missing semantic_fragility scores")
        reference = snr_per_sample[reference_method]
        comparisons = {}
        for baseline, baseline_values in snr_per_sample.items():
            if baseline == reference_method:
                continue
            comparison = {
                "rank_correlation": {},
                "deletion_auc": {},
                "topk": {},
            }
            for name in ("spearman", "kendall"):
                difference = (
                    reference[
                        "rank_correlation_to_heldout_singleton"
                    ][name]
                    - baseline_values[
                        "rank_correlation_to_heldout_singleton"
                    ][name]
                )
                values = summarize(
                    difference, cfg, bootstrap_generator
                )
                comparison["rank_correlation"][name] = values
                append_metric_row(
                    run,
                    snr_db,
                    reference_method,
                    "paired_advantage_rank_correlation",
                    name,
                    values,
                    baseline=baseline,
                )
            for name in (
                "accuracy_drop",
                "semantic_failure_rate",
                "semantic_kl",
                "prediction_inconsistency",
            ):
                difference = (
                    reference["deletion_auc"][name]
                    - baseline_values["deletion_auc"][name]
                )
                values = summarize(
                    difference, cfg, bootstrap_generator
                )
                comparison["deletion_auc"][name] = values
                append_metric_row(
                    run,
                    snr_db,
                    reference_method,
                    "paired_advantage_deletion_auc",
                    name,
                    values,
                    baseline=baseline,
                )
            for ratio in cfg.topk_ratios:
                k = max(
                    1,
                    min(group_count, int(math.ceil(group_count * ratio))),
                )
                ratio_values = {}
                for name in ADVANTAGE_METRICS:
                    if name == "prediction_inconsistency":
                        reference_metric = (
                            1.0
                            - reference["deletion_curve"][k][
                                "prediction_consistency"
                            ]
                        )
                        baseline_metric = (
                            1.0
                            - baseline_values["deletion_curve"][k][
                                "prediction_consistency"
                            ]
                        )
                    else:
                        reference_metric = reference[
                            "deletion_curve"
                        ][k][name]
                        baseline_metric = baseline_values[
                            "deletion_curve"
                        ][k][name]
                    values = summarize(
                        reference_metric - baseline_metric,
                        cfg,
                        bootstrap_generator,
                    )
                    ratio_values[name] = values
                    append_metric_row(
                        run,
                        snr_db,
                        reference_method,
                        "paired_advantage_topk",
                        name,
                        values,
                        baseline=baseline,
                        requested_ratio=ratio,
                        k=k,
                        selected_fraction=k / group_count,
                    )
                comparison["topk"][f"{ratio:g}"] = {
                    "requested_ratio": ratio,
                    "k": k,
                    "selected_fraction": k / group_count,
                    "metrics": ratio_values,
                }
            comparisons[baseline] = comparison
        snr_summary[
            "semantic_fragility_paired_advantage"
        ] = comparisons
        results["snr_results"][str(snr_db)] = snr_summary
        per_sample_results[str(snr_db)] = snr_per_sample

    output_path = run.output_dir / "ranking_results.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    per_sample_path = run.output_dir / "ranking_per_sample.pt"
    torch.save(per_sample_results, per_sample_path)
    print(f"saved {output_path}")
    print(f"saved {per_sample_path}")


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
