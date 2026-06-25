import math
from typing import Dict, Optional, Sequence

import torch
from torch import Tensor, nn

from .groups import num_groups, selected_group_mask
from .models import ConvDeepJSCC
from .quality import multiscale_ssim, per_image_mse, psnr
from .semantic import semantic_kl


@torch.no_grad()
def evaluate_topk_corruption_samples(
    jscc: ConvDeepJSCC,
    classifier: nn.Module,
    images: Tensor,
    labels: Tensor,
    scores: Tensor,
    snr_db: float,
    topk_ratio: float,
    granularity: str = "channel",
    channel_group_size: int = 1,
    generator: Optional[torch.Generator] = None,
    base_noise: Optional[Tensor] = None,
    alternate_noise: Optional[Tensor] = None,
    topk_count: Optional[int] = None,
) -> Dict[str, Tensor]:
    """Evaluate rankings with fresh AWGN draws not used to build the scores."""
    z = jscc.encode_for_channel(images)
    groups = num_groups(tuple(z.shape), granularity, channel_group_size)
    if topk_count is None:
        k = max(1, min(groups, int(math.ceil(groups * topk_ratio))))
    else:
        k = max(1, min(groups, int(topk_count)))
    selected = scores.topk(k, dim=1, largest=True).indices
    mask = selected_group_mask(
        z, selected, granularity, channel_group_size
    )

    if base_noise is None:
        base_noise = jscc.channel.sample_noise(z, snr_db, generator)
    if alternate_noise is None:
        alternate_noise = jscc.channel.sample_noise(z, snr_db, generator)
    if base_noise.shape != z.shape or alternate_noise.shape != z.shape:
        raise ValueError("Held-out noise tensors must match the latent shape")
    baseline_logits = classifier(
        jscc.decode_from_channel(z + base_noise)
    )
    corrupted_received = z + torch.where(
        mask, alternate_noise, base_noise
    )
    corrupted_logits = classifier(
        jscc.decode_from_channel(corrupted_received)
    )

    baseline_prediction = baseline_logits.argmax(dim=-1)
    corrupted_prediction = corrupted_logits.argmax(dim=-1)
    baseline_correct = baseline_prediction.eq(labels)
    corrupted_correct = corrupted_prediction.eq(labels)
    correct_to_wrong = baseline_correct & ~corrupted_correct

    return {
        "k": torch.full(
            (images.shape[0],), float(k), device=images.device
        ),
        "selected_fraction": torch.full(
            (images.shape[0],), float(k) / groups, device=images.device
        ),
        "baseline_accuracy": baseline_correct.float(),
        "corrupted_accuracy": corrupted_correct.float(),
        "accuracy_drop": (
            baseline_correct.float() - corrupted_correct.float()
        ),
        "prediction_consistency": baseline_prediction.eq(
            corrupted_prediction
        ).float(),
        "semantic_failure_rate": correct_to_wrong.float(),
        "mean_semantic_kl": semantic_kl(
            baseline_logits, corrupted_logits
        ),
    }


@torch.no_grad()
def evaluate_topk_corruption(
    jscc: ConvDeepJSCC,
    classifier: nn.Module,
    images: Tensor,
    labels: Tensor,
    scores: Tensor,
    snr_db: float,
    topk_ratio: float,
    granularity: str = "channel",
    channel_group_size: int = 1,
    generator: Optional[torch.Generator] = None,
    base_noise: Optional[Tensor] = None,
    alternate_noise: Optional[Tensor] = None,
) -> Dict[str, float]:
    samples = evaluate_topk_corruption_samples(
        jscc,
        classifier,
        images,
        labels,
        scores,
        snr_db,
        topk_ratio,
        granularity,
        channel_group_size,
        generator,
        base_noise,
        alternate_noise,
    )
    return {
        name: values.float().mean().item()
        for name, values in samples.items()
    }


def _average_ranks(values: Tensor) -> Tensor:
    order = values.argsort()
    ranks = torch.empty_like(values, dtype=torch.float32)
    sorted_values = values[order]
    start = 0
    while start < values.numel():
        stop = start + 1
        while (
            stop < values.numel()
            and sorted_values[stop].item() == sorted_values[start].item()
        ):
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def rank_correlations(
    predicted: Tensor, target: Tensor
) -> Dict[str, Tensor]:
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("predicted and target must both have shape [B, G]")
    spearman_values = []
    kendall_values = []
    for prediction_row, target_row in zip(predicted, target):
        prediction_cpu = prediction_row.detach().cpu()
        target_cpu = target_row.detach().cpu()
        x = _average_ranks(prediction_cpu)
        y = _average_ranks(target_cpu)
        centered_x = x - x.mean()
        centered_y = y - y.mean()
        denominator = (
            centered_x.square().sum().sqrt()
            * centered_y.square().sum().sqrt()
        )
        if denominator.item() > 0:
            spearman = (centered_x * centered_y).sum() / denominator
        else:
            spearman = torch.tensor(float("nan"))
        spearman_values.append(spearman)

        concordant = 0
        discordant = 0
        ties_x = 0
        ties_y = 0
        for first in range(prediction_cpu.numel()):
            for second in range(first + 1, prediction_cpu.numel()):
                delta_x = prediction_cpu[first].item() - prediction_cpu[
                    second
                ].item()
                delta_y = target_cpu[first].item() - target_cpu[second].item()
                if delta_x == 0:
                    ties_x += 1
                if delta_y == 0:
                    ties_y += 1
                product = delta_x * delta_y
                if product > 0:
                    concordant += 1
                elif product < 0:
                    discordant += 1
        total_pairs = prediction_cpu.numel() * (
            prediction_cpu.numel() - 1
        ) / 2
        kendall_denominator = math.sqrt(
            (total_pairs - ties_x) * (total_pairs - ties_y)
        )
        kendall_values.append(
            torch.tensor(
                (concordant - discordant) / kendall_denominator
                if kendall_denominator > 0
                else float("nan")
            )
        )
    return {
        "spearman": torch.stack(spearman_values),
        "kendall": torch.stack(kendall_values),
    }


def _finite_mean(values: Tensor) -> float:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return float("nan")
    return finite.float().mean().item()


def spearman_correlation(predicted: Tensor, target: Tensor) -> float:
    return _finite_mean(rank_correlations(predicted, target)["spearman"])


def kendall_correlation(predicted: Tensor, target: Tensor) -> float:
    return _finite_mean(rank_correlations(predicted, target)["kendall"])


def bootstrap_mean_ci(
    values: Tensor,
    bootstrap_samples: int,
    confidence_level: float,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, float]:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    values = values.detach().cpu().double().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "num_samples": 0,
        }
    mean = values.mean().item()
    if bootstrap_samples <= 0 or values.numel() == 1:
        return {
            "mean": mean,
            "ci_low": mean,
            "ci_high": mean,
            "num_samples": int(values.numel()),
        }
    indices = torch.randint(
        values.numel(),
        (bootstrap_samples, values.numel()),
        generator=generator,
    )
    bootstrap_means = values[indices].mean(dim=1)
    tail = (1.0 - confidence_level) / 2.0
    bounds = torch.quantile(
        bootstrap_means,
        torch.tensor(
            [tail, 1.0 - tail], dtype=bootstrap_means.dtype
        ),
    )
    return {
        "mean": mean,
        "ci_low": bounds[0].item(),
        "ci_high": bounds[1].item(),
        "num_samples": int(values.numel()),
    }


def deletion_auc(curves: Tensor, fractions: Sequence[float]) -> Tensor:
    if curves.ndim != 2 or curves.shape[1] != len(fractions):
        raise ValueError("curves must have shape [N, len(fractions)]")
    x = torch.tensor(fractions, dtype=curves.dtype, device=curves.device)
    if x.numel() < 2 or not bool(torch.all(x[1:] > x[:-1])):
        raise ValueError("fractions must be strictly increasing")
    if x[0].item() != 0.0 or x[-1].item() != 1.0:
        raise ValueError("fractions must span [0, 1]")
    return torch.trapz(curves, x=x, dim=1)


@torch.no_grad()
def reconstruction_metrics(
    classifier: nn.Module,
    perceptual_metric: nn.Module,
    images: Tensor,
    reconstruction: Tensor,
    labels: Tensor,
    ms_ssim_weights,
    ms_ssim_window_size: int,
    ms_ssim_window_sigma: float,
) -> Dict[str, Tensor]:
    if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
        raise ValueError("labels must have shape [B]")

    clean_logits = classifier(images)
    reconstruction_logits = classifier(reconstruction)
    clean_prediction = clean_logits.argmax(dim=-1)
    reconstruction_prediction = reconstruction_logits.argmax(dim=-1)
    clean_correct = clean_prediction.eq(labels)
    reconstruction_correct = reconstruction_prediction.eq(labels)
    perceptual_distance = perceptual_metric(
        images, reconstruction, normalize=True
    ).reshape(images.shape[0], -1).mean(dim=1)

    return {
        "mse": per_image_mse(images, reconstruction),
        "psnr_db": psnr(images, reconstruction),
        "ms_ssim": multiscale_ssim(
            images,
            reconstruction,
            weights=ms_ssim_weights,
            window_size=ms_ssim_window_size,
            window_sigma=ms_ssim_window_sigma,
        ),
        "lpips": perceptual_distance,
        "clean_accuracy": clean_correct.float(),
        "reconstruction_accuracy": reconstruction_correct.float(),
        "prediction_consistency": clean_prediction.eq(
            reconstruction_prediction
        ).float(),
        "semantic_failure_rate": (
            clean_correct & ~reconstruction_correct
        ).float(),
        "semantic_kl": semantic_kl(
            clean_logits, reconstruction_logits
        ),
    }
