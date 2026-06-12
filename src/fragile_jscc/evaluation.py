import math
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from .groups import num_groups, selected_group_mask
from .models import ConvDeepJSCC
from .quality import multiscale_ssim, per_image_mse, psnr
from .semantic import semantic_kl


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
    """Evaluate rankings with fresh AWGN draws not used to build the scores."""
    z = jscc.encode_for_channel(images)
    groups = num_groups(tuple(z.shape), granularity, channel_group_size)
    k = max(1, min(groups, int(math.ceil(groups * topk_ratio))))
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
        "k": float(k),
        "baseline_accuracy": baseline_correct.float().mean().item(),
        "corrupted_accuracy": corrupted_correct.float().mean().item(),
        "accuracy_drop": (
            baseline_correct.float().mean()
            - corrupted_correct.float().mean()
        ).item(),
        "prediction_consistency": baseline_prediction.eq(
            corrupted_prediction
        ).float().mean().item(),
        "semantic_failure_rate": correct_to_wrong.float().mean().item(),
        "mean_semantic_kl": semantic_kl(
            baseline_logits, corrupted_logits
        ).mean().item(),
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


def spearman_correlation(predicted: Tensor, target: Tensor) -> float:
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("predicted and target must both have shape [B, G]")
    correlations = []
    for prediction_row, target_row in zip(predicted, target):
        x = _average_ranks(prediction_row.detach().cpu())
        y = _average_ranks(target_row.detach().cpu())
        x = x - x.mean()
        y = y - y.mean()
        denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
        if denominator.item() > 0:
            correlations.append((x * y).sum() / denominator)
    if not correlations:
        return float("nan")
    return torch.stack(correlations).mean().item()


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
