from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .groups import group_mask, num_groups, reduce_by_group
from .models import ConvDeepJSCC
from .semantic import semantic_kl


@torch.no_grad()
def oracle_fragility_scores(
    jscc: ConvDeepJSCC,
    classifier: nn.Module,
    images: Tensor,
    snr_db: float,
    granularity: str = "channel",
    channel_group_size: int = 1,
    mc_samples: int = 4,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Estimate group risk by replacing its AWGN realization and measuring KL."""
    z = jscc.encode_for_channel(images)
    group_count = num_groups(
        tuple(z.shape), granularity, channel_group_size
    )
    scores = z.new_zeros((z.shape[0], group_count))

    for _ in range(mc_samples):
        base_noise = jscc.channel.sample_noise(z, snr_db, generator)
        alternate_noise = jscc.channel.sample_noise(z, snr_db, generator)
        baseline_logits = classifier(
            jscc.decode_from_channel(z + base_noise)
        )

        for group_index in range(group_count):
            mask = group_mask(
                z, group_index, granularity, channel_group_size
            )
            received = z + torch.where(
                mask, alternate_noise, base_noise
            )
            perturbed_logits = classifier(
                jscc.decode_from_channel(received)
            )
            scores[:, group_index] += semantic_kl(
                baseline_logits, perturbed_logits
            )
    return scores / float(mc_samples)


@torch.no_grad()
def activation_saliency_scores(
    jscc: ConvDeepJSCC,
    images: Tensor,
    granularity: str = "channel",
    channel_group_size: int = 1,
) -> Tensor:
    z = jscc.encode_for_channel(images)
    return reduce_by_group(z.abs(), granularity, channel_group_size)


def gradient_importance_scores(
    jscc: ConvDeepJSCC,
    classifier: nn.Module,
    images: Tensor,
    snr_db: float,
    granularity: str = "channel",
    channel_group_size: int = 1,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    with torch.no_grad():
        pseudo_targets = classifier(images).argmax(dim=-1)
        encoded = jscc.encode_for_channel(images)
        noise = jscc.channel.sample_noise(encoded, snr_db, generator)

    z = encoded.detach().requires_grad_(True)
    logits = classifier(jscc.decode_from_channel(z + noise))
    loss = F.cross_entropy(logits, pseudo_targets, reduction="sum")
    gradient = torch.autograd.grad(loss, z)[0]
    importance = (gradient * z).abs()
    return reduce_by_group(
        importance, granularity, channel_group_size
    ).detach()


def random_scores(
    jscc: ConvDeepJSCC,
    images: Tensor,
    granularity: str = "channel",
    channel_group_size: int = 1,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    with torch.no_grad():
        z = jscc.encode_for_channel(images)
    groups = num_groups(tuple(z.shape), granularity, channel_group_size)
    return torch.rand(
        (z.shape[0], groups),
        device=z.device,
        dtype=z.dtype,
        generator=generator,
    )


def all_ranking_scores(
    jscc: ConvDeepJSCC,
    classifier: nn.Module,
    images: Tensor,
    snr_db: float,
    granularity: str,
    channel_group_size: int,
    oracle_mc_samples: int,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Tensor]:
    return {
        "random": random_scores(
            jscc,
            images,
            granularity,
            channel_group_size,
            generator,
        ),
        "activation_saliency": activation_saliency_scores(
            jscc, images, granularity, channel_group_size
        ),
        "gradient_x_activation": gradient_importance_scores(
            jscc,
            classifier,
            images,
            snr_db,
            granularity,
            channel_group_size,
            generator,
        ),
        "semantic_fragility": oracle_fragility_scores(
            jscc,
            classifier,
            images,
            snr_db,
            granularity,
            channel_group_size,
            oracle_mc_samples,
            generator,
        ),
    }

