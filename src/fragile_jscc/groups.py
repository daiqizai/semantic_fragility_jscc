import math
from typing import Tuple

import torch
from torch import Tensor


def num_groups(
    latent_shape: Tuple[int, ...],
    granularity: str,
    channel_group_size: int = 1,
) -> int:
    if len(latent_shape) != 4:
        raise ValueError("Expected a BCHW latent tensor")
    _, channels, height, width = latent_shape
    if granularity == "channel":
        return math.ceil(channels / channel_group_size)
    if granularity == "spatial":
        return height * width
    raise ValueError("granularity must be 'channel' or 'spatial'")


def group_mask(
    latent: Tensor,
    group_index: int,
    granularity: str,
    channel_group_size: int = 1,
) -> Tensor:
    mask = torch.zeros_like(latent, dtype=torch.bool)
    groups = num_groups(tuple(latent.shape), granularity, channel_group_size)
    if group_index < 0 or group_index >= groups:
        raise IndexError("group_index is outside the latent grouping")

    if granularity == "channel":
        start = group_index * channel_group_size
        stop = min(start + channel_group_size, latent.shape[1])
        mask[:, start:stop] = True
    else:
        width = latent.shape[3]
        row, column = divmod(group_index, width)
        mask[:, :, row, column] = True
    return mask


def selected_group_mask(
    latent: Tensor,
    selected: Tensor,
    granularity: str,
    channel_group_size: int = 1,
) -> Tensor:
    if selected.ndim != 2 or selected.shape[0] != latent.shape[0]:
        raise ValueError("selected must have shape [batch, k]")
    mask = torch.zeros_like(latent, dtype=torch.bool)
    for batch_index in range(latent.shape[0]):
        for group_index in selected[batch_index].tolist():
            single = group_mask(
                latent[batch_index : batch_index + 1],
                int(group_index),
                granularity,
                channel_group_size,
            )
            mask[batch_index : batch_index + 1] |= single
    return mask


def reduce_by_group(
    values: Tensor,
    granularity: str,
    channel_group_size: int = 1,
) -> Tensor:
    groups = num_groups(tuple(values.shape), granularity, channel_group_size)
    scores = []
    for group_index in range(groups):
        mask = group_mask(
            values, group_index, granularity, channel_group_size
        )
        selected = values.masked_select(mask).reshape(values.shape[0], -1)
        scores.append(selected.mean(dim=1))
    return torch.stack(scores, dim=1)

