from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


CIFAR_MS_SSIM_WEIGHTS = (
    0.05168435625288417,
    0.3294877711121366,
    0.34621596677434235,
    0.2726119058606368,
)


def per_image_mse(reference: Tensor, reconstruction: Tensor) -> Tensor:
    if reference.shape != reconstruction.shape or reference.ndim != 4:
        raise ValueError(
            "reference and reconstruction must have matching BCHW shapes"
        )
    return (reference - reconstruction).square().flatten(1).mean(dim=1)


def psnr(
    reference: Tensor,
    reconstruction: Tensor,
    data_range: float = 1.0,
) -> Tensor:
    mse = per_image_mse(reference, reconstruction)
    peak = mse.new_tensor(data_range).square()
    return 10.0 * torch.log10(peak / mse.clamp_min(1e-12))


def _gaussian_kernel(
    channels: int,
    window_size: int,
    sigma: float,
    reference: Tensor,
) -> Tensor:
    coordinates = torch.arange(
        window_size, device=reference.device, dtype=reference.dtype
    )
    coordinates -= window_size // 2
    kernel_1d = torch.exp(-(coordinates.square()) / (2.0 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.expand(channels, 1, window_size, window_size)


def _ssim_components(
    reference: Tensor,
    reconstruction: Tensor,
    data_range: float,
    window_size: int,
    window_sigma: float,
) -> tuple[Tensor, Tensor]:
    kernel = _gaussian_kernel(
        reference.shape[1], window_size, window_sigma, reference
    )

    def filter_image(image: Tensor) -> Tensor:
        return F.conv2d(image, kernel, groups=image.shape[1])

    reference_mean = filter_image(reference)
    reconstruction_mean = filter_image(reconstruction)
    reference_mean_sq = reference_mean.square()
    reconstruction_mean_sq = reconstruction_mean.square()
    mean_product = reference_mean * reconstruction_mean
    reference_variance = filter_image(reference.square()) - reference_mean_sq
    reconstruction_variance = (
        filter_image(reconstruction.square()) - reconstruction_mean_sq
    )
    covariance = filter_image(reference * reconstruction) - mean_product

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    contrast_structure = (2.0 * covariance + c2) / (
        reference_variance + reconstruction_variance + c2
    )
    luminance = (2.0 * mean_product + c1) / (
        reference_mean_sq + reconstruction_mean_sq + c1
    )
    ssim_map = luminance * contrast_structure
    return (
        ssim_map.flatten(2).mean(dim=-1),
        contrast_structure.flatten(2).mean(dim=-1),
    )


def multiscale_ssim(
    reference: Tensor,
    reconstruction: Tensor,
    data_range: float = 1.0,
    weights: Sequence[float] = CIFAR_MS_SSIM_WEIGHTS,
    window_size: int = 3,
    window_sigma: float = 1.5,
) -> Tensor:
    if reference.shape != reconstruction.shape or reference.ndim != 4:
        raise ValueError(
            "reference and reconstruction must have matching BCHW shapes"
        )
    if window_size <= 0 or window_size % 2 != 1:
        raise ValueError("window_size must be a positive odd integer")
    if not weights or any(weight <= 0 for weight in weights):
        raise ValueError("weights must contain positive values")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("weights must sum to one")

    smallest_scale = min(reference.shape[-2:]) // (2 ** (len(weights) - 1))
    if smallest_scale < window_size:
        raise ValueError(
            "image is too small for the requested MS-SSIM scales and window"
        )

    reference_scale = reference
    reconstruction_scale = reconstruction
    contrast_values = []
    for scale_index in range(len(weights)):
        ssim_value, contrast_structure = _ssim_components(
            reference_scale,
            reconstruction_scale,
            data_range,
            window_size,
            window_sigma,
        )
        if scale_index < len(weights) - 1:
            contrast_values.append(torch.relu(contrast_structure))
            padding = [
                size % 2 for size in reference_scale.shape[-2:]
            ]
            reference_scale = F.avg_pool2d(
                reference_scale, kernel_size=2, padding=padding
            )
            reconstruction_scale = F.avg_pool2d(
                reconstruction_scale, kernel_size=2, padding=padding
            )

    components = torch.stack(
        contrast_values + [torch.relu(ssim_value)], dim=0
    )
    weight_tensor = components.new_tensor(weights).view(-1, 1, 1)
    return components.pow(weight_tensor).prod(dim=0).mean(dim=1)


def channel_bandwidth_ratio(source: Tensor, latent: Tensor) -> float:
    if source.ndim != 4 or latent.ndim != 4:
        raise ValueError("source and latent must be BCHW tensors")
    if source.shape[0] != latent.shape[0]:
        raise ValueError("source and latent batch sizes must match")
    source_values = source[0].numel()
    real_channel_uses = latent[0].numel()
    return real_channel_uses / source_values
