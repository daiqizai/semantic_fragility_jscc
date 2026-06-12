import math
from typing import Optional

import torch
from torch import Tensor, nn


class AWGNChannel(nn.Module):
    """Real-valued AWGN channel with per-sample signal-power calibration."""

    @staticmethod
    def noise_std(x: Tensor, snr_db: float) -> Tensor:
        dims = tuple(range(1, x.ndim))
        signal_power = x.detach().square().mean(dim=dims, keepdim=True)
        snr_linear = math.pow(10.0, float(snr_db) / 10.0)
        return torch.sqrt(signal_power / snr_linear).clamp_min(1e-12)

    def sample_noise(
        self,
        x: Tensor,
        snr_db: float,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        noise = torch.randn(
            x.shape,
            dtype=x.dtype,
            device=x.device,
            generator=generator,
        )
        return noise * self.noise_std(x, snr_db)

    def forward(
        self,
        x: Tensor,
        snr_db: float,
        noise: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        if noise is None:
            noise = self.sample_noise(x, snr_db, generator)
        if noise.shape != x.shape:
            raise ValueError("Explicit AWGN noise must have the same shape as x")
        return x + noise

