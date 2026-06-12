from typing import Tuple

import torch
from torch import Tensor, nn
from torchvision.models import resnet18

from .channels import AWGNChannel


class ConvDeepJSCC(nn.Module):
    """Small CIFAR DeepJSCC baseline with an explicit channel-latent API."""

    def __init__(self, latent_channels: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2, padding=2),
            nn.PReLU(64),
            nn.Conv2d(64, 96, 5, stride=2, padding=2),
            nn.PReLU(96),
            nn.Conv2d(96, latent_channels, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 96, 3, padding=1),
            nn.PReLU(96),
            nn.ConvTranspose2d(
                96, 64, 5, stride=2, padding=2, output_padding=1
            ),
            nn.PReLU(64),
            nn.ConvTranspose2d(
                64, 3, 5, stride=2, padding=2, output_padding=1
            ),
            nn.Sigmoid(),
        )
        self.channel = AWGNChannel()

    @staticmethod
    def power_normalize(z: Tensor) -> Tensor:
        dims = tuple(range(1, z.ndim))
        rms = z.square().mean(dim=dims, keepdim=True).sqrt()
        return z / rms.clamp_min(1e-8)

    def encode_for_channel(self, x: Tensor) -> Tensor:
        return self.power_normalize(self.encoder(x))

    def decode_from_channel(self, received: Tensor) -> Tensor:
        return self.decoder(received)

    def forward(self, x: Tensor, snr_db: float) -> Tensor:
        z = self.encode_for_channel(x)
        received = self.channel(z, snr_db)
        return self.decode_from_channel(received)


class CifarResNet18(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        network = resnet18(weights=None, num_classes=num_classes)
        network.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        network.maxpool = nn.Identity()
        self.network = network
        self.register_buffer(
            "mean", torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network((x - self.mean) / self.std)


class TinySemanticClassifier(nn.Module):
    """Fast classifier used only by tests and the no-checkpoint smoke test."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Linear(16 * 4 * 4, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x).flatten(1))


def load_model_state(model: nn.Module, checkpoint_path: str) -> Tuple[list, list]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)

