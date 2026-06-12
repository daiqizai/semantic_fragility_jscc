import torch
import torch.nn.functional as F
from torch import Tensor


def semantic_kl(reference_logits: Tensor, perturbed_logits: Tensor) -> Tensor:
    """Return KL(reference || perturbed) independently for each sample."""
    reference = F.softmax(reference_logits, dim=-1)
    divergence = F.kl_div(
        F.log_softmax(perturbed_logits, dim=-1),
        reference,
        reduction="none",
    ).sum(dim=-1)
    return divergence.clamp_min(0.0)


def prediction_consistency(
    reference_logits: Tensor, perturbed_logits: Tensor
) -> Tensor:
    return reference_logits.argmax(dim=-1).eq(perturbed_logits.argmax(dim=-1))


def freeze(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
