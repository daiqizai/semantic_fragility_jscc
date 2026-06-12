#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import lpips
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fragile_jscc.data import cifar_loaders
from fragile_jscc.evaluation import reconstruction_metrics
from fragile_jscc.experiment import ExperimentRun, load_json
from fragile_jscc.models import CifarResNet18, ConvDeepJSCC
from fragile_jscc.quality import channel_bandwidth_ratio
from fragile_jscc.semantic import freeze
from fragile_jscc.utils import resolve_device, set_seed


def mean_metrics(values):
    return {name: value.mean().item() for name, value in values.items()}


def run_dry_run(config, run: ExperimentRun, device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError("The GPU dry-run requires a CUDA device")

    set_seed(config["seed"])
    train_loader, test_loader = cifar_loaders(
        config["dataset"],
        config["data_root"],
        config["batch_size"],
        config["num_workers"],
        download=False,
    )
    train_images, train_labels = next(iter(train_loader))
    train_images = train_images.to(device, non_blocking=True)
    train_labels = train_labels.to(device, non_blocking=True)

    classifier = CifarResNet18(config["num_classes"]).to(device)
    classifier_optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=config["classifier_learning_rate"],
        momentum=config["classifier_momentum"],
    )
    classifier.train()
    classifier_optimizer.zero_grad(set_to_none=True)
    classifier_loss = F.cross_entropy(
        classifier(train_images), train_labels
    )
    classifier_loss.backward()
    classifier_optimizer.step()
    torch.save(
        {
            "model": classifier.state_dict(),
            "dry_run": True,
            "loss": classifier_loss.item(),
        },
        run.checkpoint_dir / "classifier_step.pt",
    )

    jscc = ConvDeepJSCC(config["latent_channels"]).to(device)
    jscc_optimizer = torch.optim.Adam(
        jscc.parameters(), lr=config["jscc_learning_rate"]
    )
    jscc.train()
    jscc_optimizer.zero_grad(set_to_none=True)
    reconstruction = jscc(train_images, config["snr_db"])
    jscc_loss = F.mse_loss(reconstruction, train_images)
    jscc_loss.backward()
    jscc_optimizer.step()
    torch.save(
        {
            "model": jscc.state_dict(),
            "dry_run": True,
            "loss": jscc_loss.item(),
        },
        run.checkpoint_dir / "jscc_step.pt",
    )

    test_images, test_labels = next(iter(test_loader))
    test_images = test_images.to(device, non_blocking=True)
    test_labels = test_labels.to(device, non_blocking=True)
    perceptual_metric = lpips.LPIPS(
        net=config["lpips_net"],
        version=config["lpips_version"],
        verbose=False,
    ).to(device)
    freeze(classifier)
    freeze(jscc)
    freeze(perceptual_metric)
    with torch.no_grad():
        latent = jscc.encode_for_channel(test_images)
        reconstruction = jscc.decode_from_channel(
            jscc.channel(latent, config["snr_db"])
        )
        metrics = mean_metrics(
            reconstruction_metrics(
                classifier,
                perceptual_metric,
                test_images,
                reconstruction,
                test_labels,
                config["ms_ssim_weights"],
                config["ms_ssim_window_size"],
                config["ms_ssim_window_sigma"],
            )
        )
    metrics["cbr_real_channel_uses_per_source_value"] = (
        channel_bandwidth_ratio(test_images, latent)
    )

    torch.cuda.synchronize(device)
    device_properties = torch.cuda.get_device_properties(device)
    summary = {
        "status": "passed",
        "device": str(device),
        "device_name": device_properties.name,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "classifier_train_loss": classifier_loss.item(),
        "jscc_train_mse": jscc_loss.item(),
        "evaluation": metrics,
        "max_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        "checkpoints": [
            "classifier_step.pt",
            "jscc_step.pt",
        ],
    }
    run.append_metrics(summary)
    run.write_json(run.output_dir / "summary.json", summary)
    print(
        f"device={summary['device_name']} "
        f"classifier_loss={classifier_loss.item():.6f} "
        f"jscc_mse={jscc_loss.item():.6f} "
        f"lpips={metrics['lpips']:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-S0-001_gpu_dryrun.json"
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = load_json(config_path)
    config["runtime_device"] = args.device
    device = resolve_device(args.device)
    with ExperimentRun(ROOT, config_path, config) as run:
        run_dry_run(config, run, device)


if __name__ == "__main__":
    main()
