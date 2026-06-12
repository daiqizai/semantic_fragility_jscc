#!/usr/bin/env python3
import argparse
from pathlib import Path
import random
import sys

import lpips
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fragile_jscc.data import cifar_loaders
from fragile_jscc.evaluation import reconstruction_metrics
from fragile_jscc.experiment import ExperimentRun, load_json
from fragile_jscc.models import CifarResNet18, ConvDeepJSCC, load_model_state
from fragile_jscc.quality import channel_bandwidth_ratio
from fragile_jscc.semantic import freeze
from fragile_jscc.utils import set_seed


def evaluate(
    config,
    run: ExperimentRun,
    model: ConvDeepJSCC,
    classifier: CifarResNet18,
    test_loader,
    device: torch.device,
):
    model.eval()
    freeze(classifier)
    perceptual_metric = lpips.LPIPS(
        net=config["lpips_net"],
        version=config["lpips_version"],
        verbose=False,
    ).to(device)
    freeze(perceptual_metric)

    results = {}
    for snr_db in config["test_snr_db"]:
        generator = torch.Generator(device=device)
        generator.manual_seed(config["evaluation_seed"])
        metric_sums = {}
        sample_count = 0
        cbr = None
        progress = tqdm(test_loader, desc=f"deepjscc test snr={snr_db:g}")
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                latent = model.encode_for_channel(images)
                noise = model.channel.sample_noise(
                    latent, snr_db, generator=generator
                )
                reconstruction = model.decode_from_channel(latent + noise)
                batch_metrics = reconstruction_metrics(
                    classifier,
                    perceptual_metric,
                    images,
                    reconstruction,
                    labels,
                    config["ms_ssim_weights"],
                    config["ms_ssim_window_size"],
                    config["ms_ssim_window_sigma"],
                )
            batch_cbr = channel_bandwidth_ratio(images, latent)
            if cbr is None:
                cbr = batch_cbr
            elif abs(cbr - batch_cbr) > 1e-12:
                raise RuntimeError("CBR changed between evaluation batches")
            for name, values in batch_metrics.items():
                metric_sums[name] = (
                    metric_sums.get(name, 0.0) + values.sum().item()
                )
            sample_count += images.shape[0]

        metrics = {
            name: value / sample_count
            for name, value in metric_sums.items()
        }
        metrics["cbr_real_channel_uses_per_source_value"] = cbr
        metrics["num_test_samples"] = sample_count
        results[str(snr_db)] = metrics
        run.append_metrics(
            {"split": "test", "snr_db": snr_db, **metrics}
        )
        print(
            f"snr_db={snr_db:g} psnr_db={metrics['psnr_db']:.4f} "
            f"ms_ssim={metrics['ms_ssim']:.6f} "
            f"lpips={metrics['lpips']:.6f} "
            f"accuracy={metrics['reconstruction_accuracy']:.4f}"
        )
    return results


def train(config, run: ExperimentRun, device: torch.device) -> None:
    set_seed(config["seed"])
    train_loader, test_loader = cifar_loaders(
        config["dataset"],
        config["data_root"],
        config["batch_size"],
        config["num_workers"],
        download=False,
    )
    model = ConvDeepJSCC(config["latent_channels"]).to(device)
    classifier_path = ROOT / config["classifier_checkpoint"]
    if not classifier_path.exists():
        raise FileNotFoundError(
            f"Missing classifier checkpoint: {classifier_path}. "
            "Run the classifier baseline first."
        )
    classifier = CifarResNet18(config["num_classes"])
    missing, unexpected = load_model_state(classifier, str(classifier_path))
    if missing or unexpected:
        raise RuntimeError(
            f"Incompatible classifier checkpoint: missing={missing}, "
            f"unexpected={unexpected}"
        )
    classifier.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )
    snr_min, snr_max = config["train_snr_db"]

    final_loss = None
    for epoch in range(config["epochs"]):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        progress = tqdm(
            train_loader, desc=f"deepjscc {epoch + 1}/{config['epochs']}"
        )
        for images, _ in progress:
            images = images.to(device)
            snr_db = random.uniform(snr_min, snr_max)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(images, snr_db)
            loss = F.mse_loss(reconstruction, images)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * images.shape[0]
            sample_count += images.shape[0]
            progress.set_postfix(loss=f"{loss.item():.5f}", snr=f"{snr_db:.1f}")
        scheduler.step()
        final_loss = loss_sum / sample_count
        run.append_metrics(
            {
                "epoch": epoch + 1,
                "train_mse": final_loss,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "latent_channels": config["latent_channels"],
            },
            run.checkpoint_dir / "latest.pt",
        )
    test_metrics = evaluate(
        config, run, model, classifier, test_loader, device
    )
    run.write_json(
        run.output_dir / "summary.json",
        {
            "epochs": config["epochs"],
            "final_train_mse": final_loss,
            "test_metrics": test_metrics,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-S1-005_deepjscc.json"
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = load_json(config_path)
    config["runtime_device"] = args.device
    with ExperimentRun(ROOT, config_path, config) as run:
        train(config, run, torch.device(args.device))


if __name__ == "__main__":
    main()
