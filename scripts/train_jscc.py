#!/usr/bin/env python3
import argparse
from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fragile_jscc.data import cifar_loaders
from fragile_jscc.experiment import ExperimentRun, load_json
from fragile_jscc.models import ConvDeepJSCC
from fragile_jscc.utils import set_seed


def train(config, run: ExperimentRun, device: torch.device) -> None:
    set_seed(config["seed"])
    train_loader, _ = cifar_loaders(
        config["dataset"],
        config["data_root"],
        config["batch_size"],
        config["num_workers"],
        download=False,
    )
    model = ConvDeepJSCC(config["latent_channels"]).to(device)
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
    run.write_json(
        run.output_dir / "summary.json",
        {"epochs": config["epochs"], "final_train_mse": final_loss},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-S1-002_deepjscc.json"
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
