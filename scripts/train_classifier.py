#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fragile_jscc.data import cifar_loaders
from fragile_jscc.experiment import ExperimentRun, load_json
from fragile_jscc.models import CifarResNet18
from fragile_jscc.utils import set_seed


def train(config, run: ExperimentRun, device: torch.device) -> None:
    set_seed(config["seed"])
    classes = config["num_classes"]
    train_loader, test_loader = cifar_loaders(
        config["dataset"],
        config["data_root"],
        config["batch_size"],
        config["num_workers"],
        download=False,
    )
    model = CifarResNet18(classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )

    best_accuracy = 0.0
    for epoch in range(config["epochs"]):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        progress = tqdm(
            train_loader, desc=f"classifier {epoch + 1}/{config['epochs']}"
        )
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * images.shape[0]
            sample_count += images.shape[0]
            progress.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                prediction = model(images.to(device)).argmax(dim=-1).cpu()
                correct += prediction.eq(labels).sum().item()
                total += labels.numel()
        accuracy = correct / total
        print(f"epoch={epoch + 1} test_accuracy={accuracy:.4f}")
        run.append_metrics(
            {
                "epoch": epoch + 1,
                "train_cross_entropy": loss_sum / sample_count,
                "test_accuracy": accuracy,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(
                {"model": model.state_dict(), "accuracy": accuracy},
                run.checkpoint_dir / "best.pt",
            )
    run.write_json(
        run.output_dir / "summary.json",
        {"best_test_accuracy": best_accuracy, "epochs": config["epochs"]},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-S1-001_classifier.json"
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
