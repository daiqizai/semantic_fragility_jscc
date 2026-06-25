#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fragile_jscc.models import CifarResNet18, ConvDeepJSCC, load_model_state
from fragile_jscc.quality import psnr
from fragile_jscc.semantic import freeze


def load_summary(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    rows = []
    for snr_text, metrics in sorted(
        summary["test_metrics"].items(), key=lambda item: float(item[0])
    ):
        row = {"snr_db": float(snr_text)}
        row.update(metrics)
        rows.append(row)
    return summary, rows


def save_metrics_csv(rows, output_path: Path) -> None:
    fields = [
        "snr_db",
        "psnr_db",
        "ms_ssim",
        "lpips",
        "reconstruction_accuracy",
        "prediction_consistency",
        "semantic_failure_rate",
        "semantic_kl",
        "cbr_real_channel_uses_per_source_value",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def plot_quality(rows, output_path: Path) -> None:
    snr = [row["snr_db"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), dpi=180)
    plots = [
        ("psnr_db", "PSNR (dB)", "tab:blue"),
        ("ms_ssim", "MS-SSIM", "tab:green"),
        ("lpips", "LPIPS (lower is better)", "tab:red"),
    ]
    for axis, (key, ylabel, color) in zip(axes, plots):
        axis.plot(snr, [row[key] for row in rows], marker="o", color=color)
        axis.set_xlabel("Test SNR (dB)")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
    fig.suptitle("EXP-S1-005 DeepJSCC Image Quality vs SNR")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_semantic(rows, output_path: Path) -> None:
    snr = [row["snr_db"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=180)
    axes[0].plot(
        snr,
        [row["reconstruction_accuracy"] for row in rows],
        marker="o",
        label="Reconstruction accuracy",
    )
    axes[0].plot(
        snr,
        [row["prediction_consistency"] for row in rows],
        marker="s",
        label="Prediction consistency",
    )
    axes[0].plot(
        snr,
        [row["semantic_failure_rate"] for row in rows],
        marker="^",
        label="Semantic failure rate",
    )
    axes[0].set_xlabel("Test SNR (dB)")
    axes[0].set_ylabel("Rate")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(
        snr,
        [row["semantic_kl"] for row in rows],
        marker="o",
        color="tab:purple",
    )
    axes[1].set_xlabel("Test SNR (dB)")
    axes[1].set_ylabel("Semantic KL")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("EXP-S1-005 Semantic Robustness vs SNR")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def load_models(args, device):
    jscc = ConvDeepJSCC(args.latent_channels)
    classifier = CifarResNet18(args.num_classes)
    for model, checkpoint in [
        (jscc, args.jscc_checkpoint),
        (classifier, args.classifier_checkpoint),
    ]:
        missing, unexpected = load_model_state(model, str(checkpoint))
        if missing or unexpected:
            raise RuntimeError(
                f"incompatible checkpoint {checkpoint}: "
                f"missing={missing}, unexpected={unexpected}"
            )
    jscc.to(device).eval()
    classifier.to(device).eval()
    freeze(jscc)
    freeze(classifier)
    return jscc, classifier


def predict(classifier, images):
    with torch.no_grad():
        logits = classifier(images)
        probabilities = logits.softmax(dim=-1)
        predictions = probabilities.argmax(dim=-1)
        confidence = probabilities.max(dim=-1).values
    return predictions, confidence


def make_reconstructions(args, output_dir: Path, rows) -> None:
    device = torch.device(args.device)
    jscc, classifier = load_models(args, device)
    dataset = datasets.CIFAR10(
        args.data_root,
        train=False,
        transform=transforms.ToTensor(),
        download=False,
    )
    indices = [int(item) for item in args.sample_indices.split(",")]
    images = torch.stack([dataset[index][0] for index in indices]).to(device)
    labels = torch.tensor(
        [dataset[index][1] for index in indices],
        dtype=torch.long,
        device=device,
    )
    class_names = dataset.classes
    original_predictions, original_confidence = predict(classifier, images)

    reconstructions = {}
    reconstruction_predictions = {}
    reconstruction_confidence = {}
    sample_metrics = []
    with torch.no_grad():
        latent = jscc.encode_for_channel(images)
        for row in rows:
            snr_db = row["snr_db"]
            generator = torch.Generator(device=device)
            generator.manual_seed(args.visual_seed + int(snr_db * 1000))
            noise = jscc.channel.sample_noise(latent, snr_db, generator)
            reconstruction = jscc.decode_from_channel(latent + noise)
            reconstructions[snr_db] = reconstruction.detach().cpu()
            pred, conf = predict(classifier, reconstruction)
            reconstruction_predictions[snr_db] = pred.detach().cpu()
            reconstruction_confidence[snr_db] = conf.detach().cpu()
            per_sample_psnr = psnr(images, reconstruction).detach().cpu()
            for sample_offset, index in enumerate(indices):
                sample_metrics.append(
                    {
                        "sample_index": index,
                        "label": class_names[labels[sample_offset].item()],
                        "snr_db": snr_db,
                        "psnr_db": per_sample_psnr[sample_offset].item(),
                        "original_prediction": class_names[
                            original_predictions[sample_offset].item()
                        ],
                        "original_confidence": original_confidence[
                            sample_offset
                        ].item(),
                        "reconstruction_prediction": class_names[
                            pred[sample_offset].item()
                        ],
                        "reconstruction_confidence": conf[
                            sample_offset
                        ].item(),
                    }
                )

    samples_dir = output_dir / "transmission_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    save_image(images.cpu(), samples_dir / "original_batch.png", nrow=len(indices))
    for snr_db, reconstruction in reconstructions.items():
        save_image(
            reconstruction,
            samples_dir / f"reconstruction_snr_{snr_db:g}db_batch.png",
            nrow=len(indices),
        )
    for sample_offset, index in enumerate(indices):
        save_image(
            images[sample_offset].detach().cpu(),
            samples_dir / f"sample_{sample_offset:02d}_idx_{index}_original.png",
        )
        for snr_db, reconstruction in reconstructions.items():
            save_image(
                reconstruction[sample_offset],
                samples_dir
                / f"sample_{sample_offset:02d}_idx_{index}_snr_{snr_db:g}db.png",
            )

    with (output_dir / "transmission_sample_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "sample_index",
            "label",
            "snr_db",
            "psnr_db",
            "original_prediction",
            "original_confidence",
            "reconstruction_prediction",
            "reconstruction_confidence",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sample_metrics:
            writer.writerow(row)

    snrs = [row["snr_db"] for row in rows]
    row_labels = ["Original"] + [f"{snr:g} dB" for snr in snrs]
    fig, axes = plt.subplots(
        len(row_labels),
        len(indices),
        figsize=(1.7 * len(indices), 1.65 * len(row_labels)),
        dpi=180,
    )
    for sample_offset, index in enumerate(indices):
        axes[0, sample_offset].imshow(
            images[sample_offset].detach().cpu().permute(1, 2, 0)
        )
        label = class_names[labels[sample_offset].item()]
        pred = class_names[original_predictions[sample_offset].item()]
        axes[0, sample_offset].set_title(
            f"idx {index}\nlabel {label}\npred {pred}",
            fontsize=7,
        )
    for row_index, snr_db in enumerate(snrs, start=1):
        reconstruction = reconstructions[snr_db]
        predictions = reconstruction_predictions[snr_db]
        for sample_offset in range(len(indices)):
            axes[row_index, sample_offset].imshow(
                reconstruction[sample_offset].permute(1, 2, 0)
            )
            axes[row_index, sample_offset].set_title(
                f"pred {class_names[predictions[sample_offset].item()]}",
                fontsize=7,
            )
    for row_index, row_label in enumerate(row_labels):
        axes[row_index, 0].set_ylabel(row_label, fontsize=9)
    for axis in axes.reshape(-1):
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Actual DeepJSCC Transmissions: Original vs AWGN Reconstructions")
    fig.tight_layout()
    fig.savefig(output_dir / "actual_transmission_grid.png", bbox_inches="tight")
    plt.close(fig)


def write_report_brief(summary, rows, output_path: Path) -> None:
    lines = [
        "# Current Report Assets",
        "",
        "Derived from `EXP-S1-005` CIFAR-10 AWGN DeepJSCC baseline.",
        "",
        "## Talking Points",
        "",
        "- Fixed real-valued CBR is `1/3` for all SNR points.",
        "- Classifier baseline clean accuracy is `95.29%`.",
        "- DeepJSCC reconstruction quality improves monotonically from 0 to 20 dB.",
        "- Semantic failure rate drops sharply as SNR increases, showing clear channel-conditioned semantic degradation.",
        "- These figures only support the baseline story; fragility superiority still requires `EXP-S2-002`.",
        "",
        "## Key Numbers",
        "",
        "| SNR (dB) | PSNR | MS-SSIM | LPIPS | Recon Acc | Consistency | Failure | Semantic KL |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {snr_db:g} | {psnr_db:.2f} | {ms_ssim:.4f} | {lpips:.4f} | "
            "{reconstruction_accuracy:.4f} | {prediction_consistency:.4f} | "
            "{semantic_failure_rate:.4f} | {semantic_kl:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Generated Files",
            "",
            "- `quality_vs_snr.png`",
            "- `semantic_vs_snr.png`",
            "- `actual_transmission_grid.png`",
            "- `transmission_samples/` individual original/reconstruction images",
            "- `report_metrics.csv`",
            "- `transmission_sample_metrics.csv`",
            "",
            "## Caveat",
            "",
            "`EXP-S2-002` has not run yet, so do not claim the fragility ranking is better than baselines.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs/EXP-S1-005/summary.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/EXP-S1-005/report_assets")
    parser.add_argument("--data-root", type=Path, default=Path("/data2/liulu/semantic_comm/data"))
    parser.add_argument("--jscc-checkpoint", type=Path, default=ROOT / "checkpoints/EXP-S1-005/latest.pt")
    parser.add_argument("--classifier-checkpoint", type=Path, default=ROOT / "checkpoints/EXP-S1-004/best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--sample-indices", default="0,1,2,3,4,5")
    parser.add_argument("--visual-seed", type=int, default=1007)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, rows = load_summary(args.summary)
    save_metrics_csv(rows, args.output_dir / "report_metrics.csv")
    plot_quality(rows, args.output_dir / "quality_vs_snr.png")
    plot_semantic(rows, args.output_dir / "semantic_vs_snr.png")
    make_reconstructions(args, args.output_dir, rows)
    write_report_brief(summary, rows, args.output_dir / "report_brief.md")
    manifest = {
        "source_experiment": "EXP-S1-005",
        "summary": str(args.summary),
        "jscc_checkpoint": str(args.jscc_checkpoint),
        "classifier_checkpoint": str(args.classifier_checkpoint),
        "data_root": str(args.data_root),
        "sample_indices": args.sample_indices,
        "visual_seed": args.visual_seed,
        "outputs": sorted(path.name for path in args.output_dir.iterdir()),
    }
    (args.output_dir / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
