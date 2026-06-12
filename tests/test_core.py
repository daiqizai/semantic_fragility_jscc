from pathlib import Path
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fragile_jscc.channels import AWGNChannel
from fragile_jscc.evaluation import reconstruction_metrics, spearman_correlation
from fragile_jscc.groups import group_mask, num_groups
from fragile_jscc.models import ConvDeepJSCC, TinySemanticClassifier
from fragile_jscc.quality import (
    CIFAR_MS_SSIM_WEIGHTS,
    channel_bandwidth_ratio,
    multiscale_ssim,
    psnr,
)
from fragile_jscc.scoring import oracle_fragility_scores
from fragile_jscc.semantic import freeze


class CoreTests(unittest.TestCase):
    def test_awgn_explicit_noise_is_reproducible(self):
        channel = AWGNChannel()
        x = torch.ones(2, 4, 3, 3)
        noise = channel.sample_noise(x, 5.0)
        self.assertTrue(torch.equal(channel(x, 5.0, noise), x + noise))

    def test_channel_grouping_covers_expected_channels(self):
        latent = torch.zeros(2, 5, 4, 4)
        self.assertEqual(num_groups(tuple(latent.shape), "channel", 2), 3)
        self.assertTrue(group_mask(latent, 2, "channel", 2)[:, 4].all())
        self.assertEqual(
            group_mask(latent, 2, "channel", 2).sum().item(),
            2 * 4 * 4,
        )

    def test_oracle_fragility_shape_and_finiteness(self):
        torch.manual_seed(1)
        jscc = ConvDeepJSCC(latent_channels=4)
        classifier = TinySemanticClassifier()
        freeze(jscc)
        freeze(classifier)
        images = torch.rand(2, 3, 32, 32)
        scores = oracle_fragility_scores(
            jscc,
            classifier,
            images,
            snr_db=5.0,
            granularity="channel",
            channel_group_size=2,
            mc_samples=1,
        )
        self.assertEqual(tuple(scores.shape), (2, 2))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue((scores >= 0).all())

    def test_spearman_identity_is_one(self):
        values = torch.tensor([[3.0, 1.0, 2.0], [0.0, 4.0, 2.0]])
        correlation = spearman_correlation(values, values)
        self.assertLess(abs(correlation - 1.0), 1e-6)

    def test_quality_metrics_and_cbr(self):
        torch.manual_seed(2)
        images = torch.rand(2, 3, 32, 32)
        reconstruction = images.clone()
        latent = torch.rand(2, 16, 8, 8)
        self.assertTrue((psnr(images, reconstruction) == 120.0).all())
        self.assertTrue(
            torch.allclose(
                multiscale_ssim(images, reconstruction),
                torch.ones(2),
                atol=1e-5,
            )
        )
        self.assertAlmostEqual(
            channel_bandwidth_ratio(images, latent), 1.0 / 3.0
        )

    def test_reconstruction_semantic_metrics(self):
        class DummyPerceptualMetric(nn.Module):
            def forward(self, reference, reconstruction, normalize=False):
                self.assert_normalize = normalize
                return (reference - reconstruction).abs().mean(
                    dim=(1, 2, 3), keepdim=True
                )

        torch.manual_seed(3)
        classifier = TinySemanticClassifier()
        freeze(classifier)
        images = torch.rand(2, 3, 32, 32)
        labels = classifier(images).argmax(dim=-1)
        perceptual_metric = DummyPerceptualMetric()
        metrics = reconstruction_metrics(
            classifier,
            perceptual_metric,
            images,
            images,
            labels,
            CIFAR_MS_SSIM_WEIGHTS,
            3,
            1.5,
        )
        self.assertTrue(perceptual_metric.assert_normalize)
        self.assertTrue((metrics["clean_accuracy"] == 1.0).all())
        self.assertTrue((metrics["reconstruction_accuracy"] == 1.0).all())
        self.assertTrue((metrics["prediction_consistency"] == 1.0).all())
        self.assertTrue((metrics["semantic_failure_rate"] == 0.0).all())
        self.assertTrue(
            torch.allclose(
                metrics["semantic_kl"], torch.zeros(2), atol=1e-5
            )
        )


if __name__ == "__main__":
    unittest.main()
