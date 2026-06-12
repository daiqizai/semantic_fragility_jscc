from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fragile_jscc.channels import AWGNChannel
from fragile_jscc.evaluation import spearman_correlation
from fragile_jscc.groups import group_mask, num_groups
from fragile_jscc.models import ConvDeepJSCC, TinySemanticClassifier
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


if __name__ == "__main__":
    unittest.main()
