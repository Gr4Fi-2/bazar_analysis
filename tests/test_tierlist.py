import unittest

from bazar_analysis.tierlist import _confidence_label, _tier_label, _usage_band


class TierListTests(unittest.TestCase):
    def test_confidence_label_uses_sample_size_bands(self) -> None:
        self.assertEqual(_confidence_label(100), "high")
        self.assertEqual(_confidence_label(40), "medium")
        self.assertEqual(_confidence_label(15), "low")
        self.assertEqual(_confidence_label(14), "very_low")

    def test_usage_band_uses_presence_percent(self) -> None:
        self.assertEqual(_usage_band(10.0), "core/common")
        self.assertEqual(_usage_band(5.0), "common")
        self.assertEqual(_usage_band(2.0), "uncommon")
        self.assertEqual(_usage_band(0.5), "rare")
        self.assertEqual(_usage_band(0.49), "niche")

    def test_tier_label_uses_bias_adjusted_delta(self) -> None:
        self.assertEqual(_tier_label(0.65), "S")
        self.assertEqual(_tier_label(0.35), "A")
        self.assertEqual(_tier_label(0.10), "B")
        self.assertEqual(_tier_label(-0.15), "C")
        self.assertEqual(_tier_label(-0.16), "D")


if __name__ == "__main__":
    unittest.main()
