import unittest

from bazar_analysis.utils import (
    canonical_image_url,
    derive_run_victory,
    normalize_player_rank_tier,
    parse_score_from_title,
)


class UtilsTests(unittest.TestCase):
    def test_derive_run_victory_thresholds(self) -> None:
        self.assertEqual(derive_run_victory(10, 20), ("Perfect", "Perfect Victory"))
        self.assertEqual(derive_run_victory(10, 0), ("Gold", "Gold Victory"))
        self.assertEqual(derive_run_victory(7, 0), ("Silver", "Silver Victory"))
        self.assertEqual(derive_run_victory(4, 0), ("Bronze", "Bronze Victory"))
        self.assertEqual(derive_run_victory(3, 0), ("Unfortunate", "An Unfortunate Journey"))
        self.assertEqual(derive_run_victory(None, 20), (None, None))

    def test_normalize_player_rank_tier(self) -> None:
        self.assertEqual(normalize_player_rank_tier("legend rank"), "Legendary")
        self.assertEqual(normalize_player_rank_tier("DIAMOND II"), "Diamond")
        self.assertEqual(normalize_player_rank_tier("unknown"), None)

    def test_canonical_image_url_removes_size_query_and_fragment(self) -> None:
        self.assertEqual(
            canonical_image_url("https://cdn.example/items/foo-256x256.png?cache=1#icon"),
            "https://cdn.example/items/foo.png",
        )

    def test_parse_score_from_title(self) -> None:
        self.assertEqual(parse_score_from_title("Vanessa 10-2 run"), (10, 2))
        self.assertEqual(parse_score_from_title("Jules 10 Win build"), (10, 0))
        self.assertEqual(parse_score_from_title("No score here"), (None, None))


if __name__ == "__main__":
    unittest.main()
