import unittest

import polars as pl

from bazar_analysis.analysis import _cooccurrence, _performance_by_entity


class AnalysisCoreTests(unittest.TestCase):
    def test_cooccurrence_deduplicates_values_per_board(self) -> None:
        frame = _cooccurrence([["A", "A", "B"], ["B", "A", "C"]], "left", "right")
        counts = {(row["left"], row["right"]): row["count"] for row in frame.iter_rows(named=True)}

        self.assertEqual(counts[("A", "B")], 2)
        self.assertEqual(counts[("A", "C")], 1)
        self.assertEqual(counts[("B", "C")], 1)

    def test_performance_by_entity_deduplicates_screenshot_entity_pairs(self) -> None:
        frame = pl.DataFrame(
            {
                "screenshot_id": [1, 1, 2, 3],
                "item_name": ["A", "A", "A", "B"],
                "record_wins": [10, 10, 4, 7],
                "run_victory_tier": ["Gold", "Gold", "Bronze", "Silver"],
            }
        )

        result = _performance_by_entity(frame, "item_name")
        rows = {row["item_name"]: row for row in result.iter_rows(named=True)}

        self.assertEqual(rows["A"]["run_count"], 2)
        self.assertEqual(rows["A"]["wins_10_count"], 1)
        self.assertAlmostEqual(rows["A"]["avg_wins"], 7.0)
        self.assertEqual(rows["B"]["run_count"], 1)


if __name__ == "__main__":
    unittest.main()
