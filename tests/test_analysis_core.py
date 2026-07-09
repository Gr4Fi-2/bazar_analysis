import unittest
import datetime as dt
import json

import polars as pl

from bazar_analysis.analysis import (
    _analysis_filters_from_values,
    _build_archetype_families,
    _cooccurrence,
    _filter_analysis_frame,
    _jaccard_similarity,
    _mechanic_labels_for_names,
    _parse_card_names,
    _performance_by_entity,
)


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

    def test_parse_card_names_resolves_base_ids_when_title_missing(self) -> None:
        cards_json = '[{"base_id":"abc","title":null},{"base_id":"def","title":"00000000-0000-0000-0000-000000000000"},{"base_id":"ghi","title":"Named Item"},{"base_id":"11111111-1111-1111-1111-111111111111","title":null}]'

        self.assertEqual(
            _parse_card_names(cards_json, {"abc": "Mapped Item", "def": "Mapped Other"}),
            ["Mapped Item", "Mapped Other", "Named Item"],
        )

    def test_analysis_filter_limits_heroes_and_created_at(self) -> None:
        filters = _analysis_filters_from_values(
            "jules,vanessa",
            "last3d",
            now=dt.datetime(2026, 7, 9, 12, 0, tzinfo=dt.UTC),
        )
        frame = pl.DataFrame(
            {
                "hero": ["Jules", "Vanessa", "Mak"],
                "created_at": ["2026-07-08T12:00:00+00:00", "2026-07-05T12:00:00+00:00", "2026-07-08T12:00:00+00:00"],
                "value": [1, 2, 3],
            }
        )

        result = _filter_analysis_frame(frame, filters)

        self.assertEqual(result.get_column("value").to_list(), [1])

    def test_mechanic_labels_use_name_patterns(self) -> None:
        labels = _mechanic_labels_for_names(["Froyo Cart", "Ice Swan", "Scorchpepper"])

        self.assertIn("Food", labels)
        self.assertIn("Freeze", labels)
        self.assertIn("Burn", labels)
        self.assertIn("Spice", labels)

    def test_jaccard_similarity_handles_overlap(self) -> None:
        self.assertAlmostEqual(_jaccard_similarity({"A", "B"}, {"B", "C"}), 1 / 3)
        self.assertEqual(_jaccard_similarity(set(), {"A"}), 0.0)

    def test_archetype_families_merge_similar_core_sets(self) -> None:
        frame = pl.DataFrame(
            {
                "archetype_anchor_a": ["A", "A", "X"],
                "archetype_anchor_b": ["B", "C", "Y"],
                "board_count": [10, 5, 7],
                "avg_wins": [8.0, 7.0, 5.0],
                "weighted_avg_wins": [7.8, 7.1, 5.2],
                "gold_plus_count": [6, 3, 1],
                "perfect_count": [2, 1, 0],
                "core_items_json": [
                    json.dumps([{"name": "A", "rate": 1.0}, {"name": "B", "rate": 1.0}]),
                    json.dumps([{"name": "A", "rate": 1.0}, {"name": "B", "rate": 0.8}, {"name": "C", "rate": 0.8}]),
                    json.dumps([{"name": "X", "rate": 1.0}, {"name": "Y", "rate": 1.0}]),
                ],
                "flex_items_json": ["[]", "[]", "[]"],
                "top_skills_json": ["[]", "[]", "[]"],
                "outcome_distribution_json": [
                    json.dumps([{"name": "Gold", "count": 6}]),
                    json.dumps([{"name": "Gold", "count": 3}]),
                    json.dumps([{"name": "Bronze", "count": 7}]),
                ],
                "player_rank_distribution_json": ["[]", "[]", "[]"],
            }
        )

        result = _build_archetype_families(frame)

        self.assertEqual(result.height, 2)
        first = result.row(0, named=True)
        self.assertEqual(first["board_count"], 15)
        self.assertEqual(first["cluster_count"], 2)


if __name__ == "__main__":
    unittest.main()
