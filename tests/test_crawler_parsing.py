import datetime as dt
import os
import unittest
from unittest.mock import patch

from bazar_analysis.crawler import (
    _extract_escaped_json_fragment,
    _extract_player_rank_tier,
    _find_json_fragment_end,
    _normalize_embedded_cards,
    _parse_created_timestamp,
    _should_fetch_detail_html,
)


class CrawlerParsingTests(unittest.TestCase):
    def test_parse_created_timestamp_variants(self) -> None:
        self.assertEqual(
            _parse_created_timestamp("2026-05-06T16:24:57Z"),
            dt.datetime(2026, 5, 6, 16, 24, 57, tzinfo=dt.UTC),
        )
        self.assertEqual(
            _parse_created_timestamp("Wed, 06 May 2026 16:24:57 GMT"),
            dt.datetime(2026, 5, 6, 16, 24, 57, tzinfo=dt.UTC),
        )
        self.assertIsNone(_parse_created_timestamp("not a timestamp"))

    def test_find_json_fragment_end_ignores_escaped_braces(self) -> None:
        text = r'{"text":"not a } terminator","items":[1,2]}'
        self.assertEqual(_find_json_fragment_end(text, 0), len(text))

    def test_extract_escaped_json_fragment(self) -> None:
        text = r'prefix \"run\":{\"id\":\"run_1\",\"items\":[{\"title\":\"Foo\"}]} suffix'
        self.assertEqual(
            _extract_escaped_json_fragment(text, r'\"run\":'),
            {"id": "run_1", "items": [{"title": "Foo"}]},
        )

    def test_normalize_embedded_cards(self) -> None:
        cards = _normalize_embedded_cards(
            [
                {"title": "  Big   Item ", "slotPosition": "2", "baseId": "item_1", "tierOverride": "Gold", "enchantmentOverride": "$undefined"},
                {"name": "Fallback Name", "slotPosition": "bad", "cardId": "item_2"},
                "skip me",
            ],
            "fixture",
        )

        self.assertEqual(cards[0]["title"], "Big Item")
        self.assertEqual(cards[0]["slot_position"], 2)
        self.assertIsNone(cards[0]["enchantment"])
        self.assertEqual(cards[1]["title"], "Fallback Name")
        self.assertEqual(cards[1]["slot_position"], 1)
        self.assertEqual(cards[1]["base_id"], "item_2")
        self.assertEqual(len(cards), 2)

    def test_extract_player_rank_tier_top_level_and_nested(self) -> None:
        self.assertEqual(_extract_player_rank_tier({"playerRank": "Gold II"}), "Gold")
        self.assertEqual(_extract_player_rank_tier({"profile": {"rankTier": "legendary"}}), "Legendary")
        self.assertIsNone(_extract_player_rank_tier({"playerRank": "wood"}))

    def test_should_skip_detail_html_when_api_payload_is_complete(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BAZAR_CRAWL_FETCH_DETAIL_HTML", None)
            self.assertFalse(_should_fetch_detail_html({"screenshotUrl": "/cr/run.webp", "items": [{}]}))
            self.assertTrue(_should_fetch_detail_html({"screenshotUrl": "/cr/run.webp"}))


if __name__ == "__main__":
    unittest.main()
