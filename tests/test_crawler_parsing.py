import datetime as dt
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bazar_analysis.config import get_settings
from bazar_analysis.crawler import (
    RUN_API_PAGE_SIZE,
    RunFilters,
    _api_page_path,
    _discovery_checkpoint_path,
    _extract_escaped_json_fragment,
    _extract_player_rank_tier,
    _find_json_fragment_end,
    _normalize_embedded_cards,
    _parse_created_timestamp,
    _resume_discovery,
    _should_fetch_detail_html,
    discover_runs,
)
from bazar_analysis.http_client import BazaarDBRequestError


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

    def test_interrupted_legacy_cache_resumes_at_next_page(self) -> None:
        filters = RunFilters(
            heroes={"Jules"},
            min_rank=None,
            date_range="last3d",
            pages=None,
            sort="newest",
            order="desc",
            created_after=(dt.datetime.now(dt.UTC) - dt.timedelta(days=3)).strftime("%a, %d %b %Y %H:%M:%S GMT"),
            created_before=None,
            request_delay_seconds=1.5,
        )
        now = dt.datetime.now(dt.UTC).isoformat()
        with tempfile.TemporaryDirectory() as temporary_dir:
            raw_runs_dir = Path(temporary_dir)
            settings = replace(get_settings(), raw_runs_dir=raw_runs_dir)
            for page_number in (1, 2):
                payload = [
                    {"id": f"page-{page_number}-run-{index}", "createdAt": now, "hero": "Jules"}
                    for index in range(RUN_API_PAGE_SIZE)
                ]
                _api_page_path(settings, filters, page_number).write_text(json.dumps(payload), encoding="utf-8")

            resume_page, resumed_filters = _resume_discovery(settings, filters)

            self.assertEqual(resume_page, 2)
            self.assertIsNotNone(resumed_filters.created_after)
            self.assertTrue(_discovery_checkpoint_path(settings, filters).exists())

            with patch("bazar_analysis.crawler._fetch_run_api_page", return_value=[]) as fetch:
                result = discover_runs(settings, filters)

            self.assertTrue(result.exhausted)
            self.assertEqual(len(result.runs), RUN_API_PAGE_SIZE * 2)
            self.assertEqual(fetch.call_count, 1)

    def test_discovery_surfaces_access_block_status(self) -> None:
        filters = RunFilters(
            heroes={"Jules"},
            min_rank=None,
            date_range="season15",
            pages=1,
            sort="newest",
            order="desc",
            created_after="Wed, 03 Jun 2026 16:56:45 GMT",
            created_before=None,
            request_delay_seconds=1.5,
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            settings = replace(get_settings(), raw_runs_dir=Path(temporary_dir))
            error = BazaarDBRequestError("blocked", status_code=403, retryable=False)
            with patch("bazar_analysis.crawler._fetch_run_api_page", side_effect=error):
                result = discover_runs(settings, filters)

        self.assertFalse(result.exhausted)
        self.assertEqual(result.error_status_code, 403)
        self.assertEqual(result.runs, [])


if __name__ == "__main__":
    unittest.main()
