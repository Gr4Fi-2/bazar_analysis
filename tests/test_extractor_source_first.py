import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from bazar_analysis.config import Settings, ensure_directories
from bazar_analysis.db import init_db
from bazar_analysis.extractor import extract_board_data
from bazar_analysis.utils import normalize_name


def _temp_settings() -> Settings:
    root = Path(tempfile.mkdtemp())
    data = root / "data"
    return Settings(
        project_root=root,
        data_dir=data,
        raw_dir=data / "raw",
        raw_runs_dir=data / "raw" / "runs_html",
        raw_screenshots_dir=data / "raw" / "screenshots",
        reference_dir=data / "reference",
        reference_html_dir=data / "reference" / "html",
        reference_icons_items_dir=data / "reference" / "icons" / "items",
        reference_icons_skills_dir=data / "reference" / "icons" / "skills",
        reference_browser_profile_dir=data / "reference" / "playwright_profile",
        debug_dir=data / "debug",
        debug_board_dir=data / "debug" / "board_regions",
        debug_rank_dir=data / "debug" / "rank_regions",
        debug_skill_dir=data / "debug" / "skill_regions",
        debug_crops_dir=data / "debug" / "crops",
        debug_annotated_dir=data / "debug" / "annotated",
        exports_dir=data / "exports",
        db_dir=data / "db",
        duckdb_path=data / "db" / "test.duckdb",
    )


class ExtractorSourceFirstTests(unittest.TestCase):
    def test_source_only_exact_cards_batch_insert_and_preserve_rank_reviews(self) -> None:
        settings = _temp_settings()
        ensure_directories(settings)
        conn = init_db(settings)

        conn.execute(
            """
            INSERT INTO reference_items(entity_id, name, normalized_name, slug, page_url, aliases_json, metadata_json, collected_at)
            VALUES
                ('item_a', 'Item A', ?, 'item-a', 'https://example/items/a', '[]', '{}', 'now'),
                ('item_b', 'Item B', ?, 'item-b', 'https://example/items/b', '[]', '{}', 'now')
            """,
            (normalize_name("Item A"), normalize_name("Item B")),
        )
        conn.execute(
            """
            INSERT INTO reference_skills(entity_id, name, normalized_name, slug, page_url, aliases_json, metadata_json, collected_at)
            VALUES('skill_a', 'Skill A', ?, 'skill-a', 'https://example/skills/a', '[]', '{}', 'now')
            """,
            (normalize_name("Skill A"),),
        )

        board_cards = [
            {"slot_position": 0, "title": "Item A", "base_id": "item_a"},
            {"slot_position": 1, "title": "Item A", "base_id": "item_a"},
            {"slot_position": 2, "title": "Unknown Item", "base_id": "unknown"},
        ]
        skill_cards = [{"slot_position": 0, "title": "Skill A", "base_id": "skill_a"}]
        conn.execute(
            """
            INSERT INTO runs(source_run_id, hero, run_url, title, card_hints_json, board_cards_json, skill_cards_json, crawled_at)
            VALUES('run_1', 'Jules', 'https://example/run/1', 'Fixture Run', '[]', ?, ?, 'now')
            """,
            (json.dumps(board_cards), json.dumps(skill_cards)),
        )
        run_id = conn.execute("SELECT run_id FROM runs WHERE source_run_id = 'run_1'").fetchone()["run_id"]
        conn.execute(
            "INSERT INTO screenshots(run_id, screenshot_url, is_primary) VALUES(?, 'https://example/image.jpg', 1)",
            (run_id,),
        )
        screenshot_id = conn.execute("SELECT screenshot_id FROM screenshots WHERE run_id = ?", (run_id,)).fetchone()["screenshot_id"]
        conn.execute(
            """
            INSERT INTO extracted_ranks(screenshot_id, raw_label, rank_tier, confidence, method, bbox_x, bbox_y, bbox_w, bbox_h, crop_path, top_candidates_json, status)
            VALUES(?, 'Gold', 'Gold', 1.0, 'fixture', 0, 0, 1, 1, 'rank.png', '{}', 'ok')
            """,
            (screenshot_id,),
        )
        conn.execute(
            """
            INSERT INTO review_queue(review_id, screenshot_id, detection_type, crop_path, confidence, raw_label, top_candidates_json)
            VALUES
                (1, ?, 'rank', 'rank.png', 0.5, 'Gold', '{}'),
                (2, ?, 'board_item', 'board.png', 0.1, 'Unknown', '{}')
            """,
            (screenshot_id, screenshot_id),
        )
        conn.commit()

        previous_source_only = os.environ.get("BAZAR_EXTRACT_SOURCE_ONLY")
        os.environ["BAZAR_EXTRACT_SOURCE_ONLY"] = "1"
        try:
            with redirect_stdout(StringIO()):
                result = extract_board_data(conn, settings)
        finally:
            if previous_source_only is None:
                os.environ.pop("BAZAR_EXTRACT_SOURCE_ONLY", None)
            else:
                os.environ["BAZAR_EXTRACT_SOURCE_ONLY"] = previous_source_only

        self.assertEqual(result["item_detections"], 3)
        self.assertEqual(result["skill_detections"], 1)
        board_rows = conn.execute(
            "SELECT entity_id, raw_label, duplicate_count FROM extracted_board_items ORDER BY slot_index"
        ).fetchall()
        self.assertEqual([row["duplicate_count"] for row in board_rows], [2, 2, 1])
        self.assertEqual(board_rows[0]["entity_id"], "item_a")
        self.assertIsNone(board_rows[2]["entity_id"])
        self.assertEqual(board_rows[2]["raw_label"], "Unknown Item")
        self.assertEqual(conn.execute("SELECT COUNT(*) AS count FROM extracted_ranks").fetchone()["count"], 1)
        review_types = [row["detection_type"] for row in conn.execute("SELECT detection_type FROM review_queue ORDER BY review_id").fetchall()]
        self.assertEqual(review_types, ["rank"])


if __name__ == "__main__":
    unittest.main()
