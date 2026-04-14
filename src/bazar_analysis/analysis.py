from __future__ import annotations

from itertools import combinations

import polars as pl

from .config import Settings


def _cooccurrence(rows: list[list[str]], left_name: str, right_name: str) -> pl.DataFrame:
    pairs: list[tuple[str, str]] = []
    for values in rows:
        unique_values = sorted(set(value for value in values if value))
        pairs.extend(combinations(unique_values, 2))
    if not pairs:
        return pl.DataFrame(schema={left_name: pl.String, right_name: pl.String, "count": pl.Int64})
    frame = pl.DataFrame(pairs, schema=[left_name, right_name], orient="row")
    return frame.group_by([left_name, right_name]).len(name="count").sort("count", descending=True)


def _pipeline_coverage_summary(conn) -> pl.DataFrame:
    return conn.query_pl(
        """
        WITH item_counts AS (
            SELECT
                screenshot_id,
                COUNT(*) AS board_items_total,
                SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS board_items_ok,
                SUM(CASE WHEN status = 'review' THEN 1 ELSE 0 END) AS board_items_review
            FROM extracted_board_items
            GROUP BY screenshot_id
        ),
        skill_counts AS (
            SELECT
                screenshot_id,
                COUNT(*) AS skills_total,
                SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS skills_ok,
                SUM(CASE WHEN status = 'review' THEN 1 ELSE 0 END) AS skills_review
            FROM extracted_skills
            GROUP BY screenshot_id
        ),
        rank_counts AS (
            SELECT
                screenshot_id,
                SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS rank_ok,
                SUM(CASE WHEN status = 'review' THEN 1 ELSE 0 END) AS rank_review,
                MAX(rank_tier) AS rank_tier
            FROM extracted_ranks
            GROUP BY screenshot_id
        ),
        review_counts AS (
            SELECT
                screenshot_id,
                COUNT(*) AS review_queue_total,
                SUM(CASE WHEN detection_type = 'board_item' THEN 1 ELSE 0 END) AS review_board_items,
                SUM(CASE WHEN detection_type = 'skill' THEN 1 ELSE 0 END) AS review_skills,
                SUM(CASE WHEN detection_type = 'rank' THEN 1 ELSE 0 END) AS review_ranks,
                SUM(CASE WHEN detection_type = 'screenshot_layout' THEN 1 ELSE 0 END) AS review_layout,
                SUM(CASE WHEN detection_type = 'screenshot_file' THEN 1 ELSE 0 END) AS review_files
            FROM review_queue
            GROUP BY screenshot_id
        )
        SELECT
            p.post_id,
            p.hero,
            p.title,
            s.screenshot_id,
            s.is_primary,
            s.width,
            s.height,
            CASE WHEN s.local_path IS NOT NULL THEN 1 ELSE 0 END AS has_local_path,
            CASE WHEN s.width >= 1000 AND s.height >= 600 THEN 1 ELSE 0 END AS passes_size_filter,
            COALESCE(i.board_items_total, 0) AS board_items_total,
            COALESCE(i.board_items_ok, 0) AS board_items_ok,
            COALESCE(i.board_items_review, 0) AS board_items_review,
            COALESCE(sk.skills_total, 0) AS skills_total,
            COALESCE(sk.skills_ok, 0) AS skills_ok,
            COALESCE(sk.skills_review, 0) AS skills_review,
            COALESCE(r.rank_ok, 0) AS rank_ok,
            COALESCE(r.rank_review, 0) AS rank_review,
            r.rank_tier,
            COALESCE(rv.review_queue_total, 0) AS review_queue_total,
            COALESCE(rv.review_board_items, 0) AS review_board_items,
            COALESCE(rv.review_skills, 0) AS review_skills,
            COALESCE(rv.review_ranks, 0) AS review_ranks,
            COALESCE(rv.review_layout, 0) AS review_layout,
            COALESCE(rv.review_files, 0) AS review_files
        FROM screenshots s
        JOIN posts p ON p.post_id = s.post_id
        LEFT JOIN item_counts i ON i.screenshot_id = s.screenshot_id
        LEFT JOIN skill_counts sk ON sk.screenshot_id = s.screenshot_id
        LEFT JOIN rank_counts r ON r.screenshot_id = s.screenshot_id
        LEFT JOIN review_counts rv ON rv.screenshot_id = s.screenshot_id
        ORDER BY s.screenshot_id
        """
    )


def summarize(conn, settings: Settings) -> dict[str, int]:
    item_frame = conn.query_pl(
        """
        SELECT e.screenshot_id, COALESCE(r.name, e.raw_label) AS item_name
        FROM extracted_board_items e
        LEFT JOIN reference_items r ON r.entity_id = e.entity_id
        WHERE e.status = 'ok'
        """
    )
    skill_frame = conn.query_pl(
        """
        SELECT e.screenshot_id, COALESCE(r.name, e.raw_label) AS skill_name
        FROM extracted_skills e
        LEFT JOIN reference_skills r ON r.entity_id = e.entity_id
        WHERE e.status = 'ok'
        """
    )
    rank_frame = conn.query_pl("SELECT screenshot_id, rank_tier FROM extracted_ranks WHERE status = 'ok'")

    top_items = item_frame.group_by("item_name").len(name="count").sort("count", descending=True)
    top_skills = skill_frame.group_by("skill_name").len(name="count").sort("count", descending=True)
    item_lists = item_frame.group_by("screenshot_id").agg(pl.col("item_name")).get_column("item_name").to_list() if item_frame.height else []
    item_pair_counts = _cooccurrence(item_lists, "item_a", "item_b")

    item_skill_join = item_frame.join(skill_frame, on="screenshot_id", how="inner")
    item_skill_counts = item_skill_join.group_by(["item_name", "skill_name"]).len(name="count").sort("count", descending=True)

    ranked_items = item_frame.join(rank_frame, on="screenshot_id", how="left")
    ranked_item_counts = ranked_items.filter(pl.col("rank_tier").is_not_null()).group_by(["rank_tier", "item_name"]).len(name="count").sort(["rank_tier", "count"], descending=[False, True])
    coverage = _pipeline_coverage_summary(conn)

    top_items.write_csv(settings.exports_dir / "summary_top_items.csv")
    top_skills.write_csv(settings.exports_dir / "summary_top_skills.csv")
    item_pair_counts.write_csv(settings.exports_dir / "summary_item_item_cooccurrence.csv")
    item_skill_counts.write_csv(settings.exports_dir / "summary_item_skill_cooccurrence.csv")
    ranked_item_counts.write_csv(settings.exports_dir / "summary_rank_filtered_items.csv")
    coverage.write_csv(settings.exports_dir / "summary_pipeline_coverage.csv")

    return {
        "top_items": top_items.height,
        "top_skills": top_skills.height,
        "item_item_pairs": item_pair_counts.height,
        "item_skill_pairs": item_skill_counts.height,
        "rank_filtered_rows": ranked_item_counts.height,
        "pipeline_coverage_rows": coverage.height,
    }
