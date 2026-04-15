from __future__ import annotations

import json
import math
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
                MAX(rank_tier) AS player_rank_tier
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
            run_meta.run_id,
            run_meta.hero,
            run_meta.title,
            run_meta.record_wins,
            run_meta.run_wins_label,
            run_meta.run_outcome_tier,
            run_meta.player_rank_tier AS stored_player_rank_tier,
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
            COALESCE(rank_info.rank_ok, 0) AS rank_ok,
            COALESCE(rank_info.rank_review, 0) AS rank_review,
            rank_info.player_rank_tier AS extracted_player_rank_tier,
            COALESCE(rv.review_queue_total, 0) AS review_queue_total,
            COALESCE(rv.review_board_items, 0) AS review_board_items,
            COALESCE(rv.review_skills, 0) AS review_skills,
            COALESCE(rv.review_ranks, 0) AS review_ranks,
            COALESCE(rv.review_layout, 0) AS review_layout,
            COALESCE(rv.review_files, 0) AS review_files
        FROM screenshots s
        JOIN runs run_meta ON run_meta.run_id = s.run_id
        LEFT JOIN item_counts i ON i.screenshot_id = s.screenshot_id
        LEFT JOIN skill_counts sk ON sk.screenshot_id = s.screenshot_id
        LEFT JOIN rank_counts rank_info ON rank_info.screenshot_id = s.screenshot_id
        LEFT JOIN review_counts rv ON rv.screenshot_id = s.screenshot_id
        ORDER BY s.screenshot_id
        """
    )

def _board_presence_frame(conn) -> tuple[pl.DataFrame, int]:
    frame = conn.query_pl(
        """
        SELECT
            e.screenshot_id,
            s.run_id,
            run_meta.title,
            COALESCE(ref_item.name, e.raw_label) AS item_name
        FROM extracted_board_items e
        JOIN screenshots s ON s.screenshot_id = e.screenshot_id
        JOIN runs run_meta ON run_meta.run_id = s.run_id
        LEFT JOIN reference_items ref_item ON ref_item.entity_id = e.entity_id
        WHERE e.status = 'ok'
        """
    )
    total_boards = int(frame.get_column("screenshot_id").n_unique()) if frame.height else 0
    return frame, total_boards


def _safe_log2_ratio(numerator: float, denominator: float) -> float | None:
    if numerator <= 0 or denominator <= 0:
        return None
    return math.log2(numerator / denominator)


def _systemic_item_pairs(board_frame: pl.DataFrame, total_boards: int) -> pl.DataFrame:
    if not total_boards or not board_frame.height:
        return pl.DataFrame(
            schema={
                "item_a": pl.String,
                "item_b": pl.String,
                "count": pl.Int64,
                "support": pl.Float64,
                "prevalence_a": pl.Float64,
                "prevalence_b": pl.Float64,
                "lift": pl.Float64,
                "pmi": pl.Float64,
                "npmi": pl.Float64,
                "jaccard": pl.Float64,
                "rarity_weight": pl.Float64,
                "synergy_score": pl.Float64,
            }
        )

    item_counts = (
        board_frame.group_by("item_name")
        .len(name="board_count")
        .sort("board_count", descending=True)
        .rename({"item_name": "item"})
    )
    item_count_map = dict(zip(item_counts.get_column("item").to_list(), item_counts.get_column("board_count").to_list(), strict=False))

    board_lists = board_frame.group_by("screenshot_id").agg(pl.col("item_name")).get_column("item_name").to_list()
    pair_rows: list[dict[str, float | int | str]] = []
    for values in board_lists:
        unique_values = sorted(set(value for value in values if value))
        for item_a, item_b in combinations(unique_values, 2):
            pair_rows.append({"item_a": item_a, "item_b": item_b})

    if not pair_rows:
        return pl.DataFrame(
            schema={
                "item_a": pl.String,
                "item_b": pl.String,
                "count": pl.Int64,
                "support": pl.Float64,
                "prevalence_a": pl.Float64,
                "prevalence_b": pl.Float64,
                "lift": pl.Float64,
                "pmi": pl.Float64,
                "npmi": pl.Float64,
                "jaccard": pl.Float64,
                "rarity_weight": pl.Float64,
                "synergy_score": pl.Float64,
            }
        )

    pair_counts = pl.DataFrame(pair_rows).group_by(["item_a", "item_b"]).len(name="count")

    metrics_rows: list[dict[str, float | int | str]] = []
    for row in pair_counts.iter_rows(named=True):
        item_a = row["item_a"]
        item_b = row["item_b"]
        count = int(row["count"])
        count_a = int(item_count_map[item_a])
        count_b = int(item_count_map[item_b])
        support = count / total_boards
        prevalence_a = count_a / total_boards
        prevalence_b = count_b / total_boards
        expected_support = prevalence_a * prevalence_b
        lift = support / expected_support if expected_support else None
        pmi = _safe_log2_ratio(support, expected_support)
        npmi = None
        if pmi is not None and support > 0:
            denominator = -math.log2(support)
            npmi = pmi / denominator if denominator else None
        union = count_a + count_b - count
        jaccard = count / union if union else None
        idf_a = math.log((total_boards + 1) / (count_a + 1)) + 1.0
        idf_b = math.log((total_boards + 1) / (count_b + 1)) + 1.0
        rarity_weight = math.sqrt(idf_a * idf_b)
        support_weight = math.log1p(count) * (count / (count + 2.0))
        synergy_score = (npmi or 0.0) * rarity_weight * support_weight
        metrics_rows.append(
            {
                "item_a": item_a,
                "item_b": item_b,
                "count": count,
                "support": support,
                "prevalence_a": prevalence_a,
                "prevalence_b": prevalence_b,
                "lift": lift,
                "pmi": pmi,
                "npmi": npmi,
                "jaccard": jaccard,
                "rarity_weight": rarity_weight,
                "synergy_score": synergy_score,
            }
        )

    return pl.DataFrame(metrics_rows).sort(["synergy_score", "count"], descending=[True, True])


def _systemic_item_signatures(board_frame: pl.DataFrame, pair_frame: pl.DataFrame, total_boards: int) -> pl.DataFrame:
    if not total_boards or not board_frame.height:
        return pl.DataFrame(
            schema={
                "item_name": pl.String,
                "board_count": pl.Int64,
                "prevalence": pl.Float64,
                "partner_count": pl.Int64,
                "top_partner": pl.String,
                "top_partner_count": pl.Int64,
                "top_partner_synergy": pl.Float64,
                "top3_pair_share": pl.Float64,
                "idf": pl.Float64,
                "signature_score": pl.Float64,
            }
        )

    item_counts = board_frame.group_by("item_name").len(name="board_count")
    pair_rows = pair_frame.iter_rows(named=True)
    partner_map: dict[str, list[dict[str, float | int | str]]] = {}
    for row in pair_rows:
        partner_map.setdefault(row["item_a"], []).append({"partner": row["item_b"], **row})
        partner_map.setdefault(row["item_b"], []).append({"partner": row["item_a"], **row})

    signature_rows: list[dict[str, float | int | str | None]] = []
    for row in item_counts.iter_rows(named=True):
        item_name = row["item_name"]
        board_count = int(row["board_count"])
        prevalence = board_count / total_boards
        idf = math.log((total_boards + 1) / (board_count + 1)) + 1.0
        partner_rows = sorted(partner_map.get(item_name, []), key=lambda item: (item["synergy_score"], item["count"]), reverse=True)
        meaningful_partners = [partner for partner in partner_rows if int(partner["count"]) >= 2]
        top3 = meaningful_partners[:3]
        total_pair_mass = sum(int(partner["count"]) for partner in meaningful_partners)
        top3_pair_share = (sum(int(partner["count"]) for partner in top3) / total_pair_mass) if total_pair_mass else 0.0
        top_partner = top3[0]["partner"] if top3 else None
        top_partner_count = int(top3[0]["count"]) if top3 else 0
        top_partner_synergy = float(top3[0]["synergy_score"]) if top3 else 0.0
        avg_top_synergy = (sum(float(partner["synergy_score"]) for partner in top3) / len(top3)) if top3 else 0.0
        support_weight = math.log1p(board_count)
        signature_score = idf * top3_pair_share * avg_top_synergy * support_weight
        signature_rows.append(
            {
                "item_name": item_name,
                "board_count": board_count,
                "prevalence": prevalence,
                "partner_count": len(partner_rows),
                "top_partner": top_partner,
                "top_partner_count": top_partner_count,
                "top_partner_synergy": top_partner_synergy,
                "top3_pair_share": top3_pair_share,
                "idf": idf,
                "signature_score": signature_score,
            }
        )

    return pl.DataFrame(signature_rows).sort(["signature_score", "board_count"], descending=[True, True])


def _systemic_archetypes(board_frame: pl.DataFrame, signature_frame: pl.DataFrame) -> pl.DataFrame:
    if not board_frame.height or not signature_frame.height:
        return pl.DataFrame(
            schema={
                "archetype_anchor_a": pl.String,
                "archetype_anchor_b": pl.String,
                "board_count": pl.Int64,
                "example_title": pl.String,
                "items_json": pl.String,
            }
        )

    signature_map = dict(
        zip(
            signature_frame.get_column("item_name").to_list(),
            signature_frame.get_column("signature_score").to_list(),
            strict=False,
        )
    )
    grouped = board_frame.group_by(["screenshot_id", "run_id", "title"]).agg(pl.col("item_name")).sort("screenshot_id")
    archetype_rows: list[dict[str, str | int]] = []
    for row in grouped.iter_rows(named=True):
        items = sorted(set(row["item_name"]))
        ranked_items = sorted(items, key=lambda item: (signature_map.get(item, 0.0), item), reverse=True)
        anchors = ranked_items[:2]
        if len(anchors) == 1:
            anchors = [anchors[0], anchors[0]]
        if not anchors:
            continue
        anchor_a, anchor_b = sorted(anchors)
        archetype_rows.append(
            {
                "archetype_anchor_a": anchor_a,
                "archetype_anchor_b": anchor_b,
                "example_title": row["title"],
                "items_json": json.dumps(items),
            }
        )

    archetypes = pl.DataFrame(archetype_rows)
    return (
        archetypes.group_by(["archetype_anchor_a", "archetype_anchor_b"])
        .agg(
            pl.len().alias("board_count"),
            pl.first("example_title").alias("example_title"),
            pl.first("items_json").alias("items_json"),
        )
        .sort("board_count", descending=True)
    )


def _item_source_alignment(conn, board_frame: pl.DataFrame) -> pl.DataFrame:
    item_counts = board_frame.group_by("item_name").len(name="board_count") if board_frame.height else pl.DataFrame({"item_name": [], "board_count": []})
    source_rows = conn.execute("SELECT board_cards_json FROM runs").fetchall()
    source_counter: dict[str, int] = {}
    for row in source_rows:
        for card in json.loads(row["board_cards_json"] or "[]"):
            if not isinstance(card, dict):
                continue
            item_name = card.get("title")
            if item_name:
                source_counter[item_name] = source_counter.get(item_name, 0) + 1

    names = sorted(set(item_counts.get_column("item_name").to_list()) | set(source_counter)) if item_counts.height else sorted(source_counter)
    rows: list[dict[str, float | int | str | None]] = []
    item_count_map = dict(zip(item_counts.get_column("item_name").to_list(), item_counts.get_column("board_count").to_list(), strict=False)) if item_counts.height else {}
    for name in names:
        board_count = int(item_count_map.get(name, 0))
        source_count = int(source_counter.get(name, 0))
        extraction_to_source_ratio = (board_count / source_count) if source_count else None
        rows.append(
            {
                "item_name": name,
                "board_count": board_count,
                "source_count": source_count,
                "extraction_to_source_ratio": extraction_to_source_ratio,
                "board_minus_source": board_count - source_count,
            }
        )
    return pl.DataFrame(rows).sort(["board_minus_source", "board_count"], descending=[True, True]) if rows else pl.DataFrame(
        schema={
            "item_name": pl.String,
            "board_count": pl.Int64,
            "source_count": pl.Int64,
            "extraction_to_source_ratio": pl.Float64,
            "board_minus_source": pl.Int64,
        }
    )


def systemic_analysis(conn, settings: Settings) -> dict[str, int]:
    board_frame, total_boards = _board_presence_frame(conn)
    pair_frame = _systemic_item_pairs(board_frame, total_boards)
    signature_frame = _systemic_item_signatures(board_frame, pair_frame, total_boards)
    archetype_frame = _systemic_archetypes(board_frame, signature_frame)
    source_alignment = _item_source_alignment(conn, board_frame)

    pair_frame.write_csv(settings.exports_dir / "summary_systemic_item_pairs.csv")
    signature_frame.write_csv(settings.exports_dir / "summary_systemic_item_signatures.csv")
    archetype_frame.write_csv(settings.exports_dir / "summary_systemic_archetypes.csv")
    source_alignment.write_csv(settings.exports_dir / "summary_item_source_alignment.csv")

    return {
        "boards": total_boards,
        "systemic_pairs": pair_frame.height,
        "signature_items": signature_frame.height,
        "archetypes": archetype_frame.height,
        "source_alignment_rows": source_alignment.height,
    }


def summarize(conn, settings: Settings) -> dict[str, int]:
    item_frame = conn.query_pl(
        """
        SELECT e.screenshot_id, COALESCE(r.name, e.raw_label) AS item_name
        FROM extracted_board_items e
        JOIN screenshots s ON s.screenshot_id = e.screenshot_id
        JOIN runs ON runs.run_id = s.run_id
        LEFT JOIN reference_items r ON r.entity_id = e.entity_id
        WHERE e.status = 'ok'
        """
    )
    skill_frame = conn.query_pl(
        """
        SELECT e.screenshot_id, COALESCE(r.name, e.raw_label) AS skill_name
        FROM extracted_skills e
        JOIN screenshots s ON s.screenshot_id = e.screenshot_id
        JOIN runs ON runs.run_id = s.run_id
        LEFT JOIN reference_skills r ON r.entity_id = e.entity_id
        WHERE e.status = 'ok'
        """
    )
    outcome_frame = conn.query_pl(
        """
        SELECT s.screenshot_id, r.outcome_text
        FROM screenshots s
        JOIN runs r ON r.run_id = s.run_id
        WHERE s.is_primary = 1
        """
    )

    top_items = item_frame.group_by("item_name").len(name="count").sort("count", descending=True)
    top_skills = skill_frame.group_by("skill_name").len(name="count").sort("count", descending=True)
    item_lists = item_frame.group_by("screenshot_id").agg(pl.col("item_name")).get_column("item_name").to_list() if item_frame.height else []
    item_pair_counts = _cooccurrence(item_lists, "item_a", "item_b")

    item_skill_join = item_frame.join(skill_frame, on="screenshot_id", how="inner")
    item_skill_counts = item_skill_join.group_by(["item_name", "skill_name"]).len(name="count").sort("count", descending=True)

    outcome_items = item_frame.join(outcome_frame, on="screenshot_id", how="left")
    outcome_item_counts = outcome_items.filter(pl.col("outcome_text").is_not_null()).group_by(["outcome_text", "item_name"]).len(name="count").sort(["outcome_text", "count"], descending=[False, True])
    coverage = _pipeline_coverage_summary(conn)

    top_items.write_csv(settings.exports_dir / "summary_top_items.csv")
    top_skills.write_csv(settings.exports_dir / "summary_top_skills.csv")
    item_pair_counts.write_csv(settings.exports_dir / "summary_item_item_cooccurrence.csv")
    item_skill_counts.write_csv(settings.exports_dir / "summary_item_skill_cooccurrence.csv")
    outcome_item_counts.write_csv(settings.exports_dir / "summary_outcome_filtered_items.csv")
    coverage.write_csv(settings.exports_dir / "summary_pipeline_coverage.csv")

    return {
        "top_items": top_items.height,
        "top_skills": top_skills.height,
        "item_item_pairs": item_pair_counts.height,
        "item_skill_pairs": item_skill_counts.height,
        "outcome_filtered_rows": outcome_item_counts.height,
        "pipeline_coverage_rows": coverage.height,
    }
