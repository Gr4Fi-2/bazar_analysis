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
            run_meta.record_wins,
            run_meta.run_outcome_tier,
            run_meta.player_rank_tier,
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


def _skill_presence_frame(conn) -> pl.DataFrame:
    return conn.query_pl(
        """
        SELECT
            e.screenshot_id,
            s.run_id,
            run_meta.title,
            run_meta.record_wins,
            run_meta.run_outcome_tier,
            COALESCE(ref_skill.name, e.raw_label) AS skill_name
        FROM extracted_skills e
        JOIN screenshots s ON s.screenshot_id = e.screenshot_id
        JOIN runs run_meta ON run_meta.run_id = s.run_id
        LEFT JOIN reference_skills ref_skill ON ref_skill.entity_id = e.entity_id
        WHERE e.status = 'ok'
        """
    )


def _run_meta_frame(conn) -> pl.DataFrame:
    return conn.query_pl(
        """
        SELECT
            s.screenshot_id,
            r.run_id,
            r.title,
            r.record_wins,
            r.run_wins_label,
            r.run_outcome_tier,
            r.player_rank_tier
        FROM screenshots s
        JOIN runs r ON r.run_id = s.run_id
        WHERE s.is_primary = 1
        ORDER BY s.screenshot_id
        """
    )


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


def _json_name_counts(values: list[tuple[str, float]], *, top_n: int = 8) -> str:
    ordered = [
        {"name": name, "rate": round(rate, 4)}
        for name, rate in sorted(values, key=lambda item: (-item[1], item[0]))[:top_n]
    ]
    return json.dumps(ordered, ensure_ascii=True)


def _json_counter(counter: dict[str, int]) -> str:
    ordered = [{"name": name, "count": count} for name, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]
    return json.dumps(ordered, ensure_ascii=True)


def _empty_build_cluster_outputs() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    empty_profiles = pl.DataFrame(
        schema={
            "archetype_anchor_a": pl.String,
            "archetype_anchor_b": pl.String,
            "board_count": pl.Int64,
            "avg_wins": pl.Float64,
            "median_wins": pl.Float64,
            "top_outcome": pl.String,
            "top_player_rank": pl.String,
            "example_title": pl.String,
            "core_items_json": pl.String,
            "flex_items_json": pl.String,
            "tech_items_json": pl.String,
            "top_skills_json": pl.String,
            "outcome_distribution_json": pl.String,
            "player_rank_distribution_json": pl.String,
        }
    )
    empty_components = pl.DataFrame(
        schema={
            "archetype_anchor_a": pl.String,
            "archetype_anchor_b": pl.String,
            "component_name": pl.String,
            "component_kind": pl.String,
            "presence_rate": pl.Float64,
            "board_count": pl.Int64,
            "avg_wins": pl.Float64,
        }
    )
    empty_core_builds = pl.DataFrame(
        schema={
            "core_build_key": pl.String,
            "core_item_count": pl.Int64,
            "cluster_count": pl.Int64,
            "board_count": pl.Int64,
            "avg_wins": pl.Float64,
            "core_items_json": pl.String,
            "top_flex_items_json": pl.String,
            "top_skills_json": pl.String,
            "top_outcome": pl.String,
            "outcome_distribution_json": pl.String,
            "player_rank_distribution_json": pl.String,
            "example_archetypes_json": pl.String,
        }
    )
    return empty_profiles, empty_components, empty_core_builds


def _build_core_builds(cluster_profile_frame: pl.DataFrame) -> pl.DataFrame:
    if not cluster_profile_frame.height:
        return _empty_build_cluster_outputs()[2]

    grouped: dict[tuple[str, ...], dict[str, object]] = {}
    for row in cluster_profile_frame.iter_rows(named=True):
        core_items = tuple(sorted(entry["name"] for entry in json.loads(row["core_items_json"]) if entry.get("name")))
        if not core_items:
            continue
        group = grouped.setdefault(
            core_items,
            {
                "cluster_count": 0,
                "board_count": 0,
                "wins_weighted_sum": 0.0,
                "flex_counter": {},
                "skill_counter": {},
                "outcome_counter": {},
                "rank_counter": {},
                "example_archetypes": [],
            },
        )
        board_count = int(row["board_count"])
        group["cluster_count"] = int(group["cluster_count"]) + 1
        group["board_count"] = int(group["board_count"]) + board_count
        group["wins_weighted_sum"] = float(group["wins_weighted_sum"]) + (float(row["avg_wins"] or 0.0) * board_count)
        for entry in json.loads(row["flex_items_json"]):
            if entry.get("name"):
                flex_counter = group["flex_counter"]
                flex_counter[entry["name"]] = float(flex_counter.get(entry["name"], 0.0)) + float(entry.get("rate") or 0.0) * board_count
        for entry in json.loads(row["top_skills_json"]):
            if entry.get("name"):
                skill_counter = group["skill_counter"]
                skill_counter[entry["name"]] = float(skill_counter.get(entry["name"], 0.0)) + float(entry.get("rate") or 0.0) * board_count
        for entry in json.loads(row["outcome_distribution_json"]):
            if entry.get("name"):
                outcome_counter = group["outcome_counter"]
                outcome_counter[entry["name"]] = int(outcome_counter.get(entry["name"], 0)) + int(entry.get("count") or 0)
        for entry in json.loads(row["player_rank_distribution_json"]):
            if entry.get("name"):
                rank_counter = group["rank_counter"]
                rank_counter[entry["name"]] = int(rank_counter.get(entry["name"], 0)) + int(entry.get("count") or 0)
        example_archetypes = group["example_archetypes"]
        example_archetypes.append({
            "anchors": [row["archetype_anchor_a"], row["archetype_anchor_b"]],
            "board_count": board_count,
            "avg_wins": round(float(row["avg_wins"] or 0.0), 4),
        })

    rows: list[dict[str, object]] = []
    for core_items, payload in grouped.items():
        board_count = int(payload["board_count"])
        if board_count <= 0:
            continue
        flex_entries = [
            (name, support / board_count)
            for name, support in payload["flex_counter"].items()
            if (support / board_count) >= 0.10
        ]
        skill_entries = [
            (name, support / board_count)
            for name, support in payload["skill_counter"].items()
            if (support / board_count) >= 0.10
        ]
        outcome_counter = payload["outcome_counter"]
        rank_counter = payload["rank_counter"]
        top_outcome = max(outcome_counter.items(), key=lambda item: (item[1], item[0]))[0] if outcome_counter else None
        example_archetypes = sorted(payload["example_archetypes"], key=lambda item: (-item["board_count"], -item["avg_wins"], item["anchors"]))[:8]
        rows.append(
            {
                "core_build_key": " | ".join(core_items),
                "core_item_count": len(core_items),
                "cluster_count": int(payload["cluster_count"]),
                "board_count": board_count,
                "avg_wins": float(payload["wins_weighted_sum"]) / board_count,
                "core_items_json": json.dumps(list(core_items), ensure_ascii=True),
                "top_flex_items_json": _json_name_counts(flex_entries),
                "top_skills_json": _json_name_counts(skill_entries),
                "top_outcome": top_outcome,
                "outcome_distribution_json": _json_counter(outcome_counter),
                "player_rank_distribution_json": _json_counter(rank_counter),
                "example_archetypes_json": json.dumps(example_archetypes, ensure_ascii=True),
            }
        )

    if not rows:
        return _empty_build_cluster_outputs()[2]
    return pl.DataFrame(rows).sort(["core_item_count", "board_count", "avg_wins"], descending=[True, True, True])


def _build_cluster_profiles(board_frame: pl.DataFrame, skill_frame: pl.DataFrame, signature_frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if not board_frame.height or not signature_frame.height:
        return _empty_build_cluster_outputs()

    signature_map = dict(
        zip(
            signature_frame.get_column("item_name").to_list(),
            signature_frame.get_column("signature_score").to_list(),
            strict=False,
        )
    )
    grouped_boards = board_frame.group_by(["screenshot_id", "run_id", "title", "record_wins", "run_outcome_tier", "player_rank_tier"]).agg(pl.col("item_name"))
    skill_lists = {
        row["screenshot_id"]: sorted(set(row["skill_name"]))
        for row in skill_frame.group_by("screenshot_id").agg(pl.col("skill_name")).iter_rows(named=True)
    } if skill_frame.height else {}

    archetype_rows: list[dict[str, object]] = []
    for row in grouped_boards.iter_rows(named=True):
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
                "screenshot_id": row["screenshot_id"],
                "title": row["title"],
                "record_wins": row["record_wins"],
                "run_outcome_tier": row["run_outcome_tier"],
                "items": items,
                "skills": skill_lists.get(row["screenshot_id"], []),
                "player_rank_tier": row["player_rank_tier"],
            }
        )

    if not archetype_rows:
        return _empty_build_cluster_outputs()

    profile_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    cluster_map: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in archetype_rows:
        cluster_map.setdefault((row["archetype_anchor_a"], row["archetype_anchor_b"]), []).append(row)

    for (anchor_a, anchor_b), rows in sorted(cluster_map.items()):
        board_count = len(rows)
        wins = [int(row["record_wins"]) for row in rows if row["record_wins"] is not None]
        item_counter: dict[str, int] = {}
        skill_counter: dict[str, int] = {}
        outcome_counter: dict[str, int] = {}
        rank_counter: dict[str, int] = {}
        for row in rows:
            for item in set(row["items"]):
                item_counter[item] = item_counter.get(item, 0) + 1
            for skill in set(row["skills"]):
                skill_counter[skill] = skill_counter.get(skill, 0) + 1
            outcome = row["run_outcome_tier"] or "Unknown"
            outcome_counter[outcome] = outcome_counter.get(outcome, 0) + 1
            player_rank = row["player_rank_tier"] or "Unknown"
            rank_counter[player_rank] = rank_counter.get(player_rank, 0) + 1

        core_items: list[tuple[str, float]] = []
        flex_items: list[tuple[str, float]] = []
        tech_items: list[tuple[str, float]] = []
        for item_name, count in item_counter.items():
            presence_rate = count / board_count
            if presence_rate >= 0.75:
                component_kind = "core"
                core_items.append((item_name, presence_rate))
            elif presence_rate >= 0.35:
                component_kind = "flex"
                flex_items.append((item_name, presence_rate))
            else:
                component_kind = "tech"
                tech_items.append((item_name, presence_rate))
            item_wins = [int(row["record_wins"]) for row in rows if row["record_wins"] is not None and item_name in row["items"]]
            component_rows.append(
                {
                    "archetype_anchor_a": anchor_a,
                    "archetype_anchor_b": anchor_b,
                    "component_name": item_name,
                    "component_kind": component_kind,
                    "presence_rate": presence_rate,
                    "board_count": count,
                    "avg_wins": (sum(item_wins) / len(item_wins)) if item_wins else None,
                }
            )

        top_skills = [(skill_name, count / board_count) for skill_name, count in skill_counter.items() if count / board_count >= 0.20]
        top_outcome = max(outcome_counter.items(), key=lambda item: (item[1], item[0]))[0] if outcome_counter else None
        top_player_rank = max(rank_counter.items(), key=lambda item: (item[1], item[0]))[0] if rank_counter else None
        profile_rows.append(
            {
                "archetype_anchor_a": anchor_a,
                "archetype_anchor_b": anchor_b,
                "board_count": board_count,
                "avg_wins": (sum(wins) / len(wins)) if wins else None,
                "median_wins": float(pl.Series(wins).median()) if wins else None,
                "top_outcome": top_outcome,
                "top_player_rank": top_player_rank,
                "example_title": rows[0]["title"],
                "core_items_json": _json_name_counts(core_items),
                "flex_items_json": _json_name_counts(flex_items),
                "tech_items_json": _json_name_counts(tech_items),
                "top_skills_json": _json_name_counts(top_skills),
                "outcome_distribution_json": _json_counter(outcome_counter),
                "player_rank_distribution_json": _json_counter(rank_counter),
            }
        )

    cluster_profile_frame = pl.DataFrame(profile_rows).sort(["board_count", "avg_wins"], descending=[True, True])
    cluster_component_frame = pl.DataFrame(component_rows).sort(["board_count", "presence_rate"], descending=[True, True])
    core_build_frame = _build_core_builds(cluster_profile_frame)
    return cluster_profile_frame, cluster_component_frame, core_build_frame


def _performance_by_entity(frame: pl.DataFrame, entity_column: str) -> pl.DataFrame:
    if not frame.height:
        return pl.DataFrame(
            schema={
                entity_column: pl.String,
                "run_count": pl.Int64,
                "avg_wins": pl.Float64,
                "median_wins": pl.Float64,
                "wins_10_rate": pl.Float64,
                "top_outcome": pl.String,
            }
        )
    return (
        frame.group_by(entity_column)
        .agg(
            pl.col("screenshot_id").n_unique().alias("run_count"),
            pl.col("record_wins").drop_nulls().mean().alias("avg_wins"),
            pl.col("record_wins").drop_nulls().median().alias("median_wins"),
            (pl.col("record_wins") == 10).mean().alias("wins_10_rate"),
            pl.col("run_outcome_tier").drop_nulls().mode().first().alias("top_outcome"),
        )
        .sort(["avg_wins", "run_count"], descending=[True, True])
    )


def _counts_with_performance(count_frame: pl.DataFrame, performance_frame: pl.DataFrame, entity_column: str) -> pl.DataFrame:
    if not count_frame.height:
        return pl.DataFrame(
            schema={
                entity_column: pl.String,
                "count": pl.Int64,
                "avg_wins": pl.Float64,
                "median_wins": pl.Float64,
                "wins_10_rate": pl.Float64,
                "top_outcome": pl.String,
            }
        )
    return (
        count_frame.join(performance_frame, on=entity_column, how="left")
        .sort(["count", "avg_wins"], descending=[True, True])
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
    skill_frame = _skill_presence_frame(conn)
    pair_frame = _systemic_item_pairs(board_frame, total_boards)
    signature_frame = _systemic_item_signatures(board_frame, pair_frame, total_boards)
    archetype_frame = _systemic_archetypes(board_frame, signature_frame)
    cluster_profile_frame, cluster_component_frame, core_build_frame = _build_cluster_profiles(board_frame, skill_frame, signature_frame)
    source_alignment = _item_source_alignment(conn, board_frame)

    pair_frame.write_csv(settings.exports_dir / "summary_systemic_item_pairs.csv")
    signature_frame.write_csv(settings.exports_dir / "summary_systemic_item_signatures.csv")
    archetype_frame.write_csv(settings.exports_dir / "summary_systemic_archetypes.csv")
    cluster_profile_frame.write_csv(settings.exports_dir / "summary_build_clusters.csv")
    cluster_component_frame.write_csv(settings.exports_dir / "summary_build_components.csv")
    core_build_frame.write_csv(settings.exports_dir / "summary_core_builds.csv")
    source_alignment.write_csv(settings.exports_dir / "summary_item_source_alignment.csv")

    return {
        "boards": total_boards,
        "systemic_pairs": pair_frame.height,
        "signature_items": signature_frame.height,
        "archetypes": archetype_frame.height,
        "build_clusters": cluster_profile_frame.height,
        "build_components": cluster_component_frame.height,
        "core_builds": core_build_frame.height,
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
    run_meta_frame = _run_meta_frame(conn)

    item_perf_frame = item_frame.join(run_meta_frame.select(["screenshot_id", "record_wins", "run_outcome_tier"]), on="screenshot_id", how="left") if item_frame.height else pl.DataFrame(schema={"screenshot_id": pl.Int64, "item_name": pl.String, "record_wins": pl.Int64, "run_outcome_tier": pl.String})
    skill_perf_frame = skill_frame.join(run_meta_frame.select(["screenshot_id", "record_wins", "run_outcome_tier"]), on="screenshot_id", how="left") if skill_frame.height else pl.DataFrame(schema={"screenshot_id": pl.Int64, "skill_name": pl.String, "record_wins": pl.Int64, "run_outcome_tier": pl.String})

    top_items = item_frame.group_by("item_name").len(name="count").sort("count", descending=True)
    top_skills = skill_frame.group_by("skill_name").len(name="count").sort("count", descending=True)
    item_lists = item_frame.group_by("screenshot_id").agg(pl.col("item_name")).get_column("item_name").to_list() if item_frame.height else []
    item_pair_counts = _cooccurrence(item_lists, "item_a", "item_b")

    item_skill_join = item_frame.join(skill_frame, on="screenshot_id", how="inner")
    item_skill_counts = item_skill_join.group_by(["item_name", "skill_name"]).len(name="count").sort("count", descending=True)

    outcome_items = item_frame.join(outcome_frame, on="screenshot_id", how="left")
    outcome_item_counts = outcome_items.filter(pl.col("outcome_text").is_not_null()).group_by(["outcome_text", "item_name"]).len(name="count").sort(["outcome_text", "count"], descending=[False, True])
    coverage = _pipeline_coverage_summary(conn)
    item_performance = _performance_by_entity(item_perf_frame, "item_name")
    skill_performance = _performance_by_entity(skill_perf_frame, "skill_name")
    item_counts_performance = _counts_with_performance(top_items, item_performance, "item_name")

    top_items.write_csv(settings.exports_dir / "summary_top_items.csv")
    top_skills.write_csv(settings.exports_dir / "summary_top_skills.csv")
    item_pair_counts.write_csv(settings.exports_dir / "summary_item_item_cooccurrence.csv")
    item_skill_counts.write_csv(settings.exports_dir / "summary_item_skill_cooccurrence.csv")
    outcome_item_counts.write_csv(settings.exports_dir / "summary_outcome_filtered_items.csv")
    coverage.write_csv(settings.exports_dir / "summary_pipeline_coverage.csv")
    item_performance.write_csv(settings.exports_dir / "summary_item_performance.csv")
    skill_performance.write_csv(settings.exports_dir / "summary_skill_performance.csv")
    item_counts_performance.write_csv(settings.exports_dir / "summary_item_counts_performance.csv")

    return {
        "top_items": top_items.height,
        "top_skills": top_skills.height,
        "item_item_pairs": item_pair_counts.height,
        "item_skill_pairs": item_skill_counts.height,
        "outcome_filtered_rows": outcome_item_counts.height,
        "pipeline_coverage_rows": coverage.height,
        "item_performance_rows": item_performance.height,
        "skill_performance_rows": skill_performance.height,
        "item_counts_performance_rows": item_counts_performance.height,
    }
