from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
import datetime as dt
from itertools import combinations

import polars as pl

from .config import Settings
from .crawler import SEASON_START_DATES
from .reporting import write_frame_exports


PERFORMANCE_PRIOR_RUNS = 20.0
MIN_CORE_BUILD_CLUSTER_BOARDS = 3
ARCHETYPE_FAMILY_JACCARD_THRESHOLD = 0.50
CARD_ID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", flags=re.IGNORECASE)
GOLD_PLUS_TIERS = {"Gold", "Perfect"}
MECHANIC_LABEL_PATTERNS = [
    ("Food", (r"\bfood\b", r"cake", r"batter", r"pasta", r"froyo", r"ice cream", r"sundae", r"pepper", r"spice", r"sauce", r"fruit", r"berry", r"market", r"platter", r"kitchen", r"oven", r"pan", r"skillet", r"pizza", r"burger", r"noodle", r"soup", r"honey")),
    ("Spice", (r"spice", r"pepper", r"chili", r"scorch", r"sauce")),
    ("Freeze", (r"freeze", r"freezer", r"frozen", r"frost", r"snow", r"\bice\b", r"swan", r"refrigerator", r"cool")),
    ("Burn", (r"burn", r"fire", r"fiery", r"flame", r"ignite", r"incendiary", r"scorch")),
    ("Slow", (r"slow", r"snail", r"mud", r"freeze", r"frost")),
    ("Crit", (r"crit", r"critical", r"deadly", r"keen eye")),
    ("Health/Regen", (r"health", r"heal", r"regen", r"heart", r"lifesteal", r"vital")),
    ("Weapon", (r"weapon", r"knife", r"sword", r"gun", r"pistol", r"rifle", r"cannon", r"blade", r"claw", r"harpoon", r"ramrod")),
    ("Property", (r"property", r"real estate", r"market", r"shop", r"store", r"refrigerator", r"freezer", r"cart")),
    ("Ammo", (r"ammo", r"rounds", r"bullet", r"reload", r"magazine")),
    ("Shield", (r"shield", r"armor", r"barrier", r"guard")),
    ("Haste", (r"haste", r"speed", r"quick", r"acceleration")),
    ("Poison", (r"poison", r"toxic", r"venom")),
    ("Economy", (r"gold", r"income", r"coin", r"cash", r"money", r"piggy", r"bank", r"market")),
]


@dataclass(frozen=True)
class AnalysisFilters:
    heroes: frozenset[str]
    date_range: str | None
    created_after: dt.datetime | None
    created_before: dt.datetime | None


def _parse_analysis_timestamp(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    normalized = value.strip()
    try:
        parsed = dt.datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.UTC)
        return parsed.astimezone(dt.UTC)
    except ValueError:
        pass
    try:
        return dt.datetime.strptime(normalized, "%a, %d %b %Y %H:%M:%S GMT").replace(tzinfo=dt.UTC)
    except ValueError:
        return None


def _format_gmt_timestamp(value: dt.datetime) -> str:
    return value.astimezone(dt.UTC).strftime("%a, %d %b %Y %H:%M:%S GMT")


def _created_bounds_for_analysis_date_range(date_range: str, now: dt.datetime | None = None) -> tuple[str | None, str | None]:
    now = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
    if date_range == "last24h":
        return (_format_gmt_timestamp(now - dt.timedelta(hours=24)), None)
    if date_range == "last3d":
        return (_format_gmt_timestamp(now - dt.timedelta(days=3)), None)
    if date_range == "last7d":
        return (_format_gmt_timestamp(now - dt.timedelta(days=7)), None)
    if date_range == "latest_season":
        return (SEASON_START_DATES["season15"], None)
    if date_range == "season15":
        return (SEASON_START_DATES["season15"], None)
    if date_range == "season14":
        return (SEASON_START_DATES["season14"], SEASON_START_DATES["season15"])
    if date_range == "season13":
        return (SEASON_START_DATES["season13"], SEASON_START_DATES["season14"])
    return (None, None)


def _analysis_filters_from_values(
    heroes_raw: str = "",
    date_range_raw: str = "",
    created_after_raw: str = "",
    created_before_raw: str = "",
    *,
    now: dt.datetime | None = None,
) -> AnalysisFilters:
    heroes: set[str] = set()
    for token in heroes_raw.split(","):
        hero = token.strip()
        if not hero:
            continue
        if hero.lower() in {"all", "*"}:
            heroes.clear()
            break
        heroes.add(hero.title())

    date_range = date_range_raw.strip().lower()
    default_after, default_before = _created_bounds_for_analysis_date_range(date_range, now=now) if date_range else (None, None)
    created_after = created_after_raw.strip() or default_after
    created_before = created_before_raw.strip() or default_before
    return AnalysisFilters(
        heroes=frozenset(heroes),
        date_range=date_range or None,
        created_after=_parse_analysis_timestamp(created_after),
        created_before=_parse_analysis_timestamp(created_before),
    )


def _load_analysis_filters() -> AnalysisFilters:
    return _analysis_filters_from_values(
        os.environ.get("BAZAR_ANALYSIS_HEROES", ""),
        os.environ.get("BAZAR_ANALYSIS_DATE_RANGE", ""),
        os.environ.get("BAZAR_ANALYSIS_CREATED_AFTER", ""),
        os.environ.get("BAZAR_ANALYSIS_CREATED_BEFORE", ""),
    )


def _created_at_matches(value: str | None, filters: AnalysisFilters) -> bool:
    if filters.created_after is None and filters.created_before is None:
        return True
    parsed = _parse_analysis_timestamp(value)
    if parsed is None:
        return False
    if filters.created_after is not None and parsed < filters.created_after:
        return False
    if filters.created_before is not None and parsed >= filters.created_before:
        return False
    return True


def _filter_analysis_frame(frame: pl.DataFrame, filters: AnalysisFilters | None = None) -> pl.DataFrame:
    filters = filters or _load_analysis_filters()
    if not frame.height:
        return frame
    filtered = frame
    if filters.heroes and "hero" in filtered.columns:
        filtered = filtered.filter(pl.col("hero").is_in(sorted(filters.heroes)))
    if (filters.created_after is not None or filters.created_before is not None) and "created_at" in filtered.columns:
        filtered = filtered.filter(
            pl.col("created_at").map_elements(lambda value: _created_at_matches(value, filters), return_dtype=pl.Boolean)
        )
    return filtered


def _cooccurrence(rows: list[list[str]], left_name: str, right_name: str) -> pl.DataFrame:
    pair_counts: Counter[tuple[str, str]] = Counter()
    for values in rows:
        unique_values = sorted(set(value for value in values if value))
        pair_counts.update(combinations(unique_values, 2))
    if not pair_counts:
        return pl.DataFrame(schema={left_name: pl.String, right_name: pl.String, "count": pl.Int64})
    return pl.DataFrame(
        [(left, right, count) for (left, right), count in pair_counts.items()],
        schema=[left_name, right_name, "count"],
        orient="row",
    ).sort("count", descending=True)


def _pipeline_coverage_summary(conn, filters: AnalysisFilters | None = None) -> pl.DataFrame:
    frame = conn.query_pl(
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
                SUM(CASE WHEN detection_type = 'prestige_state' THEN 1 ELSE 0 END) AS review_prestige_state,
                SUM(CASE WHEN detection_type = 'screenshot_layout' THEN 1 ELSE 0 END) AS review_layout,
                SUM(CASE WHEN detection_type = 'screenshot_file' THEN 1 ELSE 0 END) AS review_files
            FROM review_queue
            GROUP BY screenshot_id
        )
        SELECT
            run_meta.run_id,
            run_meta.hero,
            run_meta.created_at,
            run_meta.title,
            run_meta.record_wins,
            run_meta.run_wins_label,
            run_meta.run_victory_tier,
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
            COALESCE(rv.review_prestige_state, 0) AS review_prestige_state,
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
    return _filter_analysis_frame(frame, filters)

def _empty_presence_frame(entity_column: str) -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "screenshot_id": pl.Int64,
            "run_id": pl.Int64,
            "hero": pl.String,
            "created_at": pl.String,
            "title": pl.String,
            "record_wins": pl.Int64,
            "run_victory_tier": pl.String,
            "has_broken_crown": pl.Int64,
            "player_rank_tier": pl.String,
            entity_column: pl.String,
            "duplicate_count": pl.Int64,
            "source_method": pl.String,
        }
    )


def _looks_like_card_id(value: str | None) -> bool:
    return bool(value and CARD_ID_PATTERN.match(value.strip()))


def _source_card_name_lookup(conn, cards_column: str) -> dict[str, str]:
    rows = conn.execute(
        f"""
        SELECT CAST({cards_column} AS VARCHAR) AS cards_json
        FROM runs
        WHERE {cards_column} IS NOT NULL
          AND json_array_length({cards_column}) > 0
        """
    ).fetchall()
    name_counts: dict[str, Counter[str]] = {}
    for row in rows:
        try:
            cards = json.loads(row["cards_json"] or "[]")
        except json.JSONDecodeError:
            continue
        for card in cards:
            if not isinstance(card, dict):
                continue
            base_id = card.get("base_id") or card.get("cardId") or card.get("baseId")
            title = card.get("title") or card.get("name")
            if not base_id or not title or _looks_like_card_id(str(title)):
                continue
            name_counts.setdefault(str(base_id), Counter())[str(title)] += 1
    return {base_id: counter.most_common(1)[0][0] for base_id, counter in name_counts.items() if counter}


def _parse_card_names(cards_json: str | None, card_name_lookup: dict[str, str] | None = None) -> list[str]:
    if not cards_json:
        return []
    try:
        parsed = json.loads(cards_json)
    except json.JSONDecodeError:
        return []
    names: list[str] = []
    for card in parsed:
        if not isinstance(card, dict):
            continue
        base_id = card.get("base_id") or card.get("cardId") or card.get("baseId")
        title = card.get("title") or card.get("name")
        if title and not _looks_like_card_id(str(title)):
            name = title
        elif base_id and card_name_lookup:
            name = card_name_lookup.get(str(base_id), base_id)
        else:
            name = base_id or title
        if name and not _looks_like_card_id(str(name)):
            names.append(str(name))
    return names


def _source_card_presence_frame(conn, cards_column: str, entity_column: str, source_method: str) -> pl.DataFrame:
    card_name_lookup = _source_card_name_lookup(conn, cards_column)
    rows = conn.execute(
        f"""
        SELECT
            COALESCE(s.screenshot_id, -r.run_id) AS screenshot_id,
            r.run_id,
            r.hero,
            r.created_at,
            r.title,
            r.record_wins,
            r.run_victory_tier,
            r.has_broken_crown,
            r.player_rank_tier,
            CAST(r.{cards_column} AS VARCHAR) AS cards_json
        FROM runs r
        LEFT JOIN screenshots s ON s.run_id = r.run_id AND s.is_primary = 1
        WHERE r.{cards_column} IS NOT NULL
          AND json_array_length(r.{cards_column}) > 0
        ORDER BY r.run_id
        """
    ).fetchall()
    output_rows: list[dict[str, object]] = []
    for row in rows:
        name_counts = Counter(_parse_card_names(row["cards_json"], card_name_lookup))
        for name, duplicate_count in sorted(name_counts.items()):
            output_rows.append(
                {
                    "screenshot_id": row["screenshot_id"],
                    "run_id": row["run_id"],
                    "hero": row["hero"],
                    "created_at": row["created_at"],
                    "title": row["title"],
                    "record_wins": row["record_wins"],
                    "run_victory_tier": row["run_victory_tier"],
                    "has_broken_crown": row["has_broken_crown"],
                    "player_rank_tier": row["player_rank_tier"],
                    entity_column: name,
                    "duplicate_count": duplicate_count,
                    "source_method": source_method,
                }
            )
    return pl.DataFrame(output_rows) if output_rows else _empty_presence_frame(entity_column)


def _extracted_board_presence_frame(conn) -> pl.DataFrame:
    frame = conn.query_pl(
        """
        SELECT
            e.screenshot_id,
            s.run_id,
            run_meta.hero,
            run_meta.created_at,
            run_meta.title,
            run_meta.record_wins,
            run_meta.run_victory_tier,
            run_meta.has_broken_crown,
            run_meta.player_rank_tier,
            COALESCE(ref_item.name, e.raw_label) AS item_name,
            COALESCE(e.duplicate_count, 1) AS duplicate_count,
            e.method AS source_method
        FROM extracted_board_items e
        JOIN screenshots s ON s.screenshot_id = e.screenshot_id
        JOIN runs run_meta ON run_meta.run_id = s.run_id
        LEFT JOIN reference_items ref_item ON ref_item.entity_id = e.entity_id
        WHERE e.status = 'ok'
        """
    )
    return frame if frame.height else _empty_presence_frame("item_name")


def _extracted_skill_presence_frame(conn) -> pl.DataFrame:
    frame = conn.query_pl(
        """
        SELECT
            e.screenshot_id,
            s.run_id,
            run_meta.hero,
            run_meta.created_at,
            run_meta.title,
            run_meta.record_wins,
            run_meta.run_victory_tier,
            run_meta.has_broken_crown,
            run_meta.player_rank_tier,
            COALESCE(ref_skill.name, e.raw_label) AS skill_name,
            1 AS duplicate_count,
            e.method AS source_method
        FROM extracted_skills e
        JOIN screenshots s ON s.screenshot_id = e.screenshot_id
        JOIN runs run_meta ON run_meta.run_id = s.run_id
        LEFT JOIN reference_skills ref_skill ON ref_skill.entity_id = e.entity_id
        WHERE e.status = 'ok'
        """
    )
    return frame if frame.height else _empty_presence_frame("skill_name")


def _merge_source_with_extracted_fallback(source_frame: pl.DataFrame, extracted_frame: pl.DataFrame) -> pl.DataFrame:
    if source_frame.height:
        source_run_ids = source_frame.get_column("run_id").unique().to_list()
        extracted_frame = extracted_frame.filter(~pl.col("run_id").is_in(source_run_ids)) if extracted_frame.height else extracted_frame
    frames = [frame for frame in [source_frame, extracted_frame] if frame.height]
    if not frames:
        return source_frame
    return pl.concat(frames, how="diagonal_relaxed")


def _board_presence_frame(conn, filters: AnalysisFilters | None = None) -> tuple[pl.DataFrame, int]:
    frame = _merge_source_with_extracted_fallback(
        _source_card_presence_frame(conn, "board_cards_json", "item_name", "run_detail_board"),
        _extracted_board_presence_frame(conn),
    )
    frame = _filter_analysis_frame(frame, filters)
    total_boards = int(frame.get_column("screenshot_id").n_unique()) if frame.height else 0
    return frame, total_boards


def _exact_item_triplets(board_frame: pl.DataFrame) -> pl.DataFrame:
    if not board_frame.height:
        return pl.DataFrame(
            schema={
                "item_a": pl.String,
                "item_b": pl.String,
                "item_c": pl.String,
                "board_count": pl.Int64,
                "avg_wins": pl.Float64,
                "median_wins": pl.Float64,
                "gold_count": pl.Int64,
                "perfect_count": pl.Int64,
                "gold_plus_count": pl.Int64,
                "gold_plus_rate": pl.Float64,
                "perfect_rate": pl.Float64,
                "gold_broken_count": pl.Int64,
                "gold_broken_rate": pl.Float64,
                "top_outcome": pl.String,
                "example_title": pl.String,
            }
        )

    grouped_boards = board_frame.group_by(
        [
            "screenshot_id",
            "run_id",
            "title",
            "record_wins",
            "run_victory_tier",
            "has_broken_crown",
        ]
    ).agg(pl.col("item_name"))

    rows: list[dict[str, object]] = []
    for row in grouped_boards.iter_rows(named=True):
        items = sorted(set(row["item_name"]))
        for item_a, item_b, item_c in combinations(items, 3):
            rows.append(
                {
                    "item_a": item_a,
                    "item_b": item_b,
                    "item_c": item_c,
                    "title": row["title"],
                    "record_wins": row["record_wins"],
                    "run_victory_tier": row["run_victory_tier"],
                    "has_broken_crown": bool(row["has_broken_crown"]),
                }
            )

    if not rows:
        return pl.DataFrame(
            schema={
                "item_a": pl.String,
                "item_b": pl.String,
                "item_c": pl.String,
                "board_count": pl.Int64,
                "avg_wins": pl.Float64,
                "median_wins": pl.Float64,
                "gold_count": pl.Int64,
                "perfect_count": pl.Int64,
                "gold_plus_count": pl.Int64,
                "gold_plus_rate": pl.Float64,
                "perfect_rate": pl.Float64,
                "gold_broken_count": pl.Int64,
                "gold_broken_rate": pl.Float64,
                "top_outcome": pl.String,
                "example_title": pl.String,
            }
        )

    triplet_frame = pl.DataFrame(rows)
    return (
        triplet_frame.group_by(["item_a", "item_b", "item_c"])
        .agg(
            pl.len().alias("board_count"),
            pl.col("record_wins").drop_nulls().mean().alias("avg_wins"),
            pl.col("record_wins").drop_nulls().median().alias("median_wins"),
            (pl.col("run_victory_tier") == "Gold").sum().alias("gold_count"),
            (pl.col("run_victory_tier") == "Perfect").sum().alias("perfect_count"),
            ((pl.col("run_victory_tier") == "Gold") & pl.col("has_broken_crown")).sum().alias("gold_broken_count"),
            pl.col("run_victory_tier").drop_nulls().mode().first().alias("top_outcome"),
            pl.col("title").first().alias("example_title"),
        )
        .with_columns(
            (pl.col("gold_count") + pl.col("perfect_count")).alias("gold_plus_count"),
            ((pl.col("gold_count") + pl.col("perfect_count")) / pl.col("board_count")).alias("gold_plus_rate"),
            (pl.col("perfect_count") / pl.col("board_count")).alias("perfect_rate"),
            (pl.col("gold_broken_count") / pl.col("board_count")).alias("gold_broken_rate"),
        )
        .sort(["board_count", "avg_wins", "gold_plus_rate"], descending=[True, True, True])
    )


def _skill_presence_frame(conn, filters: AnalysisFilters | None = None) -> pl.DataFrame:
    frame = _merge_source_with_extracted_fallback(
        _source_card_presence_frame(conn, "skill_cards_json", "skill_name", "run_detail_skill"),
        _extracted_skill_presence_frame(conn),
    )
    return _filter_analysis_frame(frame, filters)


def _run_meta_frame(conn, filters: AnalysisFilters | None = None) -> pl.DataFrame:
    frame = conn.query_pl(
        """
        SELECT
            s.screenshot_id,
            r.run_id,
            r.hero,
            r.created_at,
            r.title,
            r.record_wins,
            r.run_wins_label,
            r.run_victory_tier,
            r.player_rank_tier
        FROM screenshots s
        JOIN runs r ON r.run_id = s.run_id
        WHERE s.is_primary = 1
        ORDER BY s.screenshot_id
        """
    )
    return _filter_analysis_frame(frame, filters)


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
        board_frame.select(["screenshot_id", "item_name"])
        .unique()
        .group_by("item_name")
        .len(name="board_count")
        .sort("board_count", descending=True)
        .rename({"item_name": "item"})
    )
    item_count_map = dict(zip(item_counts.get_column("item").to_list(), item_counts.get_column("board_count").to_list(), strict=False))

    board_lists = board_frame.group_by("screenshot_id").agg(pl.col("item_name")).get_column("item_name").to_list()
    pair_counts: Counter[tuple[str, str]] = Counter()
    for values in board_lists:
        unique_values = sorted(set(value for value in values if value))
        pair_counts.update(combinations(unique_values, 2))

    if not pair_counts:
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

    metrics_rows: list[dict[str, float | int | str]] = []
    for (item_a, item_b), count in pair_counts.items():
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

    item_counts = board_frame.select(["screenshot_id", "item_name"]).unique().group_by("item_name").len(name="board_count")
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


def _sample_confidence_label(run_count: int) -> str:
    if run_count >= 100:
        return "high"
    if run_count >= 40:
        return "medium"
    if run_count >= 15:
        return "low"
    return "very_low"


def _weighted_toward_baseline(value: float | None, count: int, baseline: float) -> float | None:
    if value is None:
        return None
    return ((float(value) * count) + (baseline * PERFORMANCE_PRIOR_RUNS)) / (count + PERFORMANCE_PRIOR_RUNS)


def _mechanic_labels_for_names(names: Iterable[str]) -> list[str]:
    text = " ".join(name.lower() for name in names if name)
    labels = [label for label, patterns in MECHANIC_LABEL_PATTERNS if any(re.search(pattern, text) for pattern in patterns)]
    return labels or ["General"]


def _mechanic_labels_json(names: Iterable[str]) -> str:
    return json.dumps(_mechanic_labels_for_names(names), ensure_ascii=True)


def _json_name_rate_entries(value: str | None) -> list[tuple[str, float]]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    entries: list[tuple[str, float]] = []
    for entry in parsed:
        if isinstance(entry, str):
            entries.append((entry, 1.0))
        elif isinstance(entry, dict) and entry.get("name"):
            entries.append((str(entry["name"]), float(entry.get("rate") or 0.0)))
    return entries


def _json_names(value: str | None) -> list[str]:
    return [name for name, _rate in _json_name_rate_entries(value)]


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _empty_archetype_family_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "family_id": pl.String,
            "family_name": pl.String,
            "cluster_count": pl.Int64,
            "board_count": pl.Int64,
            "presence_pct": pl.Float64,
            "avg_wins": pl.Float64,
            "weighted_avg_wins": pl.Float64,
            "avg_wins_delta": pl.Float64,
            "gold_plus_count": pl.Int64,
            "gold_plus_rate": pl.Float64,
            "gold_plus_delta": pl.Float64,
            "perfect_count": pl.Int64,
            "perfect_rate": pl.Float64,
            "perfect_delta": pl.Float64,
            "confidence": pl.String,
            "mechanic_labels": pl.String,
            "core_items_json": pl.String,
            "flex_items_json": pl.String,
            "top_skills_json": pl.String,
            "top_outcome": pl.String,
            "outcome_distribution_json": pl.String,
            "player_rank_distribution_json": pl.String,
            "example_archetypes_json": pl.String,
        }
    )


def _empty_archetype_report_frame(include_hero: bool = False) -> pl.DataFrame:
    schema = {
        "archetype": pl.String,
        "board_count": pl.Int64,
        "presence_pct": pl.Float64,
        "avg_wins": pl.Float64,
        "weighted_avg_wins": pl.Float64,
        "avg_wins_delta": pl.Float64,
        "gold_plus_rate": pl.Float64,
        "gold_plus_delta": pl.Float64,
        "perfect_rate": pl.Float64,
        "perfect_delta": pl.Float64,
        "confidence": pl.String,
        "mechanic_labels": pl.String,
        "core_items_json": pl.String,
        "flex_items_json": pl.String,
        "top_skills_json": pl.String,
    }
    if include_hero:
        schema = {"hero": pl.String, **schema}
    return pl.DataFrame(schema=schema)


def _archetype_report_frame(family_frame: pl.DataFrame) -> pl.DataFrame:
    include_hero = "hero" in family_frame.columns
    if not family_frame.height:
        return _empty_archetype_report_frame(include_hero)
    columns = [
        "archetype",
        "board_count",
        "presence_pct",
        "avg_wins",
        "weighted_avg_wins",
        "avg_wins_delta",
        "gold_plus_rate",
        "gold_plus_delta",
        "perfect_rate",
        "perfect_delta",
        "confidence",
        "mechanic_labels",
        "core_items_json",
        "flex_items_json",
        "top_skills_json",
    ]
    if include_hero:
        columns = ["hero", *columns]
    return (
        family_frame.with_columns(pl.col("family_name").alias("archetype"))
        .with_columns(
            pl.col("presence_pct").round(2),
            pl.col("avg_wins").round(3),
            pl.col("weighted_avg_wins").round(3),
            pl.col("avg_wins_delta").round(3),
            pl.col("gold_plus_rate").round(4),
            pl.col("gold_plus_delta").round(4),
            pl.col("perfect_rate").round(4),
            pl.col("perfect_delta").round(4),
        )
        .select(columns)
        .sort(["hero", "board_count", "weighted_avg_wins"] if include_hero else ["board_count", "weighted_avg_wins"], descending=[False, True, True] if include_hero else [True, True])
    )


def _empty_build_cluster_outputs() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    empty_profiles = pl.DataFrame(
        schema={
            "archetype_anchor_a": pl.String,
            "archetype_anchor_b": pl.String,
            "board_count": pl.Int64,
            "presence_pct": pl.Float64,
            "avg_wins": pl.Float64,
            "weighted_avg_wins": pl.Float64,
            "avg_wins_delta": pl.Float64,
            "median_wins": pl.Float64,
            "gold_plus_count": pl.Int64,
            "gold_plus_rate": pl.Float64,
            "gold_plus_delta": pl.Float64,
            "perfect_count": pl.Int64,
            "perfect_rate": pl.Float64,
            "perfect_delta": pl.Float64,
            "confidence": pl.String,
            "mechanic_labels": pl.String,
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
            "presence_pct": pl.Float64,
            "avg_wins": pl.Float64,
            "weighted_avg_wins": pl.Float64,
            "avg_wins_delta": pl.Float64,
            "gold_plus_count": pl.Int64,
            "gold_plus_rate": pl.Float64,
            "gold_plus_delta": pl.Float64,
            "perfect_count": pl.Int64,
            "perfect_rate": pl.Float64,
            "perfect_delta": pl.Float64,
            "confidence": pl.String,
            "mechanic_labels": pl.String,
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


def _build_core_builds(
    cluster_profile_frame: pl.DataFrame,
    total_boards: int,
    baseline_avg_wins: float,
    baseline_gold_plus_rate: float,
    baseline_perfect_rate: float,
) -> pl.DataFrame:
    if not cluster_profile_frame.height:
        return _empty_build_cluster_outputs()[2]

    supported_cluster_profile_frame = cluster_profile_frame.filter(
        pl.col("board_count") >= MIN_CORE_BUILD_CLUSTER_BOARDS
    )
    if not supported_cluster_profile_frame.height:
        return _empty_build_cluster_outputs()[2]

    grouped: dict[tuple[str, ...], dict[str, object]] = {}
    for row in supported_cluster_profile_frame.iter_rows(named=True):
        core_items = tuple(sorted(entry["name"] for entry in json.loads(row["core_items_json"]) if entry.get("name")))
        if not core_items:
            continue
        group = grouped.setdefault(
            core_items,
            {
                "cluster_count": 0,
                "board_count": 0,
                "wins_weighted_sum": 0.0,
                "gold_plus_count": 0,
                "perfect_count": 0,
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
        group["gold_plus_count"] = int(group["gold_plus_count"]) + int(row.get("gold_plus_count") or 0)
        group["perfect_count"] = int(group["perfect_count"]) + int(row.get("perfect_count") or 0)
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
                "presence_pct": (board_count / total_boards * 100.0) if total_boards else 0.0,
                "avg_wins": float(payload["wins_weighted_sum"]) / board_count,
                "gold_plus_count": int(payload["gold_plus_count"]),
                "perfect_count": int(payload["perfect_count"]),
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
    return (
        pl.DataFrame(rows)
        .with_columns(
            pl.struct(["avg_wins", "board_count"]).map_elements(
                lambda value: _weighted_toward_baseline(value["avg_wins"], int(value["board_count"]), baseline_avg_wins),
                return_dtype=pl.Float64,
            ).alias("weighted_avg_wins"),
            (pl.col("gold_plus_count") / pl.col("board_count")).alias("gold_plus_rate"),
            (pl.col("perfect_count") / pl.col("board_count")).alias("perfect_rate"),
        )
        .with_columns(
            pl.struct(["gold_plus_rate", "board_count"]).map_elements(
                lambda value: _weighted_toward_baseline(value["gold_plus_rate"], int(value["board_count"]), baseline_gold_plus_rate),
                return_dtype=pl.Float64,
            ).alias("weighted_gold_plus_rate"),
            pl.struct(["perfect_rate", "board_count"]).map_elements(
                lambda value: _weighted_toward_baseline(value["perfect_rate"], int(value["board_count"]), baseline_perfect_rate),
                return_dtype=pl.Float64,
            ).alias("weighted_perfect_rate"),
        )
        .with_columns(
            (pl.col("weighted_avg_wins") - pl.lit(baseline_avg_wins)).alias("avg_wins_delta"),
            (pl.col("weighted_gold_plus_rate") - pl.lit(baseline_gold_plus_rate)).alias("gold_plus_delta"),
            (pl.col("weighted_perfect_rate") - pl.lit(baseline_perfect_rate)).alias("perfect_delta"),
            pl.struct(["board_count"]).map_elements(lambda value: _sample_confidence_label(int(value["board_count"])), return_dtype=pl.String).alias("confidence"),
            pl.struct(["core_items_json", "top_flex_items_json"]).map_elements(
                lambda value: _mechanic_labels_json([*_json_names(value["core_items_json"]), *_json_names(value["top_flex_items_json"])]),
                return_dtype=pl.String,
            ).alias("mechanic_labels"),
        )
        .drop(["weighted_gold_plus_rate", "weighted_perfect_rate"])
        .sort(["board_count", "core_item_count", "weighted_avg_wins", "avg_wins"], descending=[True, True, True, True])
    )


def _board_cluster_assignments(board_frame: pl.DataFrame, signature_frame: pl.DataFrame) -> pl.DataFrame:
    if not board_frame.height or not signature_frame.height:
        return pl.DataFrame(
            schema={
                "screenshot_id": pl.Int64,
                "run_id": pl.Int64,
                "title": pl.String,
                "record_wins": pl.Int64,
                "run_victory_tier": pl.String,
                "player_rank_tier": pl.String,
                "has_broken_crown": pl.Boolean,
                "archetype_anchor_a": pl.String,
                "archetype_anchor_b": pl.String,
                "items": pl.List(pl.String),
            }
        )

    signature_map = dict(
        zip(
            signature_frame.get_column("item_name").to_list(),
            signature_frame.get_column("signature_score").to_list(),
            strict=False,
        )
    )
    grouped_boards = board_frame.group_by(
        [
            "screenshot_id",
            "run_id",
            "title",
            "record_wins",
            "run_victory_tier",
            "player_rank_tier",
            "has_broken_crown",
        ]
    ).agg(pl.col("item_name"))

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
                "run_id": row["run_id"],
                "title": row["title"],
                "record_wins": row["record_wins"],
                "run_victory_tier": row["run_victory_tier"],
                "has_broken_crown": bool(row["has_broken_crown"]),
                "items": items,
                "player_rank_tier": row["player_rank_tier"],
            }
        )

    if not archetype_rows:
        return pl.DataFrame(
            schema={
                "screenshot_id": pl.Int64,
                "run_id": pl.Int64,
                "title": pl.String,
                "record_wins": pl.Int64,
                "run_victory_tier": pl.String,
                "player_rank_tier": pl.String,
                "has_broken_crown": pl.Boolean,
                "archetype_anchor_a": pl.String,
                "archetype_anchor_b": pl.String,
                "items": pl.List(pl.String),
            }
        )
    return pl.DataFrame(archetype_rows).sort(["archetype_anchor_a", "archetype_anchor_b", "screenshot_id"])


def _build_cluster_profiles(cluster_assignment_frame: pl.DataFrame, skill_frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if not cluster_assignment_frame.height:
        return _empty_build_cluster_outputs()

    total_boards = cluster_assignment_frame.height
    all_wins = [int(row["record_wins"]) for row in cluster_assignment_frame.iter_rows(named=True) if row["record_wins"] is not None]
    baseline_avg_wins = (sum(all_wins) / len(all_wins)) if all_wins else 0.0
    baseline_gold_plus_rate = float(
        cluster_assignment_frame.select(pl.col("run_victory_tier").is_in(list(GOLD_PLUS_TIERS)).mean()).item() or 0.0
    )
    baseline_perfect_rate = float(
        cluster_assignment_frame.select((pl.col("run_victory_tier") == "Perfect").mean()).item() or 0.0
    )

    skill_lists = {
        row["screenshot_id"]: sorted(set(row["skill_name"]))
        for row in skill_frame.group_by("screenshot_id").agg(pl.col("skill_name")).iter_rows(named=True)
    } if skill_frame.height else {}

    archetype_rows = []
    for row in cluster_assignment_frame.iter_rows(named=True):
        payload = dict(row)
        payload["skills"] = skill_lists.get(row["screenshot_id"], [])
        archetype_rows.append(payload)

    profile_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    cluster_map: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in archetype_rows:
        cluster_map.setdefault((row["archetype_anchor_a"], row["archetype_anchor_b"]), []).append(row)

    for (anchor_a, anchor_b), rows in sorted(cluster_map.items()):
        board_count = len(rows)
        wins = [int(row["record_wins"]) for row in rows if row["record_wins"] is not None]
        avg_wins = (sum(wins) / len(wins)) if wins else None
        gold_plus_count = sum(1 for row in rows if row["run_victory_tier"] in GOLD_PLUS_TIERS)
        perfect_count = sum(1 for row in rows if row["run_victory_tier"] == "Perfect")
        gold_plus_rate = gold_plus_count / board_count if board_count else 0.0
        perfect_rate = perfect_count / board_count if board_count else 0.0
        weighted_avg_wins = _weighted_toward_baseline(avg_wins, board_count, baseline_avg_wins)
        weighted_gold_plus_rate = _weighted_toward_baseline(gold_plus_rate, board_count, baseline_gold_plus_rate) or 0.0
        weighted_perfect_rate = _weighted_toward_baseline(perfect_rate, board_count, baseline_perfect_rate) or 0.0
        item_counter: dict[str, int] = {}
        skill_counter: dict[str, int] = {}
        outcome_counter: dict[str, int] = {}
        rank_counter: dict[str, int] = {}
        for row in rows:
            for item in set(row["items"]):
                item_counter[item] = item_counter.get(item, 0) + 1
            for skill in set(row["skills"]):
                skill_counter[skill] = skill_counter.get(skill, 0) + 1
            outcome = row["run_victory_tier"] or "Unknown"
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
        mechanic_source_items = [name for name, _rate in core_items] + [name for name, _rate in flex_items]
        profile_rows.append(
            {
                "archetype_anchor_a": anchor_a,
                "archetype_anchor_b": anchor_b,
                "board_count": board_count,
                "presence_pct": (board_count / total_boards * 100.0) if total_boards else 0.0,
                "avg_wins": avg_wins,
                "weighted_avg_wins": weighted_avg_wins,
                "avg_wins_delta": (weighted_avg_wins - baseline_avg_wins) if weighted_avg_wins is not None else None,
                "median_wins": float(pl.Series(wins).median()) if wins else None,
                "gold_plus_count": gold_plus_count,
                "gold_plus_rate": gold_plus_rate,
                "gold_plus_delta": weighted_gold_plus_rate - baseline_gold_plus_rate,
                "perfect_count": perfect_count,
                "perfect_rate": perfect_rate,
                "perfect_delta": weighted_perfect_rate - baseline_perfect_rate,
                "confidence": _sample_confidence_label(board_count),
                "mechanic_labels": _mechanic_labels_json(mechanic_source_items or item_counter.keys()),
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
    core_build_frame = _build_core_builds(
        cluster_profile_frame,
        total_boards,
        baseline_avg_wins,
        baseline_gold_plus_rate,
        baseline_perfect_rate,
    )
    return cluster_profile_frame, cluster_component_frame, core_build_frame


def _profile_core_set(row: dict[str, object]) -> set[str]:
    core_items = set(_json_names(str(row.get("core_items_json") or "")))
    if core_items:
        return core_items
    anchors = {str(row.get("archetype_anchor_a") or ""), str(row.get("archetype_anchor_b") or "")}
    return {anchor for anchor in anchors if anchor}


def _merge_counter_from_json(counter: dict[str, float], raw_json: str | None, board_count: int) -> None:
    for name, rate in _json_name_rate_entries(raw_json):
        counter[name] = counter.get(name, 0.0) + (rate * board_count)


def _merge_count_counter_from_json(counter: dict[str, int], raw_json: str | None) -> None:
    if not raw_json:
        return
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        return
    for entry in parsed:
        if isinstance(entry, dict) and entry.get("name"):
            counter[str(entry["name"])] = counter.get(str(entry["name"]), 0) + int(entry.get("count") or 0)


def _build_archetype_families(cluster_profile_frame: pl.DataFrame) -> pl.DataFrame:
    if not cluster_profile_frame.height:
        return _empty_archetype_family_frame()

    total_boards = int(cluster_profile_frame.get_column("board_count").sum() or 0)
    if total_boards <= 0:
        return _empty_archetype_family_frame()

    baseline_avg_wins = float(
        cluster_profile_frame.select((pl.col("avg_wins") * pl.col("board_count")).sum() / pl.col("board_count").sum()).item() or 0.0
    )
    baseline_gold_plus_rate = float(cluster_profile_frame.get_column("gold_plus_count").sum() or 0) / total_boards
    baseline_perfect_rate = float(cluster_profile_frame.get_column("perfect_count").sum() or 0) / total_boards

    families: list[dict[str, object]] = []
    profile_rows = sorted(
        cluster_profile_frame.iter_rows(named=True),
        key=lambda row: (-int(row["board_count"]), -(float(row["weighted_avg_wins"] or row["avg_wins"] or 0.0)), row["archetype_anchor_a"], row["archetype_anchor_b"]),
    )
    for row in profile_rows:
        core_set = _profile_core_set(row)
        if not core_set:
            continue
        best_family: dict[str, object] | None = None
        best_score = 0.0
        for family in families:
            score = _jaccard_similarity(core_set, family["core_set"])
            if score > best_score:
                best_score = score
                best_family = family
        if best_family is None or best_score < ARCHETYPE_FAMILY_JACCARD_THRESHOLD:
            best_family = {
                "core_set": set(core_set),
                "cluster_count": 0,
                "board_count": 0,
                "wins_weighted_sum": 0.0,
                "gold_plus_count": 0,
                "perfect_count": 0,
                "core_counter": {},
                "flex_counter": {},
                "skill_counter": {},
                "outcome_counter": {},
                "rank_counter": {},
                "example_archetypes": [],
            }
            families.append(best_family)
        else:
            best_family["core_set"] = set(best_family["core_set"]) | core_set

        board_count = int(row["board_count"])
        best_family["cluster_count"] = int(best_family["cluster_count"]) + 1
        best_family["board_count"] = int(best_family["board_count"]) + board_count
        best_family["wins_weighted_sum"] = float(best_family["wins_weighted_sum"]) + (float(row["avg_wins"] or 0.0) * board_count)
        best_family["gold_plus_count"] = int(best_family["gold_plus_count"]) + int(row.get("gold_plus_count") or 0)
        best_family["perfect_count"] = int(best_family["perfect_count"]) + int(row.get("perfect_count") or 0)
        _merge_counter_from_json(best_family["core_counter"], row.get("core_items_json"), board_count)
        _merge_counter_from_json(best_family["flex_counter"], row.get("flex_items_json"), board_count)
        _merge_counter_from_json(best_family["skill_counter"], row.get("top_skills_json"), board_count)
        _merge_count_counter_from_json(best_family["outcome_counter"], row.get("outcome_distribution_json"))
        _merge_count_counter_from_json(best_family["rank_counter"], row.get("player_rank_distribution_json"))
        best_family["example_archetypes"].append(
            {
                "anchors": [row["archetype_anchor_a"], row["archetype_anchor_b"]],
                "board_count": board_count,
                "avg_wins": round(float(row["avg_wins"] or 0.0), 4),
            }
        )

    rows: list[dict[str, object]] = []
    for family in families:
        board_count = int(family["board_count"])
        if board_count <= 0:
            continue
        core_entries = [
            (name, support / board_count)
            for name, support in family["core_counter"].items()
            if (support / board_count) >= 0.35
        ]
        if not core_entries:
            core_entries = [(name, 1.0) for name in sorted(family["core_set"])]
        core_names = [name for name, _rate in sorted(core_entries, key=lambda item: (-item[1], item[0]))]
        core_name_set = set(core_names)
        flex_entries = [
            (name, support / board_count)
            for name, support in family["flex_counter"].items()
            if name not in core_name_set and (support / board_count) >= 0.10
        ]
        skill_entries = [
            (name, support / board_count)
            for name, support in family["skill_counter"].items()
            if (support / board_count) >= 0.10
        ]
        avg_wins = float(family["wins_weighted_sum"]) / board_count
        weighted_avg_wins = _weighted_toward_baseline(avg_wins, board_count, baseline_avg_wins)
        gold_plus_count = int(family["gold_plus_count"])
        perfect_count = int(family["perfect_count"])
        gold_plus_rate = gold_plus_count / board_count
        perfect_rate = perfect_count / board_count
        weighted_gold_plus_rate = _weighted_toward_baseline(gold_plus_rate, board_count, baseline_gold_plus_rate) or 0.0
        weighted_perfect_rate = _weighted_toward_baseline(perfect_rate, board_count, baseline_perfect_rate) or 0.0
        outcome_counter = family["outcome_counter"]
        rank_counter = family["rank_counter"]
        top_outcome = max(outcome_counter.items(), key=lambda item: (item[1], item[0]))[0] if outcome_counter else None
        family_name = " + ".join(core_names[:3]) if core_names else "Unknown"
        example_archetypes = sorted(family["example_archetypes"], key=lambda item: (-item["board_count"], -item["avg_wins"], item["anchors"]))[:8]
        rows.append(
            {
                "family_name": family_name,
                "cluster_count": int(family["cluster_count"]),
                "board_count": board_count,
                "presence_pct": (board_count / total_boards * 100.0) if total_boards else 0.0,
                "avg_wins": avg_wins,
                "weighted_avg_wins": weighted_avg_wins,
                "avg_wins_delta": (weighted_avg_wins - baseline_avg_wins) if weighted_avg_wins is not None else None,
                "gold_plus_count": gold_plus_count,
                "gold_plus_rate": gold_plus_rate,
                "gold_plus_delta": weighted_gold_plus_rate - baseline_gold_plus_rate,
                "perfect_count": perfect_count,
                "perfect_rate": perfect_rate,
                "perfect_delta": weighted_perfect_rate - baseline_perfect_rate,
                "confidence": _sample_confidence_label(board_count),
                "mechanic_labels": _mechanic_labels_json([*core_names, *[name for name, _rate in flex_entries]]),
                "core_items_json": _json_name_counts(core_entries),
                "flex_items_json": _json_name_counts(flex_entries),
                "top_skills_json": _json_name_counts(skill_entries),
                "top_outcome": top_outcome,
                "outcome_distribution_json": _json_counter(outcome_counter),
                "player_rank_distribution_json": _json_counter(rank_counter),
                "example_archetypes_json": json.dumps(example_archetypes, ensure_ascii=True),
            }
        )

    if not rows:
        return _empty_archetype_family_frame()
    rows = sorted(rows, key=lambda row: (-int(row["board_count"]), -(float(row["weighted_avg_wins"] or 0.0)), row["family_name"]))
    for index, row in enumerate(rows, start=1):
        row["family_id"] = f"family_{index:03d}"
    return pl.DataFrame(rows).select(list(_empty_archetype_family_frame().schema.keys()))


def _entity_shell_affinity(cluster_assignment_frame: pl.DataFrame, entity_frame: pl.DataFrame, entity_column: str) -> pl.DataFrame:
    if not cluster_assignment_frame.height or not entity_frame.height:
        return pl.DataFrame(
            schema={
                entity_column: pl.String,
                "archetype_anchor_a": pl.String,
                "archetype_anchor_b": pl.String,
                "shell_board_count": pl.Int64,
                "global_entity_board_count": pl.Int64,
                "entity_board_count": pl.Int64,
                "entity_share_of_runs": pl.Float64,
                "shell_entity_rate": pl.Float64,
                "global_entity_rate": pl.Float64,
                "lift": pl.Float64,
                "avg_wins": pl.Float64,
                "median_wins": pl.Float64,
                "top_outcome": pl.String,
            }
        )

    total_boards = cluster_assignment_frame.height
    entity_presence = (
        entity_frame.group_by("screenshot_id")
        .agg(pl.col(entity_column).unique())
        .explode(entity_column)
        .filter(pl.col(entity_column).is_not_null())
    )
    if not entity_presence.height:
        return pl.DataFrame(
            schema={
                entity_column: pl.String,
                "archetype_anchor_a": pl.String,
                "archetype_anchor_b": pl.String,
                "shell_board_count": pl.Int64,
                "global_entity_board_count": pl.Int64,
                "entity_board_count": pl.Int64,
                "entity_share_of_runs": pl.Float64,
                "shell_entity_rate": pl.Float64,
                "global_entity_rate": pl.Float64,
                "lift": pl.Float64,
                "avg_wins": pl.Float64,
                "median_wins": pl.Float64,
                "top_outcome": pl.String,
            }
        )

    shell_counts = cluster_assignment_frame.group_by(["archetype_anchor_a", "archetype_anchor_b"]).len(name="shell_board_count")
    global_counts = entity_presence.group_by(entity_column).len(name="global_entity_board_count")
    joined = entity_presence.join(
        cluster_assignment_frame.select(["screenshot_id", "archetype_anchor_a", "archetype_anchor_b", "record_wins", "run_victory_tier"]),
        on="screenshot_id",
        how="inner",
    )
    return (
        joined.group_by([entity_column, "archetype_anchor_a", "archetype_anchor_b"])
        .agg(
            pl.len().alias("entity_board_count"),
            pl.col("record_wins").drop_nulls().mean().alias("avg_wins"),
            pl.col("record_wins").drop_nulls().median().alias("median_wins"),
            pl.col("run_victory_tier").drop_nulls().mode().first().alias("top_outcome"),
        )
        .join(shell_counts, on=["archetype_anchor_a", "archetype_anchor_b"], how="left")
        .join(global_counts, on=entity_column, how="left")
        .with_columns(
            (pl.col("entity_board_count") / pl.col("global_entity_board_count")).alias("entity_share_of_runs"),
            (pl.col("entity_board_count") / pl.col("shell_board_count")).alias("shell_entity_rate"),
            (pl.col("global_entity_board_count") / pl.lit(total_boards)).alias("global_entity_rate"),
        )
        .with_columns(
            pl.when(pl.col("global_entity_rate") > 0)
            .then(pl.col("shell_entity_rate") / pl.col("global_entity_rate"))
            .otherwise(None)
            .alias("lift")
        )
        .sort([entity_column, "lift", "entity_board_count", "avg_wins"], descending=[False, True, True, True])
    )


def _performance_by_entity(frame: pl.DataFrame, entity_column: str) -> pl.DataFrame:
    schema = {
        entity_column: pl.String,
        "run_count": pl.Int64,
        "avg_wins": pl.Float64,
        "weighted_avg_wins": pl.Float64,
        "median_wins": pl.Float64,
        "wins_10_count": pl.Int64,
        "wins_10_rate": pl.Float64,
        "gold_plus_rate": pl.Float64,
        "weighted_gold_plus_rate": pl.Float64,
        "gold_plus_delta": pl.Float64,
        "perfect_rate": pl.Float64,
        "weighted_perfect_rate": pl.Float64,
        "perfect_delta": pl.Float64,
        "top_outcome": pl.String,
    }
    if not frame.height:
        return pl.DataFrame(schema=schema)

    unique_frame = frame.select(["screenshot_id", entity_column, "record_wins", "run_victory_tier"]).unique()
    global_avg_wins = unique_frame.get_column("record_wins").drop_nulls().mean()
    if global_avg_wins is None:
        global_avg_wins = 0.0
    global_gold_plus_rate = unique_frame.select(pl.col("run_victory_tier").is_in(["Gold", "Perfect"]).mean()).item()
    if global_gold_plus_rate is None:
        global_gold_plus_rate = 0.0
    global_perfect_rate = unique_frame.select((pl.col("run_victory_tier") == "Perfect").mean()).item()
    if global_perfect_rate is None:
        global_perfect_rate = 0.0

    return (
        unique_frame.group_by(entity_column)
        .agg(
            pl.len().alias("run_count"),
            pl.col("record_wins").drop_nulls().mean().alias("avg_wins"),
            pl.col("record_wins").drop_nulls().median().alias("median_wins"),
            (pl.col("record_wins") == 10).sum().alias("wins_10_count"),
            (pl.col("record_wins") == 10).mean().alias("wins_10_rate"),
            pl.col("run_victory_tier").is_in(["Gold", "Perfect"]).mean().alias("gold_plus_rate"),
            (pl.col("run_victory_tier") == "Perfect").mean().alias("perfect_rate"),
            pl.col("run_victory_tier").drop_nulls().mode().first().alias("top_outcome"),
        )
        .with_columns(
            pl.when(pl.col("avg_wins").is_not_null())
            .then(
                ((pl.col("avg_wins") * pl.col("run_count")) + (pl.lit(global_avg_wins) * PERFORMANCE_PRIOR_RUNS))
                / (pl.col("run_count") + PERFORMANCE_PRIOR_RUNS)
            )
            .otherwise(None)
            .alias("weighted_avg_wins"),
            (((pl.col("gold_plus_rate") * pl.col("run_count")) + (pl.lit(global_gold_plus_rate) * PERFORMANCE_PRIOR_RUNS))
             / (pl.col("run_count") + PERFORMANCE_PRIOR_RUNS)).alias("weighted_gold_plus_rate"),
            (((pl.col("perfect_rate") * pl.col("run_count")) + (pl.lit(global_perfect_rate) * PERFORMANCE_PRIOR_RUNS))
             / (pl.col("run_count") + PERFORMANCE_PRIOR_RUNS)).alias("weighted_perfect_rate"),
        )
        .with_columns(
            (pl.col("weighted_gold_plus_rate") - pl.lit(global_gold_plus_rate)).alias("gold_plus_delta"),
            (pl.col("weighted_perfect_rate") - pl.lit(global_perfect_rate)).alias("perfect_delta"),
        )
        .select(list(schema))
        .sort(["weighted_avg_wins", "run_count", "avg_wins"], descending=[True, True, True])
    )


def _counts_with_performance(count_frame: pl.DataFrame, performance_frame: pl.DataFrame, entity_column: str) -> pl.DataFrame:
    if not count_frame.height:
        return pl.DataFrame(
            schema={
                entity_column: pl.String,
                "count": pl.Int64,
                "avg_wins": pl.Float64,
                "weighted_avg_wins": pl.Float64,
                "median_wins": pl.Float64,
                "wins_10_count": pl.Int64,
                "wins_10_rate": pl.Float64,
                "gold_plus_rate": pl.Float64,
                "weighted_gold_plus_rate": pl.Float64,
                "gold_plus_delta": pl.Float64,
                "perfect_rate": pl.Float64,
                "weighted_perfect_rate": pl.Float64,
                "perfect_delta": pl.Float64,
                "top_outcome": pl.String,
            }
        )
    return (
        count_frame.join(performance_frame, on=entity_column, how="left")
        .sort(["count", "weighted_avg_wins", "avg_wins"], descending=[True, True, True])
    )


def _performance_by_hero(frame: pl.DataFrame, entity_column: str) -> pl.DataFrame:
    if not frame.height or "hero" not in frame.columns:
        return pl.DataFrame()
    frames: list[pl.DataFrame] = []
    heroes = sorted(hero for hero in frame.get_column("hero").drop_nulls().unique().to_list() if hero)
    for hero in heroes:
        hero_frame = frame.filter(pl.col("hero") == hero)
        performance = _performance_by_entity(hero_frame, entity_column)
        if performance.height:
            frames.append(performance.with_columns(pl.lit(hero).alias("hero")).select(["hero", *performance.columns]))
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def _counts_with_performance_by_hero(count_frame: pl.DataFrame, performance_frame: pl.DataFrame, entity_column: str) -> pl.DataFrame:
    if not count_frame.height:
        return pl.DataFrame()
    if not performance_frame.height:
        return count_frame
    return count_frame.join(performance_frame, on=["hero", entity_column], how="left").sort(
        ["hero", "count", "weighted_avg_wins", "avg_wins"],
        descending=[False, True, True, True],
    )


def _cooccurrence_by_hero(presence_frame: pl.DataFrame, entity_column: str, left_name: str, right_name: str) -> pl.DataFrame:
    if not presence_frame.height:
        return pl.DataFrame()
    frames: list[pl.DataFrame] = []
    for hero in sorted(hero for hero in presence_frame.get_column("hero").drop_nulls().unique().to_list() if hero):
        hero_presence = presence_frame.filter(pl.col("hero") == hero)
        lists = hero_presence.group_by("screenshot_id").agg(pl.col(entity_column)).get_column(entity_column).to_list()
        cooccurrence = _cooccurrence(lists, left_name, right_name)
        if cooccurrence.height:
            frames.append(cooccurrence.with_columns(pl.lit(hero).alias("hero")).select(["hero", *cooccurrence.columns]))
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def _write_if_not_empty(frame: pl.DataFrame, path) -> int:
    write_frame_exports(frame, path.with_suffix(""))
    return frame.height


def _item_source_alignment(conn, board_frame: pl.DataFrame) -> pl.DataFrame:
    item_counts = (
        board_frame.select(["screenshot_id", "item_name"]).unique().group_by("item_name").len(name="analysis_board_count")
        if board_frame.height
        else pl.DataFrame({"item_name": [], "analysis_board_count": []})
    )
    source_run_ids = set(board_frame.get_column("run_id").to_list()) if board_frame.height and "run_id" in board_frame.columns else set()
    source_rows = conn.execute("SELECT run_id, CAST(board_cards_json AS VARCHAR) AS board_cards_json FROM runs").fetchall()
    card_name_lookup = _source_card_name_lookup(conn, "board_cards_json")
    source_copy_counter: dict[str, int] = {}
    source_board_counter: dict[str, int] = {}
    for row in source_rows:
        if not source_run_ids or row["run_id"] not in source_run_ids:
            continue
        names = _parse_card_names(row["board_cards_json"], card_name_lookup)
        for item_name in names:
            source_copy_counter[item_name] = source_copy_counter.get(item_name, 0) + 1
        for item_name in set(names):
            source_board_counter[item_name] = source_board_counter.get(item_name, 0) + 1

    names = sorted(set(item_counts.get_column("item_name").to_list()) | set(source_board_counter)) if item_counts.height else sorted(source_board_counter)
    rows: list[dict[str, float | int | str | None]] = []
    item_count_map = dict(zip(item_counts.get_column("item_name").to_list(), item_counts.get_column("analysis_board_count").to_list(), strict=False)) if item_counts.height else {}
    for name in names:
        analysis_board_count = int(item_count_map.get(name, 0))
        source_board_count = int(source_board_counter.get(name, 0))
        source_copy_count = int(source_copy_counter.get(name, 0))
        analysis_to_source_board_ratio = (analysis_board_count / source_board_count) if source_board_count else None
        rows.append(
            {
                "item_name": name,
                "analysis_board_count": analysis_board_count,
                "source_board_count": source_board_count,
                "source_copy_count": source_copy_count,
                "analysis_to_source_board_ratio": analysis_to_source_board_ratio,
                "analysis_minus_source_boards": analysis_board_count - source_board_count,
            }
        )
    return pl.DataFrame(rows).sort(["analysis_minus_source_boards", "analysis_board_count"], descending=[True, True]) if rows else pl.DataFrame(
        schema={
            "item_name": pl.String,
            "analysis_board_count": pl.Int64,
            "source_board_count": pl.Int64,
            "source_copy_count": pl.Int64,
            "analysis_to_source_board_ratio": pl.Float64,
            "analysis_minus_source_boards": pl.Int64,
        }
    )


def systemic_analysis(conn, settings: Settings) -> dict[str, int]:
    filters = _load_analysis_filters()
    board_frame, total_boards = _board_presence_frame(conn, filters)
    skill_frame = _skill_presence_frame(conn, filters)
    pair_frame = _systemic_item_pairs(board_frame, total_boards)
    signature_frame = _systemic_item_signatures(board_frame, pair_frame, total_boards)
    archetype_frame = _systemic_archetypes(board_frame, signature_frame)
    cluster_assignment_frame = _board_cluster_assignments(board_frame, signature_frame)
    cluster_profile_frame, cluster_component_frame, core_build_frame = _build_cluster_profiles(cluster_assignment_frame, skill_frame)
    family_frame = _build_archetype_families(cluster_profile_frame)
    report_frame = _archetype_report_frame(family_frame)
    skill_shell_affinity_frame = _entity_shell_affinity(cluster_assignment_frame, skill_frame.select(["screenshot_id", "skill_name"]), "skill_name")
    item_shell_affinity_frame = _entity_shell_affinity(cluster_assignment_frame, board_frame.select(["screenshot_id", "item_name"]), "item_name")
    exact_triplet_frame = _exact_item_triplets(board_frame)
    source_alignment = _item_source_alignment(conn, board_frame)

    hero_pair_frames: list[pl.DataFrame] = []
    hero_signature_frames: list[pl.DataFrame] = []
    hero_archetype_frames: list[pl.DataFrame] = []
    hero_cluster_profile_frames: list[pl.DataFrame] = []
    hero_cluster_component_frames: list[pl.DataFrame] = []
    hero_core_build_frames: list[pl.DataFrame] = []
    hero_family_frames: list[pl.DataFrame] = []
    hero_skill_shell_affinity_frames: list[pl.DataFrame] = []
    hero_item_shell_affinity_frames: list[pl.DataFrame] = []
    hero_exact_triplet_frames: list[pl.DataFrame] = []
    heroes = sorted(hero for hero in board_frame.get_column("hero").drop_nulls().unique().to_list() if hero) if board_frame.height else []
    for hero in heroes:
        hero_board_frame = board_frame.filter(pl.col("hero") == hero)
        hero_skill_frame = skill_frame.filter(pl.col("hero") == hero) if skill_frame.height else skill_frame
        hero_total_boards = int(hero_board_frame.get_column("screenshot_id").n_unique()) if hero_board_frame.height else 0
        hero_pair_frame = _systemic_item_pairs(hero_board_frame, hero_total_boards)
        hero_signature_frame = _systemic_item_signatures(hero_board_frame, hero_pair_frame, hero_total_boards)
        hero_archetype_frame = _systemic_archetypes(hero_board_frame, hero_signature_frame)
        hero_cluster_assignment_frame = _board_cluster_assignments(hero_board_frame, hero_signature_frame)
        hero_cluster_profile_frame, hero_cluster_component_frame, hero_core_build_frame = _build_cluster_profiles(hero_cluster_assignment_frame, hero_skill_frame)
        hero_family_frame = _build_archetype_families(hero_cluster_profile_frame)
        hero_skill_shell_affinity_frame = _entity_shell_affinity(hero_cluster_assignment_frame, hero_skill_frame.select(["screenshot_id", "skill_name"]), "skill_name") if hero_skill_frame.height else pl.DataFrame()
        hero_item_shell_affinity_frame = _entity_shell_affinity(hero_cluster_assignment_frame, hero_board_frame.select(["screenshot_id", "item_name"]), "item_name")
        hero_exact_triplet_frame = _exact_item_triplets(hero_board_frame)

        for frame, frames in [
            (hero_pair_frame, hero_pair_frames),
            (hero_signature_frame, hero_signature_frames),
            (hero_archetype_frame, hero_archetype_frames),
            (hero_cluster_profile_frame, hero_cluster_profile_frames),
            (hero_cluster_component_frame, hero_cluster_component_frames),
            (hero_core_build_frame, hero_core_build_frames),
            (hero_family_frame, hero_family_frames),
            (hero_skill_shell_affinity_frame, hero_skill_shell_affinity_frames),
            (hero_item_shell_affinity_frame, hero_item_shell_affinity_frames),
            (hero_exact_triplet_frame, hero_exact_triplet_frames),
        ]:
            if frame.height:
                frames.append(frame.with_columns(pl.lit(hero).alias("hero")).select(["hero", *frame.columns]))

    hero_pair_frame = pl.concat(hero_pair_frames, how="diagonal_relaxed") if hero_pair_frames else pl.DataFrame()
    hero_signature_frame = pl.concat(hero_signature_frames, how="diagonal_relaxed") if hero_signature_frames else pl.DataFrame()
    hero_archetype_frame = pl.concat(hero_archetype_frames, how="diagonal_relaxed") if hero_archetype_frames else pl.DataFrame()
    hero_cluster_profile_frame = pl.concat(hero_cluster_profile_frames, how="diagonal_relaxed") if hero_cluster_profile_frames else pl.DataFrame()
    hero_cluster_component_frame = pl.concat(hero_cluster_component_frames, how="diagonal_relaxed") if hero_cluster_component_frames else pl.DataFrame()
    hero_core_build_frame = pl.concat(hero_core_build_frames, how="diagonal_relaxed") if hero_core_build_frames else pl.DataFrame()
    hero_family_frame = pl.concat(hero_family_frames, how="diagonal_relaxed") if hero_family_frames else pl.DataFrame()
    hero_report_frame = _archetype_report_frame(hero_family_frame)
    hero_skill_shell_affinity_frame = pl.concat(hero_skill_shell_affinity_frames, how="diagonal_relaxed") if hero_skill_shell_affinity_frames else pl.DataFrame()
    hero_item_shell_affinity_frame = pl.concat(hero_item_shell_affinity_frames, how="diagonal_relaxed") if hero_item_shell_affinity_frames else pl.DataFrame()
    hero_exact_triplet_frame = pl.concat(hero_exact_triplet_frames, how="diagonal_relaxed") if hero_exact_triplet_frames else pl.DataFrame()

    write_frame_exports(pair_frame, settings.exports_dir / "summary_systemic_item_pairs")
    write_frame_exports(signature_frame, settings.exports_dir / "summary_systemic_item_signatures")
    write_frame_exports(archetype_frame, settings.exports_dir / "summary_systemic_archetypes")
    write_frame_exports(cluster_profile_frame, settings.exports_dir / "summary_build_clusters")
    write_frame_exports(cluster_component_frame, settings.exports_dir / "summary_build_components")
    write_frame_exports(core_build_frame, settings.exports_dir / "summary_core_builds")
    write_frame_exports(family_frame, settings.exports_dir / "summary_archetype_families")
    write_frame_exports(report_frame, settings.exports_dir / "summary_archetype_report")
    write_frame_exports(skill_shell_affinity_frame, settings.exports_dir / "summary_skill_shell_affinity")
    write_frame_exports(item_shell_affinity_frame, settings.exports_dir / "summary_item_shell_affinity")
    write_frame_exports(exact_triplet_frame, settings.exports_dir / "summary_exact_item_triplets")
    write_frame_exports(source_alignment, settings.exports_dir / "summary_item_source_alignment")
    write_frame_exports(hero_pair_frame, settings.exports_dir / "summary_systemic_item_pairs_by_hero")
    write_frame_exports(hero_signature_frame, settings.exports_dir / "summary_systemic_item_signatures_by_hero")
    write_frame_exports(hero_archetype_frame, settings.exports_dir / "summary_systemic_archetypes_by_hero")
    write_frame_exports(hero_cluster_profile_frame, settings.exports_dir / "summary_build_clusters_by_hero")
    write_frame_exports(hero_cluster_component_frame, settings.exports_dir / "summary_build_components_by_hero")
    write_frame_exports(hero_core_build_frame, settings.exports_dir / "summary_core_builds_by_hero")
    write_frame_exports(hero_family_frame, settings.exports_dir / "summary_archetype_families_by_hero")
    write_frame_exports(hero_report_frame, settings.exports_dir / "summary_archetype_report_by_hero")
    write_frame_exports(hero_skill_shell_affinity_frame, settings.exports_dir / "summary_skill_shell_affinity_by_hero")
    write_frame_exports(hero_item_shell_affinity_frame, settings.exports_dir / "summary_item_shell_affinity_by_hero")
    write_frame_exports(hero_exact_triplet_frame, settings.exports_dir / "summary_exact_item_triplets_by_hero")

    return {
        "boards": total_boards,
        "systemic_pairs": pair_frame.height,
        "signature_items": signature_frame.height,
        "archetypes": archetype_frame.height,
        "build_clusters": cluster_profile_frame.height,
        "build_components": cluster_component_frame.height,
        "core_builds": core_build_frame.height,
        "archetype_families": family_frame.height,
        "archetype_report_rows": report_frame.height,
        "skill_shell_affinity_rows": skill_shell_affinity_frame.height,
        "item_shell_affinity_rows": item_shell_affinity_frame.height,
        "exact_item_triplets": exact_triplet_frame.height,
        "source_alignment_rows": source_alignment.height,
        "heroes": len(heroes),
        "systemic_pairs_by_hero": hero_pair_frame.height,
        "core_builds_by_hero": hero_core_build_frame.height,
        "archetype_families_by_hero": hero_family_frame.height,
        "archetype_report_rows_by_hero": hero_report_frame.height,
        "exact_item_triplets_by_hero": hero_exact_triplet_frame.height,
    }


def summarize(conn, settings: Settings) -> dict[str, int]:
    filters = _load_analysis_filters()
    item_frame, _total_boards = _board_presence_frame(conn, filters)
    skill_frame = _skill_presence_frame(conn, filters)
    outcome_frame = conn.query_pl(
        """
        SELECT COALESCE(s.screenshot_id, -r.run_id) AS screenshot_id, r.hero, r.created_at, r.outcome_text
        FROM screenshots s
        JOIN runs r ON r.run_id = s.run_id
        WHERE s.is_primary = 1
        """
    )
    outcome_frame = _filter_analysis_frame(outcome_frame, filters)
    run_meta_frame = _run_meta_frame(conn, filters)
    item_presence_frame = item_frame.select(["screenshot_id", "hero", "item_name"]).unique() if item_frame.height else item_frame.select(["screenshot_id", "hero", "item_name"])
    skill_presence_frame = skill_frame.select(["screenshot_id", "hero", "skill_name"]).unique() if skill_frame.height else skill_frame.select(["screenshot_id", "hero", "skill_name"])

    item_perf_frame = (
        item_presence_frame.join(
            run_meta_frame.select(["screenshot_id", "record_wins", "run_victory_tier"]),
            on="screenshot_id",
            how="left",
        )
        if item_presence_frame.height
        else pl.DataFrame(schema={"screenshot_id": pl.Int64, "item_name": pl.String, "record_wins": pl.Int64, "run_victory_tier": pl.String})
    )
    skill_perf_frame = (
        skill_presence_frame.join(
            run_meta_frame.select(["screenshot_id", "record_wins", "run_victory_tier"]),
            on="screenshot_id",
            how="left",
        )
        if skill_presence_frame.height
        else pl.DataFrame(schema={"screenshot_id": pl.Int64, "skill_name": pl.String, "record_wins": pl.Int64, "run_victory_tier": pl.String})
    )

    top_items = item_presence_frame.group_by("item_name").len(name="count").sort("count", descending=True)
    top_skills = skill_presence_frame.group_by("skill_name").len(name="count").sort("count", descending=True)
    top_items_by_hero = item_presence_frame.group_by(["hero", "item_name"]).len(name="count").sort(["hero", "count"], descending=[False, True]) if item_presence_frame.height else pl.DataFrame()
    top_skills_by_hero = skill_presence_frame.group_by(["hero", "skill_name"]).len(name="count").sort(["hero", "count"], descending=[False, True]) if skill_presence_frame.height else pl.DataFrame()
    item_lists = item_presence_frame.group_by("screenshot_id").agg(pl.col("item_name")).get_column("item_name").to_list() if item_presence_frame.height else []
    item_pair_counts = _cooccurrence(item_lists, "item_a", "item_b")
    item_pair_counts_by_hero = _cooccurrence_by_hero(item_presence_frame, "item_name", "item_a", "item_b")

    item_skill_join = item_presence_frame.join(skill_presence_frame, on=["screenshot_id", "hero"], how="inner")
    item_skill_counts = item_skill_join.group_by(["item_name", "skill_name"]).len(name="count").sort("count", descending=True)
    item_skill_counts_by_hero = item_skill_join.group_by(["hero", "item_name", "skill_name"]).len(name="count").sort(["hero", "count"], descending=[False, True]) if item_skill_join.height else pl.DataFrame()

    outcome_items = item_presence_frame.join(outcome_frame, on="screenshot_id", how="left")
    outcome_item_counts = outcome_items.filter(pl.col("outcome_text").is_not_null()).group_by(["outcome_text", "item_name"]).len(name="count").sort(["outcome_text", "count"], descending=[False, True])
    outcome_item_counts_by_hero = outcome_items.filter(pl.col("outcome_text").is_not_null()).group_by(["hero", "outcome_text", "item_name"]).len(name="count").sort(["hero", "outcome_text", "count"], descending=[False, False, True]) if outcome_items.height else pl.DataFrame()
    coverage = _pipeline_coverage_summary(conn, filters)
    item_performance = _performance_by_entity(item_perf_frame, "item_name")
    skill_performance = _performance_by_entity(skill_perf_frame, "skill_name")
    item_performance_by_hero = _performance_by_hero(item_perf_frame, "item_name")
    skill_performance_by_hero = _performance_by_hero(skill_perf_frame, "skill_name")
    item_counts_performance = _counts_with_performance(top_items, item_performance, "item_name")
    item_counts_performance_by_hero = _counts_with_performance_by_hero(top_items_by_hero, item_performance_by_hero, "item_name")

    write_frame_exports(top_items, settings.exports_dir / "summary_top_items")
    write_frame_exports(top_skills, settings.exports_dir / "summary_top_skills")
    write_frame_exports(top_items_by_hero, settings.exports_dir / "summary_top_items_by_hero")
    write_frame_exports(top_skills_by_hero, settings.exports_dir / "summary_top_skills_by_hero")
    write_frame_exports(item_pair_counts, settings.exports_dir / "summary_item_item_cooccurrence")
    write_frame_exports(item_pair_counts_by_hero, settings.exports_dir / "summary_item_item_cooccurrence_by_hero")
    write_frame_exports(item_skill_counts, settings.exports_dir / "summary_item_skill_cooccurrence")
    write_frame_exports(item_skill_counts_by_hero, settings.exports_dir / "summary_item_skill_cooccurrence_by_hero")
    write_frame_exports(outcome_item_counts, settings.exports_dir / "summary_outcome_filtered_items")
    write_frame_exports(outcome_item_counts_by_hero, settings.exports_dir / "summary_outcome_filtered_items_by_hero")
    write_frame_exports(coverage, settings.exports_dir / "summary_pipeline_coverage")
    write_frame_exports(item_performance, settings.exports_dir / "summary_item_performance")
    write_frame_exports(skill_performance, settings.exports_dir / "summary_skill_performance")
    write_frame_exports(item_performance_by_hero, settings.exports_dir / "summary_item_performance_by_hero")
    write_frame_exports(skill_performance_by_hero, settings.exports_dir / "summary_skill_performance_by_hero")
    write_frame_exports(item_counts_performance, settings.exports_dir / "summary_item_counts_performance")
    write_frame_exports(item_counts_performance_by_hero, settings.exports_dir / "summary_item_counts_performance_by_hero")

    return {
        "top_items": top_items.height,
        "top_skills": top_skills.height,
        "top_items_by_hero": top_items_by_hero.height,
        "top_skills_by_hero": top_skills_by_hero.height,
        "item_item_pairs": item_pair_counts.height,
        "item_item_pairs_by_hero": item_pair_counts_by_hero.height,
        "item_skill_pairs": item_skill_counts.height,
        "item_skill_pairs_by_hero": item_skill_counts_by_hero.height,
        "outcome_filtered_rows": outcome_item_counts.height,
        "outcome_filtered_rows_by_hero": outcome_item_counts_by_hero.height,
        "pipeline_coverage_rows": coverage.height,
        "item_performance_rows": item_performance.height,
        "skill_performance_rows": skill_performance.height,
        "item_performance_rows_by_hero": item_performance_by_hero.height,
        "skill_performance_rows_by_hero": skill_performance_by_hero.height,
        "item_counts_performance_rows": item_counts_performance.height,
        "item_counts_performance_rows_by_hero": item_counts_performance_by_hero.height,
    }
