from __future__ import annotations

import math

import polars as pl

from .analysis import PERFORMANCE_PRIOR_RUNS, _board_presence_frame, _load_analysis_filters, _run_meta_frame
from .config import Settings
from .reporting import write_frame_exports
from .utils import slugify


def _confidence_label(run_count: int) -> str:
    if run_count >= 100:
        return "high"
    if run_count >= 40:
        return "medium"
    if run_count >= 15:
        return "low"
    return "very_low"


def _usage_band(presence_pct: float) -> str:
    if presence_pct >= 10.0:
        return "core/common"
    if presence_pct >= 5.0:
        return "common"
    if presence_pct >= 2.0:
        return "uncommon"
    if presence_pct >= 0.5:
        return "rare"
    return "niche"


def _tier_label(score_delta: float) -> str:
    if score_delta >= 0.65:
        return "S"
    if score_delta >= 0.35:
        return "A"
    if score_delta >= 0.10:
        return "B"
    if score_delta >= -0.15:
        return "C"
    return "D"


def _tier_from_score(score: float | None, baseline_avg_wins: float) -> str:
    try:
        score_delta = float(score) - baseline_avg_wins
    except (TypeError, ValueError):
        return "D"
    if math.isnan(score_delta):
        return "D"
    return _tier_label(score_delta)


def _empty_tierlist_frame(hero: str) -> pl.DataFrame:
    baseline_prefix = slugify(hero).replace("-", "_")
    return pl.DataFrame(
        schema={
            "tier": pl.String,
            "confidence": pl.String,
            "usage_band": pl.String,
            "item_name": pl.String,
            "run_count": pl.Int64,
            "presence_pct": pl.Float64,
            "bias_adjusted_score": pl.Float64,
            "avg_wins": pl.Float64,
            "weighted_avg_wins": pl.Float64,
            "avg_wins_delta": pl.Float64,
            "gold_plus_count": pl.Int64,
            "gold_plus_pct": pl.Float64,
            "weighted_gold_plus_pct": pl.Float64,
            "gold_plus_delta_pct": pl.Float64,
            "perfect_count": pl.Int64,
            "perfect_pct": pl.Float64,
            "weighted_perfect_pct": pl.Float64,
            "perfect_delta_pct": pl.Float64,
            "top_outcome": pl.String,
            f"{baseline_prefix}_baseline_avg_wins": pl.Float64,
            f"{baseline_prefix}_baseline_gold_plus_pct": pl.Float64,
            f"{baseline_prefix}_baseline_perfect_pct": pl.Float64,
        }
    )


def build_item_tier_list(conn, settings: Settings, hero: str, min_runs: int = 1) -> dict[str, int | str]:
    frame = item_tier_list_frame(conn, hero, min_runs=min_runs)
    output_base = settings.exports_dir / f"{slugify(hero)}_item_tier_list_bias_adjusted"
    rows = write_frame_exports(frame, output_base)
    return {"hero": hero, "rows": rows, "csv": str(output_base.with_suffix(".csv")), "parquet": str(output_base.with_suffix(".parquet"))}


def item_tier_list_frame(conn, hero: str, min_runs: int = 1) -> pl.DataFrame:
    filters = _load_analysis_filters()
    board_frame, _total_boards = _board_presence_frame(conn, filters)
    if not board_frame.height:
        return _empty_tierlist_frame(hero)

    hero_frame = board_frame.filter(pl.col("hero") == hero)
    if not hero_frame.height:
        return _empty_tierlist_frame(hero)

    run_meta = _run_meta_frame(conn, filters).filter(pl.col("hero") == hero)
    if not run_meta.height:
        return _empty_tierlist_frame(hero)

    total_hero_runs = int(run_meta.get_column("screenshot_id").n_unique())
    if total_hero_runs == 0:
        return _empty_tierlist_frame(hero)

    baseline_avg_wins = run_meta.get_column("record_wins").drop_nulls().mean()
    baseline_avg_wins = float(baseline_avg_wins) if baseline_avg_wins is not None else 0.0
    baseline_gold_plus = float(run_meta.select(pl.col("run_victory_tier").is_in(["Gold", "Perfect"]).mean()).item() or 0.0)
    baseline_perfect = float(run_meta.select((pl.col("run_victory_tier") == "Perfect").mean()).item() or 0.0)

    presence = hero_frame.select(["screenshot_id", "item_name"]).unique()
    perf_frame = presence.join(
        run_meta.select(["screenshot_id", "record_wins", "run_victory_tier"]),
        on="screenshot_id",
        how="left",
    )

    if not perf_frame.height:
        return _empty_tierlist_frame(hero)

    prior = float(PERFORMANCE_PRIOR_RUNS)
    baseline_prefix = slugify(hero).replace("-", "_")
    result = (
        perf_frame.group_by("item_name")
        .agg(
            pl.len().alias("run_count"),
            pl.col("record_wins").drop_nulls().mean().alias("avg_wins"),
            pl.col("run_victory_tier").is_in(["Gold", "Perfect"]).sum().alias("gold_plus_count"),
            pl.col("run_victory_tier").is_in(["Gold", "Perfect"]).mean().alias("gold_plus_rate"),
            (pl.col("run_victory_tier") == "Perfect").sum().alias("perfect_count"),
            (pl.col("run_victory_tier") == "Perfect").mean().alias("perfect_rate"),
            pl.col("run_victory_tier").drop_nulls().mode().first().alias("top_outcome"),
        )
        .filter(pl.col("run_count") >= max(1, int(min_runs)))
        .with_columns(
            (pl.col("run_count") / pl.lit(total_hero_runs) * 100.0).alias("presence_pct"),
            (((pl.col("avg_wins") * pl.col("run_count")) + (pl.lit(baseline_avg_wins) * prior)) / (pl.col("run_count") + prior)).alias("weighted_avg_wins"),
            (((pl.col("gold_plus_rate") * pl.col("run_count")) + (pl.lit(baseline_gold_plus) * prior)) / (pl.col("run_count") + prior)).alias("weighted_gold_plus_rate"),
            (((pl.col("perfect_rate") * pl.col("run_count")) + (pl.lit(baseline_perfect) * prior)) / (pl.col("run_count") + prior)).alias("weighted_perfect_rate"),
        )
        .with_columns(
            (pl.col("weighted_avg_wins") - pl.lit(baseline_avg_wins)).alias("avg_wins_delta"),
            ((pl.col("weighted_gold_plus_rate") - pl.lit(baseline_gold_plus)) * 100.0).alias("gold_plus_delta_pct"),
            ((pl.col("weighted_perfect_rate") - pl.lit(baseline_perfect)) * 100.0).alias("perfect_delta_pct"),
        )
        .with_columns(
            (
                pl.lit(baseline_avg_wins)
                + pl.col("avg_wins_delta")
                + ((pl.col("gold_plus_delta_pct") / 100.0) * 2.0)
                + ((pl.col("perfect_delta_pct") / 100.0) * 1.0)
            ).alias("bias_adjusted_score"),
            (pl.col("gold_plus_rate") * 100.0).alias("gold_plus_pct"),
            (pl.col("weighted_gold_plus_rate") * 100.0).alias("weighted_gold_plus_pct"),
            (pl.col("perfect_rate") * 100.0).alias("perfect_pct"),
            (pl.col("weighted_perfect_rate") * 100.0).alias("weighted_perfect_pct"),
            pl.lit(baseline_avg_wins).alias(f"{baseline_prefix}_baseline_avg_wins"),
            pl.lit(baseline_gold_plus * 100.0).alias(f"{baseline_prefix}_baseline_gold_plus_pct"),
            pl.lit(baseline_perfect * 100.0).alias(f"{baseline_prefix}_baseline_perfect_pct"),
        )
        .with_columns(
            pl.struct(["run_count"]).map_elements(lambda value: _confidence_label(int(value["run_count"])), return_dtype=pl.String).alias("confidence"),
            pl.struct(["presence_pct"]).map_elements(lambda value: _usage_band(float(value["presence_pct"])), return_dtype=pl.String).alias("usage_band"),
            pl.struct(["bias_adjusted_score"]).map_elements(
                lambda value: _tier_from_score(value["bias_adjusted_score"], baseline_avg_wins),
                return_dtype=pl.String,
            ).alias("tier"),
        )
        .with_columns(
            pl.col("presence_pct").round(2),
            pl.col("bias_adjusted_score").round(3),
            pl.col("avg_wins").round(3),
            pl.col("weighted_avg_wins").round(3),
            pl.col("avg_wins_delta").round(3),
            pl.col("gold_plus_pct").round(2),
            pl.col("weighted_gold_plus_pct").round(2),
            pl.col("gold_plus_delta_pct").round(2),
            pl.col("perfect_pct").round(2),
            pl.col("weighted_perfect_pct").round(2),
            pl.col("perfect_delta_pct").round(2),
            pl.col(f"{baseline_prefix}_baseline_avg_wins").round(3),
            pl.col(f"{baseline_prefix}_baseline_gold_plus_pct").round(2),
            pl.col(f"{baseline_prefix}_baseline_perfect_pct").round(2),
        )
        .with_columns(
            pl.col("tier").map_elements(lambda value: {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4}.get(value, 99), return_dtype=pl.Int64).alias("_tier_rank")
        )
        .sort(["_tier_rank", "bias_adjusted_score", "run_count"], descending=[False, True, True])
        .drop("_tier_rank")
        .select(list(_empty_tierlist_frame(hero).schema.keys()))
    )
    return result
