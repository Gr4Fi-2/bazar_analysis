from __future__ import annotations

from pathlib import Path

import polars as pl

from .config import Settings
from .db import TABLES
from .reporting import write_frame_exports


def _scalar(conn, query: str, parameters: tuple | None = None) -> int | float | str | None:
    row = conn.execute(query, parameters).fetchone()
    if row is None or len(row) == 0:
        return None
    return row[0]


def _count_files(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern))) if path.exists() else 0


def validate_data(conn, settings: Settings) -> dict[str, int | float | str]:
    metrics: list[dict[str, str]] = []

    def add_metric(metric: str, value: int | float | str | None, status: str = "ok") -> None:
        metrics.append({"metric": metric, "value": "" if value is None else str(value), "status": status})

    for table in TABLES:
        add_metric(f"table_{table}_rows", int(_scalar(conn, f"SELECT COUNT(*) FROM {table}") or 0))

    runs = int(_scalar(conn, "SELECT COUNT(*) FROM runs") or 0)
    screenshots = int(_scalar(conn, "SELECT COUNT(*) FROM screenshots") or 0)
    board_payload_runs = int(_scalar(conn, "SELECT COUNT(*) FROM runs WHERE json_array_length(board_cards_json) > 0") or 0)
    skill_payload_runs = int(_scalar(conn, "SELECT COUNT(*) FROM runs WHERE json_array_length(skill_cards_json) > 0") or 0)
    rank_runs = int(_scalar(conn, "SELECT COUNT(*) FROM runs WHERE player_rank_tier IS NOT NULL") or 0)
    crown_runs = int(_scalar(conn, "SELECT COUNT(*) FROM runs WHERE has_broken_crown IS NOT NULL") or 0)
    pending_reviews = int(_scalar(conn, "SELECT COUNT(*) FROM review_queue WHERE status = 'pending'") or 0)
    source_board = int(_scalar(conn, "SELECT COUNT(*) FROM extracted_board_items WHERE method = 'run_detail_board'") or 0)
    source_skills = int(_scalar(conn, "SELECT COUNT(*) FROM extracted_skills WHERE method = 'run_detail_skill'") or 0)
    image_board = int(_scalar(conn, "SELECT COUNT(*) FROM extracted_board_items WHERE method <> 'run_detail_board'") or 0)
    image_skills = int(_scalar(conn, "SELECT COUNT(*) FROM extracted_skills WHERE method <> 'run_detail_skill'") or 0)

    derived_metrics = [
        ("primary_screenshots", int(_scalar(conn, "SELECT COUNT(*) FROM screenshots WHERE is_primary = 1") or 0), "ok"),
        ("runs_with_board_payload", board_payload_runs, "ok" if runs == 0 or board_payload_runs / runs >= 0.95 else "warn"),
        ("runs_with_skill_payload", skill_payload_runs, "ok" if runs == 0 or skill_payload_runs / runs >= 0.90 else "warn"),
        ("board_payload_coverage_pct", round((board_payload_runs / runs * 100.0) if runs else 0.0, 2), "ok"),
        ("skill_payload_coverage_pct", round((skill_payload_runs / runs * 100.0) if runs else 0.0, 2), "ok"),
        ("runs_with_player_rank_tier", rank_runs, "warn" if runs and rank_runs == 0 else "ok"),
        ("runs_with_broken_crown_state", crown_runs, "warn" if runs and crown_runs == 0 else "ok"),
        ("pending_review_queue_rows", pending_reviews, "warn" if pending_reviews else "ok"),
        ("source_board_detections", source_board, "ok"),
        ("source_skill_detections", source_skills, "ok"),
        ("image_board_detections", image_board, "ok"),
        ("image_skill_detections", image_skills, "ok"),
        ("export_csv_files", _count_files(settings.exports_dir, "*.csv"), "ok"),
        ("export_parquet_files", _count_files(settings.exports_dir, "*.parquet"), "ok"),
    ]
    for metric, value, status in derived_metrics:
        add_metric(metric, value, status)

    latest_created_at = _scalar(conn, "SELECT MAX(created_at) FROM runs")
    if latest_created_at:
        add_metric("latest_run_created_at", str(latest_created_at))

    frame = pl.DataFrame(metrics)
    rows = write_frame_exports(frame, settings.exports_dir / "summary_data_validation")
    warn_count = sum(1 for row in metrics if row["status"] == "warn")
    return {
        "rows": rows,
        "warnings": warn_count,
        "runs": runs,
        "screenshots": screenshots,
        "csv": str(settings.exports_dir / "summary_data_validation.csv"),
        "parquet": str(settings.exports_dir / "summary_data_validation.parquet"),
    }
