from __future__ import annotations

from .config import Settings


TABLES = [
    "runs",
    "screenshots",
    "reference_items",
    "reference_skills",
    "extracted_board_items",
    "extracted_skills",
    "extracted_ranks",
    "review_queue",
]


def export_datasets(conn, settings: Settings) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in TABLES:
        frame = conn.query_pl(f"SELECT * FROM {table}")
        counts[table] = frame.height
        csv_path = settings.exports_dir / f"{table}.csv"
        parquet_path = settings.exports_dir / f"{table}.parquet"
        frame.write_csv(csv_path)
        frame.write_parquet(parquet_path)
    return counts
