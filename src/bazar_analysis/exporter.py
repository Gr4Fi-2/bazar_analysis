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
        row = conn.execute(f"SELECT COUNT(*) AS row_count FROM {table}").fetchone()
        counts[table] = int(row["row_count"] if row else 0)
        csv_path = settings.exports_dir / f"{table}.csv"
        parquet_path = settings.exports_dir / f"{table}.parquet"
        conn.execute(f"COPY (SELECT * FROM {table}) TO {_sql_literal(str(csv_path))} (HEADER, DELIMITER ',')")
        conn.execute(f"COPY (SELECT * FROM {table}) TO {_sql_literal(str(parquet_path))} (FORMAT PARQUET)")
    return counts


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
