from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import duckdb
import polars as pl

from .config import Settings


SCHEMA = """
CREATE SEQUENCE IF NOT EXISTS runs_id_seq START 1;
CREATE TABLE IF NOT EXISTS runs (
    run_id BIGINT PRIMARY KEY DEFAULT nextval('runs_id_seq'),
    source_run_id VARCHAR NOT NULL UNIQUE,
    hero VARCHAR NOT NULL,
    run_url VARCHAR NOT NULL UNIQUE,
    created_at VARCHAR,
    title VARCHAR NOT NULL,
    profile_name VARCHAR,
    profile_url VARCHAR,
    outcome_text VARCHAR,
    record_wins INTEGER,
    record_losses INTEGER,
    rank_tier VARCHAR,
    run_outcome_tier VARCHAR,
    run_wins_label VARCHAR,
    player_rank_tier VARCHAR,
    max_health INTEGER,
    prestige INTEGER,
    level INTEGER,
    income INTEGER,
    gold INTEGER,
    html_path VARCHAR,
    card_hints_json JSON NOT NULL DEFAULT '[]',
    board_cards_json JSON NOT NULL DEFAULT '[]',
    skill_cards_json JSON NOT NULL DEFAULT '[]',
    crawled_at VARCHAR NOT NULL
);

CREATE SEQUENCE IF NOT EXISTS screenshots_id_seq START 1;
CREATE TABLE IF NOT EXISTS screenshots (
    screenshot_id BIGINT PRIMARY KEY DEFAULT nextval('screenshots_id_seq'),
    run_id BIGINT NOT NULL,
    screenshot_url VARCHAR NOT NULL,
    local_path VARCHAR,
    sha256 VARCHAR,
    width INTEGER,
    height INTEGER,
    is_primary INTEGER NOT NULL DEFAULT 0,
    downloaded_at VARCHAR,
    UNIQUE(run_id, screenshot_url)
);

CREATE TABLE IF NOT EXISTS reference_items (
    entity_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    normalized_name VARCHAR NOT NULL,
    slug VARCHAR NOT NULL,
    page_url VARCHAR NOT NULL,
    image_url VARCHAR,
    image_path VARCHAR,
    aliases_json JSON NOT NULL DEFAULT '[]',
    metadata_json JSON NOT NULL DEFAULT '{}',
    collected_at VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS reference_skills (
    entity_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    normalized_name VARCHAR NOT NULL,
    slug VARCHAR NOT NULL,
    page_url VARCHAR NOT NULL,
    image_url VARCHAR,
    image_path VARCHAR,
    aliases_json JSON NOT NULL DEFAULT '[]',
    metadata_json JSON NOT NULL DEFAULT '{}',
    collected_at VARCHAR NOT NULL
);

CREATE SEQUENCE IF NOT EXISTS extracted_board_items_id_seq START 1;
CREATE TABLE IF NOT EXISTS extracted_board_items (
    detection_id BIGINT PRIMARY KEY DEFAULT nextval('extracted_board_items_id_seq'),
    screenshot_id BIGINT NOT NULL,
    slot_index INTEGER NOT NULL,
    entity_id VARCHAR,
    raw_label VARCHAR,
    confidence DOUBLE NOT NULL,
    method VARCHAR NOT NULL,
    bbox_x INTEGER NOT NULL,
    bbox_y INTEGER NOT NULL,
    bbox_w INTEGER NOT NULL,
    bbox_h INTEGER NOT NULL,
    duplicate_count INTEGER,
    crop_path VARCHAR,
    top_candidates_json JSON NOT NULL,
    status VARCHAR NOT NULL
);

CREATE SEQUENCE IF NOT EXISTS extracted_skills_id_seq START 1;
CREATE TABLE IF NOT EXISTS extracted_skills (
    detection_id BIGINT PRIMARY KEY DEFAULT nextval('extracted_skills_id_seq'),
    screenshot_id BIGINT NOT NULL,
    slot_index INTEGER NOT NULL,
    entity_id VARCHAR,
    raw_label VARCHAR,
    confidence DOUBLE NOT NULL,
    method VARCHAR NOT NULL,
    bbox_x INTEGER NOT NULL,
    bbox_y INTEGER NOT NULL,
    bbox_w INTEGER NOT NULL,
    bbox_h INTEGER NOT NULL,
    crop_path VARCHAR,
    top_candidates_json JSON NOT NULL,
    status VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS extracted_ranks (
    screenshot_id BIGINT PRIMARY KEY,
    raw_label VARCHAR,
    rank_tier VARCHAR,
    confidence DOUBLE NOT NULL,
    method VARCHAR NOT NULL,
    bbox_x INTEGER NOT NULL,
    bbox_y INTEGER NOT NULL,
    bbox_w INTEGER NOT NULL,
    bbox_h INTEGER NOT NULL,
    crop_path VARCHAR,
    top_candidates_json JSON NOT NULL,
    status VARCHAR NOT NULL
);

CREATE SEQUENCE IF NOT EXISTS review_queue_id_seq START 1;
CREATE TABLE IF NOT EXISTS review_queue (
    review_id BIGINT PRIMARY KEY DEFAULT nextval('review_queue_id_seq'),
    screenshot_id BIGINT NOT NULL,
    detection_type VARCHAR NOT NULL,
    crop_path VARCHAR,
    confidence DOUBLE NOT NULL,
    raw_label VARCHAR,
    top_candidates_json JSON NOT NULL,
    status VARCHAR NOT NULL DEFAULT 'pending',
    notes VARCHAR,
    UNIQUE(screenshot_id, detection_type, crop_path)
);
"""


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


PIPELINE_TABLES = [
    "runs",
    "screenshots",
    "extracted_board_items",
    "extracted_skills",
    "extracted_ranks",
    "review_queue",
]


PIPELINE_SEQUENCES = [
    "runs_id_seq",
    "screenshots_id_seq",
    "extracted_board_items_id_seq",
    "extracted_skills_id_seq",
    "review_queue_id_seq",
]


LEGACY_PIPELINE_TABLES = ["posts"]


LEGACY_PIPELINE_SEQUENCES = ["posts_id_seq"]


class Row:
    def __init__(self, columns: list[str], values: tuple):
        self._columns = columns
        self._values = values
        self._index = {column: idx for idx, column in enumerate(columns)}

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._values[key]
        return self._values[self._index[key]]

    def __iter__(self) -> Iterator:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def get(self, key, default=None):
        return self[key] if key in self._index else default


class CursorWrapper:
    def __init__(self, cursor: duckdb.DuckDBPyConnection):
        self._cursor = cursor

    def fetchall(self) -> list[Row]:
        rows = self._cursor.fetchall()
        columns = [column[0] for column in self._cursor.description]
        return [Row(columns, row) for row in rows]

    def fetchone(self) -> Row | None:
        row = self._cursor.fetchone()
        if row is None:
            return None
        columns = [column[0] for column in self._cursor.description]
        return Row(columns, row)


class DatabaseConnection:
    def __init__(self, path: Path):
        self.path = path
        self._conn = duckdb.connect(str(path))

    def execute(self, query: str, parameters: tuple | list | None = None) -> CursorWrapper:
        cursor = self._conn.cursor()
        if parameters is None:
            cursor.execute(query)
        else:
            cursor.execute(query, parameters)
        return CursorWrapper(cursor)

    def executemany(self, query: str, rows: Iterable[tuple]) -> None:
        values = list(rows)
        if not values:
            return
        self._conn.executemany(query, values)

    def query_pl(self, query: str, parameters: tuple | list | None = None) -> pl.DataFrame:
        cursor = self._conn.cursor()
        if parameters is None:
            cursor.execute(query)
        else:
            cursor.execute(query, parameters)
        return cursor.pl()

    def execute_script(self, script: str) -> None:
        self._conn.execute(script)

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def _table_exists(conn: DatabaseConnection, table: str) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) AS table_count FROM information_schema.tables WHERE table_name = ?",
        (table,),
    ).fetchone()
    return bool(row and row["table_count"])


def _table_columns(conn: DatabaseConnection, table: str) -> set[str]:
    if not _table_exists(conn, table):
        return set()
    rows = conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
        (table,),
    ).fetchall()
    return {row["column_name"] for row in rows}


def _table_has_rows(conn: DatabaseConnection, table: str) -> bool:
    row = conn.execute(f"SELECT COUNT(*) AS row_count FROM {table}").fetchone()
    return bool(row and row["row_count"])


def _requires_pipeline_reset(conn: DatabaseConnection) -> bool:
    run_columns = _table_columns(conn, "runs")
    screenshot_columns = _table_columns(conn, "screenshots")
    if _table_exists(conn, "posts"):
        return True
    if run_columns and {
        "run_id",
        "source_run_id",
        "run_url",
        "created_at",
        "card_hints_json",
        "run_outcome_tier",
        "run_wins_label",
        "player_rank_tier",
        "board_cards_json",
        "skill_cards_json",
    } - run_columns:
        return True
    if screenshot_columns and "run_id" not in screenshot_columns:
        return True
    return False


def _reset_pipeline_schema(conn: DatabaseConnection) -> None:
    for table in [*PIPELINE_TABLES, *LEGACY_PIPELINE_TABLES]:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    for sequence in [*PIPELINE_SEQUENCES, *LEGACY_PIPELINE_SEQUENCES]:
        conn.execute(f"DROP SEQUENCE IF EXISTS {sequence}")
    conn.commit()


def next_id(conn: DatabaseConnection, table: str, id_column: str) -> int:
    row = conn.execute(f"SELECT COALESCE(MAX({id_column}), 0) + 1 AS next_id FROM {table}").fetchone()
    return int(row["next_id"])


def connect(database_path: Path) -> DatabaseConnection:
    return DatabaseConnection(database_path)


def init_db(settings: Settings) -> DatabaseConnection:
    conn = connect(settings.duckdb_path)
    if _requires_pipeline_reset(conn):
        _reset_pipeline_schema(conn)
    conn.execute_script(SCHEMA)
    conn.commit()
    return conn


def execute_many(conn: DatabaseConnection, query: str, rows: Iterable[tuple]) -> None:
    conn.executemany(query, rows)
    conn.commit()
