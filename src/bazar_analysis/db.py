from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path

import duckdb
import polars as pl

from .config import Settings


SCHEMA = """
CREATE SEQUENCE IF NOT EXISTS posts_id_seq START 1;
CREATE TABLE IF NOT EXISTS posts (
    post_id BIGINT PRIMARY KEY DEFAULT nextval('posts_id_seq'),
    hero VARCHAR NOT NULL,
    post_url VARCHAR NOT NULL UNIQUE,
    title VARCHAR NOT NULL,
    post_date VARCHAR,
    author VARCHAR,
    html_path VARCHAR,
    score_wins INTEGER,
    score_losses INTEGER,
    rank_title_hint VARCHAR,
    item_hints_json JSON NOT NULL DEFAULT '[]',
    crawled_at VARCHAR NOT NULL
);

CREATE SEQUENCE IF NOT EXISTS screenshots_id_seq START 1;
CREATE TABLE IF NOT EXISTS screenshots (
    screenshot_id BIGINT PRIMARY KEY DEFAULT nextval('screenshots_id_seq'),
    post_id BIGINT NOT NULL,
    screenshot_url VARCHAR NOT NULL,
    local_path VARCHAR,
    sha256 VARCHAR,
    width INTEGER,
    height INTEGER,
    is_primary INTEGER NOT NULL DEFAULT 0,
    downloaded_at VARCHAR,
    UNIQUE(post_id, screenshot_url)
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
    "posts",
    "screenshots",
    "reference_items",
    "reference_skills",
    "extracted_board_items",
    "extracted_skills",
    "extracted_ranks",
    "review_queue",
]


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


def _table_has_rows(conn: DatabaseConnection, table: str) -> bool:
    row = conn.execute(f"SELECT COUNT(*) AS row_count FROM {table}").fetchone()
    return bool(row and row["row_count"])


def _migrate_sqlite_if_needed(conn: DatabaseConnection, sqlite_path: Path) -> None:
    if conn.path.exists() and any(_table_has_rows(conn, table) for table in TABLES):
        return
    if not sqlite_path.exists():
        return

    sqlite_conn = sqlite3.connect(sqlite_path)
    try:
        for table in TABLES:
            rows = sqlite_conn.execute(f"SELECT * FROM {table}").fetchall()
            if not rows:
                continue
            columns = [column[1] for column in sqlite_conn.execute(f"PRAGMA table_info({table})").fetchall()]
            placeholders = ", ".join(["?"] * len(columns))
            column_list = ", ".join(columns)
            conn.executemany(
                f"INSERT INTO {table} ({column_list}) VALUES ({placeholders})",
                rows,
            )
        conn.commit()
    finally:
        sqlite_conn.close()


def next_id(conn: DatabaseConnection, table: str, id_column: str) -> int:
    row = conn.execute(f"SELECT COALESCE(MAX({id_column}), 0) + 1 AS next_id FROM {table}").fetchone()
    return int(row["next_id"])


def connect(database_path: Path) -> DatabaseConnection:
    return DatabaseConnection(database_path)


def init_db(settings: Settings) -> DatabaseConnection:
    database_existed = settings.duckdb_path.exists()
    conn = connect(settings.duckdb_path)
    conn.execute_script(SCHEMA)
    conn.commit()
    if not database_existed:
        _migrate_sqlite_if_needed(conn, settings.sqlite_legacy_path)
    return conn


def execute_many(conn: DatabaseConnection, query: str, rows: Iterable[tuple]) -> None:
    conn.executemany(query, rows)
    conn.commit()
