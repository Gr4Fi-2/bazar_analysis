from __future__ import annotations

from pathlib import Path

import polars as pl


def write_frame_exports(frame: pl.DataFrame, output_base: Path) -> int:
    """Write one analysis output as CSV and Parquet, returning row count."""
    output_base.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(output_base.with_suffix(".csv"))
    frame.write_parquet(output_base.with_suffix(".parquet"))
    return frame.height


def write_named_exports(frames: dict[str, pl.DataFrame], output_dir: Path) -> dict[str, int]:
    return {name: write_frame_exports(frame, output_dir / name) for name, frame in frames.items()}
