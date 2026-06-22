from __future__ import annotations

from contextlib import contextmanager
import os
import threading
import time
from typing import Callable, TypeVar

import typer

from .analysis import summarize, systemic_analysis
from .config import ensure_directories, get_settings, reset_workspace_data
from .crawler import ALL_RUN_HEROES, crawl_runs
from .db import init_db
from .downloader import download_screenshots
from .exporter import export_datasets
from .extractor import extract_board_data, extract_rank_and_crown
from .reference import build_reference_catalog


app = typer.Typer(help="Bazaar endgame board extraction pipeline")

T = TypeVar("T")


def _bootstrap():
    settings = get_settings()
    ensure_directories(settings)
    conn = init_db(settings)
    return settings, conn


def _format_elapsed(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _heartbeat_interval_seconds() -> float:
    raw_value = os.environ.get("BAZAR_PROGRESS_HEARTBEAT_SECONDS", "120").strip()
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        return 120.0


@contextmanager
def _step_progress(label: str):
    start = time.monotonic()
    interval_seconds = _heartbeat_interval_seconds()
    stop_event = threading.Event()
    thread: threading.Thread | None = None

    print(f"[progress] {label} started", flush=True)

    if interval_seconds > 0:
        def heartbeat() -> None:
            while not stop_event.wait(interval_seconds):
                elapsed = _format_elapsed(time.monotonic() - start)
                print(f"[progress] {label} still running after {elapsed}", flush=True)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()

    try:
        yield
    except Exception:
        elapsed = _format_elapsed(time.monotonic() - start)
        print(f"[progress] {label} failed after {elapsed}", flush=True)
        raise
    else:
        elapsed = _format_elapsed(time.monotonic() - start)
        print(f"[progress] {label} done after {elapsed}", flush=True)
    finally:
        stop_event.set()
        if thread is not None:
            thread.join(timeout=0.2)


def _run_step(label: str, func: Callable[..., T], *args) -> T:
    with _step_progress(label):
        return func(*args)


@contextmanager
def _temporary_env(overrides: dict[str, str]):
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@app.command("reset-data")
def reset_data_cmd() -> None:
    settings = get_settings()
    reset_workspace_data(settings)
    ensure_directories(settings)
    typer.echo({"reset": "ok"})


@app.command("crawl-runs")
def crawl_runs_cmd() -> None:
    settings, conn = _bootstrap()
    result = _run_step("crawl_runs", crawl_runs, conn, settings)
    typer.echo(result)


@app.command("build-reference")
def build_reference_cmd() -> None:
    settings, conn = _bootstrap()
    result = _run_step("build_reference", build_reference_catalog, conn, settings)
    typer.echo(result)


@app.command("download-screenshots")
def download_screenshots_cmd() -> None:
    settings, conn = _bootstrap()
    result = _run_step("download_screenshots", download_screenshots, conn, settings)
    typer.echo(result)


@app.command("extract-board-data")
def extract_board_data_cmd() -> None:
    settings, conn = _bootstrap()
    result = _run_step("extract_board_data", extract_board_data, conn, settings)
    typer.echo(result)


@app.command("extract-rank-crown")
def extract_rank_crown_cmd() -> None:
    settings, conn = _bootstrap()
    result = _run_step("extract_rank_crown", extract_rank_and_crown, conn, settings)
    typer.echo(result)


@app.command("export-datasets")
def export_datasets_cmd() -> None:
    settings, conn = _bootstrap()
    result = _run_step("export_datasets", export_datasets, conn, settings)
    typer.echo(result)


@app.command("summarize")
def summarize_cmd() -> None:
    settings, conn = _bootstrap()
    result = _run_step("summarize", summarize, conn, settings)
    typer.echo(result)


@app.command("systemic-analysis")
def systemic_analysis_cmd() -> None:
    settings, conn = _bootstrap()
    result = _run_step("systemic_analysis", systemic_analysis, conn, settings)
    typer.echo(result)


@app.command("run-analysis-fast")
def run_analysis_fast(
    refresh_extraction: bool = typer.Option(
        False,
        "--refresh-extraction",
        help="Rewrite extracted item/skill tables from run payloads before analysis.",
    ),
    export_base_datasets: bool = typer.Option(
        False,
        "--export-base-datasets",
        help="Rewrite base CSV/Parquet exports in addition to summary exports.",
    ),
    systemic: bool = typer.Option(
        True,
        "--systemic/--no-systemic",
        help="Include systemic archetype/core-build exports.",
    ),
) -> None:
    """Refresh analysis summaries from run payloads without crawling references or screenshots."""
    settings, conn = _bootstrap()
    if refresh_extraction:
        previous_source_only = os.environ.get("BAZAR_EXTRACT_SOURCE_ONLY")
        os.environ["BAZAR_EXTRACT_SOURCE_ONLY"] = "1"
        try:
            typer.echo({"extract_board_data": _run_step("extract_board_data", extract_board_data, conn, settings)})
        finally:
            if previous_source_only is None:
                os.environ.pop("BAZAR_EXTRACT_SOURCE_ONLY", None)
            else:
                os.environ["BAZAR_EXTRACT_SOURCE_ONLY"] = previous_source_only
    if export_base_datasets:
        typer.echo({"export_datasets": _run_step("export_datasets", export_datasets, conn, settings)})
    typer.echo({"summarize": _run_step("summarize", summarize, conn, settings)})
    if systemic:
        typer.echo({"systemic_analysis": _run_step("systemic_analysis", systemic_analysis, conn, settings)})


@app.command("refresh-season-fast")
def refresh_season_fast(
    heroes: str = typer.Option(
        ",".join(ALL_RUN_HEROES),
        "--heroes",
        help="Comma-separated heroes to crawl. Use 'all' for all known heroes.",
    ),
    date_range: str = typer.Option(
        "latest_season",
        "--date-range",
        help="Run date range, for example latest_season, season15, season14, last7d.",
    ),
    discovery_pages: str = typer.Option(
        "0",
        "--discovery-pages",
        help="0/all means page until the API feed is exhausted.",
    ),
    crawl_workers: int = typer.Option(
        4,
        "--crawl-workers",
        help="Parallel workers for run detail parsing. DB writes stay serial.",
    ),
    crawl_delay_seconds: float = typer.Option(
        0.10,
        "--crawl-delay-seconds",
        help="Delay before uncached BazaarDB HTTP requests.",
    ),
    use_html_cache: bool = typer.Option(
        True,
        "--use-html-cache/--no-html-cache",
        help="Reuse cached immutable run detail HTML when available.",
    ),
    systemic: bool = typer.Option(
        True,
        "--systemic/--no-systemic",
        help="Include systemic archetype/core-build exports.",
    ),
) -> None:
    """Refresh runs and source-first analysis without reference or screenshot work."""
    overrides = {
        "BAZAR_RUN_HEROES": heroes,
        "BAZAR_RUN_DATE_RANGE": date_range,
        "BAZAR_RUN_DISCOVERY_PAGES": discovery_pages,
        "BAZAR_CRAWL_WORKERS": str(max(1, crawl_workers)),
        "BAZAR_CRAWL_DELAY_SECONDS": str(max(0.0, crawl_delay_seconds)),
        "BAZAR_CRAWL_USE_HTML_CACHE": "1" if use_html_cache else "0",
        "BAZAR_EXTRACT_SOURCE_ONLY": "1",
    }
    with _temporary_env(overrides):
        settings, conn = _bootstrap()
        typer.echo({"crawl_runs": _run_step("crawl_runs", crawl_runs, conn, settings)})
        typer.echo({"extract_board_data": _run_step("extract_board_data", extract_board_data, conn, settings)})
        typer.echo({"export_datasets": _run_step("export_datasets", export_datasets, conn, settings)})
        typer.echo({"summarize": _run_step("summarize", summarize, conn, settings)})
        if systemic:
            typer.echo({"systemic_analysis": _run_step("systemic_analysis", systemic_analysis, conn, settings)})


@app.command("run-all")
def run_all() -> None:
    settings, conn = _bootstrap()
    typer.echo({"crawl_runs": _run_step("crawl_runs", crawl_runs, conn, settings)})
    typer.echo({"build_reference": _run_step("build_reference", build_reference_catalog, conn, settings)})
    typer.echo({"download_screenshots": _run_step("download_screenshots", download_screenshots, conn, settings)})
    typer.echo({"extract_board_data": _run_step("extract_board_data", extract_board_data, conn, settings)})
    typer.echo({"export_datasets": _run_step("export_datasets", export_datasets, conn, settings)})
    typer.echo({"summarize": _run_step("summarize", summarize, conn, settings)})
    typer.echo({"systemic_analysis": _run_step("systemic_analysis", systemic_analysis, conn, settings)})


if __name__ == "__main__":
    app()
