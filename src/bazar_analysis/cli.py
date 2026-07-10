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
from .tierlist import build_item_tier_list
from .validation import validate_data


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


def _analysis_env_overrides(
    heroes: str = "",
    date_range: str = "",
    created_after: str = "",
    created_before: str = "",
) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if heroes.strip():
        overrides["BAZAR_ANALYSIS_HEROES"] = heroes.strip()
    if date_range.strip():
        overrides["BAZAR_ANALYSIS_DATE_RANGE"] = date_range.strip()
    if created_after.strip():
        overrides["BAZAR_ANALYSIS_CREATED_AFTER"] = created_after.strip()
    if created_before.strip():
        overrides["BAZAR_ANALYSIS_CREATED_BEFORE"] = created_before.strip()
    return overrides


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
def summarize_cmd(
    analysis_heroes: str = typer.Option("", "--analysis-heroes", help="Optional comma-separated heroes to include in analysis outputs."),
    analysis_date_range: str = typer.Option("", "--analysis-date-range", help="Optional analysis date range such as last3d, last7d, season15."),
    analysis_created_after: str = typer.Option("", "--analysis-created-after", help="Optional hard lower created_at bound for analysis outputs."),
    analysis_created_before: str = typer.Option("", "--analysis-created-before", help="Optional hard upper created_at bound for analysis outputs."),
) -> None:
    with _temporary_env(_analysis_env_overrides(analysis_heroes, analysis_date_range, analysis_created_after, analysis_created_before)):
        settings, conn = _bootstrap()
        result = _run_step("summarize", summarize, conn, settings)
        typer.echo(result)


@app.command("systemic-analysis")
def systemic_analysis_cmd(
    analysis_heroes: str = typer.Option("", "--analysis-heroes", help="Optional comma-separated heroes to include in analysis outputs."),
    analysis_date_range: str = typer.Option("", "--analysis-date-range", help="Optional analysis date range such as last3d, last7d, season15."),
    analysis_created_after: str = typer.Option("", "--analysis-created-after", help="Optional hard lower created_at bound for analysis outputs."),
    analysis_created_before: str = typer.Option("", "--analysis-created-before", help="Optional hard upper created_at bound for analysis outputs."),
) -> None:
    with _temporary_env(_analysis_env_overrides(analysis_heroes, analysis_date_range, analysis_created_after, analysis_created_before)):
        settings, conn = _bootstrap()
        result = _run_step("systemic_analysis", systemic_analysis, conn, settings)
        typer.echo(result)


@app.command("validate-data")
def validate_data_cmd() -> None:
    settings, conn = _bootstrap()
    result = _run_step("validate_data", validate_data, conn, settings)
    typer.echo(result)


@app.command("item-tier-list")
def item_tier_list_cmd(
    hero: str = typer.Option("Jules", "--hero", help="Hero to export, for example Jules or Vanessa."),
    min_runs: int = typer.Option(1, "--min-runs", help="Minimum item sample size for inclusion."),
    analysis_date_range: str = typer.Option("", "--analysis-date-range", help="Optional analysis date range such as last3d, last7d, season15."),
    analysis_created_after: str = typer.Option("", "--analysis-created-after", help="Optional hard lower created_at bound for analysis outputs."),
    analysis_created_before: str = typer.Option("", "--analysis-created-before", help="Optional hard upper created_at bound for analysis outputs."),
) -> None:
    with _temporary_env(_analysis_env_overrides(hero, analysis_date_range, analysis_created_after, analysis_created_before)):
        settings, conn = _bootstrap()
        result = _run_step("item_tier_list", build_item_tier_list, conn, settings, hero, min_runs)
        typer.echo(result)


def _selected_heroes(raw_value: str) -> list[str]:
    heroes: list[str] = []
    for token in raw_value.split(","):
        hero = token.strip()
        if not hero:
            continue
        if hero.lower() in {"all", "*"}:
            return list(ALL_RUN_HEROES)
        heroes.append(hero.title())
    return heroes or ["Jules"]


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
    tier_list_heroes: str = typer.Option(
        "",
        "--tier-list-heroes",
        help="Optional comma-separated heroes for bias-adjusted item tier-list exports.",
    ),
    analysis_heroes: str = typer.Option("", "--analysis-heroes", help="Optional comma-separated heroes to include in analysis outputs."),
    analysis_date_range: str = typer.Option("", "--analysis-date-range", help="Optional analysis date range such as last3d, last7d, season15."),
    analysis_created_after: str = typer.Option("", "--analysis-created-after", help="Optional hard lower created_at bound for analysis outputs."),
    analysis_created_before: str = typer.Option("", "--analysis-created-before", help="Optional hard upper created_at bound for analysis outputs."),
) -> None:
    """Refresh analysis summaries from run payloads without crawling references or screenshots."""
    settings, conn = _bootstrap()
    with _temporary_env(_analysis_env_overrides(analysis_heroes, analysis_date_range, analysis_created_after, analysis_created_before)):
        if refresh_extraction:
            with _temporary_env({"BAZAR_EXTRACT_SOURCE_ONLY": "1"}):
                typer.echo({"extract_board_data": _run_step("extract_board_data", extract_board_data, conn, settings)})
        if export_base_datasets:
            typer.echo({"export_datasets": _run_step("export_datasets", export_datasets, conn, settings)})
        typer.echo({"summarize": _run_step("summarize", summarize, conn, settings)})
        if systemic:
            typer.echo({"systemic_analysis": _run_step("systemic_analysis", systemic_analysis, conn, settings)})
        if tier_list_heroes.strip():
            for hero in _selected_heroes(tier_list_heroes):
                typer.echo({"item_tier_list": _run_step(f"item_tier_list_{hero}", build_item_tier_list, conn, settings, hero, 1)})
        typer.echo({"validate_data": _run_step("validate_data", validate_data, conn, settings)})


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
        1.50,
        "--crawl-delay-seconds",
        help="Process-wide minimum gap between uncached BazaarDB HTTP requests.",
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
    tier_lists: bool = typer.Option(
        True,
        "--tier-lists/--no-tier-lists",
        help="Write bias-adjusted item tier lists for the selected heroes.",
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
        "BAZAR_ANALYSIS_HEROES": heroes,
        "BAZAR_ANALYSIS_DATE_RANGE": date_range,
    }
    with _temporary_env(overrides):
        settings, conn = _bootstrap()
        typer.echo({"crawl_runs": _run_step("crawl_runs", crawl_runs, conn, settings)})
        typer.echo({"extract_board_data": _run_step("extract_board_data", extract_board_data, conn, settings)})
        typer.echo({"export_datasets": _run_step("export_datasets", export_datasets, conn, settings)})
        typer.echo({"summarize": _run_step("summarize", summarize, conn, settings)})
        if systemic:
            typer.echo({"systemic_analysis": _run_step("systemic_analysis", systemic_analysis, conn, settings)})
        if tier_lists:
            for hero in _selected_heroes(heroes):
                typer.echo({"item_tier_list": _run_step(f"item_tier_list_{hero}", build_item_tier_list, conn, settings, hero, 1)})
        typer.echo({"validate_data": _run_step("validate_data", validate_data, conn, settings)})


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
    typer.echo({"validate_data": _run_step("validate_data", validate_data, conn, settings)})


if __name__ == "__main__":
    app()
