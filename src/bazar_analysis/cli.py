from __future__ import annotations

import os

import typer

from .analysis import summarize, systemic_analysis
from .config import ensure_directories, get_settings, reset_workspace_data
from .crawler import crawl_runs
from .db import init_db
from .downloader import download_screenshots
from .exporter import export_datasets
from .extractor import extract_board_data, extract_rank_and_crown
from .reference import build_reference_catalog


app = typer.Typer(help="Bazaar endgame board extraction pipeline")


def _bootstrap():
    settings = get_settings()
    ensure_directories(settings)
    conn = init_db(settings)
    return settings, conn


@app.command("reset-data")
def reset_data_cmd() -> None:
    settings = get_settings()
    reset_workspace_data(settings)
    ensure_directories(settings)
    typer.echo({"reset": "ok"})


@app.command("crawl-runs")
def crawl_runs_cmd() -> None:
    settings, conn = _bootstrap()
    result = crawl_runs(conn, settings)
    typer.echo(result)


@app.command("build-reference")
def build_reference_cmd() -> None:
    settings, conn = _bootstrap()
    result = build_reference_catalog(conn, settings)
    typer.echo(result)


@app.command("download-screenshots")
def download_screenshots_cmd() -> None:
    settings, conn = _bootstrap()
    result = download_screenshots(conn, settings)
    typer.echo(result)


@app.command("extract-board-data")
def extract_board_data_cmd() -> None:
    settings, conn = _bootstrap()
    result = extract_board_data(conn, settings)
    typer.echo(result)


@app.command("extract-rank-crown")
def extract_rank_crown_cmd() -> None:
    settings, conn = _bootstrap()
    result = extract_rank_and_crown(conn, settings)
    typer.echo(result)


@app.command("export-datasets")
def export_datasets_cmd() -> None:
    settings, conn = _bootstrap()
    result = export_datasets(conn, settings)
    typer.echo(result)


@app.command("summarize")
def summarize_cmd() -> None:
    settings, conn = _bootstrap()
    result = summarize(conn, settings)
    typer.echo(result)


@app.command("systemic-analysis")
def systemic_analysis_cmd() -> None:
    settings, conn = _bootstrap()
    result = systemic_analysis(conn, settings)
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
            typer.echo({"extract_board_data": extract_board_data(conn, settings)})
        finally:
            if previous_source_only is None:
                os.environ.pop("BAZAR_EXTRACT_SOURCE_ONLY", None)
            else:
                os.environ["BAZAR_EXTRACT_SOURCE_ONLY"] = previous_source_only
    if export_base_datasets:
        typer.echo({"export_datasets": export_datasets(conn, settings)})
    typer.echo({"summarize": summarize(conn, settings)})
    if systemic:
        typer.echo({"systemic_analysis": systemic_analysis(conn, settings)})


@app.command("run-all")
def run_all() -> None:
    settings, conn = _bootstrap()
    typer.echo({"crawl_runs": crawl_runs(conn, settings)})
    typer.echo({"build_reference": build_reference_catalog(conn, settings)})
    typer.echo({"download_screenshots": download_screenshots(conn, settings)})
    typer.echo({"extract_board_data": extract_board_data(conn, settings)})
    typer.echo({"export_datasets": export_datasets(conn, settings)})
    typer.echo({"summarize": summarize(conn, settings)})
    typer.echo({"systemic_analysis": systemic_analysis(conn, settings)})


if __name__ == "__main__":
    app()
