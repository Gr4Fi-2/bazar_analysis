from __future__ import annotations

import typer

from .analysis import summarize
from .config import ensure_directories, get_settings
from .crawler import crawl_posts
from .db import init_db
from .downloader import download_screenshots
from .exporter import export_datasets
from .extractor import extract_board_data
from .reference import build_reference_catalog


app = typer.Typer(help="Bazaar endgame board extraction pipeline")


def _bootstrap():
    settings = get_settings()
    ensure_directories(settings)
    conn = init_db(settings)
    return settings, conn


@app.command("crawl-posts")
def crawl_posts_cmd() -> None:
    settings, conn = _bootstrap()
    result = crawl_posts(conn, settings)
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


@app.command("run-all")
def run_all() -> None:
    settings, conn = _bootstrap()
    typer.echo({"crawl_posts": crawl_posts(conn, settings)})
    typer.echo({"build_reference": build_reference_catalog(conn, settings)})
    typer.echo({"download_screenshots": download_screenshots(conn, settings)})
    typer.echo({"extract_board_data": extract_board_data(conn, settings)})
    typer.echo({"export_datasets": export_datasets(conn, settings)})
    typer.echo({"summarize": summarize(conn, settings)})


if __name__ == "__main__":
    app()
