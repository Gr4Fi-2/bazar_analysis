from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    raw_dir: Path
    raw_posts_dir: Path
    raw_screenshots_dir: Path
    reference_dir: Path
    reference_html_dir: Path
    reference_icons_items_dir: Path
    reference_icons_skills_dir: Path
    reference_browser_profile_dir: Path
    debug_dir: Path
    debug_board_dir: Path
    debug_rank_dir: Path
    debug_skill_dir: Path
    debug_crops_dir: Path
    debug_annotated_dir: Path
    exports_dir: Path
    db_dir: Path
    duckdb_path: Path
    sqlite_legacy_path: Path


def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    return Settings(
        project_root=project_root,
        data_dir=data_dir,
        raw_dir=data_dir / "raw",
        raw_posts_dir=data_dir / "raw" / "posts_html",
        raw_screenshots_dir=data_dir / "raw" / "screenshots",
        reference_dir=data_dir / "reference",
        reference_html_dir=data_dir / "reference" / "html",
        reference_icons_items_dir=data_dir / "reference" / "icons" / "items",
        reference_icons_skills_dir=data_dir / "reference" / "icons" / "skills",
        reference_browser_profile_dir=data_dir / "reference" / "playwright_profile",
        debug_dir=data_dir / "debug",
        debug_board_dir=data_dir / "debug" / "board_regions",
        debug_rank_dir=data_dir / "debug" / "rank_regions",
        debug_skill_dir=data_dir / "debug" / "skill_regions",
        debug_crops_dir=data_dir / "debug" / "crops",
        debug_annotated_dir=data_dir / "debug" / "annotated",
        exports_dir=data_dir / "exports",
        db_dir=data_dir / "db",
        duckdb_path=data_dir / "db" / "bazar_analysis.duckdb",
        sqlite_legacy_path=data_dir / "db" / "bazar_analysis.sqlite",
    )


def ensure_directories(settings: Settings) -> None:
    for path in [
        settings.data_dir,
        settings.raw_dir,
        settings.raw_posts_dir,
        settings.raw_screenshots_dir,
        settings.reference_dir,
        settings.reference_html_dir,
        settings.reference_icons_items_dir,
        settings.reference_icons_skills_dir,
        settings.reference_browser_profile_dir,
        settings.debug_dir,
        settings.debug_board_dir,
        settings.debug_rank_dir,
        settings.debug_skill_dir,
        settings.debug_crops_dir,
        settings.debug_annotated_dir,
        settings.exports_dir,
        settings.db_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
