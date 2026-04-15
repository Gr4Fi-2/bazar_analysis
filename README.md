# bazar-analysis

 Python pipeline to crawl community runs from `bazaardb.gg/run`, build a local reference catalog from `bazaardb.gg`, extract endgame board entities from screenshots, and export analysis-ready datasets.

## Install

```bash
uv sync
```

Optional only if you explicitly want the browser fallback for BazaarDB:

```bash
uv sync --extra browser
uv run playwright install chromium
```

## Full Pipeline

```bash
uv run bazar-analysis run-all
```

Or run step-by-step:

```bash
uv run bazar-analysis reset-data
uv run bazar-analysis crawl-runs
uv run bazar-analysis build-reference
uv run bazar-analysis download-screenshots
uv run bazar-analysis extract-board-data
uv run bazar-analysis export-datasets
uv run bazar-analysis summarize
```

## Run Filters

The run crawler now uses BazaarDB's `/api/run` feed directly and supports environment-driven filters.

Common variables:

```bash
BAZAR_RUN_HEROES=Jules
BAZAR_RUN_MIN_RANK=Diamond
BAZAR_RUN_DATE_RANGE=latest_season
BAZAR_RUN_DISCOVERY_PAGES=0
BAZAR_RUN_SORT=newest
BAZAR_RUN_ORDER=desc
BAZAR_CRAWL_DELAY_SECONDS=0.35
BAZAR_DOWNLOAD_DELAY_SECONDS=0.20
```

`BAZAR_RUN_DISCOVERY_PAGES=0` means "keep paging until the API feed is exhausted".

Useful alternatives:

```bash
BAZAR_RUN_MIN_RANK=Gold
BAZAR_RUN_DATE_RANGE=last24h
BAZAR_RUN_DATE_RANGE=last7d
BAZAR_RUN_CREATED_AFTER="Wed, 01 Apr 2026 16:12:11 GMT"
BAZAR_RUN_CREATED_BEFORE="Thu, 01 May 2026 00:00:00 GMT"
```

Example:

```bash
$env:BAZAR_RUN_HEROES="Jules"
$env:BAZAR_RUN_MIN_RANK="Gold"
$env:BAZAR_RUN_DATE_RANGE="latest_season"
uv run bazar-analysis crawl-runs
```

## Outputs

- `data/raw/runs_html/`: cached run listing/detail HTML snapshots
- `data/raw/screenshots/`: downloaded build screenshots
- `data/reference/`: BazaarDB catalog snapshots and icon files
- `data/debug/`: board/rank/skill crops and annotated screenshots
- `data/exports/`: normalized parquet/csv datasets and summary tables
- `data/db/bazar_analysis.duckdb`: DuckDB working database
- `runs.csv` now includes `run_wins_label`, `run_outcome_tier`, `player_rank_tier`, `board_cards_json`, and `skill_cards_json`

## Reliability Notes

- Run crawling and screenshot discovery come directly from BazaarDB community run pages.
- Run detail HTML hydration is parsed and cached locally, so exact board items and skills come from BazaarDB's own run payload instead of image guesses whenever available.
- Reference catalog building is browser-free by default. The normal path uses `curl_cffi` browser impersonation plus BazaarDB sitemap/list pages, caches the fetched HTML locally, and downloads icon files incrementally.
- Set `BAZAR_ALLOW_PLAYWRIGHT_FALLBACK=1` only if you want Playwright as a last resort and have installed the optional `browser` extra. By default the pipeline will not open a browser.
- Board and skill extraction prefer exact run-detail card lists from BazaarDB and only fall back to image heuristics when the embedded card payload is missing.
- Rank extraction is still heuristic: it saves the top-left crop, tries template-style matching, and currently uses stored run tiers only as a bootstrap hint when no better player-rank label is available.
- Low-confidence detections are written to `review_queue` with saved crop files and top candidates instead of being silently accepted.

## Current Heuristics

- Small non-board images such as site logos are skipped into the review queue instead of being processed as real boards.
- Board and skill regions are only estimated from screenshot coordinates when BazaarDB's embedded run payload does not expose the cards directly.
- Image fallback recognition uses perceptual hash, color distance, ORB feature matches, and fuzzy name hints.
- Duplicate counts come from repeated slot predictions after normalization.
- Working storage is `duckdb`, including JSON columns that can be queried directly in SQL.
- Exports and summaries use `polars` and keep writing both CSV and Parquet.

## Remaining Gaps

- Rank symbol recognition is not a dedicated trained classifier yet.
- Skill extraction is weaker than item extraction because skill icons are smaller and less consistently placed.
- Some screenshots may need manual review when the board layout differs from the expected UI framing.
