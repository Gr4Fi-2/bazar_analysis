# bazar-analysis

 Python pipeline to crawl community runs from `bazaardb.gg/run`, build a local reference catalog from `bazaardb.gg`, extract endgame board entities from screenshots, and export analysis-ready datasets.

## Install

```bash
uv sync
```

Development/test tooling is installed by default via the `dev` dependency group:

```bash
uv run pytest
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
uv run bazar-analysis systemic-analysis
uv run bazar-analysis validate-data
```

Fast source-first refresh without reference or screenshot work:

```bash
uv run bazar-analysis refresh-season-fast --heroes Jules --date-range latest_season
```

This crawls BazaarDB runs, rewrites source-first item/skill extraction, exports base datasets, refreshes summaries/systemic analysis, writes bias-adjusted item tier lists for selected heroes, and emits a data validation report.

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
BAZAR_CRAWL_DELAY_SECONDS=1.50
BAZAR_CRAWL_RESUME_MAX_AGE_HOURS=72
BAZAR_CRAWL_FETCH_DETAIL_HTML=auto
BAZAR_DOWNLOAD_DELAY_SECONDS=0.20
```

`BAZAR_RUN_DISCOVERY_PAGES=0` means "keep paging until the API feed is exhausted".

Requests are paced process-wide, including concurrent workers. Interrupted discovery runs resume from recent cached API pages; the resume window defaults to 72 hours. HTTP 401/403 stops the remaining hero scopes immediately, while 429 and transient server errors use bounded exponential backoff and honor `Retry-After`.

`latest_season` currently maps to `season15` in code. Known explicit values are `season15`, `season14`, `season13`, `last24h`, `last3d`, and `last7d`.

`BAZAR_CRAWL_FETCH_DETAIL_HTML=auto` skips run detail HTML requests when the BazaarDB API payload already contains screenshot, item, and skill data. Set it to `1` to force detail-page fetching or `0` to never fetch detail pages.

Useful alternatives:

```bash
BAZAR_RUN_MIN_RANK=Gold
BAZAR_RUN_DATE_RANGE=last24h
BAZAR_RUN_DATE_RANGE=last7d
BAZAR_RUN_DATE_RANGE=season15
BAZAR_RUN_CREATED_AFTER="Wed, 03 Jun 2026 16:56:45 GMT"
BAZAR_RUN_CREATED_BEFORE="Thu, 09 Jul 2026 00:00:00 GMT"
```

Example:

```bash
$env:BAZAR_RUN_HEROES="Jules"
$env:BAZAR_RUN_MIN_RANK="Gold"
$env:BAZAR_RUN_DATE_RANGE="latest_season"
uv run bazar-analysis crawl-runs
```

## Analysis Helpers

Bias-adjusted item tier list for one hero:

```bash
uv run bazar-analysis item-tier-list --hero Jules
```

Scope analysis exports without changing the crawl filters:

```bash
uv run bazar-analysis run-analysis-fast --analysis-heroes Jules --analysis-date-range last3d --tier-list-heroes Jules
```

Equivalent environment variables are `BAZAR_ANALYSIS_HEROES`, `BAZAR_ANALYSIS_DATE_RANGE`, `BAZAR_ANALYSIS_CREATED_AFTER`, and `BAZAR_ANALYSIS_CREATED_BEFORE`.

`refresh-season-fast` automatically applies its `--heroes` and `--date-range` values to the analysis phase, so `--heroes Jules --date-range last3d` produces Jules/last3d-scoped summary exports.

Quick data health report:

```bash
uv run bazar-analysis validate-data
```

## Outputs

- `data/raw/runs_html/`: cached run listing/detail HTML snapshots
- `data/raw/screenshots/`: downloaded build screenshots
- `data/reference/`: BazaarDB catalog snapshots and icon files
- `data/debug/`: board/rank/skill crops and annotated screenshots
- `data/exports/`: normalized parquet/csv datasets and summary tables
- `data/db/bazar_analysis.duckdb`: DuckDB working database
- `runs.csv` now includes `run_wins_label`, `run_victory_tier`, `run_victory_label`, `player_rank_tier`, `player_rank_label`, `has_broken_crown`, `max_health`, `prestige`, `level`, `income`, `gold`, `board_cards_json`, and `skill_cards_json`
- `summary_exact_item_triplets.csv` provides BazaarDB-style exact 3-item board cores with board count, wins, and gold/perfect/broken-crown rates
- `summary_skill_shell_affinity.csv` and `summary_item_shell_affinity.csv` show where skills and items are broadly good versus shell-locked
- `summary_build_clusters.csv` and `summary_core_builds.csv` include presence, weighted wins, gold-plus/perfect deltas, confidence, and mechanic labels
- `summary_archetype_families.csv` groups similar cluster cores with Jaccard-style family merging, and `summary_archetype_report.csv` is the compact report view
- Hero-specific versions are emitted with `_by_hero` suffixes, including `summary_archetype_families_by_hero.csv` and `summary_archetype_report_by_hero.csv`
- Simple frequency and co-occurrence summaries use duplicate-safe board presence, so repeated copies of the same item on one board do not inflate population-level rates
- `summary_item_performance.csv` and `summary_skill_performance.csv` include duplicate-safe board presence, conservative weighted wins, and gold/perfect rates so low-sample entities do not dominate the top rows
- `summary_core_builds.csv` only aggregates clusters with at least three boards, avoiding one-off full-board cores
- `*_item_tier_list_bias_adjusted.csv` and `.parquet` provide per-hero item tier lists with sample confidence, usage bands, baseline-adjusted wins, gold-plus, and perfect metrics
- `summary_data_validation.csv` and `.parquet` provide table counts, payload coverage, rank/crown gaps, review queue state, and export counts
- Summary outputs are written as both CSV and Parquet

## Reliability Notes

- Run crawling and screenshot discovery come directly from BazaarDB community run pages.
- Run detail HTML hydration is parsed and cached locally, so exact board items and skills come from BazaarDB's own run payload instead of image guesses whenever available.
- Reference catalog building is browser-free by default. The normal path uses `curl_cffi` browser impersonation plus BazaarDB sitemap/list pages, caches the fetched HTML locally, and downloads icon files incrementally.
- Set `BAZAR_ALLOW_PLAYWRIGHT_FALLBACK=1` only if you want Playwright as a last resort and have installed the optional `browser` extra. By default the pipeline will not open a browser.
- Board and skill extraction prefer exact run-detail card lists from BazaarDB and only fall back to image heuristics when the embedded card payload is missing.
- Rank extraction is still heuristic: it saves the top-left crop, tries template-style matching, and currently uses stored victory tiers only as a weak bootstrap hint when no better player-rank label is available.
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
