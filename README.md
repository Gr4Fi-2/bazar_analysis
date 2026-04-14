# bazar-analysis

Python pipeline to crawl Jules build posts from `bazaar-builds.net`, build a local reference catalog from `bazaardb.gg`, extract endgame board entities from screenshots, and export analysis-ready datasets.

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
uv run bazar-analysis crawl-posts
uv run bazar-analysis build-reference
uv run bazar-analysis download-screenshots
uv run bazar-analysis extract-board-data
uv run bazar-analysis export-datasets
uv run bazar-analysis summarize
```

## Outputs

- `data/raw/posts_html/`: cached category and post HTML snapshots
- `data/raw/screenshots/`: downloaded build screenshots
- `data/reference/`: BazaarDB catalog snapshots and icon files
- `data/debug/`: board/rank/skill crops and annotated screenshots
- `data/exports/`: normalized parquet/csv datasets and summary tables
- `data/db/bazar_analysis.duckdb`: DuckDB working database

## Reliability Notes

- Post crawling and screenshot discovery are deterministic and generally reliable.
- Reference catalog building is browser-free by default. The normal path uses `curl_cffi` browser impersonation plus BazaarDB sitemap/list pages, caches the fetched HTML locally, and downloads icon files incrementally.
- Set `BAZAR_ALLOW_PLAYWRIGHT_FALLBACK=1` only if you want Playwright as a last resort and have installed the optional `browser` extra. By default the pipeline will not open a browser.
- Board extraction is heuristic: it uses fixed board/icon regions tuned to the Bazaar result screen, icon similarity against the BazaarDB catalog, and post item tags as a constrained prior.
- Rank extraction is also heuristic: it saves the top-left crop, tries template-style matching against simple textual/title cues, and falls back to post-title-derived rank hints with reduced confidence.
- Low-confidence detections are written to `review_queue` with saved crop files and top candidates instead of being silently accepted.

## Current Heuristics

- Small non-board images such as site logos are skipped into the review queue instead of being processed as real boards.
- Board and skill regions are estimated from relative screenshot coordinates tuned to the common Jules endscreen layout.
- Entity recognition uses perceptual hash, color distance, ORB feature matches, and fuzzy name hints.
- Duplicate counts come from repeated slot predictions after normalization.
- Working storage is `duckdb`, including JSON columns that can be queried directly in SQL.
- Exports and summaries use `polars` and keep writing both CSV and Parquet.

## Remaining Gaps

- Rank symbol recognition is not a dedicated trained classifier yet.
- Skill extraction is weaker than item extraction because skill icons are smaller and less consistently placed.
- Some screenshots may need manual review when the board layout differs from the expected UI framing.
