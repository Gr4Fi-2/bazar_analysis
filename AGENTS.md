# AGENTS.md

## Zweck

Dieses Repo ist eine Analyse- und Extraktionspipeline fuer Community-Runs aus The Bazaar, primaer ueber `bazaardb.gg`. Nicht raten: Erst lokale Daten, Exporte, DB-Schema und gecachte HTML-/Screenshot-Artefakte pruefen.

## Aktueller Projektkontext

- Projektname: `bazar-analysis` / Python Package `bazar_analysis`.
- Ziel: BazaarDB-Runs crawlen, Referenzkatalog fuer Items/Skills bauen, Screenshots downloaden, Board-/Skill-/Rank-Daten extrahieren und Analyse-Datasets exportieren.
- Fokus der vorhandenen Daten: aktuelle lokale DuckDB/Exports enthalten alle 7 Helden (`Vanessa`, `Karnok`, `Stelle`, `Dooley`, `Pygmalien`, `Mak`, `Jules`) mit Vanessa als groesster Gruppe. Default-Crawler-Hero im Code ist weiterhin `Jules`; fuer gezielte Crawls `BAZAR_RUN_HEROES` explizit setzen.
- Gameplay-/Mechanik-Wissen: `BAZAAR_KNOWLEDGE.md` ist die verbindliche lokale Wissensbasis fuer Spielbegriffe, Status, Engines, Scaling, Vendors, Events und BazaarDB-Quellen.
- Datenquelle: `https://bazaardb.gg/run` und `https://bazaardb.gg/api/run`.
- Lokale Haupt-DB: `data/db/bazar_analysis.duckdb`.
- Exportierte Analyse-Daten: `data/exports/*.csv` und `data/exports/*.parquet`.
- Referenz-Icons: `data/reference/icons/items/` und `data/reference/icons/skills/`.
- Rohdaten: `data/raw/runs_html/` und `data/raw/screenshots/`.
- Debug-Artefakte: `data/debug/board_regions/`, `data/debug/skill_regions/`, `data/debug/rank_regions/`, `data/debug/crops/`, `data/debug/annotated/`.

## Arbeitsweise

- Effizient arbeiten: `Glob`/`Grep` fuer Suche, parallele Reads nutzen, nicht blind grosse Dateien komplett lesen.
- Bei Analysefragen zuerst `data/exports/` und `data/db/bazar_analysis.duckdb` nutzen, nicht aus Screenshots schliessen, wenn Run-Payloads vorhanden sind.
- Bei Gameplay-, Meta-, Build-, Engine-, Scaling-, Vendor- oder Event-Fragen zuerst `BAZAAR_KNOWLEDGE.md` lesen. Keine Spielmechanikbegriffe selbst definieren; Regeltext belegen oder Unsicherheit markieren.
- Bestehende Daten nicht loeschen. `uv run bazar-analysis reset-data` entfernt Rohdaten, Referenzen, Debug, Exporte und DuckDB; nur ausfuehren, wenn explizit gewuenscht.
- Keine fremden Worktree-Aenderungen anfassen; bei Arbeitsbeginn `git status --short` pruefen. Am 2026-06-30 war `AGENTS.md` untracked und wurde als lokale Projektkontext-Datei genutzt.
- Bei Vision-/OCR-Themen beachten: Board und Skills sind source-first aus BazaarDB-Run-Payloads; Screenshot-Heuristiken sind Fallback und koennen falsch liegen.

## Setup Und Befehle

- Install: `uv sync`.
- Optional Browser-Fallback: `uv sync --extra browser` und `uv run playwright install chromium`.
- CLI Entry Point: `uv run bazar-analysis ...`.
- Full Pipeline ohne Rank-/Crown-Refresh: `uv run bazar-analysis run-all`.
- Rank-/Crown-Refresh bei Bedarf separat ausfuehren: `uv run bazar-analysis extract-rank-crown`.
- Schritte: `reset-data`, `crawl-runs`, `build-reference`, `download-screenshots`, `extract-board-data`, `extract-rank-crown`, `export-datasets`, `summarize`, `systemic-analysis`.
- Browser wird standardmaessig nicht genutzt. Playwright-Fallback nur mit `BAZAR_ALLOW_PLAYWRIGHT_FALLBACK=1`.

## Wichtige Env Vars

- `BAZAR_RUN_HEROES`: Default `Jules`, kommasepariert.
- `BAZAR_RUN_MIN_RANK`: optional; gueltige Tiers werden auf `Bronze`, `Silver`, `Gold`, `Diamond`, `Legendary` normalisiert.
- `BAZAR_RUN_DATE_RANGE`: Default `latest_season`; bekannte Werte: `latest_season`, `season14`, `season13`, `last24h`, `last3d`, `last7d`.
- `BAZAR_RUN_CREATED_AFTER` / `BAZAR_RUN_CREATED_BEFORE`: optionale harte Timestamp-Overrides fuer Discovery-Grenzen.
- `BAZAR_RUN_DISCOVERY_PAGES`: `0`/`all` bedeutet bis API erschoepft ist.
- `BAZAR_RUN_SORT`: Default `newest`; auch `wins`/`top` im Code beruecksichtigt.
- `BAZAR_RUN_ORDER`: Default `desc`.
- `BAZAR_CRAWL_DELAY_SECONDS`: Default `0.35`.
- `BAZAR_DOWNLOAD_DELAY_SECONDS`: Default `0.20`.
- `BAZAR_CURL_IMPERSONATE`: Default `firefox`.
- `BAZAR_REFERENCE_FULL=1`: breiter Sitemap-Backfill fuer Referenzen; sonst nur relevante/run-nahe Enrichment-URLs.
- `BAZAR_REFERENCE_BATCH_SIZE`: Default `25`.
- `BAZAR_REFERENCE_DELAY_MS`: Default `2500`.
- `BAZAR_ALLOW_PLAYWRIGHT_FALLBACK=1`: Playwright nur als letzter Fallback, falls optional installiert.
- `BAZAR_PLAYWRIGHT_HEADLESS=1`: Playwright headless, falls Fallback erlaubt und installiert ist.

## Code-Struktur

- `src/bazar_analysis/cli.py`: Typer CLI und Pipeline-Orchestrierung.
- `src/bazar_analysis/config.py`: Pfade und `reset_workspace_data`.
- `src/bazar_analysis/db.py`: DuckDB Schema, Tabellen, Migration/Reset bei inkompatibler Pipeline-Struktur.
- `src/bazar_analysis/crawler.py`: Run-Discovery ueber BazaarDB API, Detail-HTML-Cache, Hydration-Payload-Parsing, Board-/Skill-Karten aus Run-Payloads.
- `src/bazar_analysis/reference.py`: BazaarDB Referenzkatalog fuer Items/Skills, Icons, Sitemap/List-Pages, optional Playwright.
- `src/bazar_analysis/downloader.py`: Screenshot-Download und Validierung.
- `src/bazar_analysis/extractor.py`: Source-first Extraction, Vision-Fallback, Rank-/Crown-Erkennung, Review Queue.
- `src/bazar_analysis/vision.py`: Crop-Regionen, pHash/Farb/HSV/ORB-Matching, Rank-/Badge-Hilfen, Annotationen.
- `src/bazar_analysis/exporter.py`: Exportiert Kern-Tabellen als CSV und Parquet.
- `src/bazar_analysis/analysis.py`: Summaries, Cooccurrence, Performance, Clustering/Core Builds/Systemic Analysis.
- `src/bazar_analysis/utils.py`: Normalisierung, Victory-Tiers, URL-/JSON-Helfer.
- `BAZAAR_KNOWLEDGE.md`: Gameplay-/Analyse-Wissensbasis und Quellenrangfolge fuer Mechanikfragen.

## DB-Tabellen

- `runs`: Run-Metadaten, Hero, Record, Victory Tier, Rank, Health, Prestige, Level, Income, Gold, `card_hints_json`, `board_cards_json`, `skill_cards_json`.
- `screenshots`: Screenshot-URLs, lokale Pfade, Hash, Dimensionen, Primary-Flag.
- `reference_items` und `reference_skills`: BazaarDB Card IDs, Namen, Slugs, URLs, Icon-Pfade, Aliase, Metadata.
- `extracted_board_items`: Slot-weise Board-Items, Entity, Confidence, Methode, BBox/Crop, Duplicate Count, Status.
- `extracted_skills`: Slot-weise Skills, Entity, Confidence, Methode, BBox/Crop, Status.
- `extracted_ranks`: erkannter Player Rank pro Screenshot.
- `review_queue`: unsichere Crops/Detections fuer manuelle Pruefung.

## Datenlage Aus Aktueller DuckDB/Exports

- `runs.csv`/DuckDB `runs`: 31,282 Runs.
- Hero-Verteilung: `Vanessa` 9,512, `Karnok` 5,817, `Stelle` 4,351, `Dooley` 3,771, `Pygmalien` 3,476, `Mak` 2,247, `Jules` 2,108.
- 10-Win-Baselines lokal: `Stelle` 54.4%, `Jules` 54.3%, `Mak` 48.8%, `Vanessa` 48.6%, `Pygmalien` 46.0%, `Karnok` 43.6%, `Dooley` 41.3%.
- `screenshots.csv`/DuckDB `screenshots`: 31,282 Screenshots.
- `reference_items.csv`: 1,135 Items.
- `reference_skills.csv`: 495 Skills.
- `extracted_board_items.csv`: 193,658 Board-Item-Detections.
- `extracted_skills.csv`: 333,970 Skill-Detections.
- `extracted_ranks.csv`/DuckDB `extracted_ranks`: aktuell 0 Rank-Detections in der aktuellen DB.
- `review_queue.csv`/DuckDB `review_queue`: aktuell 0 offene Review-Eintraege in der aktuellen DB.
- `summary_pipeline_coverage.csv`: pro Run/Screenshot Coverage, Status und Review-Zaehler.
- `summary_top_items.csv`: haeufigste Items im aktuellen Datenstand z.B. Burnacuda, Flying Squirrel, Holsters, Ramrod, Zoarcid, Targeting Drone, Drone Crusher, Dive Weights, Throwing Knives, Hunter's Pack, Parts Picker, Incendiary Rounds.
- `summary_top_skills.csv`: haeufigste Skills im aktuellen Datenstand z.B. Keen Eye, Karnok's Rage, Deadly Eye, Strength, Inspired Rage, Left Eye, Fiery, Left-Handed, Quick Freeze, Right Eye, Static Acceleration, Supply Cache.
- `summary_core_builds.csv` und `summary_core_builds_by_hero.csv`: aggregierte Core Builds mit `core_items_json`, `top_flex_items_json`, `top_skills_json`, Outcome- und Rank-Verteilungen.
- `summary_systemic_archetypes.csv` und `summary_systemic_archetypes_by_hero.csv`: Archetypen ueber Anchor-Paare; fuer Meta-Fragen immer hero-spezifisch filtern oder die `_by_hero`-Exports nutzen.

## Game-/Analyse-Begriffe

- Hero: aktueller Datenstand umfasst `Vanessa`, `Karnok`, `Stelle`, `Dooley`, `Pygmalien`, `Mak` und `Jules`; Crawler-Default bleibt `Jules`.
- Victory Tier wird in `utils.derive_run_victory` abgeleitet: `Perfect` bei `record_wins >= 10` und `prestige >= 20`, `Gold` bei `record_wins >= 10`, `Silver` bei `>= 7`, `Bronze` bei `>= 4`, sonst `Unfortunate`.
- Rank-Tiers: `Bronze`, `Silver`, `Gold`, `Diamond`, `Legendary`.
- Season-Start fuer `latest_season`/`season14` im Crawler: `Wed, 06 May 2026 16:24:57 GMT`; `season13`: `Wed, 01 Apr 2026 16:12:11 GMT` bis Season 14.
- Board-Karten enthalten `slot_position`, `title`, `base_id`, `tier`, `enchantment`, `source`.
- Skills enthalten ebenfalls `slot_position`, `title`, `base_id`, `tier`, `source`.
- Duplicate Counts sind duplicate-safe gedacht: Populations-Summaries sollen wiederholte Kopien auf einem Board nicht als separate Board-Praesenz uebergewichten.

## Extraction-Regeln

- Wenn `board_cards_json`/`skill_cards_json` aus dem Run-Detail vorhanden sind, wird mit Methode `run_detail_board`/`run_detail_skill` und Confidence `1.0` eingefuegt.
- Bildbasierte Item-Erkennung wird nur genutzt, wenn keine exakten Board-Karten vorhanden sind.
- Bildbasierte Skill-Erkennung wird nur genutzt, wenn keine exakten Skill-Karten vorhanden sind.
- Fallback-Board-Region in `vision.default_regions`: relative Box `(0.38, 0.30, 0.95, 0.58)`.
- Fallback-Skill-Region: relative Box `(0.38, 0.58, 0.88, 0.73)`.
- Fallback-Rank-Region: relative Box `(0.0, 0.0, 0.23, 0.20)`.
- Board-Fallback nutzt 6 Slots in einer Reihe; Skill-Fallback nutzt 9 Slots.
- Item-Matching kombiniert pHash, Farb-Distanz, HSV-Histogramm, ORB und Name-Hints.
- Rank-Erkennung ist heuristisch per Badge-Prototype-Classifier und weiterhin schwach; Low-Confidence geht in `review_queue`.
- Crown/Prestige-State nutzt Orange-/Gray-Ratios in festen UI-Regionen; unsichere Faelle gehen in `review_queue` als `prestige_state`.

## Screenshot-Referenz Aus User-Bild

- Das gepostete Bild ist ein The-Bazaar-Endgame/Run-Screenshot mit Bazaar-Markt-UI, lila/pinker Board-Umgebung und Jules-Portrait unten.
- Sichtbare UI-Hinweise: Heldin Jules, Level-Anzeige `7`, Gold rechts unten `31`, Income/Plus-Anzeige `+9`, grosses gruenes Zahlenfeld `1733` und daneben `10`.
- Oben ist ein Shop-/Encounter-Bereich mit zentralem Gegner/Portrait, links und rechts Shop-/Board-Fliesen, Timer/Tag-Anzeige links unten im oberen UI-Bereich wirkt wie `7`.
- Sichtbare Item-/Kartenmotive in der unteren Reihe umfassen u.a. Dessert/Eis, Schneidebrett/Knife-Food-Prep, Honig/gelbe Fluessigkeit, Pfeffermuehle, Erdbeer-Korb und scharfe rote Chili/Scorchpepper-artiges Motiv.
- Sichtbare grosse Karten im mittleren Boardbereich: Spielzimmer-/Toy-Room-artiges Bild, Buffet/Serving-Platter-artiges Bild, Kuehlraum/Freezer-artiges Bild.
- Diese Bildnotizen sind nur UI-Kontext. Fuer konkrete Itemnamen immer zuerst `runs.board_cards_json`, `extracted_board_items`, `reference_items` oder BazaarDB-URLs pruefen.

## Wichtige Analyse-Outputs

- `summary_top_items.csv`: Item-Frequenzen.
- `summary_top_skills.csv`: Skill-Frequenzen.
- `summary_item_item_cooccurrence.csv`: Item-Item Cooccurrence.
- `summary_item_skill_cooccurrence.csv`: Item-Skill Cooccurrence.
- `summary_item_performance.csv` und `summary_skill_performance.csv`: duplicate-safe Performance und konservative Wins/Gold/Perfect-Raten.
- `summary_item_counts_performance.csv`: Item-Frequenzen mit Performance-Spalten in einer Tabelle.
- `summary_exact_item_triplets.csv`: exakte 3-Item-Cores im BazaarDB-Stil.
- `summary_item_shell_affinity.csv` und `summary_skill_shell_affinity.csv`: breit gute vs. shell-locked Entities.
- `summary_build_clusters.csv`, `summary_build_components.csv`, `summary_core_builds.csv` plus jeweilige `_by_hero.csv`: Cluster/Profile/Core-Build-Aggregation.
- `summary_systemic_archetypes.csv`, `summary_systemic_item_pairs.csv`, `summary_systemic_item_signatures.csv` plus jeweilige `_by_hero.csv`: systemische Archetypen und Item-Signaturen.
- `summary_item_source_alignment.csv`: Abgleich zwischen extrahierten Item-Praesenzen und `runs.board_cards_json` aus der Quelle.

## Bekannte Gaps/Risiken

- Rank-Erkennung ist kein trainierter Klassifikator; in der aktuellen DB sind Rank-Detections nicht vorhanden (`extracted_ranks` leer).
- Crown/Prestige-State kann unklare Faelle in `review_queue` erzeugen; die aktuelle DB hat aber keine offenen Review-Eintraege.
- Skill-Extraction aus Bildern ist schlechter als Item-Extraction; bei vorhandenen Payloads ist das egal.
- Screenshot-Layouts variieren: 1920x1080, 2560x1440, 3440x1440, 3840x2160 und andere Groessen kommen vor.
- Einige BazaarDB-Icons/Referenzen koennen Platzhalter oder fehlende Icons haben; `build-reference` repariert fehlende Icons inkrementell.
- Exportierte CSV-Zeilen koennen sehr lang sein wegen JSON-Spalten; besser mit Polars/DuckDB gezielt Spalten lesen statt rohe Zeilen zu interpretieren.

## Pragmatik Fuer Naechste Agents

- Nicht diskutieren, ob Daten existieren: sie existieren in `data/` und `data/exports/`.
- Bei Fragen wie "was ist auf dem Board?" zuerst `runs.csv`/DuckDB nach `run_url`, `screenshot_id`, `board_cards_json` und `extracted_board_items` suchen.
- Bei Fragen zu Meta/Builds zuerst `summary_core_builds.csv`, `summary_systemic_archetypes.csv`, `summary_item_performance.csv` und Cooccurrence-Tabellen pruefen.
- Bei Engine-/Scaling-Fragen erst `BAZAAR_KNOWLEDGE.md`, dann Card-Regeltexte und erst danach Winrate/Cooccurrence nutzen. Enabler, Payoff, Converter und Static Support getrennt ausweisen.
- Viele Summary-Exports existieren global und als `_by_hero.csv`; fuer hero-spezifische Meta/Archetypen `_by_hero` nutzen oder per DuckDB/Polars explizit `runs.hero` filtern.
- Bei Codeaenderungen minimal bleiben und bestehende Pipeline-Philosophie behalten: source-first, heuristics second, review instead of silent guessing.
