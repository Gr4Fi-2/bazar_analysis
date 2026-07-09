from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests

from .config import Settings
from .db import next_id
from .utils import (
    absolute_url,
    canonical_image_url,
    derive_run_victory,
    json_dumps,
    normalize_player_rank_tier,
    safe_stem_from_url,
)


RUNS_URL = "https://bazaardb.gg/run"
RUN_PATH_PATTERN = re.compile(r"^/run/([0-9a-f-]+)$", flags=re.IGNORECASE)
ALL_RUN_HEROES = ("Vanessa", "Stelle", "Dooley", "Pygmalien", "Jules", "Mak", "Karnok")
RANK_ORDER = {
    "Bronze": 1,
    "Silver": 2,
    "Gold": 3,
    "Diamond": 4,
    "Legendary": 5,
}
SEASON_START_DATES = {
    "season13": "Wed, 01 Apr 2026 16:12:11 GMT",
    "season14": "Wed, 06 May 2026 16:24:57 GMT",
    "season15": "Wed, 03 Jun 2026 16:56:45 GMT",
}


@dataclass
class RunRecord:
    source_run_id: str
    run_url: str
    api_payload: dict


@dataclass(frozen=True)
class RunDiscoveryResult:
    runs: list[RunRecord]
    exhausted: bool


@dataclass(frozen=True)
class RunFilters:
    heroes: set[str]
    min_rank: str | None
    date_range: str
    pages: int | None
    sort: str
    order: str
    created_after: str | None
    created_before: str | None
    request_delay_seconds: float


def _load_run_filters() -> RunFilters:
    heroes: set[str] = set()
    for hero in os.environ.get("BAZAR_RUN_HEROES", "Jules").split(","):
        token = hero.strip()
        if not token:
            continue
        if token.lower() in {"all", "*"}:
            heroes.update(ALL_RUN_HEROES)
        else:
            heroes.add(token.title())
    min_rank = normalize_player_rank_tier(os.environ.get("BAZAR_RUN_MIN_RANK", "").strip())
    date_range = os.environ.get("BAZAR_RUN_DATE_RANGE", "latest_season").strip().lower()
    pages_raw = os.environ.get("BAZAR_RUN_DISCOVERY_PAGES", "0").strip().lower()
    pages = None if pages_raw in {"", "0", "all"} else max(1, int(pages_raw))
    sort = os.environ.get("BAZAR_RUN_SORT", "newest").strip().lower()
    order = os.environ.get("BAZAR_RUN_ORDER", "desc").strip().lower()
    request_delay_seconds = max(0.0, float(os.environ.get("BAZAR_CRAWL_DELAY_SECONDS", "0.35")))
    created_after, created_before = _created_bounds_for_date_range(date_range)
    override_after = os.environ.get("BAZAR_RUN_CREATED_AFTER", "").strip()
    override_before = os.environ.get("BAZAR_RUN_CREATED_BEFORE", "").strip()
    if override_after:
        created_after = override_after
    if override_before:
        created_before = override_before
    return RunFilters(
        heroes=heroes,
        min_rank=min_rank,
        date_range=date_range,
        pages=pages,
        sort=sort,
        order=order,
        created_after=created_after,
        created_before=created_before,
        request_delay_seconds=request_delay_seconds,
    )


def _created_bounds_for_date_range(date_range: str) -> tuple[str | None, str | None]:
    now = dt.datetime.now(dt.UTC)
    if date_range == "last24h":
        return ((now - dt.timedelta(hours=24)).strftime("%a, %d %b %Y %H:%M:%S GMT"), None)
    if date_range == "last3d":
        return ((now - dt.timedelta(days=3)).strftime("%a, %d %b %Y %H:%M:%S GMT"), None)
    if date_range == "last7d":
        return ((now - dt.timedelta(days=7)).strftime("%a, %d %b %Y %H:%M:%S GMT"), None)
    if date_range == "latest_season":
        return (SEASON_START_DATES["season15"], None)
    if date_range == "season15":
        return (SEASON_START_DATES["season15"], None)
    if date_range == "season14":
        return (SEASON_START_DATES["season14"], SEASON_START_DATES["season15"])
    if date_range == "season13":
        return (SEASON_START_DATES["season13"], SEASON_START_DATES["season14"])
    return (None, None)


def _parse_created_timestamp(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    normalized = value.strip()
    try:
        parsed = dt.datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.UTC)
        return parsed.astimezone(dt.UTC)
    except ValueError:
        pass
    try:
        return dt.datetime.strptime(normalized, "%a, %d %b %Y %H:%M:%S GMT").replace(tzinfo=dt.UTC)
    except ValueError:
        return None


def _rank_meets_minimum(rank_tier: str | None, min_rank: str | None) -> bool:
    if min_rank is None:
        return True
    if rank_tier is None:
        return False
    return RANK_ORDER.get(rank_tier, 0) >= RANK_ORDER[min_rank]


def build_client() -> httpx.Client:
    return httpx.Client(
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        },
        follow_redirects=True,
        timeout=30.0,
    )


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _use_html_cache() -> bool:
    return os.environ.get("BAZAR_CRAWL_USE_HTML_CACHE", "1") != "0"


def _is_unusable_cached_html(html: str) -> bool:
    lowered = html.lower()
    return not html.strip() or ("just a moment" in lowered and "cloudflare" in lowered)


def _collapse_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _parse_int(value: str | None) -> int | None:
    if not value:
        return None
    digits = re.sub(r"[^0-9]", "", value)
    return int(digits) if digits else None


def _sort_value(run_payload: dict, sort: str) -> int | None:
    if sort == "wins":
        wins = int(run_payload.get("statWins") or 0)
        losses = run_payload.get("statLosses")
        return 100 * wins - (int(losses) if losses is not None else 10)
    if sort == "top":
        return int(run_payload.get("upvoteCount") or 0)
    return None


def _curl_get(url: str, *, timeout: int, referer: str | None, delay_seconds: float, params: list[tuple[str, str]] | None = None):
    headers = {"Accept-Language": "en-US,en;q=0.9"}
    if referer:
        headers["Referer"] = referer

    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            response = curl_requests.get(
                url,
                params=params,
                impersonate=os.environ.get("BAZAR_CURL_IMPERSONATE", "firefox"),
                timeout=timeout,
                headers=headers,
            )
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            print(f"[crawl] retry {attempt}/4 failed for {url}: {type(exc).__name__}", flush=True)
            if attempt < 4:
                time.sleep(max(0.75, delay_seconds or 0.0) * attempt)
    raise RuntimeError(f"failed to fetch {url}") from last_error


def fetch_text(client: httpx.Client, url: str, delay_seconds: float) -> str:
    return _curl_get(
        url,
        timeout=60,
        referer="https://bazaardb.gg/run",
        delay_seconds=delay_seconds,
    ).text


def _fetch_run_api_page(filters: RunFilters, cursor_payload: dict | None) -> list[dict]:
    params: list[tuple[str, str]] = [("sort", filters.sort), ("order", filters.order)]
    for hero in sorted(filters.heroes):
        params.append(("heroInclude", hero))
    if filters.created_after:
        params.append(("createdAfter", filters.created_after))
    if filters.created_before:
        params.append(("createdBefore", filters.created_before))
    if cursor_payload is not None:
        params.append(("cursorCreatedAt", cursor_payload["createdAt"]))
        params.append(("cursorId", cursor_payload["id"]))
        cursor_sort_value = _sort_value(cursor_payload, filters.sort)
        if cursor_sort_value is not None:
            params.append(("cursorSortValue", str(cursor_sort_value)))
    response = _curl_get(
        "https://bazaardb.gg/api/run",
        timeout=60,
        referer="https://bazaardb.gg/run",
        delay_seconds=filters.request_delay_seconds,
        params=params,
    )
    return response.json()


def _preceding_backslash_count(text: str, index: int) -> int:
    count = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        count += 1
        cursor -= 1
    return count


def _fragment_uses_escaped_quotes(text: str, start_index: int) -> bool:
    for index in range(start_index, len(text)):
        if text[index] == '"':
            return _preceding_backslash_count(text, index) == 1
    return False


def _is_json_string_boundary(text: str, index: int, escaped_quotes: bool) -> bool:
    if text[index] != '"':
        return False
    backslashes = _preceding_backslash_count(text, index)
    if escaped_quotes:
        return backslashes == 1
    return backslashes % 2 == 0


def _find_json_fragment_end(text: str, start_index: int) -> int | None:
    depth = 0
    in_string = False
    escaped_quotes = _fragment_uses_escaped_quotes(text, start_index)
    for index in range(start_index, len(text)):
        char = text[index]
        if _is_json_string_boundary(text, index, escaped_quotes):
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in "{[":
            depth += 1
            continue
        if char in "}]":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _extract_escaped_json_fragment(text: str, marker: str) -> dict | list | None:
    marker_index = text.find(marker)
    if marker_index == -1:
        return None
    fragment_start = marker_index + len(marker)
    while fragment_start < len(text) and text[fragment_start].isspace():
        fragment_start += 1
    if fragment_start >= len(text) or text[fragment_start] not in "[{":
        return None
    fragment_end = _find_json_fragment_end(text, fragment_start)
    if fragment_end is None:
        return None
    fragment = text[fragment_start:fragment_end]
    try:
        decoded_fragment = json.loads(f'"{fragment}"')
        return json.loads(decoded_fragment)
    except json.JSONDecodeError:
        return None


def _extract_hydrated_run_payload(soup: BeautifulSoup) -> dict | None:
    for script in soup.select("script"):
        script_text = script.string or script.get_text() or ""
        if 'self.__next_f.push' not in script_text or '\\"run\\":' not in script_text:
            continue
        payload = _extract_escaped_json_fragment(script_text, '\\"run\\":')
        if isinstance(payload, dict) and payload.get("id"):
            return payload
    return None


def _normalize_embedded_cards(cards: list[dict] | None, source: str) -> list[dict]:
    normalized_cards: list[dict] = []
    for index, card in enumerate(cards or []):
        if not isinstance(card, dict):
            continue
        title = _collapse_whitespace(str(card.get("title") or card.get("name") or "")) or None
        slot_position = card.get("slotPosition")
        try:
            slot_position = int(slot_position) if slot_position is not None else index
        except (TypeError, ValueError):
            slot_position = index
        enchantment = card.get("enchantmentOverride")
        if enchantment == "$undefined":
            enchantment = None
        normalized_cards.append(
            {
                "slot_position": slot_position,
                "title": title,
                "base_id": str(card.get("baseId") or card.get("cardId") or "").strip() or None,
                "tier": card.get("tierOverride"),
                "enchantment": enchantment,
                "source": source,
            }
        )
    return normalized_cards


def _extract_player_rank_tier(run_payload: dict | None) -> str | None:
    if not isinstance(run_payload, dict):
        return None
    for key in ["playerRank", "playerRankTier", "profileRank", "profileRankTier", "rankTier", "rank"]:
        parsed = normalize_player_rank_tier(str(run_payload.get(key) or ""))
        if parsed:
            return parsed
    for container_key in ["profile", "player", "user"]:
        container = run_payload.get(container_key)
        if not isinstance(container, dict):
            continue
        for key in ["rank", "rankTier", "profileRank", "profileRankTier"]:
            parsed = normalize_player_rank_tier(str(container.get(key) or ""))
            if parsed:
                return parsed
    return None


def _snapshot_scope_label(filters: RunFilters) -> str:
    heroes = "-".join(sorted(hero.lower().replace(" ", "-") for hero in filters.heroes)) or "all"
    label = f"{filters.date_range}_{heroes}_{filters.sort}_{filters.order}"
    return re.sub(r"[^a-z0-9_.-]+", "-", label.lower()).strip("-") or "runs"


def discover_runs(client: httpx.Client, settings: Settings, filters: RunFilters) -> RunDiscoveryResult:
    runs: list[RunRecord] = []
    seen_ids: set[str] = set()
    cursor_payload = None
    exhausted = True
    created_after_cutoff = _parse_created_timestamp(filters.created_after)
    created_before_cutoff = _parse_created_timestamp(filters.created_before)

    if filters.date_range in {"latest_season", "season15"}:
        print("[crawl] using BazaarDB API feed for latest season", flush=True)
    elif filters.date_range == "season14":
        print("[crawl] using BazaarDB API feed for Season 14", flush=True)
    elif filters.date_range == "season13":
        print("[crawl] using BazaarDB API feed for Season 13", flush=True)

    page_number = 1
    while True:
        if filters.pages is not None and page_number > filters.pages:
            exhausted = False
            break
        print(f"[crawl] api page {page_number}: sort={filters.sort} order={filters.order}", flush=True)
        try:
            page_payload = _fetch_run_api_page(filters, cursor_payload)
        except Exception as exc:
            exhausted = False
            print(f"[crawl] stopping discovery after page {page_number - 1}: {type(exc).__name__}", flush=True)
            break
        save_text(
            settings.raw_runs_dir / f"run_api_{_snapshot_scope_label(filters)}_page_{page_number}.json",
            json.dumps(page_payload, ensure_ascii=True, sort_keys=True),
        )
        page_runs = 0
        reached_created_after_cutoff = False
        for payload in page_payload:
            payload_created_at = _parse_created_timestamp(str(payload.get("createdAt") or ""))
            if (
                created_before_cutoff is not None
                and payload_created_at is not None
                and filters.sort == "newest"
                and filters.order == "desc"
                and payload_created_at > created_before_cutoff
            ):
                continue
            if (
                created_after_cutoff is not None
                and payload_created_at is not None
                and filters.sort == "newest"
                and filters.order == "desc"
                and payload_created_at <= created_after_cutoff
            ):
                reached_created_after_cutoff = True
                break
            source_run_id = str(payload["id"])
            if source_run_id in seen_ids:
                continue
            seen_ids.add(source_run_id)
            runs.append(
                RunRecord(
                    source_run_id=source_run_id,
                    run_url=f"{RUNS_URL}/{source_run_id}",
                    api_payload=payload,
                )
            )
            page_runs += 1
        print(f"[crawl] api page {page_number}: discovered {page_runs} runs, total {len(runs)}", flush=True)
        if reached_created_after_cutoff:
            print("[crawl] reached created-after cutoff; stopping discovery", flush=True)
            break
        if len(page_payload) < 20:
            break
        cursor_payload = page_payload[-1]
        page_number += 1

    return RunDiscoveryResult(runs=runs, exhausted=exhausted)


def parse_run(client: httpx.Client, settings: Settings, run: RunRecord, delay_seconds: float) -> dict:
    run_slug = safe_stem_from_url(run.run_url)
    html_path = settings.raw_runs_dir / f"run_{run_slug}.html"
    html = ""
    if _use_html_cache() and html_path.exists():
        html = html_path.read_text(encoding="utf-8")
    if _is_unusable_cached_html(html):
        html = fetch_text(client, run.run_url, delay_seconds)
        save_text(html_path, html)

    soup = BeautifulSoup(html, "html.parser")
    page_text = _collapse_whitespace(soup.get_text(" ", strip=True))
    payload = run.api_payload
    hydrated_run = _extract_hydrated_run_payload(soup) or {}

    profile_name = payload.get("username")
    profile_id = payload.get("profileId")
    profile_url = f"{RUNS_URL}/profile/{profile_id}" if profile_id else None
    hero = str(payload.get("hero") or "Unknown")
    screenshot_node = soup.select_one("img[alt$='run screenshot']") or soup.select_one("img[src*='/screenshots/']") or soup.select_one("img[src*='/cr/']")
    screenshot_url = None
    if screenshot_node and screenshot_node.get("src"):
        screenshot_url = canonical_image_url(absolute_url(run.run_url, screenshot_node["src"]))
    elif payload.get("screenshotUrl"):
        screenshot_url = canonical_image_url(absolute_url(run.run_url, str(payload.get("screenshotUrl") or "")))

    outcome_match = re.search(r"Record\s*([0-9]+)\s*[-–]\s*([0-9?]+)\s+(.+?)\s+Max Health\s*([0-9,]+)", page_text)
    record_wins = int(payload.get("statWins")) if payload.get("statWins") is not None else (int(outcome_match.group(1)) if outcome_match else None)
    losses_raw = payload.get("statLosses")
    losses_token = outcome_match.group(2) if outcome_match else None
    record_losses = int(losses_raw) if losses_raw is not None else (int(losses_token) if losses_token and losses_token.isdigit() else None)
    outcome_text = outcome_match.group(3).strip() if outcome_match else None
    max_health = int(payload.get("statMaxHealth")) if payload.get("statMaxHealth") is not None else (_parse_int(outcome_match.group(4)) if outcome_match else None)
    prestige = int(payload.get("statPrestige")) if payload.get("statPrestige") is not None else None
    level = int(payload.get("statLevel")) if payload.get("statLevel") is not None else None
    income = int(payload.get("statIncome")) if payload.get("statIncome") is not None else None
    gold = int(payload.get("statGold")) if payload.get("statGold") is not None else None
    run_victory_tier, run_victory_label = derive_run_victory(record_wins, prestige)
    player_rank_tier = _extract_player_rank_tier(hydrated_run)
    player_rank_label = player_rank_tier
    has_broken_crown = None

    board_cards = _normalize_embedded_cards(hydrated_run.get("board") or payload.get("items"), "run_page_board")
    skill_cards = _normalize_embedded_cards(hydrated_run.get("skills") or payload.get("skills"), "run_page_skill")

    card_hints: list[str] = []
    for card in [*board_cards, *skill_cards]:
        title = card.get("title")
        if title and title not in card_hints:
            card_hints.append(title)
    if not card_hints:
        for anchor in soup.select("a[href*='/card/']"):
            href = anchor.get("href") or ""
            if "/card/" not in href:
                continue
            name = Path(urlparse(absolute_url(run.run_url, href)).path).name.replace("-", " ").strip()
            if name and name not in card_hints:
                card_hints.append(name)

    wins_label = f"{record_wins} Wins" if record_wins is not None else "Unknown Record"
    title = _collapse_whitespace(" ".join(part for part in [hero, wins_label, run_victory_label or outcome_text] if part))

    return {
        "source_run_id": run.source_run_id,
        "run_url": run.run_url,
        "created_at": payload.get("createdAt"),
        "hero": hero,
        "title": title,
        "profile_name": profile_name,
        "profile_url": profile_url,
        "outcome_text": outcome_text,
        "record_wins": record_wins,
        "record_losses": record_losses,
        "run_wins_label": wins_label,
        "run_victory_tier": run_victory_tier,
        "run_victory_label": run_victory_label,
        "player_rank_tier": player_rank_tier,
        "player_rank_label": player_rank_label,
        "has_broken_crown": has_broken_crown,
        "max_health": max_health,
        "prestige": prestige,
        "level": level,
        "income": income,
        "gold": gold,
        "html_path": str(html_path),
        "card_hints": card_hints,
        "board_cards": board_cards,
        "skill_cards": skill_cards,
        "screenshot_urls": [screenshot_url] if screenshot_url else [],
    }


def _delete_screenshots(conn, screenshot_ids: list[int]) -> int:
    if not screenshot_ids:
        return 0
    placeholders = ", ".join("?" for _ in screenshot_ids)
    parameters = tuple(screenshot_ids)
    for table in ["extracted_board_items", "extracted_skills", "extracted_ranks", "review_queue"]:
        conn.execute(f"DELETE FROM {table} WHERE screenshot_id IN ({placeholders})", parameters)
    conn.execute(f"DELETE FROM screenshots WHERE screenshot_id IN ({placeholders})", parameters)
    return len(screenshot_ids)


def _delete_stale_runs(conn, active_run_ids: set[int], filters: RunFilters) -> tuple[int, int]:
    conditions: list[str] = []
    parameters: list[object] = []
    if filters.heroes:
        hero_placeholders = ", ".join("?" for _ in filters.heroes)
        conditions.append(f"hero IN ({hero_placeholders})")
        parameters.extend(sorted(filters.heroes))
    if active_run_ids:
        run_placeholders = ", ".join("?" for _ in active_run_ids)
        conditions.append(f"run_id NOT IN ({run_placeholders})")
        parameters.extend(sorted(active_run_ids))

    where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    stale_run_rows = conn.execute(f"SELECT run_id FROM runs{where_clause}", tuple(parameters)).fetchall()

    stale_run_ids = [row["run_id"] for row in stale_run_rows]
    if not stale_run_ids:
        return 0, 0

    screenshot_placeholders = ", ".join("?" for _ in stale_run_ids)
    stale_screenshot_rows = conn.execute(
        f"SELECT screenshot_id FROM screenshots WHERE run_id IN ({screenshot_placeholders})",
        tuple(stale_run_ids),
    ).fetchall()
    stale_screenshot_ids = [row["screenshot_id"] for row in stale_screenshot_rows]
    removed_screenshots = _delete_screenshots(conn, stale_screenshot_ids)

    conn.execute(f"DELETE FROM runs WHERE run_id IN ({screenshot_placeholders})", tuple(stale_run_ids))
    return len(stale_run_ids), removed_screenshots


def _is_full_discovery_scope(filters: RunFilters) -> bool:
    default_after, default_before = _created_bounds_for_date_range(filters.date_range)
    return (
        filters.date_range in {"latest_season", "season15", "season14", "season13"}
        and filters.pages is None
        and filters.min_rank is None
        and filters.created_after == default_after
        and filters.created_before == default_before
    )


def _crawl_workers() -> int:
    raw_value = os.environ.get("BAZAR_CRAWL_WORKERS", "1").strip()
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 1


def _discovery_filter_scopes(filters: RunFilters) -> list[RunFilters]:
    if len(filters.heroes) <= 1:
        return [filters]
    return [replace(filters, heroes={hero}) for hero in sorted(filters.heroes)]


def _parse_discovered_runs(client: httpx.Client, settings: Settings, runs: list[RunRecord], filters: RunFilters):
    workers = _crawl_workers()
    if workers <= 1 or len(runs) <= 1:
        for index, run in enumerate(runs, start=1):
            if index == 1 or index % 10 == 0 or index == len(runs):
                print(f"[crawl] run {index}/{len(runs)}: {run.run_url}", flush=True)
            try:
                yield run, parse_run(client, settings, run, filters.request_delay_seconds), None
            except Exception as exc:
                yield run, None, exc
        return

    print(f"[crawl] parsing run details with {workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(parse_run, client, settings, run, filters.request_delay_seconds): run for run in runs}
        completed = 0
        for future in as_completed(future_map):
            completed += 1
            run = future_map[future]
            if completed == 1 or completed % 10 == 0 or completed == len(runs):
                print(f"[crawl] run {completed}/{len(runs)} parsed: {run.run_url}", flush=True)
            try:
                yield run, future.result(), None
            except Exception as exc:
                yield run, None, exc


def _upsert_run_record(conn, run: RunRecord, record: dict, now: str) -> tuple[int, int]:
    existing_run = conn.execute("SELECT run_id FROM runs WHERE source_run_id = ?", (run.source_run_id,)).fetchone()
    if existing_run:
        run_id = existing_run["run_id"]
        conn.execute(
            """
            UPDATE runs
            SET hero = ?, run_url = ?, created_at = ?, title = ?, profile_name = ?, profile_url = ?, outcome_text = ?, record_wins = ?, record_losses = ?,
                run_wins_label = ?, run_victory_tier = ?, run_victory_label = ?, player_rank_tier = COALESCE(?, player_rank_tier), player_rank_label = COALESCE(?, player_rank_label), has_broken_crown = ?, max_health = ?, prestige = ?, level = ?, income = ?, gold = ?, html_path = ?,
                card_hints_json = ?, board_cards_json = ?, skill_cards_json = ?, crawled_at = ?
            WHERE run_id = ?
            """,
            (
                record["hero"],
                record["run_url"],
                record["created_at"],
                record["title"],
                record["profile_name"],
                record["profile_url"],
                record["outcome_text"],
                record["record_wins"],
                record["record_losses"],
                record["run_wins_label"],
                record["run_victory_tier"],
                record["run_victory_label"],
                record["player_rank_tier"],
                record["player_rank_label"],
                record["has_broken_crown"],
                record["max_health"],
                record["prestige"],
                record["level"],
                record["income"],
                record["gold"],
                record["html_path"],
                json_dumps(record["card_hints"]),
                json_dumps(record["board_cards"]),
                json_dumps(record["skill_cards"]),
                now,
                run_id,
            ),
        )
    else:
        run_id = next_id(conn, "runs", "run_id")
        conn.execute(
            """
            INSERT INTO runs(run_id, source_run_id, hero, run_url, created_at, title, profile_name, profile_url, outcome_text, record_wins, record_losses,
                             run_wins_label, run_victory_tier, run_victory_label, player_rank_tier, player_rank_label, has_broken_crown, max_health, prestige, level,
                             income, gold, html_path, card_hints_json, board_cards_json, skill_cards_json, crawled_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                record["source_run_id"],
                record["hero"],
                record["run_url"],
                record["created_at"],
                record["title"],
                record["profile_name"],
                record["profile_url"],
                record["outcome_text"],
                record["record_wins"],
                record["record_losses"],
                record["run_wins_label"],
                record["run_victory_tier"],
                record["run_victory_label"],
                record["player_rank_tier"],
                record["player_rank_label"],
                record["has_broken_crown"],
                record["max_health"],
                record["prestige"],
                record["level"],
                record["income"],
                record["gold"],
                record["html_path"],
                json_dumps(record["card_hints"]),
                json_dumps(record["board_cards"]),
                json_dumps(record["skill_cards"]),
                now,
            ),
        )

    existing_screenshots = conn.execute(
        "SELECT screenshot_id, screenshot_url FROM screenshots WHERE run_id = ?",
        (run_id,),
    ).fetchall()
    existing_screenshot_map = {row["screenshot_url"]: row["screenshot_id"] for row in existing_screenshots}
    stale_screenshot_ids = [
        screenshot_id
        for screenshot_url, screenshot_id in existing_screenshot_map.items()
        if screenshot_url not in record["screenshot_urls"]
    ]
    _delete_screenshots(conn, stale_screenshot_ids)
    inserted_screenshots = 0
    for idx, screenshot_url in enumerate(record["screenshot_urls"]):
        existing_screenshot_id = existing_screenshot_map.get(screenshot_url)
        if existing_screenshot_id is not None:
            conn.execute(
                "UPDATE screenshots SET is_primary = ? WHERE screenshot_id = ?",
                (1 if idx == 0 else 0, existing_screenshot_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO screenshots(screenshot_id, run_id, screenshot_url, is_primary)
                VALUES(?, ?, ?, ?)
                """,
                (next_id(conn, "screenshots", "screenshot_id"), run_id, screenshot_url, 1 if idx == 0 else 0),
            )
            inserted_screenshots += 1
    return run_id, inserted_screenshots


def crawl_runs(conn, settings: Settings) -> dict[str, int]:
    now = dt.datetime.utcnow().isoformat(timespec="seconds")
    filters = _load_run_filters()
    inserted_runs = 0
    inserted_screenshots = 0
    skipped_filters = 0
    run_failures = 0
    active_run_ids: set[int] = set()
    discovery_exhausted = True
    filter_scopes = _discovery_filter_scopes(filters)

    with build_client() as client:
        for scope_index, scope_filters in enumerate(filter_scopes, start=1):
            if len(filter_scopes) > 1:
                print(
                    f"[crawl] hero scope {scope_index}/{len(filter_scopes)}: {', '.join(sorted(scope_filters.heroes))}",
                    flush=True,
                )
            discovery = discover_runs(client, settings, scope_filters)
            discovery_exhausted = discovery_exhausted and discovery.exhausted
            runs = discovery.runs
            print(f"[crawl] processing {len(runs)} runs", flush=True)
            for run, record, error in _parse_discovered_runs(client, settings, runs, scope_filters):
                existing_run = conn.execute("SELECT run_id FROM runs WHERE source_run_id = ?", (run.source_run_id,)).fetchone()
                if error is not None or record is None:
                    exc = error or RuntimeError("unknown parse failure")
                    print(f"[crawl] skipping run after fetch/parse failure: {run.run_url} ({type(exc).__name__})", flush=True)
                    run_failures += 1
                    if existing_run:
                        active_run_ids.add(existing_run["run_id"])
                    continue
                if scope_filters.heroes and record["hero"] not in scope_filters.heroes:
                    skipped_filters += 1
                    continue
                if not _rank_meets_minimum(record["player_rank_tier"], scope_filters.min_rank):
                    skipped_filters += 1
                    continue
                run_id, new_screenshots = _upsert_run_record(conn, run, record, now)
                inserted_runs += 1
                inserted_screenshots += new_screenshots
                active_run_ids.add(run_id)

    if discovery_exhausted and run_failures == 0 and skipped_filters == 0 and all(_is_full_discovery_scope(scope) for scope in filter_scopes):
        removed_runs, removed_screenshots = _delete_stale_runs(conn, active_run_ids, filters)
    else:
        removed_runs = 0
        removed_screenshots = 0
        reasons = []
        if not discovery_exhausted:
            reasons.append("discovery_incomplete")
        if run_failures:
            reasons.append(f"run_failures={run_failures}")
        if skipped_filters:
            reasons.append(f"skipped_filters={skipped_filters}")
        if not all(_is_full_discovery_scope(scope) for scope in filter_scopes):
            reasons.append("incremental_or_filtered_scope")
        print(f"[crawl] skipped stale-run prune ({', '.join(reasons)})", flush=True)
    conn.commit()
    print(
        f"[crawl] done: runs={inserted_runs}, screenshots={inserted_screenshots}, skipped_filters={skipped_filters}, run_failures={run_failures}, removed_runs={removed_runs}, removed_screenshots={removed_screenshots}",
        flush=True,
    )
    return {
        "runs": inserted_runs,
        "screenshots": inserted_screenshots,
        "skipped_filters": skipped_filters,
        "run_failures": run_failures,
        "discovery_exhausted": 1 if discovery_exhausted else 0,
        "removed_runs": removed_runs,
        "removed_screenshots": removed_screenshots,
    }
