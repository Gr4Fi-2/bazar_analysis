from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests

from .config import Settings
from .utils import json_dumps, normalize_name, slugify

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


@dataclass
class ReferenceCard:
    entity_id: str
    name: str
    page_url: str
    image_url: str | None
    aliases: list[str]
    metadata: dict


def _extract_card_id(page_url: str) -> str:
    parts = [part for part in urlparse(page_url).path.split("/") if part]
    if len(parts) >= 2:
        return parts[1]
    return slugify(page_url)


def _candidate_name_from_url(card_url: str) -> str:
    return Path(urlparse(card_url).path).name.replace("-", " ")


def _open_browser_page(playwright):
    headless = os.environ.get("BAZAR_PLAYWRIGHT_HEADLESS", "0") == "1"
    browser = playwright.chromium.launch(headless=headless)
    page = browser.new_page(viewport={"width": 1600, "height": 2200})
    return browser, page


def _open_persistent_page(playwright, settings: Settings):
    headless = os.environ.get("BAZAR_PLAYWRIGHT_HEADLESS", "0") == "1"
    context = playwright.chromium.launch_persistent_context(
        user_data_dir=str(settings.reference_browser_profile_dir),
        headless=headless,
        viewport={"width": 1600, "height": 2200},
    )
    page = context.pages[0] if context.pages else context.new_page()
    return context, page


def _playwright_available() -> bool:
    return sync_playwright is not None


def _dismiss_consent(page) -> None:
    try:
        accept_button = page.get_by_role("button", name="Accept")
        if accept_button.count() > 0 and accept_button.first.is_visible():
            accept_button.first.click(timeout=5000)
            page.wait_for_timeout(1500)
    except Exception:
        pass


def _is_cloudflare_challenge(html: str) -> bool:
    lowered = html.lower()
    return "just a moment" in lowered and "cloudflare" in lowered


def _fetch_bazaardb_text(url: str) -> str:
    response = curl_requests.get(
        url,
        impersonate=os.environ.get("BAZAR_CURL_IMPERSONATE", "firefox"),
        timeout=60,
        headers={"Accept-Language": "en-US,en;q=0.9"},
    )
    response.raise_for_status()
    return response.text


def _allow_playwright_fallback() -> bool:
    return os.environ.get("BAZAR_ALLOW_PLAYWRIGHT_FALLBACK", "0") == "1"


def _load_all_cards(search_url: str, snapshot_path: Path) -> str:
    if not _playwright_available():
        raise RuntimeError("Playwright is not installed")
    headless = os.environ.get("BAZAR_PLAYWRIGHT_HEADLESS", "0") == "1"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        page = browser.new_page(viewport={"width": 1600, "height": 2200})
        page.goto(search_url, wait_until="domcontentloaded", timeout=120000)
        page.wait_for_timeout(5000)
        _dismiss_consent(page)

        for _ in range(120):
            buttons = page.get_by_text("Load more", exact=False)
            if buttons.count() == 0:
                break
            button = buttons.first
            if not button.is_visible():
                break
            button.scroll_into_view_if_needed(timeout=5000)
            button.click(timeout=10000)
            page.wait_for_timeout(1000)

        content = page.content()
        snapshot_path.write_text(content, encoding="utf-8")
        browser.close()
        return content


def _maybe_refresh_snapshot(search_url: str, snapshot_path: Path) -> str | None:
    if snapshot_path.exists():
        cached = snapshot_path.read_text(encoding="utf-8")
        if not _is_cloudflare_challenge(cached):
            return cached
    try:
        html = _fetch_bazaardb_text(search_url)
        snapshot_path.write_text(html, encoding="utf-8")
        return html
    except Exception:
        if _allow_playwright_fallback():
            try:
                return _load_all_cards(search_url, snapshot_path)
            except Exception:
                pass
        if snapshot_path.exists():
            cached = snapshot_path.read_text(encoding="utf-8")
            if not _is_cloudflare_challenge(cached):
                return cached
        return None


def _parse_catalog_html(html: str, entity_type: str) -> list[ReferenceCard]:
    soup = BeautifulSoup(html, "html.parser")
    cards: dict[str, ReferenceCard] = {}
    for heading_link in soup.select("a[href*='/card/']"):
        href = heading_link.get("href")
        if not href:
            continue
        page_url = urljoin("https://bazaardb.gg/", href)
        entity_id = _extract_card_id(page_url)
        name = heading_link.get_text(" ", strip=True)
        if not name:
            parent = heading_link.parent
            if parent:
                name = max((value.strip() for value in parent.stripped_strings if value.strip()), key=len, default="")
        name = name or _candidate_name_from_url(page_url).title()
        if len(name) > 120:
            continue
        container = heading_link
        for _ in range(5):
            if container is None:
                break
            if name and name in container.get_text(" ", strip=True):
                break
            container = container.parent
        image_url = None
        if container:
            image = container.find("img", src=re.compile(r"s\.bazaardb\.gg")) or container.find_previous("img", src=re.compile(r"s\.bazaardb\.gg"))
            if image and image.get("src"):
                image_url = urljoin("https://bazaardb.gg/", image["src"])
        aliases = [name, name.replace("-", " "), slugify(name).replace("-", " ")]
        metadata = {"entity_type": entity_type}
        existing = cards.get(entity_id)
        if existing is None or (image_url and not existing.image_url) or len(name) > len(existing.name):
            cards[entity_id] = ReferenceCard(
                entity_id=entity_id,
                name=name,
                page_url=page_url,
                image_url=image_url,
                aliases=sorted(set(alias for alias in aliases if alias)),
                metadata=metadata,
            )
    return list(cards.values())


def _upsert_reference_card(conn, card: ReferenceCard, image_path: str | None, collected_at: str) -> None:
    entity_type = card.metadata["entity_type"]
    table = f"reference_{entity_type}"
    conn.execute(
        f"""
        INSERT INTO {table}(entity_id, name, normalized_name, slug, page_url, image_url, image_path, aliases_json, metadata_json, collected_at)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(entity_id) DO UPDATE SET
            name=excluded.name,
            normalized_name=excluded.normalized_name,
            slug=excluded.slug,
            page_url=excluded.page_url,
            image_url=COALESCE(excluded.image_url, {table}.image_url),
            image_path=COALESCE(excluded.image_path, {table}.image_path),
            aliases_json=excluded.aliases_json,
            metadata_json=excluded.metadata_json,
            collected_at=excluded.collected_at
        """,
        (
            card.entity_id,
            card.name,
            normalize_name(card.name),
            slugify(card.name),
            card.page_url,
            card.image_url,
            image_path,
            json_dumps(card.aliases),
            json_dumps(card.metadata),
            collected_at,
        ),
    )


def _seed_reference_from_snapshot(conn, settings: Settings, entity_type: str, snapshot_name: str, now: str) -> int:
    snapshot_path = settings.reference_html_dir / snapshot_name
    search_url = f"https://bazaardb.gg/search?c={entity_type}"
    html = _maybe_refresh_snapshot(search_url, snapshot_path)
    if not html:
        return 0
    cards = _parse_catalog_html(html, entity_type)
    icon_dir = settings.reference_icons_items_dir if entity_type == "items" else settings.reference_icons_skills_dir
    inserted = 0
    with httpx.Client(headers={"User-Agent": "Mozilla/5.0"}, timeout=60.0, follow_redirects=True) as client:
        for card in cards:
            image_path = _download_icon(client, card.image_url, icon_dir, card.entity_id)
            _upsert_reference_card(conn, card, image_path, now)
            inserted += 1
    return inserted


def _extract_cards_from_live_page(search_url: str) -> list[ReferenceCard]:
    if not _playwright_available():
        raise RuntimeError("Playwright is not installed")
    headless = os.environ.get("BAZAR_PLAYWRIGHT_HEADLESS", "0") == "1"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        page = browser.new_page(viewport={"width": 1600, "height": 2200})
        page.goto(search_url, wait_until="domcontentloaded", timeout=120000)
        page.wait_for_timeout(5000)
        _dismiss_consent(page)

        for _ in range(120):
            buttons = page.get_by_text("Load more", exact=False)
            if buttons.count() == 0:
                break
            button = buttons.first
            if not button.is_visible():
                break
            button.scroll_into_view_if_needed(timeout=5000)
            button.click(timeout=10000)
            page.wait_for_timeout(1200)

        raw_cards = page.locator("a[href*='/card/']").evaluate_all(
            r"""
            (anchors) => anchors.map((anchor) => {
              const text = (anchor.textContent || '').replace(/\s+/g, ' ').trim();
              let current = anchor;
              let image = anchor.querySelector('img[src*=\"bazaardb\"]');
              for (let i = 0; i < 5 && !image && current; i += 1) {
                current = current.parentElement;
                if (current) {
                  image = current.querySelector('img[src*=\"bazaardb\"]');
                }
              }
              return {
                href: anchor.href,
                text,
                image_url: image ? image.src : null,
              };
            })
            """
        )
        browser.close()

    merged: dict[str, dict] = {}
    entity_type = "items" if "c=items" in search_url else "skills"
    for row in raw_cards:
        href = row.get("href")
        if not href or "/card/" not in href:
            continue
        existing = merged.setdefault(href, {"text": "", "image_url": None})
        text = (row.get("text") or "").strip()
        if text and len(text) > len(existing["text"]):
            existing["text"] = text
        if row.get("image_url") and not existing["image_url"]:
            existing["image_url"] = row["image_url"]

    cards: list[ReferenceCard] = []
    for href, payload in merged.items():
        name = payload["text"]
        if not name or len(name) > 120:
            continue
        entity_id = _extract_card_id(href)
        aliases = [name, name.replace("-", " "), slugify(name).replace("-", " ")]
        cards.append(
            ReferenceCard(
                entity_id=entity_id,
                name=name,
                page_url=href,
                image_url=payload["image_url"],
                aliases=sorted(set(alias for alias in aliases if alias)),
                metadata={"entity_type": entity_type},
            )
        )
    return cards


def _read_sitemap_urls(snapshot_path: Path) -> list[str]:
    if snapshot_path.exists():
        cached = snapshot_path.read_text(encoding="utf-8")
        urls = re.findall(r"<loc>(.*?)</loc>", cached)
        if urls:
            return urls
    text = _fetch_bazaardb_text("https://bazaardb.gg/sitemap.xml")
    snapshot_path.write_text(text, encoding="utf-8")
    return re.findall(r"<loc>(.*?)</loc>", text)


def _extract_cards_from_list_pages(list_urls: list[str], entity_type: str, settings: Settings) -> list[ReferenceCard]:
    merged: dict[str, ReferenceCard] = {}
    for list_url in list_urls:
        snapshot_path = settings.reference_html_dir / f"list_{slugify(Path(urlparse(list_url).path).name)}.html"
        try:
            html = _fetch_bazaardb_text(list_url)
            snapshot_path.write_text(html, encoding="utf-8")
        except Exception:
            if not snapshot_path.exists():
                continue
            html = snapshot_path.read_text(encoding="utf-8")
        for card in _parse_catalog_html(html, entity_type):
            card.metadata["source_list_url"] = list_url
            existing = merged.get(card.entity_id)
            if existing is None:
                merged[card.entity_id] = card
                continue
            if card.image_url and not existing.image_url:
                existing.image_url = card.image_url
            if len(card.name) > len(existing.name):
                existing.name = card.name
    return list(merged.values())


def _extract_run_card_urls(settings: Settings) -> list[str]:
    urls: set[str] = set()
    for html_path in settings.raw_runs_dir.glob("run_*.html"):
        try:
            html = html_path.read_text(encoding="utf-8")
        except Exception:
            continue
        soup = BeautifulSoup(html, "html.parser")
        for anchor in soup.select("a[href*='/card/']"):
            href = anchor.get("href")
            if not href:
                continue
            urls.add(urljoin("https://bazaardb.gg/", href))
    return sorted(urls)


def _extract_card_from_html(html: str, card_url: str) -> ReferenceCard | None:
    soup = BeautifulSoup(html, "html.parser")
    title_text = (soup.title.get_text(" ", strip=True) if soup.title else "").lower()
    entity_type = None
    if " - item - " in title_text:
        entity_type = "items"
    elif " - skill - " in title_text:
        entity_type = "skills"
    if entity_type is None:
        breadcrumb = soup.get_text(" ", strip=True).lower()
        if " items " in f" {breadcrumb} ":
            entity_type = "items"
        elif " skills " in f" {breadcrumb} ":
            entity_type = "skills"
    if entity_type is None:
        return None

    heading = soup.select_one("h1")
    name = heading.get_text(" ", strip=True) if heading else _candidate_name_from_url(card_url).title()
    image = soup.select_one("img[src*='s.bazaardb.gg']")
    image_url = image.get("src") if image and image.get("src") else None
    entity_id = _extract_card_id(card_url)
    aliases = [name, name.replace("-", " "), slugify(name).replace("-", " ")]
    return ReferenceCard(
        entity_id=entity_id,
        name=name,
        page_url=card_url,
        image_url=image_url,
        aliases=sorted(set(alias for alias in aliases if alias)),
        metadata={"entity_type": entity_type},
    )


def _extract_card_from_page(page, card_url: str) -> ReferenceCard | None:
    page.goto(card_url, wait_until="domcontentloaded", timeout=120000)
    page.wait_for_timeout(1200)
    _dismiss_consent(page)

    title = page.title()
    lowered = title.lower()
    entity_type = None
    if " - item - " in lowered:
        entity_type = "items"
    elif " - skill - " in lowered:
        entity_type = "skills"
    if entity_type is None:
        return None

    name = page.locator("h1").first.inner_text().strip() if page.locator("h1").count() else Path(card_url).name.replace("-", " ")
    image_url = None
    images = page.locator("img[src*='s.bazaardb.gg']")
    if images.count() > 0:
        image_url = images.first.get_attribute("src")
    entity_id = _extract_card_id(card_url)
    aliases = [name, name.replace("-", " "), slugify(name).replace("-", " ")]
    return ReferenceCard(
        entity_id=entity_id,
        name=name,
        page_url=card_url,
        image_url=image_url,
        aliases=sorted(set(alias for alias in aliases if alias)),
        metadata={"entity_type": entity_type},
    )


def _download_icon(client: httpx.Client, image_url: str | None, target_dir: Path, entity_id: str) -> str | None:
    if not image_url:
        return None
    if image_url.startswith("data:"):
        return None
    image_url = urljoin("https://bazaardb.gg/", image_url)
    suffix = Path(urlparse(image_url).path).suffix or ".webp"
    output_path = target_dir / f"{entity_id}{suffix}"
    if not output_path.exists():
        response = client.get(image_url)
        response.raise_for_status()
        output_path.write_bytes(response.content)
    return str(output_path)


def _repair_missing_reference_icons(conn, settings: Settings) -> dict[str, int]:
    repaired = {"items": 0, "skills": 0}
    with httpx.Client(headers={"User-Agent": "Mozilla/5.0"}, timeout=60.0, follow_redirects=True) as client:
        for entity_type, table, icon_dir in (
            ("items", "reference_items", settings.reference_icons_items_dir),
            ("skills", "reference_skills", settings.reference_icons_skills_dir),
        ):
            rows = conn.execute(
                f"SELECT entity_id, image_url, image_path FROM {table} WHERE image_url IS NOT NULL ORDER BY name"
            ).fetchall()
            missing_rows = []
            for row in rows:
                image_path = Path(row["image_path"]) if row["image_path"] else None
                if image_path and image_path.exists():
                    continue
                missing_rows.append(row)
            if missing_rows:
                print(f"[reference] repairing {len(missing_rows)} missing {entity_type} icons", flush=True)
            for index, row in enumerate(missing_rows, start=1):
                image_path = _download_icon(client, row["image_url"], icon_dir, row["entity_id"])
                conn.execute(
                    f"UPDATE {table} SET image_path = ? WHERE entity_id = ?",
                    (image_path, row["entity_id"]),
                )
                repaired[entity_type] += 1
                if index == 1 or index % 25 == 0 or index == len(missing_rows):
                    print(f"[reference] {entity_type} repair {index}/{len(missing_rows)}", flush=True)
    conn.commit()
    return repaired


def build_reference_catalog(conn, settings: Settings) -> dict[str, int]:
    now = dt.datetime.utcnow().isoformat(timespec="seconds")
    counts: dict[str, int] = {"items": 0, "skills": 0}
    repaired = _repair_missing_reference_icons(conn, settings)
    print(f"[reference] repaired icons: {repaired}", flush=True)

    counts["items"] += _seed_reference_from_snapshot(conn, settings, "items", "items.html", now)
    counts["skills"] += _seed_reference_from_snapshot(conn, settings, "skills", "skills.html", now)
    print(f"[reference] seeded snapshots: {counts}", flush=True)

    sitemap_urls = _read_sitemap_urls(settings.reference_html_dir / "sitemap.xml")
    item_list_urls = [url for url in sitemap_urls if "/list/" in url and "-items-" in url]
    skill_list_urls = [url for url in sitemap_urls if "/list/" in url and "-skills-" in url]
    for entity_type, list_urls in (("items", item_list_urls), ("skills", skill_list_urls)):
        icon_dir = settings.reference_icons_items_dir if entity_type == "items" else settings.reference_icons_skills_dir
        cards = _extract_cards_from_list_pages(list_urls, entity_type, settings)
        print(f"[reference] syncing {len(cards)} {entity_type} cards from list pages", flush=True)
        with httpx.Client(headers={"User-Agent": "Mozilla/5.0"}, timeout=60.0, follow_redirects=True) as client:
            for index, card in enumerate(cards, start=1):
                image_path = _download_icon(client, card.image_url, icon_dir, card.entity_id)
                _upsert_reference_card(conn, card, image_path, now)
                if index == 1 or index % 50 == 0 or index == len(cards):
                    print(f"[reference] {entity_type} list sync {index}/{len(cards)}", flush=True)
        counts[entity_type] += len(cards)
    card_urls = [url for url in sitemap_urls if "/card/" in url]

    known_ids = {
        row[0]
        for row in conn.execute(
            "SELECT entity_id FROM reference_items UNION SELECT entity_id FROM reference_skills"
        ).fetchall()
    }

    existing_missing_urls = [
        row[0]
        for row in conn.execute(
            "SELECT page_url FROM reference_items WHERE image_path IS NULL ORDER BY name"
        ).fetchall()
    ]
    run_card_urls = _extract_run_card_urls(settings)
    enrichment_urls = list(dict.fromkeys([*existing_missing_urls, *run_card_urls]))

    # The default run should always cover cards that appear in the crawled runs.
    # Only the broader sitemap backfill is gated behind the explicit full flag.
    if os.environ.get("BAZAR_REFERENCE_FULL", "0") == "1":
        enrichment_urls = list(dict.fromkeys([*enrichment_urls, *sorted(card_urls)]))

    batch_size = int(os.environ.get("BAZAR_REFERENCE_BATCH_SIZE", "25"))
    page_delay_ms = int(os.environ.get("BAZAR_REFERENCE_DELAY_MS", "2500"))
    if os.environ.get("BAZAR_REFERENCE_FULL", "0") == "1":
        enrichment_urls = enrichment_urls[: max(batch_size, len(run_card_urls) + len(existing_missing_urls))]

    with httpx.Client(headers={"User-Agent": "Mozilla/5.0"}, timeout=60.0, follow_redirects=True) as client:
        print(f"[reference] enriching {len(enrichment_urls)} card pages", flush=True)
        for index, card_url in enumerate(enrichment_urls, start=1):
            card = None
            try:
                html = _fetch_bazaardb_text(card_url)
                card = _extract_card_from_html(html, card_url)
            except Exception:
                if _allow_playwright_fallback() and _playwright_available():
                    try:
                        with sync_playwright() as playwright:
                            context, page = _open_persistent_page(playwright, settings)
                            try:
                                card = _extract_card_from_page(page, card_url)
                            finally:
                                context.close()
                    except Exception:
                        card = None
            if card is None:
                continue
            entity_type = card.metadata["entity_type"]
            icon_dir = settings.reference_icons_items_dir if entity_type == "items" else settings.reference_icons_skills_dir
            image_path = _download_icon(client, card.image_url, icon_dir, card.entity_id)
            _upsert_reference_card(conn, card, image_path, now)
            known_ids.add(card.entity_id)
            counts[entity_type] += 1
            if index == 1 or index % 10 == 0 or index == len(enrichment_urls):
                print(f"[reference] enrichment {index}/{len(enrichment_urls)}", flush=True)
            if (counts["items"] + counts["skills"]) % 25 == 0:
                conn.commit()
    conn.commit()
    print(f"[reference] done: {counts}", flush=True)
    return counts
