from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from .config import Settings
from .db import next_id
from .utils import absolute_url, canonical_image_url, json_dumps, parse_rank_from_title, parse_score_from_title, safe_stem_from_url, slugify


BASE_CATEGORY_URL = "https://bazaar-builds.net/category/builds/jules-builds/"


@dataclass
class PostRecord:
    post_url: str
    title: str
    post_date: str | None


def build_client() -> httpx.Client:
    return httpx.Client(
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        },
        follow_redirects=True,
        timeout=30.0,
    )


def fetch_text(client: httpx.Client, url: str) -> str:
    response = client.get(url)
    response.raise_for_status()
    return response.text


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def discover_category_posts(client: httpx.Client, settings: Settings) -> list[PostRecord]:
    posts: list[PostRecord] = []
    seen_urls: set[str] = set()

    for page_number in range(1, 100):
        page_url = BASE_CATEGORY_URL if page_number == 1 else f"{BASE_CATEGORY_URL}page/{page_number}/"
        print(f"[crawl] category page {page_number}: {page_url}", flush=True)
        html = fetch_text(client, page_url)
        save_text(settings.raw_posts_dir / f"jules_category_page_{page_number}.html", html)
        soup = BeautifulSoup(html, "html.parser")

        page_posts: list[PostRecord] = []
        for heading in soup.select("h3 a[href]"):
            title = heading.get_text(" ", strip=True)
            if not title or "Jules" not in title:
                continue
            url = absolute_url(page_url, heading["href"])
            if url in seen_urls:
                continue
            article = heading.find_parent(["article", "div", "li"])
            date_text = None
            if article:
                date_candidate = article.get_text(" ", strip=True)
                month_names = [
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December",
                ]
                for month in month_names:
                    if month in date_candidate:
                        idx = date_candidate.index(month)
                        date_text = date_candidate[idx:].split("Read More", 1)[0].strip()
                        break
            page_posts.append(PostRecord(post_url=url, title=title, post_date=date_text))

        if not page_posts:
            print(f"[crawl] page {page_number}: no more posts, stopping", flush=True)
            break

        for post in page_posts:
            seen_urls.add(post.post_url)
            posts.append(post)

        print(f"[crawl] page {page_number}: discovered {len(page_posts)} posts, total {len(posts)}", flush=True)

    return posts


def parse_post(client: httpx.Client, settings: Settings, post: PostRecord) -> dict:
    html = fetch_text(client, post.post_url)
    post_slug = safe_stem_from_url(post.post_url)
    html_path = settings.raw_posts_dir / f"post_{post_slug}.html"
    save_text(html_path, html)

    soup = BeautifulSoup(html, "html.parser")
    title = (soup.select_one("h1") or soup.select_one("title"))
    canonical_title = title.get_text(" ", strip=True) if title else post.title

    author_node = soup.select_one("a[rel='author'], .author a")
    author = author_node.get_text(" ", strip=True) if author_node else None

    date_node = soup.select_one("time")
    post_date = date_node.get("datetime") if date_node and date_node.get("datetime") else post.post_date

    item_hints: list[str] = []
    for link in soup.select("a[href*='/tag/']"):
        text = link.get_text(" ", strip=True)
        if text and text.lower() not in {"jules"}:
            item_hints.append(text)

    content = soup.select_one("article") or soup
    screenshot_urls: list[str] = []
    for image in content.select("img[src]"):
        src = image.get("src") or ""
        if "/wp-content/uploads/" not in src:
            continue
        if any(size in src for size in ["150x150", "300x", "500x"]):
            continue
        absolute = absolute_url(post.post_url, src)
        absolute = canonical_image_url(absolute)
        if absolute not in screenshot_urls:
            screenshot_urls.append(absolute)

    if not screenshot_urls:
        og_image = soup.select_one("meta[property='og:image']")
        if og_image and og_image.get("content"):
            screenshot_urls.append(canonical_image_url(og_image["content"]))

    wins, losses = parse_score_from_title(canonical_title)
    rank_hint = parse_rank_from_title(canonical_title)

    return {
        "post_url": post.post_url,
        "title": canonical_title,
        "post_date": post_date,
        "author": author,
        "html_path": str(html_path),
        "item_hints": sorted(set(item_hints)),
        "screenshot_urls": screenshot_urls,
        "score_wins": wins,
        "score_losses": losses,
        "rank_title_hint": rank_hint,
    }


def crawl_posts(conn, settings: Settings) -> dict[str, int]:
    now = dt.datetime.utcnow().isoformat(timespec="seconds")
    inserted_posts = 0
    inserted_screenshots = 0

    with build_client() as client:
        posts = discover_category_posts(client, settings)
        print(f"[crawl] processing {len(posts)} posts", flush=True)
        for index, post in enumerate(posts, start=1):
            if index == 1 or index % 10 == 0 or index == len(posts):
                print(f"[crawl] post {index}/{len(posts)}: {post.post_url}", flush=True)
            record = parse_post(client, settings, post)
            existing_post = conn.execute("SELECT post_id FROM posts WHERE post_url = ?", (record["post_url"],)).fetchone()
            if existing_post:
                post_id = existing_post["post_id"]
                conn.execute(
                    """
                    UPDATE posts
                    SET hero = ?, title = ?, post_date = ?, author = ?, html_path = ?, score_wins = ?, score_losses = ?, rank_title_hint = ?, item_hints_json = ?, crawled_at = ?
                    WHERE post_id = ?
                    """,
                    (
                        "Jules",
                        record["title"],
                        record["post_date"],
                        record["author"],
                        record["html_path"],
                        record["score_wins"],
                        record["score_losses"],
                        record["rank_title_hint"],
                        json_dumps(record["item_hints"]),
                        now,
                        post_id,
                    ),
                )
            else:
                post_id = next_id(conn, "posts", "post_id")
                conn.execute(
                    """
                    INSERT INTO posts(post_id, hero, post_url, title, post_date, author, html_path, score_wins, score_losses, rank_title_hint, item_hints_json, crawled_at)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        post_id,
                        "Jules",
                        record["post_url"],
                        record["title"],
                        record["post_date"],
                        record["author"],
                        record["html_path"],
                        record["score_wins"],
                        record["score_losses"],
                        record["rank_title_hint"],
                        json_dumps(record["item_hints"]),
                        now,
                    ),
                )
            inserted_posts += 1

            for idx, screenshot_url in enumerate(record["screenshot_urls"]):
                existing_screenshot = conn.execute(
                    "SELECT screenshot_id FROM screenshots WHERE post_id = ? AND screenshot_url = ?",
                    (post_id, screenshot_url),
                ).fetchone()
                if existing_screenshot:
                    conn.execute(
                        "UPDATE screenshots SET is_primary = ? WHERE screenshot_id = ?",
                        (1 if idx == 0 else 0, existing_screenshot["screenshot_id"]),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO screenshots(screenshot_id, post_id, screenshot_url, is_primary)
                        VALUES(?, ?, ?, ?)
                        """,
                        (next_id(conn, "screenshots", "screenshot_id"), post_id, screenshot_url, 1 if idx == 0 else 0),
                    )
                inserted_screenshots += 1

    conn.commit()
    print(f"[crawl] done: posts={inserted_posts}, screenshots={inserted_screenshots}", flush=True)
    return {"posts": inserted_posts, "screenshots": inserted_screenshots}
