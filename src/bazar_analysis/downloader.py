from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import httpx
from PIL import Image

from .config import Settings


def _read_image_metadata(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        return rgb_image.size


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_and_validate_image(client: httpx.Client, url: str, output_path: Path, attempts: int = 3) -> tuple[bytes, int, int]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = client.get(url)
            response.raise_for_status()
            content = response.content
            output_path.write_bytes(content)
            width, height = _read_image_metadata(output_path)
            return content, width, height
        except Exception as exc:
            last_error = exc
            output_path.unlink(missing_ok=True)
            print(f"[download] retry {attempt}/{attempts} failed for {url}: {type(exc).__name__}", flush=True)
    raise RuntimeError(f"failed to download valid image after {attempts} attempts: {url}") from last_error


def download_screenshots(conn, settings: Settings) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT screenshot_id, screenshot_url, post_id, local_path, sha256
        FROM screenshots
        ORDER BY screenshot_id
        """
    ).fetchall()

    downloaded = 0
    skipped = 0
    repaired = 0
    failed = 0
    print(f"[download] checking {len(rows)} screenshots", flush=True)
    with httpx.Client(
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0 Safari/537.36",
            "Referer": "https://bazaar-builds.net/",
        },
        follow_redirects=True,
        timeout=60.0,
    ) as client:
        for index, row in enumerate(rows, start=1):
            screenshot_id = row["screenshot_id"]
            url = row["screenshot_url"]
            suffix = Path(url).suffix or ".jpg"
            output_path = settings.raw_screenshots_dir / f"screenshot_{screenshot_id}{suffix}"
            db_path = Path(row["local_path"]) if row["local_path"] else None
            existing_path = None
            if db_path and db_path.exists():
                existing_path = db_path
            elif output_path.exists():
                existing_path = output_path

            if existing_path is not None:
                try:
                    width, height = _read_image_metadata(existing_path)
                    skipped += 1
                except Exception:
                    existing_path.unlink(missing_ok=True)
                    try:
                        content, width, height = _download_and_validate_image(client, url, output_path)
                        downloaded += 1
                    except Exception as exc:
                        failed += 1
                        print(f"[download] giving up on screenshot {screenshot_id}: {exc}", flush=True)
                        conn.execute(
                            "UPDATE screenshots SET local_path = NULL, sha256 = NULL, width = NULL, height = NULL WHERE screenshot_id = ?",
                            (screenshot_id,),
                        )
                        continue
                else:
                    if existing_path != output_path or str(existing_path) != row["local_path"] or not row["sha256"]:
                        repaired += 1
                    content = None
            else:
                try:
                    content, width, height = _download_and_validate_image(client, url, output_path)
                    downloaded += 1
                except Exception as exc:
                    failed += 1
                    print(f"[download] giving up on screenshot {screenshot_id}: {exc}", flush=True)
                    conn.execute(
                        "UPDATE screenshots SET local_path = NULL, sha256 = NULL, width = NULL, height = NULL WHERE screenshot_id = ?",
                        (screenshot_id,),
                    )
                    continue

            sha256 = hashlib.sha256(content).hexdigest() if content is not None else (row["sha256"] or _sha256_file(output_path))

            conn.execute(
                """
                UPDATE screenshots
                SET local_path = ?, sha256 = ?, width = ?, height = ?, downloaded_at = ?
                WHERE screenshot_id = ?
                """,
                (
                    str(output_path),
                    sha256,
                    width,
                    height,
                    dt.datetime.utcnow().isoformat(timespec="seconds"),
                    screenshot_id,
                ),
            )
            if index == 1 or index % 25 == 0 or index == len(rows):
                print(
                    f"[download] {index}/{len(rows)} downloaded={downloaded} skipped={skipped} repaired={repaired}",
                    flush=True,
                )
    conn.commit()
    print(f"[download] done: downloaded={downloaded}, skipped={skipped}, repaired={repaired}, failed={failed}", flush=True)
    return {"downloaded": downloaded, "skipped": skipped, "repaired": repaired, "failed": failed}
