from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def slugify(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return text or "unknown"


def normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def canonical_image_url(url: str) -> str:
    parsed = urlparse(url)
    path = re.sub(r"-\d+x\d+(?=\.[a-zA-Z0-9]+$)", "", parsed.path)
    return parsed._replace(path=path, query="", fragment="").geturl()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def safe_stem_from_url(url: str) -> str:
    parsed = urlparse(url)
    stem = Path(parsed.path).stem
    return slugify(stem)


def absolute_url(base_url: str, maybe_relative: str) -> str:
    return urljoin(base_url, maybe_relative)


def parse_score_from_title(title: str) -> tuple[int | None, int | None]:
    match = re.search(r"(\d+)[-–](\d+)", title)
    if not match:
        if re.search(r"\b10\s+Win\b", title, flags=re.IGNORECASE):
            return 10, 0
        return None, None
    return int(match.group(1)), int(match.group(2))


def parse_rank_from_title(title: str) -> str | None:
    lowered = title.lower()
    for tier in ["legend", "diamond", "gold", "silver", "bronze"]:
        if tier in lowered:
            return tier.title()
    return None
