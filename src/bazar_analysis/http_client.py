from __future__ import annotations

import datetime as dt
from email.utils import parsedate_to_datetime
import os
import random
import threading
import time
from typing import Any

from curl_cffi import requests as curl_requests


RETRYABLE_STATUS_CODES = {408, 425, 429}


class BazaarDBRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds


class _RequestPacer:
    """Enforce a process-wide minimum gap, including across worker threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_request_at: float | None = None

    def wait(self, delay_seconds: float) -> None:
        delay_seconds = max(0.0, delay_seconds)
        with self._lock:
            now = time.monotonic()
            if self._last_request_at is not None:
                remaining = delay_seconds - (now - self._last_request_at)
                if remaining > 0:
                    time.sleep(remaining)
            self._last_request_at = time.monotonic()


_REQUEST_PACER = _RequestPacer()


def _retry_after_seconds(value: str | None, *, now: dt.datetime | None = None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=dt.UTC)
    current = now or dt.datetime.now(dt.UTC)
    return max(0.0, (retry_at.astimezone(dt.UTC) - current.astimezone(dt.UTC)).total_seconds())


def _backoff_seconds(attempt: int, retry_after_seconds: float | None) -> float:
    if retry_after_seconds is not None:
        return retry_after_seconds
    base = min(60.0, 2.0 ** max(0, attempt - 1))
    return base + random.uniform(0.0, min(0.5, base * 0.25))


def bazaardb_get(
    url: str,
    *,
    timeout: int | float = 60,
    referer: str | None = None,
    delay_seconds: float = 1.5,
    params: list[tuple[str, str]] | None = None,
    max_attempts: int = 4,
    log_prefix: str = "http",
    headers: dict[str, str] | None = None,
) -> Any:
    request_headers = {"Accept-Language": "en-US,en;q=0.9", **(headers or {})}
    if referer:
        request_headers["Referer"] = referer
    max_attempts = max(1, max_attempts)

    last_error: BazaarDBRequestError | None = None
    for attempt in range(1, max_attempts + 1):
        _REQUEST_PACER.wait(delay_seconds)
        try:
            response = curl_requests.get(
                url,
                params=params,
                impersonate=os.environ.get("BAZAR_CURL_IMPERSONATE", "firefox"),
                timeout=timeout,
                headers=request_headers,
                allow_redirects=True,
            )
        except Exception as exc:
            last_error = BazaarDBRequestError(
                f"request failed for {url}: {type(exc).__name__}",
                retryable=True,
            )
        else:
            status_code = int(response.status_code)
            if 200 <= status_code < 400:
                return response
            retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
            retryable = status_code in RETRYABLE_STATUS_CODES or 500 <= status_code < 600
            last_error = BazaarDBRequestError(
                f"HTTP {status_code} for {url}",
                status_code=status_code,
                retryable=retryable,
                retry_after_seconds=retry_after,
            )
            if not retryable:
                print(f"[{log_prefix}] non-retryable HTTP {status_code}; stopping requests to {url}", flush=True)
                raise last_error

        if attempt >= max_attempts:
            break
        wait_seconds = _backoff_seconds(attempt, last_error.retry_after_seconds)
        print(
            f"[{log_prefix}] retry {attempt}/{max_attempts} for {url} in {wait_seconds:.1f}s",
            flush=True,
        )
        time.sleep(wait_seconds)

    if last_error is None:  # pragma: no cover - defensive guard
        last_error = BazaarDBRequestError(f"request failed for {url}", retryable=True)
    raise last_error
