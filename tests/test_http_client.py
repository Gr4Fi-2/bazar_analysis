import datetime as dt
import unittest
from unittest.mock import Mock, patch

from bazar_analysis.http_client import BazaarDBRequestError, _retry_after_seconds, bazaardb_get


class _FakeResponse:
    def __init__(self, status_code: int, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.headers = headers or {}


class HTTPClientTests(unittest.TestCase):
    def test_retry_after_supports_seconds_and_http_dates(self) -> None:
        now = dt.datetime(2026, 7, 10, 12, 0, tzinfo=dt.UTC)

        self.assertEqual(_retry_after_seconds("7", now=now), 7.0)
        self.assertEqual(_retry_after_seconds("Fri, 10 Jul 2026 12:00:09 GMT", now=now), 9.0)
        self.assertIsNone(_retry_after_seconds("invalid", now=now))

    @patch("bazar_analysis.http_client._REQUEST_PACER.wait")
    @patch("bazar_analysis.http_client.curl_requests.get")
    def test_non_retryable_403_stops_immediately(self, get: Mock, pace: Mock) -> None:
        get.return_value = _FakeResponse(403)

        with self.assertRaises(BazaarDBRequestError) as raised:
            bazaardb_get("https://bazaardb.gg/api/run", max_attempts=4)

        self.assertEqual(raised.exception.status_code, 403)
        self.assertFalse(raised.exception.retryable)
        self.assertEqual(get.call_count, 1)
        pace.assert_called_once_with(1.5)

    @patch("bazar_analysis.http_client.time.sleep")
    @patch("bazar_analysis.http_client._REQUEST_PACER.wait")
    @patch("bazar_analysis.http_client.curl_requests.get")
    def test_429_respects_retry_after(self, get: Mock, pace: Mock, sleep: Mock) -> None:
        get.side_effect = [_FakeResponse(429, {"Retry-After": "7"}), _FakeResponse(200)]

        response = bazaardb_get("https://bazaardb.gg/api/run", max_attempts=2)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(get.call_count, 2)
        self.assertEqual(pace.call_count, 2)
        sleep.assert_called_once_with(7.0)


if __name__ == "__main__":
    unittest.main()
