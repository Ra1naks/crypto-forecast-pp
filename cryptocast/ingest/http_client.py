"""HTTP-клиент: таймауты, ретраи с экспоненциальным backoff, 429 и Retry-After."""
from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable

import requests

log = logging.getLogger(__name__)
RETRY_STATUSES = {408, 418, 425, 429, 500, 502, 503, 504}


class SourceUnavailable(RuntimeError):
    """Источник недоступен после исчерпания попыток."""


class HttpClient:
    def __init__(
        self,
        timeout: float = 30,
        max_retries: int = 5,
        backoff_base: float = 2.0,
        backoff_max: float = 120.0,
        user_agent: str = "cryptocast-study/1.0",
        sleep: Callable[[float], None] = time.sleep,
        session: requests.Session | None = None,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.sleep = sleep
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def _delay(self, attempt: int, response: requests.Response | None) -> float:
        if response is not None:
            header = response.headers.get("Retry-After")
            if header:
                try:
                    return min(float(header), self.backoff_max)
                except ValueError:
                    pass  # формат HTTP-date: используем backoff
        return min(self.backoff_base * 2**attempt, self.backoff_max) + random.uniform(0, 0.5)

    def get(self, url: str, params: dict | None = None, timeout: float | None = None):
        last_error = "unknown"
        for attempt in range(self.max_retries + 1):
            response = None
            try:
                response = self.session.get(url, params=params, timeout=timeout or self.timeout)
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    return response
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                if response.status_code not in RETRY_STATUSES:
                    raise SourceUnavailable(last_error)
            if attempt < self.max_retries:
                delay = self._delay(attempt, response)
                log.warning("Повтор %s/%s через %.1f c: %s", attempt + 1, self.max_retries, delay, last_error[:120])
                self.sleep(delay)
        raise SourceUnavailable(last_error)
