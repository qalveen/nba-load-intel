"""Rate-limited, retrying wrapper around nba_api endpoint calls.

nba_api endpoints fetch data on construction, so we wrap the constructor
itself with a rate limiter + exponential backoff. This is the piece that
makes the pipeline survive stats.nba.com throttling.
"""
import time
import logging
from typing import Any

import config

log = logging.getLogger(__name__)


class RateLimiter:
    """Enforce a minimum delay between successive requests."""

    def __init__(self, delay: float = config.REQUEST_DELAY_SECONDS):
        self.delay = delay
        self._last = 0.0

    def wait(self):
        elapsed = time.monotonic() - self._last
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last = time.monotonic()


_limiter = RateLimiter()


def call_endpoint(endpoint_cls, **kwargs) -> Any:
    """Instantiate an nba_api endpoint with rate limiting + backoff.

    Retries on any exception (timeouts, throttles, transient network errors)
    with exponential backoff. Raises after MAX_RETRIES.
    """
    last_err = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        _limiter.wait()
        try:
            return endpoint_cls(timeout=config.REQUEST_TIMEOUT, **kwargs)
        except Exception as e:  # noqa: BLE001 - nba_api raises broadly
            last_err = e
            sleep_for = config.BACKOFF_BASE ** attempt
            log.warning(
                "call to %s failed (attempt %d/%d): %s -- backing off %.1fs",
                endpoint_cls.__name__, attempt, config.MAX_RETRIES, e, sleep_for,
            )
            time.sleep(sleep_for)
    raise RuntimeError(
        f"endpoint {endpoint_cls.__name__} failed after "
        f"{config.MAX_RETRIES} retries"
    ) from last_err
