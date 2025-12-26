# =====================================================================
# retry_utils.py — Desk-lead Retry Engine (async)
# =====================================================================
# Goals:
# - Exponential backoff + jitter
# - Optional retry_on predicate (exception-based)
# - Optional on_retry callback (logging/metrics)
# - Hard timeout support per attempt (optional)
# - Safe defaults for exchange APIs (429/5xx/network)
# =====================================================================

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Awaitable, Callable, Optional, TypeVar

T = TypeVar("T")


def _default_retry_on(exc: Exception) -> bool:
    """
    Default: retry on network/timeout-like errors.
    We keep it generic: callers can pass a stricter predicate.
    """
    # asyncio
    if isinstance(exc, (asyncio.TimeoutError, ConnectionError)):
        return True

    # aiohttp is optional; do not import hard dependency here
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()

    # common transient classes
    if "clientconnectorerror" in name or "clientoserror" in name:
        return True
    if "serverdisconnectederror" in name:
        return True
    if "clientpayloaderror" in name:
        return True
    if "timeout" in name:
        return True

    # common transient messages
    if "too many requests" in msg or "http 429" in msg:
        return True
    if "temporarily unavailable" in msg:
        return True
    if "connection reset" in msg or "connection aborted" in msg:
        return True

    return False


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    retries: int = 4,
    base_delay: float = 0.35,
    max_delay: float = 8.0,
    jitter: float = 0.12,
    timeout_s: Optional[float] = None,
    retry_on: Optional[Callable[[Exception], bool]] = None,
    on_retry: Optional[Callable[[int, Exception, float], Any]] = None,
) -> T:
    """
    Executes async fn() with retries.

    Args:
      retries: total attempts (>=1). Example retries=4 => up to 4 tries.
      base_delay: initial backoff seconds.
      max_delay: cap for backoff delay.
      jitter: random added seconds (0..jitter).
      timeout_s: per-attempt timeout (None = no timeout).
      retry_on: predicate(exc)->bool. If returns False => raise immediately.
      on_retry: callback(attempt_index, exc, sleep_s). Called before sleeping.

    Returns:
      Result of fn() if successful.

    Raises:
      Last exception if all attempts fail or retry_on says no retry.
    """
    if retries <= 0:
        raise ValueError("retries must be >= 1")

    pred = retry_on or _default_retry_on

    last_exc: Optional[Exception] = None
    t_start = time.time()

    for attempt in range(1, retries + 1):
        try:
            if timeout_s is not None and timeout_s > 0:
                return await asyncio.wait_for(fn(), timeout=timeout_s)
            return await fn()
        except Exception as exc:
            last_exc = exc

            # no retry if predicate says so
            if not pred(exc):
                raise

            # out of attempts
            if attempt >= retries:
                raise

            # compute backoff
            # attempt=1 => 0 * base (but we still sleep base_delay)
            backoff = base_delay * (2 ** (attempt - 1))
            sleep_s = min(max_delay, backoff) + random.random() * float(jitter)

            if on_retry:
                try:
                    on_retry(attempt, exc, sleep_s)
                except Exception:
                    # never let logging callbacks break retry behavior
                    pass

            await asyncio.sleep(sleep_s)

    # should never reach here
    if last_exc:
        raise last_exc
    raise RuntimeError("retry_async failed without exception")
