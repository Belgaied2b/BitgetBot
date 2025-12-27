from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import aiohttp

LOGGER = logging.getLogger(__name__)

DERIBIT_BASE = os.getenv("DERIBIT_BASE", "https://www.deribit.com/api/v2")
OPTIONS_TTL_S = float(os.getenv("OPTIONS_TTL_S", "120"))
HTTP_TIMEOUT_S = float(os.getenv("OPTIONS_HTTP_TIMEOUT_S", "10"))

# Deribit volatility index candles:
# method/endpoint: public/get_volatility_index_data
# Docs show GET style on /api/v2/public/... and WS JSON-RPC.
# We'll use GET for simplicity.
# Parameters: currency, start_timestamp, end_timestamp, resolution (seconds or "1D").


@dataclass
class OptionsSnapshot:
    ts: float
    dvol_btc: Optional[float]
    dvol_eth: Optional[float]
    regime: str
    raw: Dict[str, Any]


class OptionsCache:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._snap: Optional[OptionsSnapshot] = None

    async def get(self, force: bool = False) -> OptionsSnapshot:
        now = time.time()
        async with self._lock:
            if (not force) and self._snap and (now - self._snap.ts) < OPTIONS_TTL_S:
                return self._snap

        snap = await _fetch_options_snapshot()
        async with self._lock:
            self._snap = snap
        return snap


async def _fetch_dvol(currency: str, resolution: int = 3600) -> Tuple[Optional[float], Dict[str, Any]]:
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - int(24 * 3600 * 1000)  # last 24h window
    params = {
        "currency": currency,
        "start_timestamp": str(start_ts),
        "end_timestamp": str(end_ts),
        "resolution": str(int(resolution)),
    }
    url = f"{DERIBIT_BASE}/public/get_volatility_index_data"

    raw: Dict[str, Any] = {}
    try:
        timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as resp:
                raw = await resp.json() if resp.status == 200 else {}
    except Exception as e:
        LOGGER.debug("[OPTIONS] dvol fetch %s failed: %s", currency, e)
        return None, {}

    # Expect: {"jsonrpc":"2.0","result":{"data":[[ts,o,h,l,c],...], ...}}
    close_v = None
    try:
        res = raw.get("result") if isinstance(raw, dict) else None
        data = (res or {}).get("data") if isinstance(res, dict) else None
        if isinstance(data, list) and data:
            last = data[-1]
            if isinstance(last, (list, tuple)) and len(last) >= 5:
                close_v = float(last[4])
    except Exception:
        close_v = None
    return close_v, (raw if isinstance(raw, dict) else {})


async def _fetch_options_snapshot() -> OptionsSnapshot:
    t0 = time.time()
    dvol_btc, raw_btc = await _fetch_dvol("BTC", resolution=3600)
    dvol_eth, raw_eth = await _fetch_dvol("ETH", resolution=3600)

    # regime heuristics
    vals = [v for v in [dvol_btc, dvol_eth] if isinstance(v, (int, float))]
    avg = sum(vals) / len(vals) if vals else None

    regime = "unknown"
    try:
        if avg is None:
            regime = "unknown"
        elif avg >= 70:
            regime = "high_vol"
        elif avg <= 45:
            regime = "low_vol"
        else:
            regime = "mid_vol"
    except Exception:
        regime = "unknown"

    dt_ms = int((time.time() - t0) * 1000)
    LOGGER.info("[OPTIONS] ms=%d dvol_btc=%s dvol_eth=%s regime=%s",
                dt_ms,
                f"{dvol_btc:.3g}" if dvol_btc else None,
                f"{dvol_eth:.3g}" if dvol_eth else None,
                regime)

    return OptionsSnapshot(
        ts=time.time(),
        dvol_btc=dvol_btc,
        dvol_eth=dvol_eth,
        regime=regime,
        raw={"btc": raw_btc, "eth": raw_eth},
    )


def score_options_context(opt: Any, bias: str) -> Dict[str, Any]:
    """
    Non-blocking options vol context.

    - breakout/trend trades like *mid/high vol*
    - mean-reversion likes low vol

    We keep it simple: +1 if vol is mid/high, 0 otherwise.
    """
    b = (bias or "").upper()
    if not isinstance(opt, dict) and not isinstance(opt, OptionsSnapshot):
        return {"ok": True, "score": 0, "regime": "unknown", "reason": "no_options"}

    regime = opt.regime if isinstance(opt, OptionsSnapshot) else str(opt.get("regime") or "unknown")
    score = 0
    reason = "neutral"
    if regime in ("mid_vol", "high_vol"):
        score = 1
        reason = "vol_supports_trend"
    elif regime == "low_vol":
        score = 0
        reason = "low_vol_neutral"
    else:
        score = 0
        reason = "unknown"

    return {"ok": True, "score": int(score), "regime": regime, "reason": reason}
