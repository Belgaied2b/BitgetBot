from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import aiohttp

LOGGER = logging.getLogger(__name__)

# CoinGecko global data endpoint (free tier works without key)
# Docs: https://docs.coingecko.com/reference/crypto-global
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

MACRO_TTL_S = float(os.getenv("MACRO_TTL_S", "75"))  # keep low: macro doesn't need tick-level
HTTP_TIMEOUT_S = float(os.getenv("MACRO_HTTP_TIMEOUT_S", "10"))


@dataclass
class MacroSnapshot:
    ts: float
    total_mcap_usd: Optional[float]
    mcap_change_24h_pct: Optional[float]
    btc_dominance_pct: Optional[float]
    eth_dominance_pct: Optional[float]
    btc_mcap_usd: Optional[float]
    total2_mcap_usd: Optional[float]
    raw: Dict[str, Any]


class MacroCache:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._snap: Optional[MacroSnapshot] = None

    async def get(self, force: bool = False) -> MacroSnapshot:
        now = time.time()
        async with self._lock:
            if (not force) and self._snap and (now - self._snap.ts) < MACRO_TTL_S:
                return self._snap

        snap = await _fetch_macro_snapshot()
        async with self._lock:
            self._snap = snap
        return snap


async def _fetch_macro_snapshot() -> MacroSnapshot:
    url = f"{COINGECKO_BASE}/global"
    t0 = time.time()
    raw: Dict[str, Any] = {}
    try:
        timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                raw = await resp.json() if resp.status == 200 else {}
    except Exception as e:
        LOGGER.warning("[MACRO] fetch failed: %s", e)
        raw = {}

    data = raw.get("data") if isinstance(raw, dict) else None
    total_mcap_usd = None
    mcap_change_24h_pct = None
    btc_dom = None
    eth_dom = None

    try:
        if isinstance(data, dict):
            total = (data.get("total_market_cap") or {}).get("usd")
            if total is not None:
                total_mcap_usd = float(total)
            mchg = data.get("market_cap_change_percentage_24h_usd")
            if mchg is not None:
                mcap_change_24h_pct = float(mchg)
            pct = data.get("market_cap_percentage") or {}
            if isinstance(pct, dict):
                if pct.get("btc") is not None:
                    btc_dom = float(pct.get("btc"))
                if pct.get("eth") is not None:
                    eth_dom = float(pct.get("eth"))
    except Exception:
        pass

    btc_mcap = None
    total2 = None
    try:
        if total_mcap_usd is not None and btc_dom is not None:
            btc_mcap = total_mcap_usd * (btc_dom / 100.0)
            btc_mcap = float(btc_mcap)
            total2 = float(total_mcap_usd - btc_mcap)
    except Exception:
        pass

    dt_ms = int((time.time() - t0) * 1000)
    LOGGER.info("[MACRO] ok=%s ms=%d total_mcap=%s btc_dom=%s mcap24h=%s",
                bool(total_mcap_usd), dt_ms,
                f"{total_mcap_usd:.3g}" if total_mcap_usd else None,
                f"{btc_dom:.3g}" if btc_dom else None,
                f"{mcap_change_24h_pct:.3g}" if mcap_change_24h_pct else None)

    return MacroSnapshot(
        ts=time.time(),
        total_mcap_usd=total_mcap_usd,
        mcap_change_24h_pct=mcap_change_24h_pct,
        btc_dominance_pct=btc_dom,
        eth_dominance_pct=eth_dom,
        btc_mcap_usd=btc_mcap,
        total2_mcap_usd=total2,
        raw=raw if isinstance(raw, dict) else {},
    )


def score_macro_alignment(macro: Any, bias: str, symbol: str = "") -> Dict[str, Any]:
    """
    Non-blocking 'desk macro' score.

    Returns:
      { ok: bool, score: int in [-1..+1], regime: str, reason: str }
    """
    b = (bias or "").upper()
    if not isinstance(macro, dict) and not isinstance(macro, MacroSnapshot):
        return {"ok": True, "score": 0, "regime": "unknown", "reason": "no_macro"}

    # normalize input
    if isinstance(macro, MacroSnapshot):
        btc_dom = macro.btc_dominance_pct
        mcap24 = macro.mcap_change_24h_pct
    else:
        btc_dom = macro.get("btc_dominance_pct")
        mcap24 = macro.get("mcap_change_24h_pct")

    try:
        btc_dom = float(btc_dom) if btc_dom is not None else None
    except Exception:
        btc_dom = None
    try:
        mcap24 = float(mcap24) if mcap24 is not None else None
    except Exception:
        mcap24 = None

    # simple regime
    regime = "risk_on"
    if btc_dom is not None and btc_dom >= 58.0:
        regime = "risk_off"
    if mcap24 is not None and mcap24 <= -2.0:
        regime = "risk_off"

    # score
    score = 0
    reason = "neutral"
    if b == "LONG":
        if regime == "risk_off" and not str(symbol).upper().startswith("BTC"):
            score = -1
            reason = "risk_off_long_alt"
        elif regime == "risk_on":
            score = +1
            reason = "risk_on_long"
    elif b == "SHORT":
        if regime == "risk_off":
            score = +1
            reason = "risk_off_short"
        else:
            score = 0
            reason = "risk_on_short_neutral"

    return {"ok": True, "score": int(score), "regime": regime, "reason": reason}
