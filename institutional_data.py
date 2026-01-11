# =====================================================================
# institutional_data.py — Ultra Desk 3.1 (BITGET ONLY / max institutional)
# Bitget USDT Perpetual Futures (public endpoints + optional WS hub)
#
# Objectif:
# - Plus AUCUNE dépendance Binance (REST/WS/symbols)
# - Priorité aux snapshots du institutional_ws_hub (Bitget)
# - Fallback REST Bitget si le WS hub est absent/instable
# - Maintien des clés legacy (openInterest, fundingRate, ...) pour compat
#
# Robustesse:
# - Rate limiter global (semaphore + pacing)
# - Soft cooldown sur erreurs 429/5xx
# - Backoff par symbole
# - Shared aiohttp session
#
# Modes:
# - LIGHT: OI + funding + basis (si dispo)
# - NORMAL: + tape delta + orderbook + spread + depth bands + large prints
# - FULL: + funding stats (hist) + build-up (price ret vs OI) + realized vol (si candles dispo)
#
# Notes:
# - Liquidations: uniquement via WS HUB si fourni (sinon None)
# - Orderflow WS interne: volontairement NON inclus ici (tu as déjà institutional_ws_hub)
# =====================================================================

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import aiohttp

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Optional external WS hub (read-only) — expected to be Bitget-based
# ---------------------------------------------------------------------
_WS_HUB = None
try:
    from institutional_ws_hub import HUB as _WS_HUB  # type: ignore
except Exception:
    _WS_HUB = None

INST_USE_WS_HUB = str(os.getenv("INST_USE_WS_HUB", "1")).strip() == "1"
WS_STALE_SEC = float(os.getenv("INST_WS_STALE_SEC", "15"))

# ---------------------------------------------------------------------
# Bitget REST
# ---------------------------------------------------------------------
BITGET_API_BASE = str(os.getenv("BITGET_API_BASE", "https://api.bitget.com")).rstrip("/")
BITGET_PRODUCT_TYPE = str(os.getenv("BITGET_PRODUCT_TYPE", "usdt-futures")).strip()  # v2 uses productType

# ---------------------------------------------------------------------
# Normalisation (rolling z-scores)
# ---------------------------------------------------------------------
INST_NORM_ENABLED = str(os.getenv("INST_NORM_ENABLED", "1")).strip() == "1"
INST_NORM_MIN_POINTS = int(float(os.getenv("INST_NORM_MIN_POINTS", "20")))
INST_NORM_WINDOW = int(float(os.getenv("INST_NORM_WINDOW", "120")))

# ---------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------
INST_MODE = str(os.getenv("INST_MODE", "LIGHT")).upper().strip()
if INST_MODE not in ("LIGHT", "NORMAL", "FULL"):
    INST_MODE = "LIGHT"

# ---------------------------------------------------------------------
# Liquidations (Bitget) — only if WS HUB provides it
# ---------------------------------------------------------------------
INST_INCLUDE_LIQUIDATIONS = str(os.getenv("INST_INCLUDE_LIQUIDATIONS", "0")).strip() == "1"
_LIQ_MIN_NOTIONAL_USD = float(os.getenv("INST_LIQ_MIN_NOTIONAL_USD", "50000"))

# ---------------------------------------------------------------------
# Global rate limiting + cooldown
# ---------------------------------------------------------------------
_HTTP_CONCURRENCY = max(1, int(os.getenv("BITGET_HTTP_CONCURRENCY", os.getenv("BINANCE_HTTP_CONCURRENCY", "3"))))
_HTTP_MIN_INTERVAL_SEC = float(os.getenv("BITGET_MIN_INTERVAL_SEC", os.getenv("BINANCE_MIN_INTERVAL_SEC", "0.12")))
_HTTP_TIMEOUT_S = float(os.getenv("BITGET_HTTP_TIMEOUT_S", os.getenv("BINANCE_HTTP_TIMEOUT_S", "10")))
_HTTP_RETRIES = max(0, int(os.getenv("BITGET_HTTP_RETRIES", os.getenv("BINANCE_HTTP_RETRIES", "2"))))

_SOFT_COOLDOWN_MS_DEFAULT = int(float(os.getenv("BITGET_SOFT_COOLDOWN_SEC", "15")) * 1000)

_HTTP_SEM = asyncio.Semaphore(_HTTP_CONCURRENCY)
_PACE_LOCK = asyncio.Lock()
_LAST_REQ_TS = 0.0

_SOFT_UNTIL_MS = 0

# Per-symbol backoff
_SYM_STATE: Dict[str, "SymbolBackoff"] = {}

# ---------------------------------------------------------------------
# Shared session
# ---------------------------------------------------------------------
_SESSION: Optional[aiohttp.ClientSession] = None
_SESSION_LOCK = asyncio.Lock()

# ---------------------------------------------------------------------
# Light caches
# ---------------------------------------------------------------------
_CONTRACTS_CACHE: Tuple[float, Set[str]] = (0.0, set())
_CONTRACTS_TTL = float(os.getenv("BITGET_CONTRACTS_TTL", "900"))

_DEPTH_CACHE: Dict[Tuple[str, int], Tuple[float, Any]] = {}
_TRADES_CACHE: Dict[Tuple[str, int], Tuple[float, Any]] = {}
_FUNDING_CACHE: Dict[str, Tuple[float, Any]] = {}
_PRICE_CACHE: Dict[str, Tuple[float, Any]] = {}
_FUNDING_HIST_CACHE: Dict[Tuple[str, int], Tuple[float, Any]] = {}
_OI_CACHE: Dict[str, Tuple[float, Any]] = {}

DEPTH_TTL = float(os.getenv("INST_DEPTH_TTL", "10"))
TRADES_TTL = float(os.getenv("INST_TRADES_TTL", "10"))
FUNDING_TTL = float(os.getenv("INST_FUNDING_TTL", "30"))
PRICE_TTL = float(os.getenv("INST_PRICE_TTL", "10"))
FUNDING_HIST_TTL = float(os.getenv("INST_FUNDING_HIST_TTL", "300"))
OI_TTL = float(os.getenv("INST_OI_TTL", "30"))

# scoring version (1/2/3)
INST_SCORE_VERSION = str(os.getenv("INST_SCORE_VERSION", "2")).strip()

# v3 quality thresholds
INST_SPREAD_MAX_BPS = float(os.getenv("INST_SPREAD_MAX_BPS", "12.0"))
INST_DEPTH_MIN_USD_25BPS = float(os.getenv("INST_DEPTH_MIN_USD_25BPS", "250000"))
INST_LARGE_TRADE_USD = float(os.getenv("INST_LARGE_TRADE_USD", "250000"))

# store OI history internally (for oi_hist_slope replacement)
_OI_SERIES: Dict[str, Deque[Tuple[int, float]]] = {}  # symbol -> deque[(ts_ms, oi)]
_OI_SERIES_MAXLEN = int(float(os.getenv("INST_OI_SERIES_MAXLEN", "80")))


def _now_ms() -> int:
    return int(time.time() * 1000)


def _is_soft_blocked() -> bool:
    return _now_ms() < int(_SOFT_UNTIL_MS)


def _set_soft_cooldown(ms_from_now: int, reason: str) -> None:
    global _SOFT_UNTIL_MS
    until = _now_ms() + int(ms_from_now)
    if until > _SOFT_UNTIL_MS:
        _SOFT_UNTIL_MS = until
    LOGGER.warning("[INST] BITGET SOFT COOLDOWN until_ms=%s reason=%s", _SOFT_UNTIL_MS, reason)


async def _get_session() -> aiohttp.ClientSession:
    global _SESSION
    if _SESSION is not None and not _SESSION.closed:
        return _SESSION

    async with _SESSION_LOCK:
        if _SESSION is not None and not _SESSION.closed:
            return _SESSION
        timeout = aiohttp.ClientTimeout(total=float(_HTTP_TIMEOUT_S))
        connector = aiohttp.TCPConnector(limit=50, ttl_dns_cache=300, enable_cleanup_closed=True)
        _SESSION = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return _SESSION


async def close_institutional_session() -> None:
    global _SESSION
    if _SESSION is not None and not _SESSION.closed:
        try:
            await _SESSION.close()
        except Exception:
            pass
    _SESSION = None


async def _pace() -> None:
    global _LAST_REQ_TS
    async with _PACE_LOCK:
        now = time.time()
        wait = float(_HTTP_MIN_INTERVAL_SEC) - (now - float(_LAST_REQ_TS))
        if wait > 0:
            await asyncio.sleep(wait)
        _LAST_REQ_TS = time.time()


@dataclass
class SymbolBackoff:
    until_ms: int = 0
    errors: int = 0

    def blocked(self) -> bool:
        return _now_ms() < int(self.until_ms)

    def mark_ok(self) -> None:
        self.errors = 0
        self.until_ms = 0

    def mark_err(self, base_ms: int = 1200, cap_ms: int = 120_000) -> None:
        self.errors += 1
        mult = 1.7 ** min(self.errors, 8)
        cd = int(min(cap_ms, base_ms * mult))
        self.until_ms = max(self.until_ms, _now_ms() + cd)


def _sym_key(symbol: Optional[str]) -> Optional[str]:
    if not symbol:
        return None
    return str(symbol).upper()


def _get_sym_state(symbol: Optional[str]) -> Optional[SymbolBackoff]:
    k = _sym_key(symbol)
    if not k:
        return None
    st = _SYM_STATE.get(k)
    if st is None:
        st = SymbolBackoff()
        _SYM_STATE[k] = st
    return st


def _unwrap_bitget(data: Any) -> Any:
    """
    Bitget v2 style:
      {"code":"00000","msg":"success","requestTime":...,"data": ...}
    """
    if isinstance(data, dict) and "data" in data and "code" in data:
        return data.get("data")
    return data


async def _http_get(path: str, params: Optional[Dict[str, Any]] = None, *, symbol: Optional[str] = None) -> Any:
    if _is_soft_blocked():
        return None

    st = _get_sym_state(symbol)
    if st is not None and st.blocked():
        return None

    url = f"{BITGET_API_BASE}{path}"
    session = await _get_session()

    async with _HTTP_SEM:
        for attempt in range(0, _HTTP_RETRIES + 1):
            await _pace()
            try:
                async with session.get(url, params=params) as resp:
                    status = int(resp.status)
                    raw = await resp.read()
                    try:
                        txt = raw.decode("utf-8", errors="ignore")
                    except Exception:
                        txt = str(raw)[:500]

                    parsed: Any = None
                    try:
                        parsed = json.loads(txt) if txt else None
                    except Exception:
                        parsed = None

                    if status != 200:
                        if status == 429:
                            _set_soft_cooldown(_SOFT_COOLDOWN_MS_DEFAULT, reason=f"{path} 429")
                            if st is not None:
                                st.mark_err(base_ms=2500)
                            return None
                        if 500 <= status <= 599:
                            _set_soft_cooldown(3000, reason=f"{path} {status}")
                            if st is not None:
                                st.mark_err(base_ms=1800)
                            if attempt < _HTTP_RETRIES:
                                await asyncio.sleep(min(2.5, 0.6 * (1.8 ** attempt)))
                                continue
                            return None

                        if st is not None:
                            st.mark_err(base_ms=1500)
                        LOGGER.warning("[INST] HTTP %s GET %s params=%s resp=%s", status, path, params, (txt or "")[:200])
                        return None

                    if st is not None:
                        st.mark_ok()
                    return parsed

            except asyncio.TimeoutError:
                if st is not None:
                    st.mark_err(base_ms=1600)
                if attempt < _HTTP_RETRIES:
                    await asyncio.sleep(min(2.5, 0.6 * (1.8 ** attempt)))
                    continue
                LOGGER.error("[INST] Timeout GET %s params=%s", path, params)
                return None
            except Exception as e:
                if st is not None:
                    st.mark_err(base_ms=2000)
                if attempt < _HTTP_RETRIES:
                    await asyncio.sleep(min(2.5, 0.6 * (1.8 ** attempt)))
                    continue
                LOGGER.error("[INST] Exception GET %s params=%s: %s", path, params, e)
                return None
    return None


# =====================================================================
# WS HUB snapshot reader (preferred)
# =====================================================================

def _ws_hub_running() -> bool:
    if _WS_HUB is None:
        return False
    try:
        v = getattr(_WS_HUB, "is_running", False)
        if callable(v):
            return bool(v())
        return bool(v)
    except Exception:
        return False


def _ws_snapshot(symbol: str) -> Optional[Dict[str, Any]]:
    if not INST_USE_WS_HUB or _WS_HUB is None:
        return None
    try:
        if not _ws_hub_running():
            return None
        snap = _WS_HUB.get_snapshot(symbol)
        if not isinstance(snap, dict) or not snap.get("available"):
            return None
        ts = snap.get("ts")
        if ts is None:
            return None
        if (time.time() - float(ts)) > float(WS_STALE_SEC):
            return None
        return snap
    except Exception:
        return None


# =====================================================================
# Symbol mapping (Bitget V2 futures tends to use "BTCUSDT" without suffix)
# =====================================================================

def _normalize_symbol_candidate(symbol: str) -> str:
    s = (symbol or "").upper().strip()
    s = s.replace("-", "").replace("_", "")
    # common suffixes from older Bitget naming
    s = s.replace("UMCBL", "").replace("DMCBL", "").replace("CMCBL", "")
    s = s.replace("PERP", "")
    return s


async def _get_bitget_contract_symbols() -> Set[str]:
    ts0, cached = _CONTRACTS_CACHE
    now = time.time()
    if cached and (now - ts0) < float(_CONTRACTS_TTL):
        return cached

    # Bitget v2: contracts list
    data = await _http_get("/api/v2/mix/market/contracts", params={"productType": BITGET_PRODUCT_TYPE}, symbol=None)
    payload = _unwrap_bitget(data)

    out: Set[str] = set()
    if isinstance(payload, list):
        for it in payload:
            try:
                if not isinstance(it, dict):
                    continue
                sym = it.get("symbol")
                if sym:
                    out.add(_normalize_symbol_candidate(str(sym)))
            except Exception:
                continue

    # update cache even if empty (avoid hammering)
    global _CONTRACTS_CACHE
    _CONTRACTS_CACHE = (now, out)
    if out:
        LOGGER.info("[INST] Bitget contracts loaded: %d", len(out))
    else:
        LOGGER.warning("[INST] Bitget contracts empty (API failure or format change)")
    return out


def _map_symbol_to_bitget(symbol: str, known: Set[str]) -> Optional[str]:
    """
    Return normalized symbol like "BTCUSDT".
    - Accepts "BTCUSDT_UMCBL", "BTC-USDT", "BTCUSDT" etc.
    """
    s = _normalize_symbol_candidate(symbol)
    if s in known:
        return s
    if s.startswith("1000") and s[4:] in known:
        return s[4:]
    m = re.match(r"^(\d+)([A-Z].+)$", s)
    if m and m.group(2) in known:
        return m.group(2)
    # if we couldn't load contracts (known empty), allow pass-through
    if not known and s.endswith("USDT"):
        return s
    return None


# =====================================================================
# Bitget REST fetchers
# =====================================================================

async def _fetch_open_interest(sym: str) -> Optional[float]:
    now = time.time()
    cached = _OI_CACHE.get(sym)
    if cached is not None and (now - cached[0]) < OI_TTL:
        return cached[1]  # type: ignore

    data = await _http_get(
        "/api/v2/mix/market/open-interest",
        params={"symbol": sym, "productType": BITGET_PRODUCT_TYPE},
        symbol=sym,
    )
    payload = _unwrap_bitget(data)

    oi: Optional[float] = None
    try:
        if isinstance(payload, dict):
            # common keys seen across exchanges: "openInterest" / "size"
            v = payload.get("openInterest")
            if v is None:
                v = payload.get("size")
            if v is None:
                v = payload.get("holdingAmount")
            if v is not None:
                oi = float(v)
    except Exception:
        oi = None

    _OI_CACHE[sym] = (now, oi)
    return oi


async def _fetch_current_funding(sym: str) -> Optional[float]:
    now = time.time()
    cached = _FUNDING_CACHE.get(sym)
    if cached is not None and (now - cached[0]) < FUNDING_TTL:
        return cached[1]  # type: ignore

    data = await _http_get(
        "/api/v2/mix/market/current-fundRate",
        params={"symbol": sym, "productType": BITGET_PRODUCT_TYPE},
        symbol=sym,
    )
    payload = _unwrap_bitget(data)

    fr: Optional[float] = None
    try:
        if isinstance(payload, dict):
            v = payload.get("fundingRate")
            if v is None:
                v = payload.get("fundRate")
            if v is not None:
                fr = float(v)
    except Exception:
        fr = None

    _FUNDING_CACHE[sym] = (now, fr)
    return fr


async def _fetch_next_funding_time_ms(sym: str) -> Optional[int]:
    data = await _http_get(
        "/api/v2/mix/market/funding-time",
        params={"symbol": sym, "productType": BITGET_PRODUCT_TYPE},
        symbol=sym,
    )
    payload = _unwrap_bitget(data)
    try:
        if isinstance(payload, dict):
            v = payload.get("nextFundingTime")
            if v is None:
                v = payload.get("fundingTime")
            if v is None:
                v = payload.get("ts")
            if v is not None:
                return int(float(v))
    except Exception:
        return None
    return None


async def _fetch_symbol_prices(sym: str) -> Optional[Dict[str, Any]]:
    now = time.time()
    cached = _PRICE_CACHE.get(sym)
    if cached is not None and (now - cached[0]) < PRICE_TTL:
        return cached[1]  # type: ignore

    data = await _http_get(
        "/api/v2/mix/market/symbol-price",
        params={"symbol": sym, "productType": BITGET_PRODUCT_TYPE},
        symbol=sym,
    )
    payload = _unwrap_bitget(data)
    if not isinstance(payload, dict):
        _PRICE_CACHE[sym] = (now, None)
        return None

    _PRICE_CACHE[sym] = (now, payload)
    return payload


async def _fetch_merge_depth(sym: str, limit: int = 100) -> Optional[Dict[str, Any]]:
    cache_key = (sym, int(limit))
    now = time.time()
    cached = _DEPTH_CACHE.get(cache_key)
    if cached is not None and (now - cached[0]) < DEPTH_TTL:
        return cached[1]  # type: ignore

    data = await _http_get(
        "/api/v2/mix/market/merge-depth",
        params={"symbol": sym, "productType": BITGET_PRODUCT_TYPE, "limit": int(limit)},
        symbol=sym,
    )
    payload = _unwrap_bitget(data)
    if not isinstance(payload, dict):
        return None

    _DEPTH_CACHE[cache_key] = (now, payload)
    return payload


async def _fetch_recent_trades(sym: str, limit: int = 200) -> Optional[List[Dict[str, Any]]]:
    cache_key = (sym, int(limit))
    now = time.time()
    cached = _TRADES_CACHE.get(cache_key)
    if cached is not None and (now - cached[0]) < TRADES_TTL:
        return cached[1]  # type: ignore

    data = await _http_get(
        "/api/v2/mix/market/fills",
        params={"symbol": sym, "productType": BITGET_PRODUCT_TYPE, "limit": int(limit)},
        symbol=sym,
    )
    payload = _unwrap_bitget(data)
    if not isinstance(payload, list) or not payload:
        return None

    _TRADES_CACHE[cache_key] = (now, payload)
    return payload


async def _fetch_funding_history(sym: str, limit: int = 30) -> Optional[List[Dict[str, Any]]]:
    cache_key = (sym, int(limit))
    now = time.time()
    cached = _FUNDING_HIST_CACHE.get(cache_key)
    if cached is not None and (now - cached[0]) < FUNDING_HIST_TTL:
        return cached[1]  # type: ignore

    data = await _http_get(
        "/api/v2/mix/market/history-fundRate",
        params={"symbol": sym, "productType": BITGET_PRODUCT_TYPE, "limit": int(limit)},
        symbol=sym,
    )
    payload = _unwrap_bitget(data)
    if not isinstance(payload, list) or not payload:
        return None

    _FUNDING_HIST_CACHE[cache_key] = (now, payload)
    return payload


# =====================================================================
# Metrics helpers (robust parsing)
# =====================================================================

def _mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    if np is not None:
        try:
            return float(np.mean(vals)), float(np.std(vals))
        except Exception:
            pass
    try:
        m = sum(vals) / len(vals)
        v = sum((x - m) ** 2 for x in vals) / len(vals)
        return float(m), float(v ** 0.5)
    except Exception:
        return None, None


def _compute_funding_stats(funding_hist: Optional[List[Dict[str, Any]]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        if not funding_hist:
            return None, None, None
        rates: List[float] = []
        for x in funding_hist[-60:]:
            v = None
            if isinstance(x, dict):
                v = x.get("fundingRate")
                if v is None:
                    v = x.get("fundRate")
            if v is None:
                continue
            try:
                rates.append(float(v))
            except Exception:
                continue
        if len(rates) < 6:
            return None, None, None
        mean, stdv = _mean_std(rates)
        if mean is None or stdv is None:
            return None, None, None
        std = stdv if stdv > 1e-12 else 0.0
        last = float(rates[-1])
        z = float((last - mean) / std) if std > 0 else None
        return float(mean), (float(std) if std > 0 else None), z
    except Exception:
        return None, None, None


def _depth_best(depth: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Returns: (bid_p, bid_q, ask_p, ask_q) from merge-depth payload
    Tries multiple layouts.
    """
    try:
        bids = depth.get("bids")
        asks = depth.get("asks")
        if isinstance(depth.get("data"), dict):
            bids = bids or depth["data"].get("bids")
            asks = asks or depth["data"].get("asks")

        if not isinstance(bids, list) or not isinstance(asks, list) or not bids or not asks:
            return None, None, None, None

        b0 = bids[0]
        a0 = asks[0]
        # support ["price","size"] or {"price":..,"size":..}
        if isinstance(b0, (list, tuple)) and len(b0) >= 2:
            bp, bq = float(b0[0]), float(b0[1])
        elif isinstance(b0, dict):
            bp, bq = float(b0.get("price") or b0.get("px")), float(b0.get("size") or b0.get("sz"))
        else:
            return None, None, None, None

        if isinstance(a0, (list, tuple)) and len(a0) >= 2:
            ap, aq = float(a0[0]), float(a0[1])
        elif isinstance(a0, dict):
            ap, aq = float(a0.get("price") or a0.get("px")), float(a0.get("size") or a0.get("sz"))
        else:
            return None, None, None, None

        return bp, bq, ap, aq
    except Exception:
        return None, None, None, None


def _compute_spread_bps(bp: Optional[float], ap: Optional[float]) -> Optional[float]:
    try:
        if bp is None or ap is None:
            return None
        bp = float(bp); ap = float(ap)
        if bp <= 0 or ap <= 0 or ap <= bp:
            return None
        mid = (bp + ap) / 2.0
        return float(((ap - bp) / mid) * 10000.0) if mid > 0 else None
    except Exception:
        return None


def _compute_microprice(bp: Optional[float], bq: Optional[float], ap: Optional[float], aq: Optional[float]) -> Optional[float]:
    try:
        if bp is None or ap is None or bq is None or aq is None:
            return None
        bp = float(bp); ap = float(ap); bq = float(bq); aq = float(aq)
        den = bq + aq
        if bp <= 0 or ap <= 0 or den <= 0:
            return None
        return float(((bp * aq) + (ap * bq)) / den)
    except Exception:
        return None


def _compute_book_skew(bq: Optional[float], aq: Optional[float]) -> Optional[float]:
    try:
        if bq is None or aq is None:
            return None
        bq = float(bq); aq = float(aq)
        den = bq + aq
        if den <= 0:
            return None
        return float((bq - aq) / den)
    except Exception:
        return None


def _compute_depth_usd_bands(depth: Dict[str, Any], band_bps: float) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Returns: (imbalance, bid_usd, ask_usd, total_usd) within +/- band_bps around mid.
    """
    try:
        bids = depth.get("bids")
        asks = depth.get("asks")
        if isinstance(depth.get("data"), dict):
            bids = bids or depth["data"].get("bids")
            asks = asks or depth["data"].get("asks")
        if not isinstance(bids, list) or not isinstance(asks, list) or not bids or not asks:
            return None, None, None, None

        bp, _, ap, _ = _depth_best(depth)
        if bp is None or ap is None:
            return None, None, None, None
        mid = (float(bp) + float(ap)) / 2.0
        if mid <= 0:
            return None, None, None, None

        band = float(band_bps) / 10000.0
        lo = mid * (1.0 - band)
        hi = mid * (1.0 + band)

        bid_val = 0.0
        ask_val = 0.0

        for lv in bids:
            if isinstance(lv, (list, tuple)) and len(lv) >= 2:
                p = float(lv[0]); q = float(lv[1])
            elif isinstance(lv, dict):
                p = float(lv.get("price") or lv.get("px"))
                q = float(lv.get("size") or lv.get("sz"))
            else:
                continue
            if p < lo:
                break
            bid_val += p * q

        for lv in asks:
            if isinstance(lv, (list, tuple)) and len(lv) >= 2:
                p = float(lv[0]); q = float(lv[1])
            elif isinstance(lv, dict):
                p = float(lv.get("price") or lv.get("px"))
                q = float(lv.get("size") or lv.get("sz"))
            else:
                continue
            if p > hi:
                break
            ask_val += p * q

        total = bid_val + ask_val
        if total <= 0:
            return None, float(bid_val), float(ask_val), 0.0
        imb = (bid_val - ask_val) / total
        return float(imb), float(bid_val), float(ask_val), float(total)
    except Exception:
        return None, None, None, None


def _trade_fields(t: Dict[str, Any]) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[str]]:
    """
    Return: (ts_ms, price, size, side)
    side expected buy/sell
    """
    try:
        ts = t.get("ts")
        if ts is None:
            ts = t.get("time")
        if ts is None:
            ts = t.get("T")
        ts_ms = int(float(ts)) if ts is not None else None

        price = t.get("price")
        if price is None:
            price = t.get("px")
        if price is None:
            price = t.get("p")
        pf = float(price) if price is not None else None

        size = t.get("size")
        if size is None:
            size = t.get("sz")
        if size is None:
            size = t.get("q")
        qf = float(size) if size is not None else None

        side = t.get("side")
        if side is None:
            side = t.get("direction")
        if side is None:
            side = t.get("S")
        sf = str(side).upper() if side is not None else None

        # normalize side variants
        if sf in ("BUY", "B", "LONG"):
            sf = "BUY"
        elif sf in ("SELL", "S", "SHORT"):
            sf = "SELL"

        return ts_ms, pf, qf, sf
    except Exception:
        return None, None, None, None


def _compute_tape_delta(trades: List[Dict[str, Any]], window_sec: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns: (delta_qty, delta_notional, cvd_notional) over window_sec
      delta_* normalized in [-1,+1]
      cvd_notional = sum(buy_notional - sell_notional) (raw)
    """
    try:
        if not trades:
            return None, None, None
        cutoff = _now_ms() - int(window_sec) * 1000

        buy_q = 0.0
        sell_q = 0.0
        buy_n = 0.0
        sell_n = 0.0

        for t in reversed(trades):
            ts, p, q, side = _trade_fields(t if isinstance(t, dict) else {})
            if ts is None or ts < cutoff:
                break
            if p is None or q is None or p <= 0 or q <= 0:
                continue
            notional = float(p) * float(q)
            if side == "BUY":
                buy_q += float(q)
                buy_n += notional
            elif side == "SELL":
                sell_q += float(q)
                sell_n += notional

        den_q = buy_q + sell_q
        den_n = buy_n + sell_n

        delta_q = ((buy_q - sell_q) / den_q) if den_q > 0 else None
        delta_n = ((buy_n - sell_n) / den_n) if den_n > 0 else None
        cvd_n = (buy_n - sell_n) if (buy_n + sell_n) > 0 else None
        return float(delta_q) if delta_q is not None else None, float(delta_n) if delta_n is not None else None, float(cvd_n) if cvd_n is not None else None
    except Exception:
        return None, None, None


def _compute_large_trade_imbalance(trades: List[Dict[str, Any]], window_sec: int, threshold_usd: float) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    try:
        if not trades:
            return None, None, None
        cutoff = _now_ms() - int(window_sec) * 1000
        thr = float(threshold_usd)

        buy = 0
        sell = 0

        for t in reversed(trades):
            ts, p, q, side = _trade_fields(t if isinstance(t, dict) else {})
            if ts is None or ts < cutoff:
                break
            if p is None or q is None or p <= 0 or q <= 0:
                continue
            notional = float(p) * float(q)
            if notional < thr:
                continue
            if side == "BUY":
                buy += 1
            elif side == "SELL":
                sell += 1

        den = buy + sell
        if den <= 0:
            return None, int(buy), int(sell)
        return float((buy - sell) / den), int(buy), int(sell)
    except Exception:
        return None, None, None


def _compute_oi_slope(sym: str, new_oi: Optional[float]) -> Optional[float]:
    if new_oi is None:
        return None
    dq = _OI_SERIES.get(sym)
    if dq is None:
        dq = deque(maxlen=_OI_SERIES_MAXLEN)
        _OI_SERIES[sym] = dq

    now = _now_ms()
    dq.append((now, float(new_oi)))

    # slope over last ~6 points (or fewer)
    if len(dq) < 2:
        return 0.0
    a = float(dq[0][1])
    b = float(dq[-1][1])
    den = abs(a) if abs(a) > 1e-12 else max(abs(b), 1e-12)
    return float((b - a) / den)


def _compute_oi_hist_slope(sym: str, points: int = 12) -> Optional[float]:
    dq = _OI_SERIES.get(sym)
    if not dq or len(dq) < 6:
        return None
    sub = list(dq)[-max(6, int(points)):]
    a = float(sub[0][1])
    b = float(sub[-1][1])
    den = abs(a) if abs(a) > 1e-12 else max(abs(b), 1e-12)
    return float((b - a) / den)


def _classify_tape(delta: Optional[float]) -> str:
    if delta is None:
        return "unknown"
    x = float(delta)
    if x >= 0.35:
        return "strong_buy"
    if x >= 0.12:
        return "buy"
    if x <= -0.35:
        return "strong_sell"
    if x <= -0.12:
        return "sell"
    return "neutral"


def _classify_orderbook(imb: Optional[float]) -> str:
    if imb is None:
        return "unknown"
    x = float(imb)
    if x >= 0.35:
        return "strong_bid"
    if x >= 0.12:
        return "bid"
    if x <= -0.35:
        return "strong_ask"
    if x <= -0.12:
        return "ask"
    return "balanced"


def _classify_funding(funding_rate: Optional[float], z: Optional[float] = None) -> str:
    if funding_rate is None:
        return "unknown"
    fr = float(funding_rate)
    if z is not None and abs(float(z)) >= 2.2:
        return "extreme"
    if fr <= -0.0015:
        return "very_negative"
    if fr <= -0.0005:
        return "negative"
    if fr < 0.0005:
        return "neutral"
    if fr < 0.0015:
        return "positive"
    return "very_positive"


def _classify_basis(basis_pct: Optional[float]) -> str:
    if basis_pct is None:
        return "unknown"
    b = float(basis_pct)
    if b >= 0.002:
        return "contango_strong"
    if b >= 0.0006:
        return "contango"
    if b <= -0.002:
        return "backwardation_strong"
    if b <= -0.0006:
        return "backwardation"
    return "flat"


def _classify_crowding(bias: str, funding_rate: Optional[float], basis_pct: Optional[float], funding_z: Optional[float]) -> str:
    if funding_rate is None and basis_pct is None:
        return "unknown"

    b = (bias or "").upper()
    fr = float(funding_rate) if funding_rate is not None else 0.0
    bs = float(basis_pct) if basis_pct is not None else 0.0

    crowded_long = (fr >= 0.001) or (bs >= 0.0015) or (funding_z is not None and funding_z >= 2.0)
    crowded_short = (fr <= -0.001) or (bs <= -0.0015) or (funding_z is not None and funding_z <= -2.0)

    if b == "LONG":
        if crowded_long:
            return "long_crowded_risky"
        if crowded_short:
            return "short_crowded_favorable"
        return "balanced"

    if b == "SHORT":
        if crowded_long:
            return "long_crowded_favorable"
        if crowded_short:
            return "short_crowded_risky"
        return "balanced"

    return "unknown"


def _classify_flow(cvd_slope: Optional[float], tape_5m: Optional[float]) -> str:
    if tape_5m is not None:
        return _classify_tape(tape_5m)
    if cvd_slope is None:
        return "unknown"
    x = float(cvd_slope)
    if x >= 1.0:
        return "strong_buy"
    if x >= 0.2:
        return "buy"
    if x <= -1.0:
        return "strong_sell"
    if x <= -0.2:
        return "sell"
    return "neutral"


def _classify_liq(delta_ratio: Optional[float], total_usd: Optional[float]) -> str:
    if delta_ratio is None or total_usd is None:
        return "unknown"
    if float(total_usd) < float(_LIQ_MIN_NOTIONAL_USD):
        return "low"
    return _classify_tape(delta_ratio)


def _classify_build_up(bias: str, price_ret_1h: Optional[float], oi_slope: Optional[float]) -> str:
    try:
        if price_ret_1h is None or oi_slope is None:
            return "unknown"
        pr = float(price_ret_1h)
        oi = float(oi_slope)
        if pr > 0 and oi > 0:
            return "long_build_up"
        if pr < 0 and oi > 0:
            return "short_build_up"
        if pr > 0 and oi < 0:
            return "short_covering"
        if pr < 0 and oi < 0:
            return "long_liquidation"
        return "flat"
    except Exception:
        return "unknown"


# =====================================================================
# Normalization (rolling z-scores)
# =====================================================================

@dataclass
class _RollingZ:
    maxlen: int = INST_NORM_WINDOW
    values: Deque[float] = field(default_factory=lambda: deque(maxlen=INST_NORM_WINDOW))

    def update_and_z(self, v: Optional[float]) -> Optional[float]:
        if v is None or not INST_NORM_ENABLED:
            return None
        try:
            vf = float(v)
        except Exception:
            return None

        vals = list(self.values)
        z: Optional[float] = None
        if len(vals) >= int(INST_NORM_MIN_POINTS):
            mean, std = _mean_std(vals)
            if mean is not None and std is not None and std > 1e-12:
                z = float((vf - float(mean)) / float(std))
        self.values.append(vf)
        return z


_NORM: Dict[str, Dict[str, _RollingZ]] = {}


def _norm_state(sym: str) -> Dict[str, _RollingZ]:
    st = _NORM.get(sym)
    if st is None:
        st = {}
        _NORM[sym] = st
    return st


def _norm_update(sym: str, metric: str, value: Optional[float]) -> Optional[float]:
    if not INST_NORM_ENABLED:
        return None
    st = _norm_state(sym)
    rz = st.get(metric)
    if rz is None:
        rz = _RollingZ()
        st[metric] = rz
    return rz.update_and_z(value)


# =====================================================================
# Scoring (v1/v2/v3) — repris de ton design
# =====================================================================

def _score_institutional_v1(
    bias: str,
    *,
    oi_slope: Optional[float],
    oi_hist_slope: Optional[float],
    cvd_slope: Optional[float],
    tape_5m: Optional[float],
    funding_rate: Optional[float],
    funding_z: Optional[float],
    basis_pct: Optional[float],
    ob_25bps: Optional[float],
    liq_delta_ratio_5m: Optional[float],
    liq_total_usd_5m: Optional[float],
) -> Tuple[int, Dict[str, int], Dict[str, Any]]:
    b = (bias or "").upper()
    comp: Dict[str, int] = {"flow": 0, "oi": 0, "crowding": 0, "orderbook": 0}
    meta: Dict[str, Any] = {}

    flow_points = 0

    if tape_5m is not None:
        x = float(tape_5m)
        if b == "LONG":
            if x >= 0.35:
                flow_points += 2
            elif x >= 0.12:
                flow_points += 1
        else:
            if x <= -0.35:
                flow_points += 2
            elif x <= -0.12:
                flow_points += 1

    if cvd_slope is not None:
        x = float(cvd_slope)
        if b == "LONG":
            if x >= 1.0:
                flow_points += 2
            elif x >= 0.2:
                flow_points += 1
        else:
            if x <= -1.0:
                flow_points += 2
            elif x <= -0.2:
                flow_points += 1

    if liq_delta_ratio_5m is not None and liq_total_usd_5m is not None and float(liq_total_usd_5m) >= float(_LIQ_MIN_NOTIONAL_USD):
        x = float(liq_delta_ratio_5m)
        if b == "LONG":
            if x >= 0.35:
                flow_points += 2
            elif x >= 0.12:
                flow_points += 1
        else:
            if x <= -0.35:
                flow_points += 2
            elif x <= -0.12:
                flow_points += 1
        meta["liq_used"] = True
    else:
        meta["liq_used"] = False

    if flow_points >= 3:
        comp["flow"] = 2
    elif flow_points >= 1:
        comp["flow"] = 1

    oi_ok = False
    if oi_slope is not None:
        x = float(oi_slope)
        if b == "LONG" and x >= 0.008:
            oi_ok = True
        if b == "SHORT" and x <= -0.008:
            oi_ok = True
    if (not oi_ok) and oi_hist_slope is not None:
        x = float(oi_hist_slope)
        if b == "LONG" and x >= 0.012:
            oi_ok = True
        if b == "SHORT" and x <= -0.012:
            oi_ok = True
    if oi_ok:
        comp["oi"] = 1

    if funding_rate is not None:
        fr = float(funding_rate)
        if b == "LONG" and fr < -0.0005:
            comp["crowding"] = 1
        if b == "SHORT" and fr > 0.0005:
            comp["crowding"] = 1
    if funding_z is not None:
        z = float(funding_z)
        if b == "LONG" and z <= -1.6:
            comp["crowding"] = max(comp["crowding"], 1)
        if b == "SHORT" and z >= 1.6:
            comp["crowding"] = max(comp["crowding"], 1)
    if basis_pct is not None:
        bs = float(basis_pct)
        if b == "LONG" and bs < -0.0006:
            comp["crowding"] = max(comp["crowding"], 1)
        if b == "SHORT" and bs > 0.0006:
            comp["crowding"] = max(comp["crowding"], 1)

    if ob_25bps is not None:
        x = float(ob_25bps)
        if b == "LONG" and x >= 0.12:
            comp["orderbook"] = 1
        if b == "SHORT" and x <= -0.12:
            comp["orderbook"] = 1

    total = int(comp["flow"] + comp["oi"] + comp["crowding"] + comp["orderbook"])
    total = max(0, min(4, total))
    meta["raw_components_sum"] = int(comp["flow"] + comp["oi"] + comp["crowding"] + comp["orderbook"])
    return total, comp, meta


def _score_institutional_v2_regime(
    bias: str,
    *,
    crowding_regime: str,
    tape_5m: Optional[float],
    ob_imb: Optional[float],
    oi_slope: Optional[float],
    funding_rate: Optional[float],
    funding_z: Optional[float],
    basis_pct: Optional[float],
    liq_delta_ratio_5m: Optional[float],
    liq_total_usd_5m: Optional[float],
    tape_5m_z: Optional[float],
    ob_imb_z: Optional[float],
    oi_slope_z: Optional[float],
    basis_pct_z: Optional[float],
    liq_total_z: Optional[float],
) -> Tuple[int, Dict[str, int], Dict[str, Any]]:
    b = (bias or "").upper().strip()
    comp: Dict[str, int] = {"flow": 0, "oi": 0, "crowding": 0, "orderbook": 0}
    meta: Dict[str, Any] = {"used_z": {}, "penalties": []}

    # FLOW
    flow_points = 0
    used_z_flow = False

    if tape_5m_z is not None:
        used_z_flow = True
        if b == "LONG":
            if tape_5m_z >= 1.5:
                flow_points += 2
            elif tape_5m_z >= 0.7:
                flow_points += 1
        elif b == "SHORT":
            if tape_5m_z <= -1.5:
                flow_points += 2
            elif tape_5m_z <= -0.7:
                flow_points += 1
    elif tape_5m is not None:
        x = float(tape_5m)
        if b == "LONG":
            if x >= 0.35:
                flow_points += 2
            elif x >= 0.12:
                flow_points += 1
        elif b == "SHORT":
            if x <= -0.35:
                flow_points += 2
            elif x <= -0.12:
                flow_points += 1

    liq_used = False
    if liq_delta_ratio_5m is not None and liq_total_usd_5m is not None and float(liq_total_usd_5m) >= float(_LIQ_MIN_NOTIONAL_USD):
        liq_used = True
        boost_ok = True
        if liq_total_z is not None and liq_total_z < 0.7:
            boost_ok = False

        if boost_ok:
            x = float(liq_delta_ratio_5m)
            if b == "LONG" and x >= 0.35:
                flow_points += 1
            if b == "SHORT" and x <= -0.35:
                flow_points += 1

    if flow_points >= 2:
        comp["flow"] = 2
    elif flow_points >= 1:
        comp["flow"] = 1

    meta["used_z"]["flow"] = bool(used_z_flow)
    meta["liq_used"] = bool(liq_used)

    # OI
    used_z_oi = False
    oi_ok = False
    if oi_slope_z is not None:
        used_z_oi = True
        if b == "LONG" and oi_slope_z >= 0.8:
            oi_ok = True
        elif b == "SHORT" and oi_slope_z <= -0.8:
            oi_ok = True
    elif oi_slope is not None:
        x = float(oi_slope)
        if b == "LONG" and x >= 0.008:
            oi_ok = True
        elif b == "SHORT" and x <= -0.008:
            oi_ok = True
    if oi_ok:
        comp["oi"] = 1
    meta["used_z"]["oi"] = bool(used_z_oi)

    # CROWDING (contrarian)
    crowding_ok = False
    if funding_z is not None:
        z = float(funding_z)
        if b == "LONG" and z <= -1.6:
            crowding_ok = True
        if b == "SHORT" and z >= 1.6:
            crowding_ok = True
    if (not crowding_ok) and funding_rate is not None:
        fr = float(funding_rate)
        if b == "LONG" and fr < -0.0005:
            crowding_ok = True
        if b == "SHORT" and fr > 0.0005:
            crowding_ok = True
    if (not crowding_ok) and basis_pct is not None and basis_pct_z is not None:
        if b == "LONG" and float(basis_pct) < -0.0006 and float(basis_pct_z) <= -0.7:
            crowding_ok = True
        if b == "SHORT" and float(basis_pct) > 0.0006 and float(basis_pct_z) >= 0.7:
            crowding_ok = True

    if crowding_ok:
        comp["crowding"] = 1

    penalty = 0
    if isinstance(crowding_regime, str) and crowding_regime.endswith("_risky"):
        if comp["flow"] < 2:
            meta["penalties"].append("crowding_risky_without_strong_flow")
            penalty = 1

    # ORDERBOOK
    used_z_ob = False
    ob_ok = False
    if ob_imb_z is not None:
        used_z_ob = True
        if b == "LONG" and ob_imb_z >= 0.7:
            ob_ok = True
        elif b == "SHORT" and ob_imb_z <= -0.7:
            ob_ok = True
    elif ob_imb is not None:
        x = float(ob_imb)
        if b == "LONG" and x >= 0.12:
            ob_ok = True
        elif b == "SHORT" and x <= -0.12:
            ob_ok = True
    if ob_ok:
        comp["orderbook"] = 1
    meta["used_z"]["orderbook"] = bool(used_z_ob)

    total_raw = int(comp["flow"] + comp["oi"] + comp["crowding"] + comp["orderbook"])
    total = max(0, min(4, total_raw - int(penalty)))
    meta["penalty_points"] = int(penalty)
    meta["raw_sum"] = int(total_raw)
    return total, comp, meta


def _score_institutional_v3_quality(
    bias: str,
    *,
    base_score: int,
    base_components: Dict[str, int],
    crowding_regime: str,
    spread_bps: Optional[float],
    depth_usd_25bps: Optional[float],
    build_up_regime: str,
    large_trade_imb_5m: Optional[float],
) -> Tuple[int, Dict[str, int], Dict[str, Any]]:
    b = (bias or "").upper().strip()
    score = int(base_score)
    meta: Dict[str, Any] = {"penalties": [], "bonuses": []}
    comp = dict(base_components or {})

    if spread_bps is not None and float(spread_bps) > float(INST_SPREAD_MAX_BPS):
        score -= 1
        meta["penalties"].append("spread_too_wide")

    if depth_usd_25bps is not None and float(depth_usd_25bps) < float(INST_DEPTH_MIN_USD_25BPS):
        score -= 1
        meta["penalties"].append("depth_too_thin_25bps")

    if isinstance(crowding_regime, str) and crowding_regime.endswith("_risky"):
        if int(comp.get("flow", 0)) < 2:
            score -= 1
            meta["penalties"].append("crowding_risky_no_strong_flow")

    aligns = False
    if b == "LONG" and build_up_regime == "long_build_up":
        aligns = True
    if b == "SHORT" and build_up_regime == "short_build_up":
        aligns = True

    if aligns and int(comp.get("flow", 0)) >= 1 and int(comp.get("oi", 0)) >= 1:
        score += 1
        meta["bonuses"].append("build_up_aligns")

    if large_trade_imb_5m is not None:
        x = float(large_trade_imb_5m)
        if b == "LONG" and x >= 0.35:
            score += 1
            meta["bonuses"].append("large_prints_support")
        if b == "SHORT" and x <= -0.35:
            score += 1
            meta["bonuses"].append("large_prints_support")

    score = max(0, min(4, int(score)))
    return score, comp, meta


def _components_ok_count(components: Dict[str, int]) -> int:
    try:
        return int(sum(1 for v in (components or {}).values() if int(v) > 0))
    except Exception:
        return 0


def _available_components_list(payload: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    if payload.get("oi") is not None:
        out.append("oi")
    if payload.get("funding_rate") is not None:
        out.append("funding")
    if payload.get("basis_pct") is not None:
        out.append("basis")
    if payload.get("tape_delta_5m") is not None:
        out.append("tape")
    if payload.get("orderbook_imb_25bps") is not None:
        out.append("orderbook")
    if payload.get("cvd_notional_5m") is not None:
        out.append("cvd_notional")
    if payload.get("funding_z") is not None:
        out.append("funding_hist")
    if payload.get("liq_total_usd_5m") is not None:
        out.append("liquidations")
    if payload.get("spread_bps") is not None:
        out.append("spread")
    if payload.get("depth_usd_25bps") is not None:
        out.append("depth")
    if payload.get("build_up_regime") not in (None, "unknown"):
        out.append("build_up")
    if payload.get("ws_snapshot_used"):
        out.append("ws_hub")
    if payload.get("normalization_enabled"):
        out.append("norm")
    return out


# =====================================================================
# MAIN API
# =====================================================================

async def compute_full_institutional_analysis(
    symbol: str,
    bias: str,
    *,
    include_liquidations: bool = False,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    bias = (bias or "").upper().strip()
    eff_mode = (mode or INST_MODE).upper().strip()
    if eff_mode not in ("LIGHT", "NORMAL", "FULL"):
        eff_mode = "LIGHT"

    warnings: List[str] = []
    sources: Dict[str, str] = {}

    use_liq = bool(INST_INCLUDE_LIQUIDATIONS or include_liquidations)

    if _is_soft_blocked():
        warnings.append(f"bitget_soft_cooldown_until_ms={_SOFT_UNTIL_MS}")

    # Resolve Bitget symbol
    known = await _get_bitget_contract_symbols()
    bitget_symbol = _map_symbol_to_bitget(symbol, known)
    if bitget_symbol is None:
        return {
            "institutional_score": 0,
            "institutional_score_raw": 0,
            "institutional_score_v2": 0,
            "institutional_score_v3": 0,
            "exchange": "BITGET",
            "exchange_symbol": None,
            "binance_symbol": None,  # legacy key, kept to avoid KeyError downstream
            "available": False,
            "warnings": ["symbol_not_mapped_to_bitget"],
            "score_components": {"flow": 0, "oi": 0, "crowding": 0, "orderbook": 0},
            "score_meta": {"mode": eff_mode},
            "available_components": [],
            "available_components_count": 0,
            "ban": {"soft_until_ms": int(_SOFT_UNTIL_MS)},
            "ws_snapshot_used": False,
            "normalization_enabled": bool(INST_NORM_ENABLED),
            "data_sources": {},
            # legacy/compat
            "openInterest": None,
            "fundingRate": None,
        }

    # Prefer WS hub snapshot
    ws_snap = _ws_snapshot(bitget_symbol)
    ws_used = bool(ws_snap is not None)

    # Fields
    oi_value: Optional[float] = None
    oi_slope: Optional[float] = None
    oi_hist_slope: Optional[float] = None
    oi_usd: Optional[float] = None

    funding_rate: Optional[float] = None
    funding_mean: Optional[float] = None
    funding_std: Optional[float] = None
    funding_z: Optional[float] = None
    next_funding_time_ms: Optional[int] = None

    basis_pct: Optional[float] = None
    mark_price: Optional[float] = None
    index_price: Optional[float] = None

    tape_1m: Optional[float] = None
    tape_5m: Optional[float] = None
    tape_1m_notional: Optional[float] = None
    tape_5m_notional: Optional[float] = None
    cvd_notional_5m: Optional[float] = None

    ob_25: Optional[float] = None
    ob_10: Optional[float] = None

    spread_bps: Optional[float] = None
    microprice: Optional[float] = None
    book_skew: Optional[float] = None

    depth_bid_usd_10: Optional[float] = None
    depth_ask_usd_10: Optional[float] = None
    depth_usd_10: Optional[float] = None
    depth_bid_usd_25: Optional[float] = None
    depth_ask_usd_25: Optional[float] = None
    depth_usd_25: Optional[float] = None
    depth_bid_usd_50: Optional[float] = None
    depth_ask_usd_50: Optional[float] = None
    depth_usd_50: Optional[float] = None
    depth_ratio_25: Optional[float] = None

    large_trade_imb_5m: Optional[float] = None
    large_buy_cnt_5m: Optional[int] = None
    large_sell_cnt_5m: Optional[int] = None

    # Liquidations via WS hub only (if present)
    liq_buy_usd_5m: Optional[float] = None
    liq_sell_usd_5m: Optional[float] = None
    liq_total_usd_5m: Optional[float] = None
    liq_delta_ratio_5m: Optional[float] = None
    liq_regime: str = "unknown"

    # FULL extras (best-effort; no guaranteed candles parsing here)
    price_ret_1h: Optional[float] = None
    realized_vol_24h: Optional[float] = None
    build_up_regime: str = "unknown"

    # -----------------------------------------------------------------
    # 1) From WS hub snapshot if available
    # -----------------------------------------------------------------
    if ws_snap is not None:
        try:
            # allow multiple possible key names
            oi_value = ws_snap.get("oi") or ws_snap.get("open_interest") or ws_snap.get("openInterest") or ws_snap.get("holdingAmount")
            if oi_value is not None:
                oi_value = float(oi_value)
                sources["oi"] = "ws"

            funding_rate = ws_snap.get("funding_rate") or ws_snap.get("fundingRate") or ws_snap.get("funding")
            if funding_rate is not None:
                funding_rate = float(funding_rate)
                sources["funding_rate"] = "ws"

            mark_price = ws_snap.get("mark_price") or ws_snap.get("markPrice")
            index_price = ws_snap.get("index_price") or ws_snap.get("indexPrice")
            if mark_price is not None:
                mark_price = float(mark_price)
            if index_price is not None:
                index_price = float(index_price)

            if basis_pct is None and mark_price is not None and index_price is not None and float(index_price) > 0:
                basis_pct = (float(mark_price) - float(index_price)) / float(index_price)
                sources["basis_pct"] = "ws"

            next_funding_time_ms = ws_snap.get("next_funding_time_ms") or ws_snap.get("nextFundingTime")
            if next_funding_time_ms is not None:
                next_funding_time_ms = int(float(next_funding_time_ms))
                sources["next_funding_time"] = "ws"

            # microstructure
            for k_in, out_ref in [
                ("tape_delta_1m", "tape_1m"),
                ("tape_delta_5m", "tape_5m"),
                ("tape_delta_1m_notional", "tape_1m_notional"),
                ("tape_delta_5m_notional", "tape_5m_notional"),
            ]:
                v = ws_snap.get(k_in)
                if v is None:
                    continue
                if out_ref == "tape_1m":
                    tape_1m = float(v)
                elif out_ref == "tape_5m":
                    tape_5m = float(v)
                elif out_ref == "tape_1m_notional":
                    tape_1m_notional = float(v)
                elif out_ref == "tape_5m_notional":
                    tape_5m_notional = float(v)

            if tape_5m is not None:
                sources["tape"] = "ws"

            ob_25 = ws_snap.get("orderbook_imbalance") or ws_snap.get("orderbook_imb_25bps") or ws_snap.get("orderbookImb25bps")
            if ob_25 is not None:
                ob_25 = float(ob_25)
                ob_10 = float(ob_25)
                sources["orderbook"] = "ws"

            spread_bps = ws_snap.get("spread_bps")
            if spread_bps is not None:
                spread_bps = float(spread_bps)
                sources["spread"] = "ws"

            microprice = ws_snap.get("microprice")
            if microprice is not None:
                microprice = float(microprice)

            book_skew = ws_snap.get("book_skew")
            if book_skew is not None:
                book_skew = float(book_skew)

            cvd_notional_5m = ws_snap.get("cvd_notional_5m")
            if cvd_notional_5m is not None:
                cvd_notional_5m = float(cvd_notional_5m)
                sources["cvd_notional_5m"] = "ws"

            if use_liq:
                liq_total_usd_5m = ws_snap.get("liq_total_usd_5m") or ws_snap.get("liq_notional_5m")
                liq_buy_usd_5m = ws_snap.get("liq_buy_usd_5m")
                liq_sell_usd_5m = ws_snap.get("liq_sell_usd_5m")
                liq_delta_ratio_5m = ws_snap.get("liq_delta_ratio_5m")
                if liq_total_usd_5m is not None:
                    liq_total_usd_5m = float(liq_total_usd_5m)
                if liq_buy_usd_5m is not None:
                    liq_buy_usd_5m = float(liq_buy_usd_5m)
                if liq_sell_usd_5m is not None:
                    liq_sell_usd_5m = float(liq_sell_usd_5m)
                if liq_delta_ratio_5m is not None:
                    liq_delta_ratio_5m = float(liq_delta_ratio_5m)
                if liq_total_usd_5m is not None:
                    liq_regime = _classify_liq(liq_delta_ratio_5m, liq_total_usd_5m)
                    sources["liquidations"] = "ws"
        except Exception:
            warnings.append("ws_snapshot_parse_error")

    # -----------------------------------------------------------------
    # 2) REST fallback (Bitget) for missing critical fields
    # -----------------------------------------------------------------
    # OI
    if oi_value is None:
        oi_value = await _fetch_open_interest(bitget_symbol)
        sources["oi"] = sources.get("oi", "rest") if oi_value is not None else sources.get("oi", "none")
        if oi_value is None:
            warnings.append("no_oi")

    if oi_value is not None:
        oi_slope = _compute_oi_slope(bitget_symbol, oi_value)
        oi_hist_slope = _compute_oi_hist_slope(bitget_symbol, points=12)

    # Funding + next funding time
    if funding_rate is None:
        funding_rate = await _fetch_current_funding(bitget_symbol)
        sources["funding_rate"] = sources.get("funding_rate", "rest") if funding_rate is not None else sources.get("funding_rate", "none")
        if funding_rate is None:
            warnings.append("no_funding")

    if next_funding_time_ms is None:
        next_funding_time_ms = await _fetch_next_funding_time_ms(bitget_symbol)
        sources["next_funding_time"] = sources.get("next_funding_time", "rest") if next_funding_time_ms is not None else sources.get("next_funding_time", "none")

    # Prices (mark/index) to compute basis
    if (mark_price is None or index_price is None) and basis_pct is None:
        pr = await _fetch_symbol_prices(bitget_symbol)
        if isinstance(pr, dict):
            try:
                mp = pr.get("markPrice") or pr.get("mark_price")
                ip = pr.get("indexPrice") or pr.get("index_price")
                if mp is not None:
                    mark_price = float(mp)
                if ip is not None:
                    index_price = float(ip)
                if mark_price is not None and index_price is not None and float(index_price) > 0:
                    basis_pct = (float(mark_price) - float(index_price)) / float(index_price)
                    sources["basis_pct"] = sources.get("basis_pct", "rest")
            except Exception:
                warnings.append("price_parse_error")

    # NORMAL/FULL: depth + trades
    depth: Optional[Dict[str, Any]] = None
    trades: Optional[List[Dict[str, Any]]] = None

    if eff_mode in ("NORMAL", "FULL"):
        depth, trades = await asyncio.gather(
            _fetch_merge_depth(bitget_symbol, limit=100),
            _fetch_recent_trades(bitget_symbol, limit=250),
        )

        if isinstance(depth, dict):
            bp, bq, ap, aq = _depth_best(depth)
            spread_bps = spread_bps if spread_bps is not None else _compute_spread_bps(bp, ap)
            microprice = microprice if microprice is not None else _compute_microprice(bp, bq, ap, aq)
            book_skew = book_skew if book_skew is not None else _compute_book_skew(bq, aq)

            # depth bands
            ob_10, b10, a10, t10 = _compute_depth_usd_bands(depth, 10.0)
            ob_25, b25, a25, t25 = _compute_depth_usd_bands(depth, 25.0)
            _, b50, a50, t50 = _compute_depth_usd_bands(depth, 50.0)

            depth_bid_usd_10, depth_ask_usd_10, depth_usd_10 = b10, a10, t10
            depth_bid_usd_25, depth_ask_usd_25, depth_usd_25 = b25, a25, t25
            depth_bid_usd_50, depth_ask_usd_50, depth_usd_50 = b50, a50, t50

            if depth_usd_25 is not None and depth_bid_usd_25 is not None and depth_ask_usd_25 is not None and float(depth_usd_25) > 0:
                depth_ratio_25 = float((float(depth_bid_usd_25) - float(depth_ask_usd_25)) / float(depth_usd_25))

            sources["orderbook"] = sources.get("orderbook", "rest")
            sources["spread"] = sources.get("spread", "rest")
            sources["depth"] = sources.get("depth", "rest")
        else:
            warnings.append("no_depth")

        if isinstance(trades, list) and trades:
            d1q, d1n, _ = _compute_tape_delta(trades, window_sec=60)
            d5q, d5n, cvd5n = _compute_tape_delta(trades, window_sec=300)

            tape_1m = tape_1m if tape_1m is not None else d1q
            tape_5m = tape_5m if tape_5m is not None else d5q
            tape_1m_notional = tape_1m_notional if tape_1m_notional is not None else d1n
            tape_5m_notional = tape_5m_notional if tape_5m_notional is not None else d5n
            cvd_notional_5m = cvd_notional_5m if cvd_notional_5m is not None else cvd5n

            if large_trade_imb_5m is None:
                large_trade_imb_5m, large_buy_cnt_5m, large_sell_cnt_5m = _compute_large_trade_imbalance(
                    trades, window_sec=300, threshold_usd=float(INST_LARGE_TRADE_USD)
                )

            sources["tape"] = sources.get("tape", "rest")
            sources["large_trades"] = sources.get("large_trades", "rest")
        else:
            warnings.append("no_trades")

    # FULL: funding stats + build-up (best-effort)
    if eff_mode == "FULL":
        fh = await _fetch_funding_history(bitget_symbol, limit=30)
        if isinstance(fh, list) and fh:
            funding_mean, funding_std, funding_z = _compute_funding_stats(fh)
        else:
            warnings.append("no_funding_hist")
        # build-up needs price_ret_1h; not implemented robustly here => keep unknown unless WS provides it
        try:
            pr1h = None
            if ws_snap is not None:
                pr1h = ws_snap.get("price_return_1h")
                if pr1h is not None:
                    price_ret_1h = float(pr1h)
            build_up_regime = _classify_build_up(bias, price_ret_1h, oi_slope)
        except Exception:
            build_up_regime = "unknown"

    # derive oi_usd if possible
    try:
        if oi_value is not None and mark_price is not None:
            oi_usd = float(oi_value) * float(mark_price)
    except Exception:
        oi_usd = None

    # Normalization (rolling z)
    oi_slope_z = _norm_update(bitget_symbol, "oi_slope", oi_slope)
    tape_5m_z = _norm_update(bitget_symbol, "tape_5m", tape_5m)
    ob_imb_z = _norm_update(bitget_symbol, "ob_imb", ob_25)
    basis_pct_z = _norm_update(bitget_symbol, "basis_pct", basis_pct)
    spread_bps_z = _norm_update(bitget_symbol, "spread_bps", spread_bps)
    depth_25_z = _norm_update(bitget_symbol, "depth_25", depth_usd_25)
    liq_total_z = _norm_update(bitget_symbol, "liq_total", liq_total_usd_5m)

    # Regimes
    funding_regime = _classify_funding(funding_rate, z=funding_z)
    basis_regime = _classify_basis(basis_pct)
    crowding_regime = _classify_crowding(bias, funding_rate, basis_pct, funding_z)
    flow_regime = _classify_flow(None, tape_5m)
    ob_regime = _classify_orderbook(ob_25)

    # scoring v1 + v2 + v3
    inst_score_raw, components_raw, meta_raw = _score_institutional_v1(
        bias,
        oi_slope=oi_slope,
        oi_hist_slope=oi_hist_slope,
        cvd_slope=None,
        tape_5m=tape_5m,
        funding_rate=funding_rate,
        funding_z=funding_z,
        basis_pct=basis_pct,
        ob_25bps=ob_25,
        liq_delta_ratio_5m=liq_delta_ratio_5m,
        liq_total_usd_5m=liq_total_usd_5m,
    )

    inst_score_v2, components_v2, meta_v2 = _score_institutional_v2_regime(
        bias,
        crowding_regime=crowding_regime,
        tape_5m=tape_5m,
        ob_imb=ob_25,
        oi_slope=oi_slope,
        funding_rate=funding_rate,
        funding_z=funding_z,
        basis_pct=basis_pct,
        liq_delta_ratio_5m=liq_delta_ratio_5m,
        liq_total_usd_5m=liq_total_usd_5m,
        tape_5m_z=tape_5m_z,
        ob_imb_z=ob_imb_z,
        oi_slope_z=oi_slope_z,
        basis_pct_z=basis_pct_z,
        liq_total_z=liq_total_z,
    )

    inst_score_v3, components_v3, meta_v3 = _score_institutional_v3_quality(
        bias,
        base_score=inst_score_v2,
        base_components=components_v2,
        crowding_regime=crowding_regime,
        spread_bps=spread_bps,
        depth_usd_25bps=depth_usd_25,
        build_up_regime=build_up_regime,
        large_trade_imb_5m=large_trade_imb_5m,
    )

    inst_score_main = inst_score_v2
    components_main = components_v2
    meta_main = {"scoring": "v2_regime", "v2": meta_v2, "v1_raw": meta_raw}

    if INST_SCORE_VERSION == "1":
        inst_score_main = inst_score_raw
        components_main = components_raw
        meta_main = {"scoring": "v1_raw", "v1_raw": meta_raw}
    elif INST_SCORE_VERSION == "3":
        inst_score_main = inst_score_v3
        components_main = components_v3
        meta_main = {"scoring": "v3_quality", "v3": meta_v3, "v2": meta_v2, "v1_raw": meta_raw}

    ok_count = _components_ok_count(components_main)

    score_meta: Dict[str, Any] = {
        "mode": eff_mode,
        "ok_count": int(ok_count),
        "ws_snapshot_used": bool(ws_used),
        "norm_enabled": bool(INST_NORM_ENABLED),
        "norm_min_points": int(INST_NORM_MIN_POINTS),
        "norm_window": int(INST_NORM_WINDOW),
        "spread_max_bps_v3": float(INST_SPREAD_MAX_BPS),
        "depth_min_usd_25bps_v3": float(INST_DEPTH_MIN_USD_25BPS),
        "large_trade_usd_threshold": float(INST_LARGE_TRADE_USD),
    }
    score_meta.update(meta_main)

    available = any(
        [
            oi_value is not None,
            funding_rate is not None,
            basis_pct is not None,
            tape_5m is not None,
            ob_25 is not None,
            (liq_total_usd_5m is not None and liq_total_usd_5m > 0.0),
            spread_bps is not None,
            depth_usd_25 is not None,
        ]
    )

    liquidation_intensity: Optional[float] = None
    try:
        if liq_total_usd_5m is not None:
            liquidation_intensity = float(liq_total_usd_5m)
    except Exception:
        liquidation_intensity = None

    payload: Dict[str, Any] = {
        "institutional_score": int(inst_score_main),
        "institutional_score_raw": int(inst_score_raw),
        "institutional_score_v2": int(inst_score_v2),
        "institutional_score_v3": int(inst_score_v3),

        "exchange": "BITGET",
        "exchange_symbol": bitget_symbol,
        "binance_symbol": bitget_symbol,  # legacy alias kept to avoid KeyError downstream

        "available": bool(available),

        "oi": oi_value,
        "oi_usd": oi_usd,
        "oi_slope": oi_slope,
        "oi_slope_z": oi_slope_z,
        "oi_hist_slope": oi_hist_slope,

        "funding_rate": funding_rate,
        "funding_regime": funding_regime,
        "funding_mean": funding_mean,
        "funding_std": funding_std,
        "funding_z": funding_z,
        "next_funding_time_ms": next_funding_time_ms,

        "basis_pct": basis_pct,
        "basis_pct_z": basis_pct_z,
        "basis_regime": basis_regime,

        "mark_price": mark_price,
        "index_price": index_price,

        "tape_delta_1m": tape_1m,
        "tape_delta_5m": tape_5m,
        "tape_delta_1m_notional": tape_1m_notional,
        "tape_delta_5m_notional": tape_5m_notional,
        "tape_delta_5m_z": tape_5m_z,
        "tape_regime": _classify_tape(tape_5m),

        "cvd_notional_5m": cvd_notional_5m,

        "large_trade_imb_5m": large_trade_imb_5m,
        "large_buy_cnt_5m": large_buy_cnt_5m,
        "large_sell_cnt_5m": large_sell_cnt_5m,

        "orderbook_imb_10bps": ob_10,
        "orderbook_imb_25bps": ob_25,
        "orderbook_imb_25bps_z": ob_imb_z,
        "orderbook_regime": ob_regime,

        "spread_bps": spread_bps,
        "spread_bps_z": spread_bps_z,
        "microprice": microprice,
        "book_skew": book_skew,

        "depth_bid_usd_10bps": depth_bid_usd_10,
        "depth_ask_usd_10bps": depth_ask_usd_10,
        "depth_usd_10bps": depth_usd_10,

        "depth_bid_usd_25bps": depth_bid_usd_25,
        "depth_ask_usd_25bps": depth_ask_usd_25,
        "depth_usd_25bps": depth_usd_25,
        "depth_25bps_z": depth_25_z,
        "depth_ratio_25bps": depth_ratio_25,

        "depth_bid_usd_50bps": depth_bid_usd_50,
        "depth_ask_usd_50bps": depth_ask_usd_50,
        "depth_usd_50bps": depth_usd_50,

        "crowding_regime": crowding_regime,
        "flow_regime": flow_regime,

        "liq_buy_usd_5m": liq_buy_usd_5m,
        "liq_sell_usd_5m": liq_sell_usd_5m,
        "liq_total_usd_5m": liq_total_usd_5m,
        "liq_total_usd_5m_z": liq_total_z,
        "liq_delta_ratio_5m": liq_delta_ratio_5m,
        "liq_regime": liq_regime,
        "liquidation_intensity": liquidation_intensity,

        "price_return_1h": price_ret_1h,
        "realized_vol_24h": realized_vol_24h,
        "build_up_regime": build_up_regime,

        "warnings": warnings,

        "score_components": components_main,
        "score_components_raw": components_raw,
        "score_components_v2": components_v2,
        "score_components_v3": components_v3,
        "score_meta": score_meta,

        "available_components": [],
        "available_components_count": 0,

        "ban": {"soft_until_ms": int(_SOFT_UNTIL_MS)},
        "ws_snapshot_used": bool(ws_used),
        "normalization_enabled": bool(INST_NORM_ENABLED),
        "data_sources": sources,

        # Legacy/compat keys (KEEP!)
        "openInterest": oi_value,
        "fundingRate": funding_rate,
        "basisPct": basis_pct,
        "tapeDelta5m": tape_5m,
        "orderbookImb25bps": ob_25,
        "cvdSlope": None,
    }

    payload["available_components"] = _available_components_list(payload)
    payload["available_components_count"] = int(len(payload["available_components"]))

    st = _get_sym_state(bitget_symbol)
    if st is not None:
        payload["symbol_cooldown_until_ms"] = int(st.until_ms)
        payload["symbol_errors"] = int(st.errors)

    return payload


# Alias
async def compute_institutional(
    symbol: str,
    bias: str,
    *,
    mode: Optional[str] = None,
    include_liquidations: bool = False,
) -> Dict[str, Any]:
    return await compute_full_institutional_analysis(symbol, bias, include_liquidations=include_liquidations, mode=mode)


def get_ban_state() -> Dict[str, int]:
    return {"soft_until_ms": int(_SOFT_UNTIL_MS)}
