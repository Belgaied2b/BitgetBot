# =====================================================================
# institutional_data.py — Ultra Desk 3.2 (Bitget-only, public endpoints)
# Bitget USDT-M Futures (Mix) — REST + optional external WS hub (read-only)
#
# Confirmed endpoints used (Bitget API docs):
# - GET /api/v2/mix/market/merge-depth              (orderbook)         ✅
# - GET /api/v2/mix/market/open-interest            (open interest)     ✅
# - GET /api/v2/mix/market/current-fund-rate        (current funding)   ✅
# - GET /api/v2/mix/market/history-fund-rate        (funding history)   ✅
#
# Notes:
# - Keeps legacy keys to avoid KeyError: openInterest, fundingRate, binance_symbol...
# - Adds bitget_symbol too (preferred).
# - Designed to raise inst_score in LIGHT even if WS hub is unstable.
# =====================================================================

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import aiohttp

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

LOGGER = logging.getLogger(__name__)

BITGET_API_BASE = "https://api.bitget.com"

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
# Product type (Bitget v2 request examples use lower-case values)
# USDT-M Futures: "usdt-futures"
# ---------------------------------------------------------------------
INST_BITGET_PRODUCT_TYPE = str(os.getenv("INST_BITGET_PRODUCT_TYPE", "usdt-futures")).strip()

# ---------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------
INST_MODE = str(os.getenv("INST_MODE", "LIGHT")).upper().strip()
if INST_MODE not in ("LIGHT", "NORMAL", "FULL"):
    INST_MODE = "LIGHT"

# ---------------------------------------------------------------------
# Feature toggles (defaults chosen to reduce inst_gate_low)
# ---------------------------------------------------------------------
# If you really want old behavior, set these to 0 via env.
INST_ENABLE_OPEN_INTEREST = str(os.getenv("INST_ENABLE_OPEN_INTEREST", "1")).strip() == "1"
INST_ENABLE_CURRENT_FUNDING = str(os.getenv("INST_ENABLE_CURRENT_FUNDING", "1")).strip() == "1"

# Tape/Trades remain off by default (not needed to fix your current gate issue).
INST_ENABLE_RECENT_FILLS = str(os.getenv("INST_ENABLE_RECENT_FILLS", "0")).strip() == "1"
INST_ENABLE_CANDLES = str(os.getenv("INST_ENABLE_CANDLES", "0")).strip() == "1"

# ---------------------------------------------------------------------
# Normalisation (rolling z-scores)
# ---------------------------------------------------------------------
INST_NORM_ENABLED = str(os.getenv("INST_NORM_ENABLED", "1")).strip() == "1"
INST_NORM_MIN_POINTS = int(float(os.getenv("INST_NORM_MIN_POINTS", "20")))
INST_NORM_WINDOW = int(float(os.getenv("INST_NORM_WINDOW", "120")))

# ---------------------------------------------------------------------
# Simple caching (avoid duplicate calls inside same scan burst)
# ---------------------------------------------------------------------
_CACHE_TTL_SEC = float(os.getenv("INST_CACHE_TTL_SEC", "12"))
_CACHE: Dict[Tuple[str, str], Tuple[float, Any]] = {}

# ---------------------------------------------------------------------
# Global rate limiting + retries
# ---------------------------------------------------------------------
_HTTP_CONCURRENCY = max(1, int(os.getenv("BITGET_HTTP_CONCURRENCY", "4")))
_HTTP_MIN_INTERVAL_SEC = float(os.getenv("BITGET_MIN_INTERVAL_SEC", "0.08"))
_HTTP_TIMEOUT_S = float(os.getenv("BITGET_HTTP_TIMEOUT_S", "10"))
_HTTP_RETRIES = max(0, int(os.getenv("BITGET_HTTP_RETRIES", "2")))

_HTTP_SEM = asyncio.Semaphore(_HTTP_CONCURRENCY)
_PACE_LOCK = asyncio.Lock()
_LAST_REQ_TS = 0.0

# Per-symbol backoff
_SYM_STATE: Dict[str, "SymbolBackoff"] = {}

# OI rolling history for slope
_OI_SERIES: Dict[str, Deque[Tuple[int, float]]] = {}
_OI_SERIES_MAXLEN = int(float(os.getenv("INST_OI_SERIES_MAXLEN", "10")))

# ---------------------------------------------------------------------
# Shared session
# ---------------------------------------------------------------------
_SESSION: Optional[aiohttp.ClientSession] = None
_SESSION_LOCK = asyncio.Lock()


def _now_ms() -> int:
    return int(time.time() * 1000)


async def _get_session() -> aiohttp.ClientSession:
    global _SESSION
    if _SESSION is not None and not _SESSION.closed:
        return _SESSION

    async with _SESSION_LOCK:
        if _SESSION is not None and not _SESSION.closed:
            return _SESSION
        timeout = aiohttp.ClientTimeout(total=float(_HTTP_TIMEOUT_S))
        connector = aiohttp.TCPConnector(limit=60, ttl_dns_cache=300, enable_cleanup_closed=True)
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


def _get_sym_state(symbol: Optional[str]) -> Optional[SymbolBackoff]:
    if not symbol:
        return None
    k = str(symbol).upper().strip()
    if not k:
        return None
    st = _SYM_STATE.get(k)
    if st is None:
        st = SymbolBackoff()
        _SYM_STATE[k] = st
    return st


def _cache_get(key: Tuple[str, str]) -> Any:
    try:
        ts, val = _CACHE.get(key, (0.0, None))
        if val is None:
            return None
        if (time.time() - float(ts)) <= float(_CACHE_TTL_SEC):
            return val
    except Exception:
        return None
    return None


def _cache_set(key: Tuple[str, str], val: Any) -> None:
    try:
        _CACHE[key] = (time.time(), val)
    except Exception:
        pass


async def _http_get(path: str, params: Optional[Dict[str, Any]] = None, *, symbol: Optional[str] = None) -> Any:
    """
    Safe GET with:
    - concurrency semaphore
    - pacing
    - per-symbol backoff
    - retries on transient errors (timeouts / 5xx)
    """
    st = _get_sym_state(symbol)
    if st is not None and st.blocked():
        return None

    url = BITGET_API_BASE + path
    session = await _get_session()

    async with _HTTP_SEM:
        for attempt in range(0, _HTTP_RETRIES + 1):
            await _pace()
            try:
                async with session.get(url, params=params) as resp:
                    status = resp.status
                    raw = await resp.read()
                    try:
                        txt = raw.decode("utf-8", errors="ignore")
                    except Exception:
                        txt = str(raw)[:500]

                    try:
                        data = json.loads(txt) if txt else None
                    except Exception:
                        data = None

                    if status != 200:
                        if st is not None:
                            st.mark_err(base_ms=1800)
                        if status == 429:
                            await asyncio.sleep(1.0)
                            return None
                        if 500 <= status <= 599 and attempt < _HTTP_RETRIES:
                            await asyncio.sleep(min(2.5, 0.6 * (1.8 ** attempt)))
                            continue
                        LOGGER.warning("[INST] HTTP %s GET %s params=%s resp=%s", status, path, params, (txt or "")[:200])
                        return None

                    # Bitget: {code:"00000", msg:"success", data:...}
                    if isinstance(data, dict):
                        code = str(data.get("code") or "")
                        if code and code != "00000":
                            if st is not None:
                                st.mark_err(base_ms=1600)
                            LOGGER.warning("[INST] BITGET code=%s msg=%s path=%s params=%s", code, data.get("msg"), path, params)
                            return None

                    if st is not None:
                        st.mark_ok()
                    return data

            except asyncio.TimeoutError:
                if st is not None:
                    st.mark_err(base_ms=1600)
                if attempt < _HTTP_RETRIES:
                    await asyncio.sleep(min(2.5, 0.6 * (1.8 ** attempt)))
                    continue
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


# ---------------------------------------------------------------------
# WS hub snapshot reader (optional)
# ---------------------------------------------------------------------
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
# Metrics helpers (orderbook)
# =====================================================================
def _compute_orderbook_band_metrics(depth: Dict[str, Any], band_bps: float = 25.0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns: (imbalance, bid_usd, ask_usd) within +/- band_bps around mid.
    Expects depth["bids"], depth["asks"] lists of [price, size] (strings or floats).
    """
    try:
        bids = depth.get("bids") or []
        asks = depth.get("asks") or []
        if not bids or not asks:
            return None, None, None

        b0p = float(bids[0][0])
        a0p = float(asks[0][0])
        if a0p <= 0 or b0p <= 0:
            return None, None, None
        mid = (b0p + a0p) / 2.0
        band = float(band_bps) / 10000.0
        lo = mid * (1.0 - band)
        hi = mid * (1.0 + band)

        bid_val = 0.0
        ask_val = 0.0

        for p, q in bids:
            pf = float(p)
            if pf < lo:
                break
            bid_val += pf * float(q)

        for p, q in asks:
            pf = float(p)
            if pf > hi:
                break
            ask_val += pf * float(q)

        den = bid_val + ask_val
        if den <= 0:
            return None, float(bid_val), float(ask_val)
        imb = float((bid_val - ask_val) / den)
        return imb, float(bid_val), float(ask_val)
    except Exception:
        return None, None, None


def _compute_spread_bps_from_depth(depth: Dict[str, Any]) -> Optional[float]:
    try:
        bids = depth.get("bids") or []
        asks = depth.get("asks") or []
        if not bids or not asks:
            return None
        bp = float(bids[0][0])
        ap = float(asks[0][0])
        if bp <= 0 or ap <= 0 or ap <= bp:
            return None
        mid = (bp + ap) / 2.0
        if mid <= 0:
            return None
        return float(((ap - bp) / mid) * 10000.0)
    except Exception:
        return None


def _compute_microprice_from_depth(depth: Dict[str, Any]) -> Optional[float]:
    try:
        bids = depth.get("bids") or []
        asks = depth.get("asks") or []
        if not bids or not asks:
            return None
        bp = float(bids[0][0])
        bq = float(bids[0][1])
        ap = float(asks[0][0])
        aq = float(asks[0][1])
        den = bq + aq
        if bp <= 0 or ap <= 0 or den <= 0:
            return None
        return float(((bp * aq) + (ap * bq)) / den)
    except Exception:
        return None


# =====================================================================
# Normalization (rolling z-scores)
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
# Bitget endpoints (confirmed by docs)
# =====================================================================

def _normalize_symbol(s: str) -> str:
    # Accept BTC-USDT, BTC_USDT, BTCUSDT_PERP -> BTCUSDT (best-effort)
    x = (s or "").upper().strip()
    x = x.replace("-", "").replace("_", "")
    x = x.replace("PERP", "")
    return x


async def _fetch_merge_depth(symbol: str) -> Optional[Dict[str, Any]]:
    # GET /api/v2/mix/market/merge-depth?productType=usdt-futures&symbol=BTCUSDT
    cache_key = ("merge-depth", symbol)
    cached = _cache_get(cache_key)
    if isinstance(cached, dict):
        return cached

    data = await _http_get(
        "/api/v2/mix/market/merge-depth",
        params={"productType": INST_BITGET_PRODUCT_TYPE, "symbol": symbol},
        symbol=symbol,
    )
    if not isinstance(data, dict):
        return None
    d = data.get("data")
    if not isinstance(d, dict):
        return None

    bids = d.get("bids")
    asks = d.get("asks")
    if not isinstance(bids, list) or not isinstance(asks, list):
        return None

    out = {"bids": bids, "asks": asks}
    _cache_set(cache_key, out)
    return out


async def _fetch_open_interest(symbol: str) -> Tuple[Optional[float], Optional[int]]:
    # GET /api/v2/mix/market/open-interest?symbol=BTCUSDT&productType=usdt-futures
    cache_key = ("open-interest", symbol)
    cached = _cache_get(cache_key)
    if isinstance(cached, tuple) and len(cached) == 2:
        return cached[0], cached[1]

    data = await _http_get(
        "/api/v2/mix/market/open-interest",
        params={"productType": INST_BITGET_PRODUCT_TYPE, "symbol": symbol},
        symbol=symbol,
    )
    if not isinstance(data, dict):
        return None, None
    d = data.get("data")
    if not isinstance(d, dict):
        return None, None

    ts_ms: Optional[int] = None
    try:
        ts_ms = int(float(d.get("ts"))) if d.get("ts") is not None else None
    except Exception:
        ts_ms = None

    lst = d.get("openInterestList")
    if not isinstance(lst, list) or not lst:
        return None, ts_ms

    size = None
    try:
        size = float(lst[0].get("size"))
    except Exception:
        size = None

    _cache_set(cache_key, (size, ts_ms))
    return size, ts_ms


async def _fetch_current_funding(symbol: str) -> Tuple[Optional[float], Optional[int]]:
    # GET /api/v2/mix/market/current-fund-rate?symbol=BTCUSDT&productType=usdt-futures
    cache_key = ("current-fund-rate", symbol)
    cached = _cache_get(cache_key)
    if isinstance(cached, tuple) and len(cached) == 2:
        return cached[0], cached[1]

    data = await _http_get(
        "/api/v2/mix/market/current-fund-rate",
        params={"productType": INST_BITGET_PRODUCT_TYPE, "symbol": symbol},
        symbol=symbol,
    )
    if not isinstance(data, dict):
        return None, None

    arr = data.get("data")
    if not isinstance(arr, list) or not arr:
        return None, None

    item = None
    for x in arr:
        if isinstance(x, dict) and str(x.get("symbol") or "").upper() == symbol:
            item = x
            break
    if item is None and isinstance(arr[0], dict):
        item = arr[0]

    fr = None
    nxt = None
    if isinstance(item, dict):
        try:
            fr = float(item.get("fundingRate"))
        except Exception:
            fr = None
        try:
            nxt = int(float(item.get("nextUpdate"))) if item.get("nextUpdate") is not None else None
        except Exception:
            nxt = None

    _cache_set(cache_key, (fr, nxt))
    return fr, nxt


async def _fetch_funding_history(symbol: str, limit: int = 30) -> Optional[List[Dict[str, Any]]]:
    # GET /api/v2/mix/market/history-fund-rate?symbol=BTCUSDT&productType=usdt-futures
    data = await _http_get(
        "/api/v2/mix/market/history-fund-rate",
        params={"productType": INST_BITGET_PRODUCT_TYPE, "symbol": symbol, "pageSize": str(int(limit)), "pageNo": "1"},
        symbol=symbol,
    )
    if not isinstance(data, dict):
        return None
    d = data.get("data")
    if not isinstance(d, list) or not d:
        return None
    return d


def _compute_funding_stats_bitget(hist: Optional[List[Dict[str, Any]]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Returns: (mean, std, zscore_last) using fundingRate from history-fund-rate."""
    try:
        if not hist:
            return None, None, None
        rates: List[float] = []
        for x in hist[-24:]:
            try:
                rates.append(float(x.get("fundingRate")))
            except Exception:
                continue
        if len(rates) < 5:
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


# =====================================================================
# Regimes / scoring helpers
# =====================================================================
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


def _components_ok_count(components: Dict[str, int]) -> int:
    try:
        return int(sum(1 for v in (components or {}).values() if int(v) > 0))
    except Exception:
        return 0


def _available_components_list(payload: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    if payload.get("orderbook_imb_25bps") is not None:
        out.append("orderbook")
    if payload.get("spread_bps") is not None:
        out.append("spread")
    if payload.get("depth_usd_25bps") is not None:
        out.append("depth")
    if payload.get("oi") is not None:
        out.append("oi")
    if payload.get("funding_rate") is not None:
        out.append("funding")
    if payload.get("funding_z") is not None:
        out.append("funding_hist")
    if payload.get("tape_delta_5m") is not None:
        out.append("tape")
    if payload.get("ws_snapshot_used"):
        out.append("ws_hub")
    if payload.get("normalization_enabled"):
        out.append("norm")
    return out


def _update_oi_series(sym: str, ts_ms: Optional[int], oi: Optional[float]) -> Optional[float]:
    """
    Returns an oi_slope (per hour) computed from the oldest->latest points in series.
    """
    if oi is None:
        return None

    series = _OI_SERIES.get(sym)
    if series is None:
        series = deque(maxlen=_OI_SERIES_MAXLEN)
        _OI_SERIES[sym] = series

    t = int(ts_ms) if ts_ms is not None else _now_ms()
    series.append((t, float(oi)))

    if len(series) < 2:
        return None

    t0, v0 = series[0]
    t1, v1 = series[-1]
    dt_ms = max(0, int(t1) - int(t0))
    if dt_ms <= 0:
        return None

    dt_h = dt_ms / 3_600_000.0
    if dt_h <= 1e-9:
        return None
    return float((v1 - v0) / dt_h)


# =====================================================================
# MAIN API
# =====================================================================
async def compute_full_institutional_analysis(
    symbol: str,
    bias: str,
    *,
    include_liquidations: bool = False,  # kept for signature compatibility
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Bitget-only institutional analysis (public data).
    Keeps legacy keys to avoid breaking existing code.
    """
    bias = (bias or "").upper().strip()
    eff_mode = (mode or INST_MODE).upper().strip()
    if eff_mode not in ("LIGHT", "NORMAL", "FULL"):
        eff_mode = "LIGHT"

    sym = _normalize_symbol(symbol)
    warnings: List[str] = []
    sources: Dict[str, str] = {}

    # WS snapshot (if hub provides it)
    ws_snap = _ws_snapshot(sym)
    ws_used = bool(ws_snap is not None)

    # -----------------------------------------------------------------
    # Depth (REST, confirmed)
    # -----------------------------------------------------------------
    depth = await _fetch_merge_depth(sym)
    if depth is None:
        warnings.append("no_depth")
        sources["depth"] = "none"
    else:
        sources["depth"] = "bitget_rest"

    ob_25 = None
    spread_bps = None
    microprice = None
    depth_bid_usd_25 = depth_ask_usd_25 = depth_usd_25 = None

    if isinstance(depth, dict):
        ob_25, b25, a25 = _compute_orderbook_band_metrics(depth, band_bps=25.0)
        depth_bid_usd_25, depth_ask_usd_25 = b25, a25
        if b25 is not None and a25 is not None:
            depth_usd_25 = float(b25 + a25)
        spread_bps = _compute_spread_bps_from_depth(depth)
        microprice = _compute_microprice_from_depth(depth)

    # -----------------------------------------------------------------
    # Current funding (REST confirmed) + optional WS override
    # -----------------------------------------------------------------
    funding_rate: Optional[float] = None
    next_funding_time_ms: Optional[int] = None

    if INST_ENABLE_CURRENT_FUNDING:
        fr, nxt = await _fetch_current_funding(sym)
        if fr is not None:
            funding_rate = fr
            next_funding_time_ms = nxt
            sources["funding_rate"] = "bitget_rest"
        else:
            warnings.append("no_current_funding")
            sources["funding_rate"] = "none"

    # WS override if present (keeps your hub as priority when available)
    if ws_snap is not None:
        try:
            if ws_snap.get("funding_rate") is not None:
                funding_rate = float(ws_snap.get("funding_rate"))
                sources["funding_rate"] = "ws_hub"
            if ws_snap.get("next_funding_time_ms") is not None:
                next_funding_time_ms = int(float(ws_snap.get("next_funding_time_ms")))
        except Exception:
            warnings.append("ws_parse_error")

    # -----------------------------------------------------------------
    # Open interest (REST confirmed)
    # -----------------------------------------------------------------
    oi_value: Optional[float] = None
    oi_ts_ms: Optional[int] = None
    oi_slope: Optional[float] = None

    if INST_ENABLE_OPEN_INTEREST:
        oi_value, oi_ts_ms = await _fetch_open_interest(sym)
        if oi_value is None:
            warnings.append("no_open_interest")
            sources["oi"] = "none"
        else:
            sources["oi"] = "bitget_rest"
            oi_slope = _update_oi_series(sym, oi_ts_ms, oi_value)

    # -----------------------------------------------------------------
    # Funding history stats (FULL only)
    # -----------------------------------------------------------------
    funding_mean = funding_std = funding_z = None
    if eff_mode == "FULL":
        hist = await _fetch_funding_history(sym, limit=30)
        if hist is None:
            warnings.append("no_funding_hist")
            sources["funding_hist"] = "none"
        else:
            funding_mean, funding_std, funding_z = _compute_funding_stats_bitget(hist)
            sources["funding_hist"] = "bitget_rest"

    # -----------------------------------------------------------------
    # Optional tape from WS (kept)
    # -----------------------------------------------------------------
    tape_5m = None
    if ws_snap is not None:
        try:
            if ws_snap.get("tape_delta_5m") is not None:
                tape_5m = float(ws_snap.get("tape_delta_5m"))
                sources["tape"] = "ws_hub"
        except Exception:
            warnings.append("ws_parse_error")

    # -----------------------------------------------------------------
    # Normalization
    # -----------------------------------------------------------------
    ob_imb_z = _norm_update(sym, "ob_imb", ob_25)
    spread_bps_z = _norm_update(sym, "spread_bps", spread_bps)
    depth_25_z = _norm_update(sym, "depth_25", depth_usd_25)
    oi_z = _norm_update(sym, "oi", oi_value)
    funding_z2 = _norm_update(sym, "funding_rate", funding_rate)

    funding_regime = _classify_funding(funding_rate, z=(funding_z if funding_z is not None else funding_z2))
    ob_regime = _classify_orderbook(ob_25)

    # -----------------------------------------------------------------
    # Scoring (0..4) compatible with your gate logic:
    # components: flow, oi, crowding, orderbook
    # -----------------------------------------------------------------
    components = {"flow": 0, "oi": 0, "crowding": 0, "orderbook": 0}

    # orderbook component: if we have at least one robust microstructure metric
    if (ob_25 is not None) or (spread_bps is not None) or (depth_usd_25 is not None):
        components["orderbook"] = 1

    # flow: strong imbalance (doesn't need tape)
    if ob_25 is not None and abs(float(ob_25)) >= 0.12:
        components["flow"] = 1
    elif tape_5m is not None and abs(float(tape_5m)) > 0:
        components["flow"] = 1

    # oi: open interest present
    if oi_value is not None:
        components["oi"] = 1

    # crowding: current funding or funding history zscore
    if funding_rate is not None or funding_z is not None:
        components["crowding"] = 1

    score = _components_ok_count(components)
    score = max(0, min(4, int(score)))
    ok_count = int(score)

    payload: Dict[str, Any] = {
        # primary score fields
        "institutional_score": int(score),
        "institutional_score_raw": int(score),
        "institutional_score_v2": int(score),
        "institutional_score_v3": int(score),

        # symbols: keep both
        "bitget_symbol": sym,
        "binance_symbol": sym,  # legacy compat

        "available": bool(depth is not None or ws_used),

        # OI
        "oi": oi_value,
        "oi_slope": oi_slope,
        "oi_z": oi_z,

        # CVD placeholders kept
        "cvd_slope": None,
        "cvd_notional_5m": None,

        # Funding
        "funding_rate": funding_rate,
        "funding_regime": funding_regime,
        "funding_mean": funding_mean,
        "funding_std": funding_std,
        "funding_z": funding_z,
        "funding_z2": funding_z2,
        "next_funding_time_ms": next_funding_time_ms,

        # Basis placeholders kept
        "basis_pct": None,
        "basis_regime": "unknown",

        # Tape
        "tape_delta_1m": None,
        "tape_delta_5m": tape_5m,
        "tape_regime": "unknown",

        # Orderbook metrics
        "orderbook_imb_25bps": ob_25,
        "orderbook_imb_25bps_z": ob_imb_z,
        "orderbook_regime": ob_regime,

        "spread_bps": spread_bps,
        "spread_bps_z": spread_bps_z,
        "microprice": microprice,

        "depth_bid_usd_25bps": depth_bid_usd_25,
        "depth_ask_usd_25bps": depth_ask_usd_25,
        "depth_usd_25bps": depth_usd_25,
        "depth_25bps_z": depth_25_z,

        # Meta / regimes placeholders
        "crowding_regime": "unknown",
        "flow_regime": "unknown",

        "warnings": warnings,

        "score_components": components,
        "score_meta": {
            "mode": eff_mode,
            "ok_count": int(ok_count),
            "ws_snapshot_used": bool(ws_used),
            "norm_enabled": bool(INST_NORM_ENABLED),
            "norm_min_points": int(INST_NORM_MIN_POINTS),
            "norm_window": int(INST_NORM_WINDOW),
            "bitget_product_type": INST_BITGET_PRODUCT_TYPE,
            "cache_ttl_sec": float(_CACHE_TTL_SEC),
        },

        "available_components": [],
        "available_components_count": 0,

        "ws_snapshot_used": bool(ws_used),
        "orderflow_ws_used": bool(ws_used),  # keep boolean semantics
        "normalization_enabled": bool(INST_NORM_ENABLED),
        "data_sources": sources,

        # Legacy/compat keys (KEEP!)
        "openInterest": oi_value,
        "fundingRate": funding_rate,
        "basisPct": None,
        "tapeDelta5m": tape_5m,
        "orderbookImb25bps": ob_25,
        "cvdSlope": None,
    }

    comps = _available_components_list(payload)
    payload["available_components"] = comps
    payload["available_components_count"] = int(len(comps))

    st = _get_sym_state(sym)
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
