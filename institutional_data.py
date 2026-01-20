# institutional_data.py
# =====================================================================
# Ultra Desk 3.1 (Bitget-only, public endpoints)
# Bitget USDT-M Futures (Mix) — REST + optional external WS hub
#
# Confirmed endpoints used (Bitget API docs):
# - GET /api/v2/mix/market/merge-depth              (orderbook)
# - GET /api/v2/mix/market/history-fund-rate        (funding hist)
# - GET /api/v2/mix/market/current-fund-rate        (current funding)  [fallback]
# - GET /api/v2/mix/market/open-interest            (open interest)    [optional flag]
#
# Improvements in this version:
# ✅ Derived metrics (no extra endpoints): OI % change (15m/1h), funding change (1h), funding flip flag
# ✅ Uses WS hub values when available to compute derived metrics even if REST OI is disabled
# ✅ Optional OI REST fallback when WS missing (env: INST_OI_FALLBACK_WHEN_WS_MISSING=1)
# ✅ Safer caches + symbol-normalized series storage
# ✅ Keeps legacy keys to avoid KeyError: openInterest, fundingRate, binance_symbol...
# ✅ NEW: prefer settings.py values when present (avoid split-brain defaults)
# ✅ NEW: z-score clamp via INST_NORM_CLIP
# ✅ NEW: quality_score + quality_flags in score_meta
#
# v6 HOTFIX:
# ✅ FIX inst_unavailable caused by merge-depth returning empty/unsupported payload unless precision/limit are provided
#    - Always request merge-depth with precision + limit
#    - Retry once with alternate precision if book is not usable
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

# ---------------------------------------------------------------------
# Robust numeric parsing (Bitget sometimes returns numeric fields as strings)
# ---------------------------------------------------------------------

def _to_float(x):
    """Best-effort float conversion.

    Handles ints/floats, numeric strings (including commas), and returns None on failure.
    """
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            v = float(x)
            return v
        # numpy scalars
        try:
            import numpy as _np  # local import to avoid hard dependency
            if isinstance(x, (_np.floating, _np.integer)):
                return float(x)
        except Exception:
            pass
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return None
            # common formatting artifacts
            s = s.replace(',', '')
            return float(s)
        return float(x)
    except Exception:
        return None


def _maybe_json_load(v):
    """If v is a JSON string, attempt to load it."""
    try:
        if isinstance(v, str):
            s = v.strip()
            if s and (s[0] in '{[') and (s[-1] in '}]'):
                import json as _json
                return _json.loads(s)
        return v
    except Exception:
        return v


def _norm_book_level(level):
    """Normalize one orderbook level to (price, qty) floats.

    Supported shapes:
      - [price, qty]
      - {price: .., size: ..}
      - "price,qty" or "price:qty" strings
    """
    try:
        if level is None:
            return None

        # Levels as strings: "p,q" or "p: q"
        if isinstance(level, str):
            s = level.strip()
            if ',' in s:
                a, b = s.split(',', 1)
                p = _to_float(a)
                q = _to_float(b)
                if p is None or q is None or p <= 0 or q <= 0:
                    return None
                return (float(p), float(q))
            if ':' in s:
                a, b = s.split(':', 1)
                p = _to_float(a)
                q = _to_float(b)
                if p is None or q is None or p <= 0 or q <= 0:
                    return None
                return (float(p), float(q))
            return None

        # Levels as list/tuple
        if isinstance(level, (list, tuple)):
            if len(level) < 2:
                return None
            p = _to_float(level[0])
            q = _to_float(level[1])
            if p is None or q is None or p <= 0 or q <= 0:
                return None
            return (float(p), float(q))

        # Levels as dict
        if isinstance(level, dict):
            # common key variants
            p = _to_float(level.get("price") or level.get("p") or level.get("px"))
            q = _to_float(level.get("size") or level.get("qty") or level.get("q") or level.get("amount"))
            if p is None or q is None or p <= 0 or q <= 0:
                return None
            return (float(p), float(q))

        return None
    except Exception:
        return None


def _normalize_book_side(side: Any, *, is_bids: bool) -> List[Tuple[float, float]]:
    """Normalize one side of the orderbook to sorted (price, qty) tuples."""
    out: List[Tuple[float, float]] = []
    try:
        if not isinstance(side, list):
            return out
        for lvl in side:
            t = _norm_book_level(lvl)
            if t is None:
                continue
            out.append(t)
        # sort: bids desc, asks asc
        out.sort(key=lambda x: x[0], reverse=bool(is_bids))
        return out
    except Exception:
        return out


# ---------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------

def _cfg(name: str, default: Any) -> Any:
    """Prefer settings.py attribute if present, else env/default."""
    try:
        import settings as _settings
        if hasattr(_settings, name):
            return getattr(_settings, name)
    except Exception:
        pass
    return default


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip() == "1"


def _env_int(name: str, default: str) -> int:
    try:
        return int(str(os.getenv(name, default)).strip())
    except Exception:
        return int(default)


def _env_float(name: str, default: str) -> float:
    try:
        return float(str(os.getenv(name, default)).strip())
    except Exception:
        return float(default)


# ---------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------

BITGET_BASE_URL = str(_cfg("BITGET_BASE_URL", os.getenv("BITGET_BASE_URL", "https://api.bitget.com"))).rstrip("/")
INST_BITGET_PRODUCT_TYPE = str(_cfg("INST_BITGET_PRODUCT_TYPE", os.getenv("INST_BITGET_PRODUCT_TYPE", "usdt-futures"))).strip()

INST_TRACE_HTTP = bool(_cfg("INST_TRACE_HTTP", _env_flag("INST_TRACE_HTTP", "0")))
INST_DEBUG = bool(_cfg("INST_DEBUG", _env_flag("INST_DEBUG", "0")))
INST_HTTP_TIMEOUT_S = float(_cfg("INST_HTTP_TIMEOUT_S", _env_float("INST_HTTP_TIMEOUT_S", "8.0")))

INST_USE_WS_HUB = bool(_cfg("INST_USE_WS_HUB", str(os.getenv("INST_USE_WS_HUB", "1")).strip() == "1"))
INST_WS_HUB_URL = str(_cfg("INST_WS_HUB_URL", os.getenv("INST_WS_HUB_URL", "ws://127.0.0.1:8765"))).strip()

INST_ENABLE_FUNDING_HIST = bool(_cfg("INST_ENABLE_FUNDING_HIST", str(os.getenv("INST_ENABLE_FUNDING_HIST", "1")).strip() == "1"))
INST_ENABLE_CURRENT_FUNDING = bool(_cfg("INST_ENABLE_CURRENT_FUNDING", str(os.getenv("INST_ENABLE_CURRENT_FUNDING", "1")).strip() == "1"))
INST_ENABLE_OPEN_INTEREST = bool(_cfg("INST_ENABLE_OPEN_INTEREST", str(os.getenv("INST_ENABLE_OPEN_INTEREST", "1")).strip() == "1"))
INST_OI_FALLBACK_WHEN_WS_MISSING = bool(_cfg("INST_OI_FALLBACK_WHEN_WS_MISSING", str(os.getenv("INST_OI_FALLBACK_WHEN_WS_MISSING", "1")).strip() == "1"))

INST_DERIVED_ENABLED = bool(_cfg("INST_DERIVED_ENABLED", str(os.getenv("INST_DERIVED_ENABLED", "1")).strip() == "1"))

# z-score windows & clamps
INST_NORM_WINDOW = int(_cfg("INST_NORM_WINDOW", _env_int("INST_NORM_WINDOW", "200")))
INST_NORM_MIN_SAMPLES = int(_cfg("INST_NORM_MIN_SAMPLES", _env_int("INST_NORM_MIN_SAMPLES", "30")))
INST_NORM_CLIP = float(_cfg("INST_NORM_CLIP", _env_float("INST_NORM_CLIP", "4.0")))

# Derived/Tape settings (kept for compatibility, even if not used heavily)
INST_TAPE_WINDOW = int(_cfg("INST_TAPE_WINDOW", _env_int("INST_TAPE_WINDOW", "120")))
INST_TAPE_SLOPE_POINTS = int(_cfg("INST_TAPE_SLOPE_POINTS", _env_int("INST_TAPE_SLOPE_POINTS", "12")))

# merge-depth parameters (v6 fix)
INST_MERGE_DEPTH_LIMIT = int(_cfg("INST_MERGE_DEPTH_LIMIT", _env_int("INST_MERGE_DEPTH_LIMIT", "50")))
INST_MERGE_DEPTH_PRECISION_PRIMARY = str(_cfg("INST_MERGE_DEPTH_PRECISION_PRIMARY", os.getenv("INST_MERGE_DEPTH_PRECISION_PRIMARY", "scale0"))).strip()
INST_MERGE_DEPTH_PRECISION_RETRY = str(_cfg("INST_MERGE_DEPTH_PRECISION_RETRY", os.getenv("INST_MERGE_DEPTH_PRECISION_RETRY", "scale1"))).strip()


# ---------------------------------------------------------------------
# Simple in-process session cache
# ---------------------------------------------------------------------

_AIOHTTP_SESSION: Optional[aiohttp.ClientSession] = None

async def _get_session() -> aiohttp.ClientSession:
    global _AIOHTTP_SESSION
    if _AIOHTTP_SESSION is None or _AIOHTTP_SESSION.closed:
        timeout = aiohttp.ClientTimeout(total=float(INST_HTTP_TIMEOUT_S))
        _AIOHTTP_SESSION = aiohttp.ClientSession(timeout=timeout)
    return _AIOHTTP_SESSION


# ---------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------

async def _http_get(path: str, params: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
    url = f"{BITGET_BASE_URL}{path}"
    try:
        sess = await _get_session()
        if INST_TRACE_HTTP:
            LOGGER.info("[INST_HTTP_REQ] sym=%s attempt=0 url=%s params=%s", symbol, url, json.dumps(params, separators=(',', ':')))
        async with sess.get(url, params=params) as resp:
            status = resp.status
            try:
                data = await resp.json(content_type=None)
            except Exception:
                txt = await resp.text()
                if INST_DEBUG:
                    LOGGER.warning("[INST_HTTP_NONJSON] sym=%s status=%s path=%s body=%s", symbol, status, path, txt[:500])
                return None

        if not isinstance(data, dict):
            return None

        # normalize possible JSON-in-string in data field
        if "data" in data:
            data["data"] = _maybe_json_load(data.get("data"))

        if INST_TRACE_HTTP:
            LOGGER.info(
                "[INST_HTTP_RESP] sym=%s status=%s path=%s code=%s msg=%s",
                symbol,
                status,
                path,
                data.get("code"),
                data.get("msg"),
            )

        # Bitget success code is usually "00000"
        if str(data.get("code")) != "00000":
            if INST_DEBUG:
                LOGGER.warning("[INST_HTTP_BAD] sym=%s path=%s code=%s msg=%s", symbol, path, data.get("code"), data.get("msg"))
            return None

        return data
    except Exception as e:
        if INST_DEBUG:
            LOGGER.exception("[INST_HTTP_EXC] sym=%s path=%s err=%s", symbol, path, e)
        return None


# ---------------------------------------------------------------------
# WS hub client (optional)
# ---------------------------------------------------------------------

class _WSHubClient:
    def __init__(self, url: str):
        self.url = url
        self._ws = None
        self._lock = asyncio.Lock()
        self._last_ok = 0.0
        self._cache: Dict[str, Any] = {}

    async def _connect(self):
        import websockets  # type: ignore
        self._ws = await websockets.connect(self.url, ping_interval=10, ping_timeout=10)

    async def _ensure(self):
        async with self._lock:
            if self._ws is None:
                await self._connect()

    async def get(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            await self._ensure()
            if self._ws is None:
                return None
            req = {"op": "get", "symbol": symbol}
            await self._ws.send(json.dumps(req))
            raw = await asyncio.wait_for(self._ws.recv(), timeout=2.5)
            msg = json.loads(raw)
            if isinstance(msg, dict) and msg.get("ok") is True and isinstance(msg.get("data"), dict):
                self._cache[symbol] = msg["data"]
                self._last_ok = time.time()
                return msg["data"]
            return None
        except Exception:
            return None

    @property
    def ok_recent(self) -> bool:
        return (time.time() - float(self._last_ok)) < 60.0


_WS_HUB: Optional[_WSHubClient] = None

def _ws_hub() -> Optional[_WSHubClient]:
    global _WS_HUB
    if not INST_USE_WS_HUB:
        return None
    if _WS_HUB is None:
        _WS_HUB = _WSHubClient(INST_WS_HUB_URL)
    return _WS_HUB


# ---------------------------------------------------------------------
# Normalization store (z-score across time, per symbol)
# ---------------------------------------------------------------------

@dataclass
class _Series:
    dq: Deque[Tuple[int, float]] = field(default_factory=lambda: deque(maxlen=INST_NORM_WINDOW))

    def push(self, ts_ms: int, v: float):
        self.dq.append((int(ts_ms), float(v)))

    def values(self) -> List[float]:
        return [x[1] for x in self.dq]

    def value_at_or_before(self, ts_ms: int) -> Optional[Tuple[int, float]]:
        last = None
        for t, v in self.dq:
            if t <= ts_ms:
                last = (t, v)
            else:
                break
        return last


@dataclass
class _NormState:
    ob_imb: _Series = field(default_factory=_Series)
    spread_bps: _Series = field(default_factory=_Series)
    depth_25: _Series = field(default_factory=_Series)
    oi: _Series = field(default_factory=_Series)
    funding: _Series = field(default_factory=_Series)
    tape_5m: _Series = field(default_factory=_Series)

_DERIVED: Dict[str, _NormState] = {}

def _derived_state(sym: str) -> _NormState:
    if sym not in _DERIVED:
        _DERIVED[sym] = _NormState()
    return _DERIVED[sym]


def _zscore(vals: List[float], x: float) -> Optional[float]:
    try:
        if np is None:
            return None
        if len(vals) < int(INST_NORM_MIN_SAMPLES):
            return None
        arr = np.asarray(vals, dtype=float)
        mu = float(np.nanmean(arr))
        sd = float(np.nanstd(arr))
        if not (sd > 0):
            return None
        z = (float(x) - mu) / sd
        # clamp
        clip = float(INST_NORM_CLIP)
        if clip > 0:
            z = float(max(-clip, min(clip, z)))
        return float(z)
    except Exception:
        return None


def _norm_update(sym: str, key: str, v: Optional[float]) -> Optional[float]:
    try:
        if v is None:
            return None
        st = _derived_state(sym)
        ts_ms = int(time.time() * 1000)
        series = getattr(st, key, None)
        if series is None:
            return None
        series.push(ts_ms, float(v))
        z = _zscore(series.values(), float(v))
        return z
    except Exception:
        return None


def _pct_change(series: _Series, now_ts_ms: int, horizon_s: float) -> Optional[float]:
    try:
        old = series.value_at_or_before(int(now_ts_ms - int(horizon_s * 1000)))
        if old is None:
            return None
        _, v0 = old
        vals = series.values()
        if not vals:
            return None
        v1 = float(vals[-1])
        if v0 == 0.0:
            return None
        return float((v1 - float(v0)) / float(v0))
    except Exception:
        return None


def _delta(series: _Series, now_ts_ms: int, horizon_s: float) -> Optional[float]:
    try:
        old = series.value_at_or_before(int(now_ts_ms - int(horizon_s * 1000)))
        if old is None:
            return None
        _, v0 = old
        vals = series.values()
        if not vals:
            return None
        v1 = float(vals[-1])
        return float(v1 - float(v0))
    except Exception:
        return None


def _series_slope_per_min(series: _Series, points: int = 12) -> Optional[float]:
    try:
        if np is None:
            return None
        pts = list(series.dq)[-int(points):]
        if len(pts) < 3:
            return None
        xs = []
        ys = []
        t0 = pts[0][0]
        for t, v in pts:
            xs.append((t - t0) / 60000.0)  # minutes
            ys.append(float(v))
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        a = float(np.polyfit(x, y, 1)[0])
        return a
    except Exception:
        return None


# ---------------------------------------------------------------------
# Bitget public endpoint fetchers (tolerant but strict where necessary)
# ---------------------------------------------------------------------

async def _fetch_merge_depth(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch merge-depth with enforced precision/limit (v6 fix).

    Bitget merge-depth can return empty/non-usable payload unless precision+limit are provided.
    We strictly require non-empty bids+asks after normalization.

    Retry policy:
      1) precision=PRIMARY (default scale0), limit=INST_MERGE_DEPTH_LIMIT
      2) precision=RETRY   (default scale1), limit=INST_MERGE_DEPTH_LIMIT   (only if #1 unusable)
    """
    tries = [
        (INST_MERGE_DEPTH_PRECISION_PRIMARY, int(INST_MERGE_DEPTH_LIMIT)),
        (INST_MERGE_DEPTH_PRECISION_RETRY, int(INST_MERGE_DEPTH_LIMIT)),
    ]

    for i, (precision, limit) in enumerate(tries):
        data = await _http_get(
            "/api/v2/mix/market/merge-depth",
            params={
                "productType": INST_BITGET_PRODUCT_TYPE,
                "symbol": symbol,
                "precision": str(precision),
                "limit": str(int(limit)),
            },
            symbol=symbol,
        )
        if not isinstance(data, dict):
            continue

        raw = data.get("data")
        d = None
        if isinstance(raw, dict):
            d = raw
        elif isinstance(raw, list) and raw and isinstance(raw[0], dict):
            d = raw[0]

        if not isinstance(d, dict):
            if INST_DEBUG:
                LOGGER.warning("[INST_DEPTH_PARSE] sym=%s try=%s reason=no_data_dict", symbol, i)
            continue

        bids = _maybe_json_load(d.get("bids"))
        asks = _maybe_json_load(d.get("asks"))

        if not isinstance(bids, list) or not isinstance(asks, list):
            if INST_DEBUG:
                LOGGER.warning("[INST_DEPTH_PARSE] sym=%s try=%s reason=bids_asks_not_list", symbol, i)
            continue

        nbids = _normalize_book_side(bids, is_bids=True)
        nasks = _normalize_book_side(asks, is_bids=False)

        if not nbids or not nasks:
            # only retry on first attempt
            if INST_DEBUG:
                LOGGER.warning(
                    "[INST_DEPTH_EMPTY] sym=%s try=%s precision=%s limit=%s nbids=%s nasks=%s",
                    symbol, i, precision, limit, len(nbids), len(nasks)
                )
            continue

        return {
            "bids": nbids,
            "asks": nasks,
            "precision_used": str(precision),
            "limit_used": int(limit),
        }

    return None


async def _fetch_current_funding_rate(symbol: str) -> Tuple[Optional[float], Optional[int]]:
    """Return (funding_rate, next_funding_time_ms)."""
    data = await _http_get(
        "/api/v2/mix/market/current-fund-rate",
        params={"productType": INST_BITGET_PRODUCT_TYPE, "symbol": symbol},
        symbol=symbol,
    )
    if not isinstance(data, dict):
        return None, None

    raw = data.get("data")
    d0 = None
    if isinstance(raw, list) and raw:
        if isinstance(raw[0], dict):
            d0 = raw[0]
    elif isinstance(raw, dict):
        d0 = raw

    if not isinstance(d0, dict):
        if INST_DEBUG:
            LOGGER.warning("[INST_FUND_PARSE] sym=%s reason=no_data", symbol)
        return None, None

    fr = _to_float(d0.get("fundingRate") or d0.get("funding_rate"))
    nft = _to_float(d0.get("nextUpdateTime") or d0.get("nextFundingTime") or d0.get("next_funding_time"))
    next_ms = int(nft) if nft is not None else None

    if fr is None and INST_DEBUG:
        LOGGER.warning("[INST_FUND_PARSE] sym=%s reason=no_fundingRate keys=%s", symbol, list(d0.keys())[:20])

    return (float(fr) if fr is not None else None), next_ms


async def _fetch_funding_hist(symbol: str, limit: int = 50) -> Tuple[Optional[float], Optional[float]]:
    """Return (mean, std) of historical funding rate values."""
    if not INST_ENABLE_FUNDING_HIST:
        return None, None

    data = await _http_get(
        "/api/v2/mix/market/history-fund-rate",
        params={"productType": INST_BITGET_PRODUCT_TYPE, "symbol": symbol, "pageSize": str(int(limit))},
        symbol=symbol,
    )
    if not isinstance(data, dict):
        return None, None

    raw = data.get("data")
    if not isinstance(raw, list) or not raw:
        return None, None

    vals: List[float] = []
    for it in raw[: int(limit)]:
        if not isinstance(it, dict):
            continue
        fr = _to_float(it.get("fundingRate") or it.get("funding_rate"))
        if fr is None:
            continue
        vals.append(float(fr))

    if not vals:
        return None, None

    if np is None:
        # fallback (no numpy): simple mean, std naive
        mu = sum(vals) / max(1, len(vals))
        var = sum((v - mu) ** 2 for v in vals) / max(1, len(vals))
        return float(mu), float(var ** 0.5)

    arr = np.asarray(vals, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


async def _fetch_open_interest(symbol: str) -> Optional[float]:
    """Return open interest size (float)."""
    if not INST_ENABLE_OPEN_INTEREST:
        return None

    data = await _http_get(
        "/api/v2/mix/market/open-interest",
        params={"productType": INST_BITGET_PRODUCT_TYPE, "symbol": symbol},
        symbol=symbol,
    )
    if not isinstance(data, dict):
        return None

    raw = data.get("data")

    # observed shapes:
    # - dict { openInterestList: [ {...} ] }
    # - list [ {...} ]
    # - dict with 'data' list (rare)
    arr = None
    if isinstance(raw, dict):
        arr = raw.get("openInterestList") or raw.get("data") or raw.get("list")
    elif isinstance(raw, list):
        arr = raw

    if not isinstance(arr, list) or not arr:
        if INST_DEBUG:
            LOGGER.warning("[INST_OI_PARSE] sym=%s reason=empty_list", symbol)
        return None

    d0 = arr[0] if isinstance(arr[0], dict) else None
    if not isinstance(d0, dict):
        return None

    size = _to_float(d0.get("size") or d0.get("openInterest") or d0.get("open_interest"))
    if size is None and INST_DEBUG:
        LOGGER.warning("[INST_OI_PARSE] sym=%s reason=no_size keys=%s", symbol, list(d0.keys())[:20])

    return float(size) if size is not None else None


# ---------------------------------------------------------------------
# Orderbook metrics
# ---------------------------------------------------------------------

def _mid_from_book(book: Dict[str, Any]) -> Optional[float]:
    try:
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        if not bids or not asks:
            return None
        bb = float(bids[0][0])
        aa = float(asks[0][0])
        if bb <= 0 or aa <= 0:
            return None
        return (bb + aa) / 2.0
    except Exception:
        return None


def _spread_bps(book: Dict[str, Any]) -> Optional[float]:
    try:
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        if not bids or not asks:
            return None
        bb = float(bids[0][0])
        aa = float(asks[0][0])
        mid = (bb + aa) / 2.0
        if mid <= 0:
            return None
        return float(((aa - bb) / mid) * 10000.0)
    except Exception:
        return None


def _depth_usd_25bps(book: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute depth within ±25bps around mid, return (total_usd, bid_usd, ask_usd)."""
    try:
        mid = _mid_from_book(book)
        if mid is None or mid <= 0:
            return None, None, None

        lo = mid * (1.0 - 0.0025)
        hi = mid * (1.0 + 0.0025)

        bids = book.get("bids") or []
        asks = book.get("asks") or []

        bid_notional = 0.0
        for p, q in bids:
            if float(p) < lo:
                break
            bid_notional += float(p) * float(q)

        ask_notional = 0.0
        for p, q in asks:
            if float(p) > hi:
                break
            ask_notional += float(p) * float(q)

        total = bid_notional + ask_notional
        return float(total), float(bid_notional), float(ask_notional)
    except Exception:
        return None, None, None


def _orderbook_imbalance_25bps(book: Dict[str, Any]) -> Optional[float]:
    """Return (bid_depth - ask_depth) / (bid_depth + ask_depth) within ±25bps of mid."""
    try:
        total, bid, ask = _depth_usd_25bps(book)
        if total is None or total <= 0 or bid is None or ask is None:
            return None
        denom = float(bid) + float(ask)
        if denom <= 0:
            return None
        return float((float(bid) - float(ask)) / denom)
    except Exception:
        return None


# ---------------------------------------------------------------------
# Funding regimes / crowding
# ---------------------------------------------------------------------

def _classify_funding(fr: Optional[float], z: Optional[float] = None) -> str:
    if fr is None and z is None:
        return "unknown"
    x = fr if fr is not None else 0.0
    # coarse buckets
    if abs(x) < 0.00005:
        return "flat"
    if x > 0:
        return "positive"
    return "negative"


def _classify_orderbook(ob_imb: Optional[float]) -> str:
    if ob_imb is None:
        return "unknown"
    if ob_imb > 0.2:
        return "bid_dominant"
    if ob_imb < -0.2:
        return "ask_dominant"
    return "balanced"


def _classify_flow(tape: Optional[float], z: Optional[float]) -> str:
    # tape is optional in this pack
    if tape is None and z is None:
        return "unknown"
    return "neutral"


def _classify_crowding(fr: Optional[float], z: Optional[float]) -> str:
    if fr is None and z is None:
        return "unknown"
    # heuristic: funding extreme => crowded
    x = abs(fr) if fr is not None else 0.0
    if x >= 0.001:
        return "overcrowded"
    if x >= 0.0005:
        return "crowded"
    return "normal"


# ---------------------------------------------------------------------
# Quality scoring (meta)
# ---------------------------------------------------------------------

def _quality_assessment(
    *,
    ok_count: int,
    ws_used: bool,
    depth_ok: bool,
    funding_ok: bool,
    oi_ok: bool,
    warnings: List[str],
) -> Tuple[int, List[str]]:
    flags: List[str] = []
    score = 100

    if not depth_ok:
        flags.append("no_depth")
        score -= 40
    if not funding_ok:
        flags.append("no_funding")
        score -= 25
    if not oi_ok:
        flags.append("no_oi")
        score -= 25

    if ws_used:
        flags.append("ws_used")
        score += 5

    if ok_count <= 0:
        flags.append("no_components")
        score -= 20

    for w in warnings:
        if w:
            flags.append(str(w))

    score = max(0, min(100, int(score)))
    return score, flags


# ---------------------------------------------------------------------
# Symbol normalize (keep legacy name `binance_symbol`)
# ---------------------------------------------------------------------

def _normalize_symbol(symbol: str) -> str:
    s = str(symbol or "").strip().upper()
    s = s.replace("-", "").replace("_", "")
    s = s.replace("PERP", "")
    # Bitget uses e.g. BTCUSDT, keep it
    return s


# ---------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------

async def compute_full_institutional_analysis(
    symbol: str,
    *,
    bias: Optional[str] = None,
    mode: str = "LIGHT",
    require_liquidations: bool = False,
    version: str = "UltraDesk3.2-bitget-only+ws-hub-max+funding-fallback+oi-optin+derived-2026-01-19",
) -> Dict[str, Any]:
    sym = _normalize_symbol(symbol)
    mode = str(mode or "LIGHT").upper()
    warnings: List[str] = []

    ws_used = False
    ws_payload = None

    if INST_USE_WS_HUB:
        hub = _ws_hub()
        if hub is None:
            pass
        else:
            try:
                ws_payload = await hub.get(sym)
                if ws_payload:
                    ws_used = True
            except Exception:
                ws_payload = None
                ws_used = False

    # REST orderbook (mandatory for inst availability)
    depth = await _fetch_merge_depth(sym)

    # Funding
    funding_rate = None
    next_funding_time_ms = None
    funding_mean = None
    funding_std = None
    funding_z = None

    if INST_ENABLE_CURRENT_FUNDING:
        fr, nft = await _fetch_current_funding_rate(sym)
        funding_rate = fr
        next_funding_time_ms = nft

    if INST_ENABLE_FUNDING_HIST and (mode in ("NORMAL", "FULL")):
        try:
            mu, sd = await _fetch_funding_hist(sym, limit=50)
            funding_mean = mu
            funding_std = sd
            if funding_rate is not None and mu is not None and sd is not None and sd > 0:
                funding_z = float((float(funding_rate) - float(mu)) / float(sd))
        except Exception:
            pass

    # Open interest
    oi_value = None
    if INST_ENABLE_OPEN_INTEREST:
        oi_value = await _fetch_open_interest(sym)
    elif (not ws_used) and INST_OI_FALLBACK_WHEN_WS_MISSING:
        oi_value = await _fetch_open_interest(sym)

    # Derived: keep time series if enabled
    snap_ts_ms = int(time.time() * 1000)
    oi_change_15m_pct = None
    oi_change_1h_pct = None
    funding_change_1h = None
    funding_flip = None

    if INST_DERIVED_ENABLED:
        try:
            st = _derived_state(sym)
            if oi_value is not None:
                st.oi.push(snap_ts_ms, float(oi_value))
            if funding_rate is not None:
                st.funding.push(snap_ts_ms, float(funding_rate))

            oi_change_15m_pct = _pct_change(st.oi, snap_ts_ms, horizon_s=900.0)
            oi_change_1h_pct = _pct_change(st.oi, snap_ts_ms, horizon_s=3600.0)
            funding_change_1h = _delta(st.funding, snap_ts_ms, horizon_s=3600.0)

            if funding_rate is not None:
                old = st.funding.value_at_or_before(int(snap_ts_ms - 3600_000))
                if old is not None:
                    _, fr0 = old
                    fr1 = float(funding_rate)
                    if fr0 == 0.0:
                        funding_flip = None
                    else:
                        funding_flip = bool((float(fr0) > 0 and fr1 < 0) or (float(fr0) < 0 and fr1 > 0))
        except Exception:
            pass

    # Orderbook metrics
    ob_25 = None
    spread_bps = None
    depth_usd_25 = None
    depth_bid_usd_25 = None
    depth_ask_usd_25 = None

    if depth is not None:
        ob_25 = _orderbook_imbalance_25bps(depth)
        spread_bps = _spread_bps(depth)
        t, b, a = _depth_usd_25bps(depth)
        depth_usd_25, depth_bid_usd_25, depth_ask_usd_25 = t, b, a

    # z updates
    ob_imb_z = _norm_update(sym, "ob_imb", ob_25)
    spread_bps_z = _norm_update(sym, "spread_bps", spread_bps)
    depth_25_z = _norm_update(sym, "depth_25", depth_usd_25)
    oi_z = _norm_update(sym, "oi", oi_value) if oi_value is not None else None
    funding_z2 = _norm_update(sym, "funding", funding_rate) if funding_rate is not None else None

    funding_regime = _classify_funding(funding_rate, z=funding_z if funding_z is not None else funding_z2)
    ob_regime = _classify_orderbook(ob_25)
    flow_regime = _classify_flow(None, None)
    crowding_regime = _classify_crowding(funding_rate, funding_z2 if funding_z2 is not None else funding_z)

    # score components (simple availability gating)
    components = {"flow": 0, "oi": 0, "crowding": 0, "orderbook": 0}
    score = 0

    if ob_25 is not None:
        components["orderbook"] = 1
        score += 1

    if funding_rate is not None or funding_z is not None:
        components["crowding"] = 1
        score += 1

    if oi_value is not None:
        components["oi"] = 1
        score += 1

    score = max(0, min(4, int(score)))
    ok_count = int(sum(1 for k, v in components.items() if int(v) > 0))

    quality_score, quality_flags = _quality_assessment(
        ok_count=int(ok_count),
        ws_used=bool(ws_used),
        depth_ok=bool(depth is not None),
        funding_ok=bool((funding_rate is not None) or (funding_z is not None)),
        oi_ok=bool(oi_value is not None),
        warnings=warnings,
    )

    payload: Dict[str, Any] = {
        "institutional_score": int(score),
        "institutional_score_raw": int(score),
        "institutional_score_v2": int(score),
        "institutional_score_v3": int(score),

        # legacy key name kept for compatibility with existing code
        "binance_symbol": sym,

        # availability: must have at least one real institutional signal source
        "available": bool(depth is not None or ws_used or (funding_rate is not None) or (oi_value is not None)),

        # Open Interest
        "oi": oi_value,
        "openInterest": oi_value,  # legacy alias
        "oi_slope": oi_change_1h_pct,
        "oi_change_15m_pct": oi_change_15m_pct,
        "oi_change_1h_pct": oi_change_1h_pct,

        # Funding
        "funding_rate": funding_rate,
        "fundingRate": funding_rate,  # legacy alias
        "funding_regime": funding_regime,
        "funding_mean": funding_mean,
        "funding_std": funding_std,
        "funding_z": funding_z,
        "next_funding_time_ms": next_funding_time_ms,
        "funding_change_1h": funding_change_1h,
        "funding_flip": funding_flip,

        # Orderbook / Liquidity
        "orderbook_imb_25bps": ob_25,
        "orderbook_imb_25bps_z": ob_imb_z,
        "spread_bps": spread_bps,
        "spread_bps_z": spread_bps_z,
        "depth_usd_25bps": depth_usd_25,
        "depth_bid_usd_25bps": depth_bid_usd_25,
        "depth_ask_usd_25bps": depth_ask_usd_25,
        "depth_usd_25bps_z": depth_25_z,
        "orderbook_regime": ob_regime,

        # Flow placeholders
        "tape_delta_5m": None,
        "tape_regime": flow_regime,
        "cvd_notional_5m": None,
        "cvd_slope": None,

        # Regimes
        "crowding_regime": crowding_regime,

        # Scoring meta
        "score_components": components,
        "ok_count": int(ok_count),
        "gate": int(ok_count),  # same scale as ok_count for caller
        "quality_score": int(quality_score),
        "quality_flags": list(quality_flags),

        # Diagnostics
        "sources": {
            "depth": "bitget_rest" if depth is not None else None,
            "funding_rate": "bitget_rest" if funding_rate is not None else None,
            "next_funding_time": "bitget_rest" if next_funding_time_ms is not None else None,
            "oi": "bitget_rest" if oi_value is not None else None,
            "ws": "ws_hub" if ws_used else None,
        },

        "warnings": warnings,
        "version": version,
        "mode": mode,
        "bias": bias,
    }

    return payload
