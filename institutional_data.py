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

INST_VERSION = "UltraDesk3.2-bitget-only+ws-hub-max+funding-fallback+oi-optin+derived-2026-01-19"

BITGET_API_BASE = str(os.getenv("BITGET_API_BASE", "https://api.bitget.com")).strip()

# ---------------------------------------------------------------------
# Optional: prefer settings.py as source of truth (prevents split-brain defaults)
# ---------------------------------------------------------------------
try:
    import settings as _SETTINGS  # type: ignore
except Exception:  # pragma: no cover
    _SETTINGS = None  # type: ignore


def _cfg(name: str, default: Any) -> Any:
    """Return settings.NAME if available, else default."""
    try:
        if _SETTINGS is not None and hasattr(_SETTINGS, name):
            return getattr(_SETTINGS, name)
    except Exception:
        pass
    return default


# ---------------------------------------------------------------------
# Debug flags
# ---------------------------------------------------------------------
INST_DEBUG = str(os.getenv("INST_DEBUG", "0")).strip() == "1"
INST_TRACE_HTTP = str(os.getenv("INST_TRACE_HTTP", "0")).strip() == "1"
INST_TRACE_WS = str(os.getenv("INST_TRACE_WS", "0")).strip() == "1"
INST_TRACE_PAYLOAD = str(os.getenv("INST_TRACE_PAYLOAD", "0")).strip() == "1"

# ---------------------------------------------------------------------
# Optional external WS hub (read-only) — expected to be Bitget-based
# ---------------------------------------------------------------------
_WS_HUB = None
try:
    from institutional_ws_hub import HUB as _WS_HUB  # type: ignore
except Exception:
    _WS_HUB = None

INST_USE_WS_HUB = bool(_cfg("INST_USE_WS_HUB", str(os.getenv("INST_USE_WS_HUB", "1")).strip() == "1"))
WS_STALE_SEC = float(_cfg("WS_STALE_SEC", float(os.getenv("INST_WS_STALE_SEC", "15"))))

# ---------------------------------------------------------------------
# Institutional MAX additions
# - Prefer WS hub for depth/spread when available (reduces REST load + improves freshness).
# - Compute flow/microstructure regimes from WS tape + derived series.
# ---------------------------------------------------------------------
INST_DEPTH_USE_WS_IF_AVAILABLE = bool(_cfg("INST_DEPTH_USE_WS_IF_AVAILABLE", str(os.getenv("INST_DEPTH_USE_WS_IF_AVAILABLE", "1")).strip() == "1"))
# In FULL mode you may want REST depth always (true orderbook imbalance at band); keep enabled by default.
INST_DEPTH_REST_REQUIRED_IN_FULL = bool(_cfg("INST_DEPTH_REST_REQUIRED_IN_FULL", str(os.getenv("INST_DEPTH_REST_REQUIRED_IN_FULL", "1")).strip() == "1"))

# Flow regime thresholds (z-score on tape_delta_5m and/or absolute USDT notional)
INST_FLOW_Z_STRONG = float(os.getenv("INST_FLOW_Z_STRONG", "1.0"))
INST_FLOW_Z_EXTREME = float(os.getenv("INST_FLOW_Z_EXTREME", "2.0"))
INST_FLOW_ABS_STRONG_USDT = float(os.getenv("INST_FLOW_ABS_STRONG_USDT", "50000"))
INST_FLOW_ABS_EXTREME_USDT = float(os.getenv("INST_FLOW_ABS_EXTREME_USDT", "150000"))

# Crowding thresholds (funding z-score, realtime)
INST_CROWD_Z_STRONG = float(os.getenv("INST_CROWD_Z_STRONG", "1.5"))
INST_CROWD_Z_EXTREME = float(os.getenv("INST_CROWD_Z_EXTREME", "2.5"))

# Derived slope configuration
INST_TAPE_SLOPE_POINTS = int(float(os.getenv("INST_TAPE_SLOPE_POINTS", "12")))  # number of samples

# ---------------------------------------------------------------------
# Product type (Bitget v2 uses lower-case values in request examples)
# USDT-M Futures: "usdt-futures"
# ---------------------------------------------------------------------
INST_BITGET_PRODUCT_TYPE = str(os.getenv("INST_BITGET_PRODUCT_TYPE", "usdt-futures")).strip()

# ---------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------
INST_MODE = str(_cfg("INST_MODE", os.getenv("INST_MODE", "LIGHT"))).upper().strip()
if INST_MODE not in ("LIGHT", "NORMAL", "FULL"):
    INST_MODE = "LIGHT"

# ---------------------------------------------------------------------
# Optional features
# - Current funding: enabled by default (fixes funding_rate_ok=False when WS hub isn’t running)
# - Open interest: opt-in (can add load on big scans)
# - OI fallback when WS missing: optional, still respects TTL cache
# ---------------------------------------------------------------------
INST_ENABLE_OPEN_INTEREST = bool(_cfg("INST_ENABLE_OPEN_INTEREST", str(os.getenv("INST_ENABLE_OPEN_INTEREST", "0")).strip() == "1"))
INST_ENABLE_CURRENT_FUNDING = bool(_cfg("INST_ENABLE_CURRENT_FUNDING", str(os.getenv("INST_ENABLE_CURRENT_FUNDING", "1")).strip() == "1"))
INST_OI_FALLBACK_WHEN_WS_MISSING = bool(_cfg("INST_OI_FALLBACK_WHEN_WS_MISSING", str(os.getenv("INST_OI_FALLBACK_WHEN_WS_MISSING", "0")).strip() == "1"))

# kept for compatibility (not implemented in this file)
INST_ENABLE_RECENT_FILLS = bool(_cfg("INST_ENABLE_RECENT_FILLS", str(os.getenv("INST_ENABLE_RECENT_FILLS", "0")).strip() == "1"))
INST_ENABLE_CANDLES = bool(_cfg("INST_ENABLE_CANDLES", str(os.getenv("INST_ENABLE_CANDLES", "0")).strip() == "1"))

# ---------------------------------------------------------------------
# Normalisation (rolling z-scores)
# ---------------------------------------------------------------------
INST_NORM_ENABLED = bool(_cfg("INST_NORM_ENABLED", str(os.getenv("INST_NORM_ENABLED", "1")).strip() == "1"))
INST_NORM_MIN_POINTS = int(_cfg("INST_NORM_MIN_POINTS", int(float(os.getenv("INST_NORM_MIN_POINTS", "20")))))
INST_NORM_WINDOW = int(_cfg("INST_NORM_WINDOW", int(float(os.getenv("INST_NORM_WINDOW", "120")))))
INST_NORM_CLIP = float(os.getenv("INST_NORM_CLIP", "5.0"))

# ---------------------------------------------------------------------
# Derived series (no extra endpoints)
# ---------------------------------------------------------------------
INST_DERIVED_ENABLED = str(os.getenv("INST_DERIVED_ENABLED", "1")).strip() == "1"
INST_DERIVED_MAX_AGE_S = float(os.getenv("INST_DERIVED_MAX_AGE_S", "10800"))  # 3h
INST_DERIVED_MAXLEN = int(float(os.getenv("INST_DERIVED_MAXLEN", "480")))     # ~8 min sampling @1s; or ~8h @60s

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

# ---------------------------------------------------------------------
# Shared session
# ---------------------------------------------------------------------
_SESSION: Optional[aiohttp.ClientSession] = None
_SESSION_LOCK = asyncio.Lock()

# ---------------------------------------------------------------------
# Small caches (reduce load when scanning many symbols)
# ---------------------------------------------------------------------
_FUNDING_CACHE: Dict[str, Tuple[float, float, Optional[int]]] = {}  # sym -> (funding_rate, ts_s, next_update_ms)
_OI_CACHE: Dict[str, Tuple[float, float]] = {}  # sym -> (oi_size, ts_s)

_FUNDING_TTL_S = float(os.getenv("INST_FUNDING_TTL_S", "45"))
_OI_TTL_S = float(os.getenv("INST_OI_TTL_S", "45"))


def _now_ms() -> int:
    return int(time.time() * 1000)


def _jex(x: Any, maxlen: int = 240) -> str:
    """safe short json-ish excerpt for logs"""
    try:
        s = json.dumps(x, ensure_ascii=False)
    except Exception:
        s = str(x)
    s = s.replace("\n", " ")
    return s[:maxlen]


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
        if INST_TRACE_HTTP:
            LOGGER.info("[INST_HTTP_SKIP] sym=%s blocked_until_ms=%s path=%s", symbol, st.until_ms, path)
        return None

    url = BITGET_API_BASE + path
    session = await _get_session()

    async with _HTTP_SEM:
        for attempt in range(0, _HTTP_RETRIES + 1):
            await _pace()
            try:
                if INST_TRACE_HTTP:
                    LOGGER.info("[INST_HTTP_REQ] sym=%s attempt=%s url=%s params=%s", symbol, attempt, url, _jex(params))

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

                    if INST_TRACE_HTTP:
                        code = None
                        msg = None
                        if isinstance(data, dict):
                            code = data.get("code")
                            msg = data.get("msg")
                        LOGGER.info("[INST_HTTP_RESP] sym=%s status=%s path=%s code=%s msg=%s", symbol, status, path, code, msg)

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

                    # Bitget typically returns {code:"00000", msg:"success", data:...}
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
        if INST_TRACE_WS:
            LOGGER.info("[INST_WS] sym=%s hub_disabled=%s hub_none=%s", symbol, (not INST_USE_WS_HUB), (_WS_HUB is None))
        return None
    try:
        if not _ws_hub_running():
            if INST_TRACE_WS:
                LOGGER.info("[INST_WS] sym=%s hub_not_running", symbol)
            return None

        snap = _WS_HUB.get_snapshot(symbol)
        if not isinstance(snap, dict) or not snap.get("available"):
            if INST_TRACE_WS:
                LOGGER.info("[INST_WS] sym=%s snap_unavailable snap=%s", symbol, _jex(snap))
            return None

        ts = snap.get("ts")
        if ts is None:
            if INST_TRACE_WS:
                LOGGER.info("[INST_WS] sym=%s snap_missing_ts snap=%s", symbol, _jex(snap))
            return None

        age = (time.time() - float(ts))
        if age > float(WS_STALE_SEC):
            if INST_TRACE_WS:
                LOGGER.info("[INST_WS] sym=%s snap_stale age=%.3fs stale_sec=%.3f", symbol, age, float(WS_STALE_SEC))
            return None

        if INST_TRACE_WS:
            LOGGER.info("[INST_WS] sym=%s snap_used age=%.3fs keys=%s", symbol, age, list(snap.keys()))
        return snap
    except Exception as e:
        if INST_TRACE_WS:
            LOGGER.info("[INST_WS] sym=%s snap_error=%s", symbol, e)
        return None


# =====================================================================
# Metrics helpers (orderbook)
# =====================================================================
def _compute_orderbook_band_metrics(
    depth: Dict[str, Any], band_bps: float = 25.0
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
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
                try:
                    z = float(max(-float(INST_NORM_CLIP), min(float(INST_NORM_CLIP), float(z))))
                except Exception:
                    pass

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
# Derived series (OI/funding change without new endpoints)
# =====================================================================
@dataclass
class _Series:
    maxlen: int = INST_DERIVED_MAXLEN
    pts: Deque[Tuple[int, float]] = field(default_factory=lambda: deque(maxlen=INST_DERIVED_MAXLEN))

    def append(self, ts_ms: int, v: float) -> None:
        self.pts.append((int(ts_ms), float(v)))

    def purge_older_than(self, cutoff_ms: int) -> None:
        while self.pts and int(self.pts[0][0]) < int(cutoff_ms):
            self.pts.popleft()

    def last(self) -> Optional[Tuple[int, float]]:
        return self.pts[-1] if self.pts else None

    def value_at_or_before(self, ts_ms: int) -> Optional[Tuple[int, float]]:
        if not self.pts:
            return None
        for t, v in reversed(self.pts):
            if int(t) <= int(ts_ms):
                return int(t), float(v)
        return None


@dataclass
class _DerivedState:
    oi: _Series = field(default_factory=_Series)
    funding: _Series = field(default_factory=_Series)
    tape_5m: _Series = field(default_factory=_Series)
    spread_bps: _Series = field(default_factory=_Series)
_DERIVED: Dict[str, _DerivedState] = {}


def _derived_state(sym: str) -> _DerivedState:
    st = _DERIVED.get(sym)
    if st is None:
        st = _DerivedState()
        _DERIVED[sym] = st
    return st


def _update_derived(sym: str, ts_ms: int, *, oi: Optional[float], funding: Optional[float], tape_5m: Optional[float] = None, spread_bps: Optional[float] = None) -> None:
    if not INST_DERIVED_ENABLED:
        return
    if not sym:
        return
    st = _derived_state(sym)
    cutoff = int(ts_ms - (float(INST_DERIVED_MAX_AGE_S) * 1000.0))
    try:
        st.oi.purge_older_than(cutoff)
        st.funding.purge_older_than(cutoff)
        st.tape_5m.purge_older_than(cutoff)
        st.spread_bps.purge_older_than(cutoff)
    except Exception:
        pass

    try:
        if oi is not None:
            st.oi.append(ts_ms, float(oi))
    except Exception:
        pass

    try:
        if funding is not None:
            st.funding.append(ts_ms, float(funding))
    except Exception:
        pass

    try:
        if tape_5m is not None:
            st.tape_5m.append(ts_ms, float(tape_5m))
    except Exception:
        pass

    try:
        if spread_bps is not None:
            st.spread_bps.append(ts_ms, float(spread_bps))
    except Exception:
        pass


def _pct_change(series: _Series, now_ms: int, horizon_s: float) -> Optional[float]:
    try:
        if not series.pts:
            return None
        last = series.last()
        if last is None:
            return None
        _, v1 = last
        t0_target = int(now_ms - float(horizon_s) * 1000.0)
        old = series.value_at_or_before(t0_target)
        if old is None:
            return None
        _, v0 = old
        v0f = float(v0)
        if abs(v0f) <= 1e-12:
            return None
        return float(((float(v1) - v0f) / v0f) * 100.0)
    except Exception:
        return None


def _delta(series: _Series, now_ms: int, horizon_s: float) -> Optional[float]:
    try:
        if not series.pts:
            return None
        last = series.last()
        if last is None:
            return None
        _, v1 = last
        t0_target = int(now_ms - float(horizon_s) * 1000.0)
        old = series.value_at_or_before(t0_target)
        if old is None:
            return None
        _, v0 = old
        return float(float(v1) - float(v0))
    except Exception:
        return None


def _series_slope_per_min(series: _Series, points: int = 12) -> Optional[float]:
    """Return slope (units per minute) for the last `points` samples of a derived series.
    Uses numpy when available; falls back to simple delta/time.
    """
    try:
        pts = list(series.pts)[-max(2, int(points)):]
        if len(pts) < 2:
            return None
        # convert to minutes relative to first
        t0 = float(pts[0][0])
        xs = [(float(ts) - t0) / 60000.0 for ts, _ in pts]
        ys = [float(v) for _, v in pts]
        if np is not None and len(xs) >= 3:
            x = np.asarray(xs, dtype=float)
            y = np.asarray(ys, dtype=float)
            # slope of y = a*x + b
            a = float(np.polyfit(x, y, 1)[0])
            return a
        # fallback: delta / minutes
        dt = float(xs[-1] - xs[0])
        if dt <= 1e-9:
            return None
        return float((ys[-1] - ys[0]) / dt)
    except Exception:
        return None


# =====================================================================
# Bitget confirmed endpoints
# =====================================================================
async def _fetch_merge_depth(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch merge-depth with tolerant parsing.

    Observed shapes:
      - data: { bids: [...], asks: [...] }
      - data: [ { bids: [...], asks: [...] } ]
    """
    data = await _http_get(
        "/api/v2/mix/market/merge-depth",
        params={"productType": INST_BITGET_PRODUCT_TYPE, "symbol": symbol},
        symbol=symbol,
    )
    if not isinstance(data, dict):
        return None

    raw = data.get("data")
    d = None
    if isinstance(raw, dict):
        d = raw
    elif isinstance(raw, list) and raw and isinstance(raw[0], dict):
        d = raw[0]

    if not isinstance(d, dict):
        return None

    bids = d.get("bids")
    asks = d.get("asks")
    if not isinstance(bids, list) or not isinstance(asks, list):
        return None

    return {"bids": bids, "asks": asks}


async def _fetch_funding_history(symbol: str, limit: int = 30) -> Optional[List[Dict[str, Any]]]:
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


def _parse_next_update_ms(d0: Dict[str, Any]) -> Optional[int]:
    for k in ("nextUpdate", "nextUpdateTime", "nextFundingTime", "nextSettleTime", "next_settle_time", "next_funding_time"):
        if k in d0 and d0.get(k) is not None:
            try:
                return int(float(d0.get(k)))
            except Exception:
                continue
    return None


async def _fetch_current_funding_rate(symbol: str) -> Tuple[Optional[float], Optional[int]]:
    """Fetch current funding rate (Bitget v2) with tolerant parsing.

    Observed shapes across deployments:
      - data: [ { fundingRate: '...', nextFundingTime: '...' } ]
      - data: { fundingRate: '...', nextFundingTime: '...' }
      - data: [ { capitalRate: '...' } ]  (alias)
    """
    now_s = time.time()
    c = _FUNDING_CACHE.get(symbol)
    if c is not None:
        fr, ts_s, next_ms = c
        if (now_s - ts_s) <= float(_FUNDING_TTL_S):
            return float(fr), next_ms

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
        d0 = raw[0]
    elif isinstance(raw, dict):
        d0 = raw

    if not isinstance(d0, dict):
        return None, None

    fr = None
    for k in ("fundingRate", "capitalRate", "funding_rate"):
        if k in d0 and d0.get(k) is not None:
            try:
                fr = float(d0.get(k))
                break
            except Exception:
                fr = None

    next_ms = _parse_next_update_ms(d0)

    if fr is not None:
        _FUNDING_CACHE[symbol] = (float(fr), now_s, next_ms)

    return fr, next_ms


async def _fetch_open_interest(symbol: str) -> Optional[float]:
    """Fetch open interest (Bitget v2) with tolerant parsing.

    Observed shapes:
      - data: { openInterestList: [ { size: '...' } ] }
      - data: [ { size: '...' } ]
      - data: { size: '...' } or { openInterest: '...' }
    """
    now_s = time.time()
    c = _OI_CACHE.get(symbol)
    if c is not None:
        oi, ts_s = c
        if (now_s - ts_s) <= float(_OI_TTL_S):
            return float(oi)

    data = await _http_get(
        "/api/v2/mix/market/open-interest",
        params={"productType": INST_BITGET_PRODUCT_TYPE, "symbol": symbol},
        symbol=symbol,
    )
    if not isinstance(data, dict):
        return None

    raw = data.get("data")

    # Helper: extract a numeric OI from a dict
    def _extract_from_dict(d: dict) -> Optional[float]:
        if not isinstance(d, dict):
            return None
        # common keys
        for k in ("size", "openInterest", "open_interest", "holdingAmount", "holding", "amount", "oi"):
            if k in d and d.get(k) is not None:
                try:
                    v = float(d.get(k))
                    return v
                except Exception:
                    pass
        return None

    oi = None

    if isinstance(raw, dict):
        # official doc shape: openInterestList
        lst = raw.get("openInterestList")
        if isinstance(lst, list) and lst:
            oi = _extract_from_dict(lst[0])
        if oi is None:
            oi = _extract_from_dict(raw)

    elif isinstance(raw, list) and raw:
        # sometimes directly a list of dicts
        if isinstance(raw[0], dict) and "openInterestList" in raw[0]:
            lst = raw[0].get("openInterestList")
            if isinstance(lst, list) and lst:
                oi = _extract_from_dict(lst[0])
        if oi is None and isinstance(raw[0], dict):
            oi = _extract_from_dict(raw[0])

    if oi is not None:
        _OI_CACHE[symbol] = (float(oi), now_s)

    return oi


def _compute_funding_stats_bitget(hist: Optional[List[Dict[str, Any]]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
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
# MAIN API
# =====================================================================
def _normalize_symbol(s: str) -> str:
    x = (s or "").upper().strip()
    x = x.replace("-", "").replace("_", "")
    x = x.replace("PERP", "")
    return x


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


def _quality_assessment(
    *,
    ok_count: int,
    ws_used: bool,
    depth_ok: bool,
    funding_ok: bool,
    oi_ok: bool,
    warnings: List[str],
) -> Tuple[int, List[str]]:
    """Heuristic completeness score (0..100) for data reliability."""
    score = 100
    flags: List[str] = []

    if ok_count <= 0:
        score -= 60
        flags.append("no_components_ok")
    elif ok_count == 1:
        score -= 35
        flags.append("only_1_component_ok")
    elif ok_count == 2:
        score -= 15
        flags.append("only_2_components_ok")

    if not ws_used:
        score -= 10
        flags.append("ws_not_used")
    if not depth_ok:
        score -= 25
        flags.append("no_orderbook")
    if not funding_ok:
        score -= 15
        flags.append("no_funding")
    if not oi_ok:
        score -= 15
        flags.append("no_oi")

    if warnings:
        score -= min(20, 5 * len(warnings))
        flags.append("warnings_present")

    score = max(0, min(100, int(score)))
    return score, flags


def _available_components_list(payload: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    if payload.get("orderbook_imb_25bps") is not None:
        out.append("orderbook")
    if payload.get("spread_bps") is not None:
        out.append("spread")
    if payload.get("depth_usd_25bps") is not None:
        out.append("depth")
    if payload.get("funding_z") is not None:
        out.append("funding_hist")
    if payload.get("ws_snapshot_used"):
        out.append("ws_hub")
    if payload.get("normalization_enabled"):
        out.append("norm")
    if payload.get("oi") is not None:
        out.append("oi")
    if payload.get("funding_rate") is not None:
        out.append("funding")
    if payload.get("tape_delta_5m") is not None:
        out.append("tape")
    if payload.get("oi_change_1h_pct") is not None:
        out.append("oi_change_1h")
    if payload.get("funding_change_1h") is not None:
        out.append("funding_change_1h")
    return out


async def compute_full_institutional_analysis(
    symbol: str,
    bias: str,
    *,
    include_liquidations: bool = False,  # kept for signature compatibility
    mode: Optional[str] = None,
    ) -> Dict[str, Any]:
    _ = include_liquidations  # not used in bitget-only build

    bias = (bias or "").upper().strip()
    eff_mode = (mode or INST_MODE).upper().strip()
    if eff_mode not in ("LIGHT", "NORMAL", "FULL"):
        eff_mode = "LIGHT"

    sym = _normalize_symbol(symbol)
    warnings: List[str] = []
    sources: Dict[str, str] = {}

    if INST_DEBUG:
        LOGGER.info(
            "[INST_DIAG_START] sym=%s bias=%s mode=%s version=%s base=%s productType=%s ws_hub=%s",
            sym,
            bias,
            eff_mode,
            INST_VERSION,
            BITGET_API_BASE,
            INST_BITGET_PRODUCT_TYPE,
            INST_USE_WS_HUB,
        )

    ws_snap = _ws_snapshot(sym)
    ws_used = bool(ws_snap is not None)

    # WS hub extras (institutional): depth/spread/mid diagnostics
    ws_spread_bps: Optional[float] = None
    ws_bid_depth_usd: Optional[float] = None
    ws_ask_depth_usd: Optional[float] = None
    ws_mid: Optional[float] = None

    now_ms = _now_ms()
    snap_ts_ms = now_ms
    if isinstance(ws_snap, dict) and ws_snap.get("ts") is not None:
        try:
            snap_ts_ms = int(float(ws_snap.get("ts")) * 1000.0)
        except Exception:
            snap_ts_ms = now_ms

    # Depth source selection (institutional):
    # - If WS hub is available and enabled, we can skip REST depth in LIGHT/NORMAL to reduce load and improve freshness.
    # - In FULL mode, REST depth is still preferred to compute true band metrics unless disabled.
    depth: Optional[Dict[str, Any]] = None
    if not (INST_DEPTH_USE_WS_IF_AVAILABLE and ws_used and (eff_mode != "FULL" or (not INST_DEPTH_REST_REQUIRED_IN_FULL))):
        depth = await _fetch_merge_depth(sym)
        if depth is None:
            sources["depth"] = "none"
        else:
            sources["depth"] = "bitget_rest"
    else:
        # WS proxy mode (depth fields may still be available via hub snapshot)
        sources["depth"] = "ws_hub_proxy"

    if depth is None:
        if ws_used:
            warnings.append("depth_proxy_ws")
        else:
            warnings.append("no_depth")

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
    elif ws_used and (ws_bid_depth_usd is not None or ws_ask_depth_usd is not None or ws_spread_bps is not None):
        # WS proxy metrics: use top-of-book depth + spread if REST depth is skipped/unavailable.
        # These are not exact "25bps band" metrics, but they are directionally useful and much fresher.
        b25 = ws_bid_depth_usd
        a25 = ws_ask_depth_usd
        depth_bid_usd_25, depth_ask_usd_25 = b25, a25
        if b25 is not None and a25 is not None:
            depth_usd_25 = float(b25 + a25)

        if ws_spread_bps is not None:
            spread_bps = ws_spread_bps
        if ws_mid is not None:
            microprice = ws_mid

        # Proxy orderbook imbalance from depth asymmetry
        if b25 is not None and a25 is not None and (b25 + a25) > 0:
            ob_25 = float((b25 - a25) / (b25 + a25))
        else:
            ob_25 = None
    funding_mean = funding_std = funding_z = None
    if eff_mode == "FULL":
        hist = await _fetch_funding_history(sym, limit=30)
        if hist is None:
            warnings.append("no_funding_hist")
            sources["funding_hist"] = "none"
        else:
            funding_mean, funding_std, funding_z = _compute_funding_stats_bitget(hist)
            sources["funding_hist"] = "bitget_rest"

    funding_rate: Optional[float] = None
    next_funding_time_ms: Optional[int] = None
    tape_5m: Optional[float] = None
    oi_value: Optional[float] = None

    if ws_snap is not None:
        try:
            if ws_snap.get("funding_rate") is not None:
                funding_rate = float(ws_snap.get("funding_rate"))
                sources["funding_rate"] = "ws_hub"

            if ws_snap.get("next_funding_time_ms") is not None:
                next_funding_time_ms = int(float(ws_snap.get("next_funding_time_ms")))
                sources["next_funding_time"] = "ws_hub"

            if ws_snap.get("tape_delta_5m") is not None:
                tape_5m = float(ws_snap.get("tape_delta_5m"))
                sources["tape"] = "ws_hub"

            if ws_snap.get("open_interest") is not None:
                oi_value = float(ws_snap.get("open_interest"))
                sources["oi"] = "ws_hub"

            # WS depth/spread (if hub provides it)
            if ws_snap.get("spread_bps") is not None:
                ws_spread_bps = float(ws_snap.get("spread_bps"))
                sources["spread"] = "ws_hub"

            if ws_snap.get("bid_depth_usd") is not None:
                ws_bid_depth_usd = float(ws_snap.get("bid_depth_usd"))
                sources["depth_bid"] = "ws_hub"
            if ws_snap.get("ask_depth_usd") is not None:
                ws_ask_depth_usd = float(ws_snap.get("ask_depth_usd"))
                sources["depth_ask"] = "ws_hub"
            if ws_snap.get("mid") is not None:
                ws_mid = float(ws_snap.get("mid"))
                sources["mid"] = "ws_hub"

            qf = ws_snap.get("quality_flags")
            if isinstance(qf, list) and qf:
                # Keep as warnings for observability (not necessarily a veto)
                for x in qf[:6]:
                    try:
                        warnings.append(f"ws_{str(x)}")
                    except Exception:
                        pass
        except Exception:
            warnings.append("ws_parse_error")

    if funding_rate is None and INST_ENABLE_CURRENT_FUNDING:
        fr, nu = await _fetch_current_funding_rate(sym)
        if fr is not None:
            funding_rate = float(fr)
            sources["funding_rate"] = "bitget_rest"
        else:
            warnings.append("no_current_funding")
            sources["funding_rate"] = sources.get("funding_rate", "none")

        if nu is not None:
            next_funding_time_ms = int(nu)
            sources["next_funding_time"] = sources.get("next_funding_time", "bitget_rest")

    need_oi_rest = bool(INST_ENABLE_OPEN_INTEREST) or (INST_OI_FALLBACK_WHEN_WS_MISSING and (oi_value is None))
    if oi_value is None and need_oi_rest:
        oi = await _fetch_open_interest(sym)
        if oi is not None:
            oi_value = float(oi)
            sources["oi"] = "bitget_rest" if sources.get("oi") != "ws_hub" else "ws_hub"
        else:
            warnings.append("no_open_interest")
            sources["oi"] = sources.get("oi", "none")

    if INST_ENABLE_RECENT_FILLS and tape_5m is None:
        warnings.append("recent_fills_flag_on_but_not_implemented_here")
    if INST_ENABLE_CANDLES:
        warnings.append("candles_flag_on_but_not_implemented_here")

    if INST_DERIVED_ENABLED:
        _update_derived(sym, snap_ts_ms, oi=oi_value, funding=funding_rate, tape_5m=tape_5m, spread_bps=spread_bps)

    oi_change_15m_pct = None
    oi_change_1h_pct = None
    funding_change_1h = None
    funding_flip = None

    if INST_DERIVED_ENABLED:
        try:
            st = _derived_state(sym)
            oi_change_15m_pct = _pct_change(st.oi, snap_ts_ms, horizon_s=900.0)
            oi_change_1h_pct = _pct_change(st.oi, snap_ts_ms, horizon_s=3600.0)
            funding_change_1h = _delta(st.funding, snap_ts_ms, horizon_s=3600.0)
        except Exception:
            pass

        try:
            if funding_rate is not None:
                st = _derived_state(sym)
                old = st.funding.value_at_or_before(int(snap_ts_ms - 3600_000))
                if old is not None:
                    _, fr0 = old
                    fr1 = float(funding_rate)
                    if fr0 == 0.0:
                        funding_flip = None
                    else:
                        funding_flip = bool((float(fr0) > 0 and fr1 < 0) or (float(fr0) < 0 and fr1 > 0))
        except Exception:
            funding_flip = None

    ob_imb_z = _norm_update(sym, "ob_imb", ob_25)
    spread_bps_z = _norm_update(sym, "spread_bps", spread_bps)
    depth_25_z = _norm_update(sym, "depth_25", depth_usd_25)
    oi_z = _norm_update(sym, "oi", oi_value) if oi_value is not None else None
    funding_z2 = _norm_update(sym, "funding", funding_rate) if funding_rate is not None else None
    tape_z = _norm_update(sym, "tape_5m", tape_5m) if tape_5m is not None else None

    funding_regime = _classify_funding(funding_rate, z=funding_z if funding_z is not None else funding_z2)
    ob_regime = _classify_orderbook(ob_25)

    flow_regime = _classify_flow(tape_5m, tape_z)
    crowding_regime = _classify_crowding(funding_rate, funding_z2 if funding_z2 is not None else funding_z)

    cvd_slope = None
    if INST_DERIVED_ENABLED:
        try:
            st_d = _derived_state(sym)
            cvd_slope = _series_slope_per_min(st_d.tape_5m, points=int(INST_TAPE_SLOPE_POINTS))
        except Exception:
            cvd_slope = None

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

    if tape_5m is not None:
        components["flow"] = 1
        score += 1

    score = max(0, min(4, int(score)))
    ok_count = _components_ok_count(components)

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

        "binance_symbol": sym,

        "available": bool(depth is not None or ws_used or (funding_rate is not None) or (oi_value is not None)),

        "oi": oi_value,
        "oi_slope": oi_change_1h_pct,

        "cvd_slope": cvd_slope,
        "cvd_notional_5m": tape_5m,

        "funding_rate": funding_rate,
        "funding_regime": funding_regime,
        "funding_mean": funding_mean,
        "funding_std": funding_std,
        "funding_z": funding_z,
        "next_funding_time_ms": next_funding_time_ms,

        "basis_pct": None,
        "basis_regime": "unknown",

        "tape_delta_1m": None,
        "tape_delta_5m": tape_5m,
        "tape_regime": flow_regime,

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

        "crowding_regime": crowding_regime,
        "flow_regime": flow_regime,

        "oi_change_15m_pct": oi_change_15m_pct,
        "oi_change_1h_pct": oi_change_1h_pct,
        "funding_change_1h": funding_change_1h,
        "funding_flip_1h": funding_flip,

        "liq_1h_usdt": None,

        "oi_z": oi_z,
        "funding_z_rt": funding_z2,

        "warnings": warnings,

        "score_components": components,
        "score_meta": {
            "mode": eff_mode,
            "ok_count": int(ok_count),
            "quality_score": int(quality_score),
            "quality_flags": list(quality_flags),
            "ws_snapshot_used": bool(ws_used),
            "norm_enabled": bool(INST_NORM_ENABLED),
            "norm_min_points": int(INST_NORM_MIN_POINTS),
            "norm_window": int(INST_NORM_WINDOW),
            "bitget_product_type": INST_BITGET_PRODUCT_TYPE,
            "inst_version": INST_VERSION,
            "derived_enabled": bool(INST_DERIVED_ENABLED),
            "derived_max_age_s": float(INST_DERIVED_MAX_AGE_S),
        },

        "available_components": [],
        "available_components_count": 0,

        "ws_snapshot_used": bool(ws_used),
        "orderflow_ws_used": bool(ws_used),
        "normalization_enabled": bool(INST_NORM_ENABLED),
        "data_sources": sources,

        "openInterest": oi_value,
        "fundingRate": funding_rate,
        "basisPct": None,
        "tapeDelta5m": tape_5m,
        "orderbookImb25bps": ob_25,
        "cvdSlope": cvd_slope,
    }

    comps = _available_components_list(payload)
    payload["available_components"] = comps
    payload["available_components_count"] = int(len(comps))

    st = _get_sym_state(sym)
    if st is not None:
        payload["symbol_cooldown_until_ms"] = int(st.until_ms)
        payload["symbol_errors"] = int(st.errors)

    if INST_DEBUG:
        LOGGER.info(
            "[INST_DIAG_DONE] sym=%s ws_used=%s depth_ok=%s funding_rate_ok=%s funding_hist_ok=%s oi_ok=%s score=%s comps=%s sources=%s warnings=%s",
            sym,
            bool(ws_used),
            bool(depth is not None),
            bool(funding_rate is not None),
            bool(funding_z is not None),
            bool(oi_value is not None),
            int(score),
            comps,
            sources,
            warnings,
        )

    if INST_TRACE_PAYLOAD:
        LOGGER.info(
            "[INST_PAYLOAD] sym=%s ob_imb=%s spread_bps=%s depth_usd_25=%s funding_rate=%s funding_z=%s oi=%s oi_chg_1h=%s tape_5m=%s",
            sym,
            None if ob_25 is None else float(ob_25),
            spread_bps,
            depth_usd_25,
            funding_rate,
            funding_z,
            oi_value,
            oi_change_1h_pct,
            tape_5m,
        )

    return payload


async def compute_institutional(
    symbol: str,
    bias: str,
    *,
    mode: Optional[str] = None,
    include_liquidations: bool = False,
) -> Dict[str, Any]:
    return await compute_full_institutional_analysis(symbol, bias, include_liquidations=include_liquidations, mode=mode)
