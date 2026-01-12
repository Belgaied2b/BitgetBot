# =====================================================================
# institutional_data.py — Ultra Desk 3.1 (Bitget-only, public endpoints)
# Bitget USDT-M Futures (Mix) — REST only + optional external WS hub
#
# Confirmed endpoints used (Bitget API docs):
# - GET /api/v2/mix/market/merge-depth              (orderbook)  ✅
# - GET /api/v2/mix/market/history-fund-rate        (funding hist)✅
#
# Everything else is optional / best-effort (can be disabled).
# Keeps legacy keys to avoid KeyError: openInterest, fundingRate, binance_symbol...
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

INST_VERSION = "UltraDesk3.1-bitget-only+trace-2026-01-12"

BITGET_API_BASE = "https://api.bitget.com"

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

INST_USE_WS_HUB = str(os.getenv("INST_USE_WS_HUB", "1")).strip() == "1"
WS_STALE_SEC = float(os.getenv("INST_WS_STALE_SEC", "15"))

# ---------------------------------------------------------------------
# Product type (Bitget v2 uses lower-case values in request examples)
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
# Optional “best-effort” features (disabled by default)
# (In your current code, these are intentionally not implemented as real fetches)
# ---------------------------------------------------------------------
INST_ENABLE_OPEN_INTEREST = str(os.getenv("INST_ENABLE_OPEN_INTEREST", "0")).strip() == "1"
INST_ENABLE_CURRENT_FUNDING = str(os.getenv("INST_ENABLE_CURRENT_FUNDING", "0")).strip() == "1"
INST_ENABLE_RECENT_FILLS = str(os.getenv("INST_ENABLE_RECENT_FILLS", "0")).strip() == "1"
INST_ENABLE_CANDLES = str(os.getenv("INST_ENABLE_CANDLES", "0")).strip() == "1"

# ---------------------------------------------------------------------
# Normalisation (rolling z-scores)
# ---------------------------------------------------------------------
INST_NORM_ENABLED = str(os.getenv("INST_NORM_ENABLED", "1")).strip() == "1"
INST_NORM_MIN_POINTS = int(float(os.getenv("INST_NORM_MIN_POINTS", "20")))
INST_NORM_WINDOW = int(float(os.getenv("INST_NORM_WINDOW", "120")))

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
def _compute_orderbook_band_metrics(depth: Dict[str, Any], band_bps: float = 25.0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        bids = depth.get("bids") or []
        asks = depth.get("asks") or []
        if not bids or not asks:
            return None, None, None

        b0p = float(bids[0][0]); a0p = float(asks[0][0])
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
        bp = float(bids[0][0]); bq = float(bids[0][1])
        ap = float(asks[0][0]); aq = float(asks[0][1])
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
# Bitget confirmed endpoints (docs)
# =====================================================================

async def _fetch_merge_depth(symbol: str) -> Optional[Dict[str, Any]]:
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
    return out


async def compute_full_institutional_analysis(
    symbol: str,
    bias: str,
    *,
    include_liquidations: bool = False,  # kept for signature compatibility
    mode: Optional[str] = None,
) -> Dict[str, Any]:
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
            sym, bias, eff_mode, INST_VERSION, BITGET_API_BASE, INST_BITGET_PRODUCT_TYPE, INST_USE_WS_HUB
        )

    ws_snap = _ws_snapshot(sym)
    ws_used = bool(ws_snap is not None)

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

    funding_mean = funding_std = funding_z = None
    if eff_mode == "FULL":
        hist = await _fetch_funding_history(sym, limit=30)
        if hist is None:
            warnings.append("no_funding_hist")
            sources["funding_hist"] = "none"
        else:
            funding_mean, funding_std, funding_z = _compute_funding_stats_bitget(hist)
            sources["funding_hist"] = "bitget_rest"

    funding_rate = None
    tape_5m = None
    if ws_snap is not None:
        try:
            if ws_snap.get("funding_rate") is not None:
                funding_rate = float(ws_snap.get("funding_rate"))
                sources["funding_rate"] = "ws_hub"
            if ws_snap.get("tape_delta_5m") is not None:
                tape_5m = float(ws_snap.get("tape_delta_5m"))
                sources["tape"] = "ws_hub"
        except Exception:
            warnings.append("ws_parse_error")

    # Optional best-effort (still not implemented here)
    oi_value = None
    if INST_ENABLE_OPEN_INTEREST:
        warnings.append("open_interest_not_implemented_in_this_file")
        sources["oi"] = "flag_enabled_but_not_implemented"

    if INST_ENABLE_CURRENT_FUNDING and funding_rate is None:
        warnings.append("current_funding_flag_on_but_ws_missing")
        sources["funding_rate"] = sources.get("funding_rate", "flag_on_but_missing")

    if INST_ENABLE_RECENT_FILLS and tape_5m is None:
        warnings.append("recent_fills_flag_on_but_ws_missing")
        sources["tape"] = sources.get("tape", "flag_on_but_missing")

    # Normalization
    ob_imb_z = _norm_update(sym, "ob_imb", ob_25)
    spread_bps_z = _norm_update(sym, "spread_bps", spread_bps)
    depth_25_z = _norm_update(sym, "depth_25", depth_usd_25)

    funding_regime = _classify_funding(funding_rate, z=funding_z)
    ob_regime = _classify_orderbook(ob_25)

    components = {"flow": 0, "oi": 0, "crowding": 0, "orderbook": 0}
    score = 0

    if ob_25 is not None:
        components["orderbook"] = 1
        score += 1

    if funding_rate is not None or funding_z is not None:
        components["crowding"] = 1
        score += 1

    score = max(0, min(4, int(score)))
    ok_count = _components_ok_count(components)

    payload: Dict[str, Any] = {
        "institutional_score": int(score),
        "institutional_score_raw": int(score),
        "institutional_score_v2": int(score),
        "institutional_score_v3": int(score),

        # legacy name kept (your logs expect binance_symbol=...)
        "binance_symbol": sym,

        "available": bool(depth is not None or ws_used),

        "oi": oi_value,
        "oi_slope": None,
        "cvd_slope": None,
        "cvd_notional_5m": None,

        "funding_rate": funding_rate,
        "funding_regime": funding_regime,
        "funding_mean": funding_mean,
        "funding_std": funding_std,
        "funding_z": funding_z,
        "next_funding_time_ms": None,

        "basis_pct": None,
        "basis_regime": "unknown",

        "tape_delta_1m": None,
        "tape_delta_5m": tape_5m,
        "tape_regime": "unknown",

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
            "inst_version": INST_VERSION,
        },

        "available_components": [],
        "available_components_count": 0,

        "ws_snapshot_used": bool(ws_used),
        "orderflow_ws_used": bool(ws_used),
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
            "[INST_PAYLOAD] sym=%s ob_imb=%.6f spread_bps=%s depth_usd_25=%s funding_rate=%s funding_z=%s",
            sym,
            float(ob_25) if ob_25 is not None else float("nan"),
            spread_bps,
            depth_usd_25,
            funding_rate,
            funding_z,
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
