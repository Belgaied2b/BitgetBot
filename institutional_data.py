# =====================================================================
# institutional_data.py — Ultra Desk Institutional (BITGET) + Liquidations (BINANCE)
# =====================================================================
# Goal:
#   - Institutional layer driven by Bitget Futures (OI / Funding / Basis / Tape / Orderbook)
#   - Liquidations remain on Binance Futures (single all-market WS stream)
#
# Robustesse scan multi-coins :
# - Rate limiter global (semaphore + pacing)
# - Soft cooldown after 429/5xx
# - Backoff per symbole (évite de marteler le même coin)
# - Shared aiohttp session (pas de session par call)
# - Modes LIGHT/NORMAL/FULL (scanner pass1/pass2)
#
# Output (compat):
# - Keeps keys used by analyze_signal.py:
#     institutional_score, flow_regime, crowding_regime, cvd_slope, liquidation_intensity,
#     binance_symbol, score_meta, score_components, warnings, available_components(_count)
# - Adds legacy aliases to avoid KeyError in older logs:
#     openInterest, fundingRate, basisPct, tapeDelta5m, orderbookImb25bps, cvdSlope
# =====================================================================

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import aiohttp

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Bitget REST (institutional)
# ---------------------------------------------------------------------
BITGET_API_BASE = "https://api.bitget.com"

# productType (Bitget expects lowercase in requests, docs show e.g. "usdt-futures")
_DEFAULT_PRODUCT_TYPE_RAW = os.getenv("PRODUCT_TYPE", "USDT-FUTURES")

try:
    from settings import PRODUCT_TYPE as _SETTINGS_PRODUCT_TYPE  # type: ignore
    if isinstance(_SETTINGS_PRODUCT_TYPE, str) and _SETTINGS_PRODUCT_TYPE.strip():
        _DEFAULT_PRODUCT_TYPE_RAW = _SETTINGS_PRODUCT_TYPE.strip()
except Exception:
    pass


def _product_type_param() -> str:
    v = str(os.getenv("BITGET_PRODUCT_TYPE", _DEFAULT_PRODUCT_TYPE_RAW)).strip()
    # Accept either "USDT-FUTURES" or "usdt-futures"
    v = v.replace("_", "-").replace(" ", "").upper()
    if v in ("USDT-FUTURES", "USDTFUTURES"):
        return "usdt-futures"
    if v in ("COIN-FUTURES", "COINFUTURES"):
        return "coin-futures"
    if v in ("USDC-FUTURES", "USDCFUTURES"):
        return "usdc-futures"
    # fallback: best effort
    return v.lower()


# Modes (env) + overrides
INST_MODE = str(os.getenv("INST_MODE", "LIGHT")).upper().strip()
if INST_MODE not in ("LIGHT", "NORMAL", "FULL"):
    INST_MODE = "LIGHT"

# Liquidations (Binance WS)
INST_INCLUDE_LIQUIDATIONS = str(os.getenv("INST_INCLUDE_LIQUIDATIONS", "0")).strip() == "1"
_LIQ_WINDOW_SEC = int(float(os.getenv("INST_LIQ_WINDOW_SEC", "300")))      # default 5m
_LIQ_STORE_SEC = int(float(os.getenv("INST_LIQ_STORE_SEC", "900")))        # default 15m
_LIQ_MIN_NOTIONAL_USD = float(os.getenv("INST_LIQ_MIN_NOTIONAL_USD", "50000"))

# Bitget limiter/retry
_BITGET_CONCURRENCY = max(1, int(os.getenv("BITGET_HTTP_CONCURRENCY", "6")))
_BITGET_MIN_INTERVAL_SEC = float(os.getenv("BITGET_MIN_INTERVAL_SEC", "0.08"))
_BITGET_HTTP_TIMEOUT_S = float(os.getenv("BITGET_HTTP_TIMEOUT_S", os.getenv("BITGET_HTTP_TIMEOUT", "10")))
_BITGET_HTTP_RETRIES = max(0, int(os.getenv("BITGET_HTTP_RETRIES", "2")))
_BITGET_SOFT_COOLDOWN_MS_DEFAULT = int(float(os.getenv("BITGET_SOFT_COOLDOWN_SEC", "8")) * 1000)

_HTTP_SEM = asyncio.Semaphore(_BITGET_CONCURRENCY)
_PACE_LOCK = asyncio.Lock()
_LAST_REQ_TS = 0.0

# Bitget soft cooldown (ms timestamps)
_BITGET_SOFT_UNTIL_MS = 0

# Per-symbol backoff (Bitget)
_SYM_STATE: Dict[str, "SymbolBackoff"] = {}

# ---------------------------------------------------------------------
# Binance WS liquidations (kept)
# ---------------------------------------------------------------------
BINANCE_FSTREAM_WS_BASE = "wss://fstream.binance.com/ws"
BINANCE_FAPI_BASE = "https://fapi.binance.com"

_RE_BAN_UNTIL = re.compile(r"banned until (\d+)", re.IGNORECASE)
_BINANCE_HARD_BAN_UNTIL_MS = 0
_BINANCE_SOFT_UNTIL_MS = 0
_HARD_BAN_FALLBACK_MS = int(float(os.getenv("BINANCE_HARD_BAN_FALLBACK_MIN", "15")) * 60_000)
_BINANCE_SYMBOLS: Optional[Set[str]] = None
_BINANCE_SYMBOLS_TS: float = 0.0
BINANCE_SYMBOLS_TTL = float(os.getenv("BINANCE_SYMBOLS_TTL_S", "900"))

# shared session
_SESSION: Optional[aiohttp.ClientSession] = None
_SESSION_LOCK = asyncio.Lock()

# liquidations ws worker state
_LIQ_TASK: Optional[asyncio.Task] = None
_LIQ_START_LOCK = asyncio.Lock()
_LIQ_STOP: Optional[asyncio.Event] = None
_LIQ_LOCK = asyncio.Lock()
_LIQ_EVENTS: Dict[str, Deque[Tuple[int, str, float]]] = {}  # symbol -> deque[(ts_ms, side, notional_usd)]


def _now_ms() -> int:
    return int(time.time() * 1000)


# =====================================================================
# Shared session
# =====================================================================

async def _get_session() -> aiohttp.ClientSession:
    global _SESSION
    if _SESSION is not None and not _SESSION.closed:
        return _SESSION

    async with _SESSION_LOCK:
        if _SESSION is not None and not _SESSION.closed:
            return _SESSION
        timeout = aiohttp.ClientTimeout(total=float(_BITGET_HTTP_TIMEOUT_S))
        connector = aiohttp.TCPConnector(limit=50, ttl_dns_cache=300, enable_cleanup_closed=True)
        _SESSION = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return _SESSION


async def close_institutional_session() -> None:
    """Optionnel: à appeler au shutdown (arrête WS liq + ferme la session HTTP)."""
    global _SESSION, _LIQ_TASK, _LIQ_STOP

    try:
        if _LIQ_STOP is not None:
            _LIQ_STOP.set()
        if _LIQ_TASK is not None:
            _LIQ_TASK.cancel()
            try:
                await _LIQ_TASK
            except Exception:
                pass
    except Exception:
        pass
    _LIQ_TASK = None

    if _SESSION is not None and not _SESSION.closed:
        try:
            await _SESSION.close()
        except Exception:
            pass
    _SESSION = None


# =====================================================================
# Bitget pacing + backoff
# =====================================================================

def _is_bitget_soft_blocked() -> bool:
    return _now_ms() < int(_BITGET_SOFT_UNTIL_MS)


def _set_bitget_soft_cooldown(ms_from_now: int, reason: str) -> None:
    global _BITGET_SOFT_UNTIL_MS
    until = _now_ms() + int(ms_from_now)
    if until > _BITGET_SOFT_UNTIL_MS:
        _BITGET_SOFT_UNTIL_MS = until
    LOGGER.warning("[INST] BITGET SOFT COOLDOWN until_ms=%s reason=%s", _BITGET_SOFT_UNTIL_MS, reason)


async def _pace() -> None:
    global _LAST_REQ_TS
    async with _PACE_LOCK:
        now = time.time()
        wait = float(_BITGET_MIN_INTERVAL_SEC) - (now - float(_LAST_REQ_TS))
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

    def mark_err(self, base_ms: int = 900, cap_ms: int = 60_000) -> None:
        self.errors += 1
        mult = 1.7 ** min(self.errors, 8)
        cd = int(min(cap_ms, base_ms * mult))
        self.until_ms = max(self.until_ms, _now_ms() + cd)


def _sym_key(symbol: Optional[str]) -> Optional[str]:
    if not symbol:
        return None
    return str(symbol).upper().strip()


def _get_sym_state(symbol: Optional[str]) -> Optional[SymbolBackoff]:
    k = _sym_key(symbol)
    if not k:
        return None
    st = _SYM_STATE.get(k)
    if st is None:
        st = SymbolBackoff()
        _SYM_STATE[k] = st
    return st


async def _bitget_http_get(path: str, params: Optional[Dict[str, Any]] = None, *, symbol: Optional[str] = None) -> Any:
    """
    Safe GET with:
    - concurrency semaphore
    - pacing
    - per-symbol backoff
    - soft cooldown on 429/5xx
    - small retries on timeouts / 5xx
    """
    if _is_bitget_soft_blocked():
        return None

    st = _get_sym_state(symbol)
    if st is not None and st.blocked():
        return None

    url = BITGET_API_BASE + path
    session = await _get_session()

    async with _HTTP_SEM:
        for attempt in range(0, _BITGET_HTTP_RETRIES + 1):
            await _pace()
            try:
                async with session.get(url, params=params) as resp:
                    status = int(resp.status)
                    raw = await resp.read()
                    try:
                        txt = raw.decode("utf-8", errors="ignore")
                    except Exception:
                        txt = str(raw)[:800]

                    try:
                        data: Any = json.loads(txt) if txt else None
                    except Exception:
                        data = None

                    if status != 200:
                        if status == 429:
                            _set_bitget_soft_cooldown(_BITGET_SOFT_COOLDOWN_MS_DEFAULT, reason=f"{path} 429")
                            if st is not None:
                                st.mark_err(base_ms=2_000)
                            return None
                        if 500 <= status <= 599:
                            _set_bitget_soft_cooldown(3_000, reason=f"{path} {status}")
                            if st is not None:
                                st.mark_err(base_ms=1_500)
                            if attempt < _BITGET_HTTP_RETRIES:
                                await asyncio.sleep(min(2.5, 0.5 * (1.8 ** attempt)))
                                continue
                            return None
                        if st is not None:
                            st.mark_err(base_ms=1_200)
                        LOGGER.warning("[INST] BITGET HTTP %s GET %s params=%s resp=%s", status, path, params, (txt or "")[:200])
                        return None

                    # Bitget error codes: expect dict with code "00000"
                    if isinstance(data, dict):
                        code = str(data.get("code") or "")
                        if code and code != "00000":
                            msg = str(data.get("msg") or "")
                            # treat as soft cooldown if rate-limit style
                            if "too many" in msg.lower() or "frequency" in msg.lower():
                                _set_bitget_soft_cooldown(_BITGET_SOFT_COOLDOWN_MS_DEFAULT, reason=f"{path} code={code}")
                            if st is not None:
                                st.mark_err(base_ms=1_200)
                            LOGGER.warning("[INST] BITGET API error code=%s path=%s msg=%s", code, path, msg[:180])
                            return None

                    if st is not None:
                        st.mark_ok()
                    return data

            except asyncio.TimeoutError:
                if st is not None:
                    st.mark_err(base_ms=1_500)
                if attempt < _BITGET_HTTP_RETRIES:
                    await asyncio.sleep(min(2.5, 0.5 * (1.8 ** attempt)))
                    continue
                LOGGER.error("[INST] BITGET Timeout GET %s params=%s", path, params)
                return None
            except Exception as e:
                if st is not None:
                    st.mark_err(base_ms=1_800)
                if attempt < _BITGET_HTTP_RETRIES:
                    await asyncio.sleep(min(2.5, 0.5 * (1.8 ** attempt)))
                    continue
                LOGGER.error("[INST] BITGET Exception GET %s params=%s: %s", path, params, e)
                return None

    return None


# =====================================================================
# Bitget light caches (TTL seconds)
# =====================================================================
_OI_CACHE: Dict[Tuple[str, str], Tuple[float, Any]] = {}
_FUNDING_CACHE: Dict[Tuple[str, str], Tuple[float, Any]] = {}
_PRICE_CACHE: Dict[Tuple[str, str], Tuple[float, Any]] = {}
_TAKER_CACHE: Dict[Tuple[str, str, str, int], Tuple[float, Any]] = {}
_DEPTH_CACHE: Dict[Tuple[str, str, int], Tuple[float, Any]] = {}

OI_TTL = float(os.getenv("INST_OI_TTL", "30"))
FUNDING_TTL = float(os.getenv("INST_FUNDING_TTL", "30"))
PRICE_TTL = float(os.getenv("INST_PRICE_TTL", "10"))
TAKER_TTL = float(os.getenv("INST_TAKER_TTL", "20"))
DEPTH_TTL = float(os.getenv("INST_DEPTH_TTL", "5"))

# OI slope memory: symbol -> (ts, oi)
_OI_HISTORY: Dict[str, Tuple[float, float]] = {}


# =====================================================================
# Bitget endpoints
# =====================================================================

async def _fetch_open_interest(symbol: str, product_type: str) -> Optional[float]:
    now = time.time()
    key = (symbol, product_type)
    cached = _OI_CACHE.get(key)
    if cached is not None and (now - cached[0]) < OI_TTL:
        return cached[1]  # type: ignore

    data = await _bitget_http_get(
        "/api/v2/mix/market/open-interest",
        params={"symbol": symbol, "productType": product_type},
        symbol=symbol,
    )
    if not isinstance(data, dict):
        return None
    payload = data.get("data")
    # docs: data.openInterestList[0].size
    try:
        lst = payload.get("openInterestList") if isinstance(payload, dict) else None
        if isinstance(lst, list) and lst:
            size = lst[0].get("size")
            oi = float(size)
        else:
            return None
    except Exception:
        return None

    _OI_CACHE[key] = (now, oi)
    return oi


async def _fetch_current_funding(symbol: str, product_type: str) -> Optional[float]:
    now = time.time()
    key = (symbol, product_type)
    cached = _FUNDING_CACHE.get(key)
    if cached is not None and (now - cached[0]) < FUNDING_TTL:
        return cached[1]  # type: ignore

    data = await _bitget_http_get(
        "/api/v2/mix/market/current-fund-rate",
        params={"symbol": symbol, "productType": product_type},
        symbol=symbol,
    )
    if not isinstance(data, dict):
        return None
    arr = data.get("data")
    # docs: data is a list with fundingRate
    try:
        if isinstance(arr, list) and arr:
            fr = float(arr[0].get("fundingRate"))
        else:
            return None
    except Exception:
        return None

    _FUNDING_CACHE[key] = (now, fr)
    return fr


async def _fetch_symbol_price(symbol: str, product_type: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns: (last_price, index_price, mark_price)
    docs: GET /api/v2/mix/market/symbol-price -> data: [{price,indexPrice,markPrice}]
    """
    now = time.time()
    key = (symbol, product_type)
    cached = _PRICE_CACHE.get(key)
    if cached is not None and (now - cached[0]) < PRICE_TTL:
        return cached[1]  # type: ignore

    data = await _bitget_http_get(
        "/api/v2/mix/market/symbol-price",
        params={"symbol": symbol, "productType": product_type},
        symbol=symbol,
    )
    if not isinstance(data, dict):
        return None, None, None
    arr = data.get("data")
    try:
        if isinstance(arr, list) and arr:
            item = arr[0]
            last_p = float(item.get("price")) if item.get("price") is not None else None
            idx_p = float(item.get("indexPrice")) if item.get("indexPrice") is not None else None
            mark_p = float(item.get("markPrice")) if item.get("markPrice") is not None else None
        else:
            return None, None, None
    except Exception:
        return None, None, None

    out = (last_p, idx_p, mark_p)
    _PRICE_CACHE[key] = (now, out)
    return out


async def _fetch_taker_buy_sell(symbol: str, product_type: str, period: str = "5m", limit: int = 50) -> Optional[List[Dict[str, Any]]]:
    """
    Bitget: GET /api/v2/mix/market/taker-buy-sell
      params: symbol, productType, period (e.g. 5m), limit
      returns data: list[{buyVolume, sellVolume, ts}]
    """
    now = time.time()
    key = (symbol, product_type, period, int(limit))
    cached = _TAKER_CACHE.get(key)
    if cached is not None and (now - cached[0]) < TAKER_TTL:
        return cached[1]  # type: ignore

    data = await _bitget_http_get(
        "/api/v2/mix/market/taker-buy-sell",
        params={"symbol": symbol, "productType": product_type, "period": period, "limit": int(limit)},
        symbol=symbol,
    )
    if not isinstance(data, dict):
        return None
    arr = data.get("data")
    if not isinstance(arr, list) or not arr:
        return None

    _TAKER_CACHE[key] = (now, arr)
    return arr


async def _fetch_merge_depth(symbol: str, product_type: str, limit: int = 100) -> Optional[Dict[str, Any]]:
    """
    Bitget: GET /api/v2/mix/market/merge-depth
      returns data: {bids:[[price,size],...], asks:[[price,size],...], ts:...}
    """
    now = time.time()
    key = (symbol, product_type, int(limit))
    cached = _DEPTH_CACHE.get(key)
    if cached is not None and (now - cached[0]) < DEPTH_TTL:
        return cached[1]  # type: ignore

    data = await _bitget_http_get(
        "/api/v2/mix/market/merge-depth",
        params={"symbol": symbol, "productType": product_type, "limit": int(limit)},
        symbol=symbol,
    )
    if not isinstance(data, dict):
        return None
    payload = data.get("data")
    if not isinstance(payload, dict):
        return None
    bids = payload.get("bids")
    asks = payload.get("asks")
    if not isinstance(bids, list) or not isinstance(asks, list):
        return None

    _DEPTH_CACHE[key] = (now, payload)
    return payload


# =====================================================================
# Helpers: metrics
# =====================================================================

def _compute_delta_ratio(buy: float, sell: float) -> Optional[float]:
    try:
        b = float(buy)
        s = float(sell)
        den = b + s
        if den <= 0:
            return None
        return float((b - s) / den)
    except Exception:
        return None


def _compute_tape_from_taker(arr: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns: (buy_vol, sell_vol, delta_ratio) using the LAST point in arr.
    """
    try:
        if not arr:
            return None, None, None
        last = arr[-1]
        buy = float(last.get("buyVolume") or 0.0)
        sell = float(last.get("sellVolume") or 0.0)
        dr = _compute_delta_ratio(buy, sell)
        return float(buy), float(sell), dr
    except Exception:
        return None, None, None


def _compute_cvd_slope_from_taker(arr: List[Dict[str, Any]], window: int = 30) -> Optional[float]:
    """
    Approx CVD slope using taker buy/sell series:
      delta = buy - sell
      cvd(t) = sum(delta)
      slope = (cvd_end - cvd_start) / max(abs(cvd_start), abs(cvd_end), 1e-9)
    """
    try:
        if not arr or len(arr) < max(8, window):
            return None
        sub = arr[-window:]
        cvd = 0.0
        series: List[float] = []
        for x in sub:
            buy = float(x.get("buyVolume") or 0.0)
            sell = float(x.get("sellVolume") or 0.0)
            cvd += (buy - sell)
            series.append(cvd)
        if len(series) < 6:
            return None
        start = float(series[0])
        end = float(series[-1])
        den = max(abs(start), abs(end), 1e-9)
        return float((end - start) / den)
    except Exception:
        return None


def _compute_orderbook_imbalance(depth: Dict[str, Any], band_bps: float = 25.0) -> Optional[float]:
    """Imbalance in [-1,+1] within +/- band_bps around mid."""
    try:
        bids = depth.get("bids") or []
        asks = depth.get("asks") or []
        if not bids or not asks:
            return None

        b0p = float(bids[0][0])
        a0p = float(asks[0][0])
        if a0p <= 0 or b0p <= 0:
            return None

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
            return None
        return float((bid_val - ask_val) / den)
    except Exception:
        return None


# =====================================================================
# Regimes + scoring (compatible)
# =====================================================================

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


def _score_institutional(
    bias: str,
    *,
    oi_slope: Optional[float],
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
    if payload.get("cvd_slope") is not None:
        out.append("cvd")
    if payload.get("liq_total_usd_5m") is not None:
        out.append("liquidations")
    return out


# =====================================================================
# Binance: symbols mapping (for liquidation metrics only)
# =====================================================================

def _binance_is_hard_banned() -> bool:
    return _now_ms() < int(_BINANCE_HARD_BAN_UNTIL_MS)


def _binance_is_soft_blocked() -> bool:
    return _now_ms() < int(_BINANCE_SOFT_UNTIL_MS)


def _set_binance_hard_ban_until(ms: int, reason: str) -> None:
    global _BINANCE_HARD_BAN_UNTIL_MS
    ms = int(ms)
    if ms > _BINANCE_HARD_BAN_UNTIL_MS:
        _BINANCE_HARD_BAN_UNTIL_MS = ms
    LOGGER.error("[INST] BINANCE HARD BAN until_ms=%s reason=%s", ms, reason)


def _set_binance_soft_cooldown(ms_from_now: int, reason: str) -> None:
    global _BINANCE_SOFT_UNTIL_MS
    until = _now_ms() + int(ms_from_now)
    if until > _BINANCE_SOFT_UNTIL_MS:
        _BINANCE_SOFT_UNTIL_MS = until
    LOGGER.warning("[INST] BINANCE SOFT COOLDOWN until_ms=%s reason=%s", _BINANCE_SOFT_UNTIL_MS, reason)


async def _binance_http_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    if _binance_is_hard_banned() or _binance_is_soft_blocked():
        return None

    url = BINANCE_FAPI_BASE + path
    session = await _get_session()

    try:
        async with session.get(url, params=params) as resp:
            status = int(resp.status)
            raw = await resp.read()
            try:
                txt = raw.decode("utf-8", errors="ignore")
            except Exception:
                txt = str(raw)[:800]
            try:
                data = json.loads(txt) if txt else None
            except Exception:
                data = None

            if status != 200:
                msg = ""
                code = None
                if isinstance(data, dict):
                    msg = str(data.get("msg") or "")
                    code = data.get("code")

                if isinstance(code, int) and code == -1003:
                    low = msg.lower()
                    if "banned until" in low:
                        m = _RE_BAN_UNTIL.search(low)
                        if m:
                            _set_binance_hard_ban_until(int(m.group(1)), reason=f"{path} -1003 {msg[:120]}")
                        else:
                            _set_binance_hard_ban_until(_now_ms() + _HARD_BAN_FALLBACK_MS, reason=f"{path} -1003 no_ts")
                    else:
                        _set_binance_soft_cooldown(20_000, reason=f"{path} -1003")
                    return None

                if status == 418:
                    raw_msg = (msg or txt or "")
                    m = _RE_BAN_UNTIL.search(raw_msg)
                    if m:
                        _set_binance_hard_ban_until(int(m.group(1)), reason=f"{path} 418")
                    else:
                        _set_binance_hard_ban_until(_now_ms() + _HARD_BAN_FALLBACK_MS, reason=f"{path} 418 no_ts")
                    return None

                if status == 429:
                    _set_binance_soft_cooldown(20_000, reason=f"{path} 429")
                    return None

                if 500 <= status <= 599:
                    _set_binance_soft_cooldown(5_000, reason=f"{path} {status}")
                    return None

                return None

            return data
    except Exception:
        return None


async def _get_binance_symbols() -> Set[str]:
    global _BINANCE_SYMBOLS, _BINANCE_SYMBOLS_TS
    now = time.time()
    if _BINANCE_SYMBOLS is not None and (now - _BINANCE_SYMBOLS_TS) < BINANCE_SYMBOLS_TTL:
        return _BINANCE_SYMBOLS

    data = await _binance_http_get("/fapi/v1/exchangeInfo", params=None)
    symbols: Set[str] = set()
    if not isinstance(data, dict) or "symbols" not in data:
        _BINANCE_SYMBOLS = _BINANCE_SYMBOLS or set()
        return _BINANCE_SYMBOLS

    for s in data.get("symbols", []):
        try:
            if s.get("status") != "TRADING":
                continue
            if s.get("contractType") != "PERPETUAL":
                continue
            if s.get("quoteAsset") != "USDT":
                continue
            sym = str(s.get("symbol", "")).upper()
            if sym:
                symbols.add(sym)
        except Exception:
            continue

    _BINANCE_SYMBOLS = symbols
    _BINANCE_SYMBOLS_TS = now
    return _BINANCE_SYMBOLS


def _map_symbol_to_binance(symbol: str, binance_symbols: Set[str]) -> Optional[str]:
    s = (symbol or "").upper().replace("-", "").replace("_", "")
    s = s.replace("UMCBL", "").replace("USDTM", "USDT")
    if s in binance_symbols:
        return s
    if s.startswith("1000"):
        alt = s[4:]
        if alt in binance_symbols:
            return alt
    return None


# =====================================================================
# Binance liquidations WS (single shared worker)
# =====================================================================

def _liq_stream_url() -> str:
    return f"{BINANCE_FSTREAM_WS_BASE}/!forceOrder@arr"


async def _liq_add_event(symbol: str, ts_ms: int, side: str, notional_usd: float) -> None:
    try:
        sym = str(symbol or "").upper().strip()
        if not sym:
            return
        s = str(side or "").upper().strip()
        if s not in ("BUY", "SELL"):
            return
        n = float(notional_usd)
        if not (n > 0.0):
            return

        cutoff_store = _now_ms() - int(_LIQ_STORE_SEC) * 1000
        async with _LIQ_LOCK:
            dq = _LIQ_EVENTS.get(sym)
            if dq is None:
                dq = deque(maxlen=6000)
                _LIQ_EVENTS[sym] = dq
            dq.append((int(ts_ms), s, float(n)))

            while dq and int(dq[0][0]) < cutoff_store:
                dq.popleft()

            if not dq:
                _LIQ_EVENTS.pop(sym, None)
    except Exception:
        return


async def _liq_metrics(symbol: str, window_sec: int) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    try:
        sym = str(symbol or "").upper().strip()
        if not sym:
            return None, None, None, None
        cutoff = _now_ms() - int(window_sec) * 1000

        buy = 0.0
        sell = 0.0
        async with _LIQ_LOCK:
            dq = _LIQ_EVENTS.get(sym)
            if not dq:
                return None, None, None, None

            for ts_ms, side, notional in reversed(dq):
                if int(ts_ms) < cutoff:
                    break
                if side == "BUY":
                    buy += float(notional)
                elif side == "SELL":
                    sell += float(notional)

        total = buy + sell
        if total <= 0:
            return float(buy), float(sell), 0.0, None
        delta_ratio = (buy - sell) / total
        return float(buy), float(sell), float(total), float(delta_ratio)
    except Exception:
        return None, None, None, None


async def _liq_worker() -> None:
    global _LIQ_STOP
    backoff = 1.0
    url = _liq_stream_url()
    LOGGER.info("[INST_LIQ] WS worker start url=%s", url)

    while _LIQ_STOP is not None and (not _LIQ_STOP.is_set()):
        try:
            session = await _get_session()
            async with session.ws_connect(url, heartbeat=30) as ws:
                LOGGER.info("[INST_LIQ] WS connected")
                backoff = 1.0
                async for msg in ws:
                    if _LIQ_STOP is not None and _LIQ_STOP.is_set():
                        break

                    if msg.type == aiohttp.WSMsgType.TEXT:
                        raw = msg.data
                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        try:
                            raw = msg.data.decode("utf-8", errors="ignore")
                        except Exception:
                            continue
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
                    else:
                        continue

                    try:
                        payload = json.loads(raw) if raw else None
                    except Exception:
                        continue

                    if isinstance(payload, dict) and "data" in payload and isinstance(payload.get("data"), (dict, list)):
                        payload = payload.get("data")

                    events: List[Any] = []
                    if isinstance(payload, list):
                        events = payload
                    elif isinstance(payload, dict):
                        events = [payload]

                    for ev in events:
                        try:
                            if not isinstance(ev, dict):
                                continue
                            o = ev.get("o")
                            if not isinstance(o, dict):
                                continue
                            sym = o.get("s")
                            side = o.get("S")
                            qty = o.get("q")
                            price = o.get("ap") or o.get("p")
                            ts = o.get("T") or ev.get("E") or _now_ms()

                            qf = float(qty)
                            pf = float(price)
                            notional = qf * pf
                            await _liq_add_event(str(sym), int(ts), str(side), float(notional))
                        except Exception:
                            continue

        except asyncio.CancelledError:
            break
        except Exception as e:
            LOGGER.warning("[INST_LIQ] WS error: %s", e)

        if _LIQ_STOP is not None and _LIQ_STOP.is_set():
            break
        await asyncio.sleep(min(60.0, backoff))
        backoff = min(60.0, backoff * 2.0)

    LOGGER.info("[INST_LIQ] WS worker stopped")


async def _ensure_liq_stream() -> None:
    global _LIQ_TASK, _LIQ_STOP
    if _LIQ_STOP is None or _LIQ_STOP.is_set():
        _LIQ_STOP = asyncio.Event()

    if _LIQ_TASK is not None and (not _LIQ_TASK.done()):
        return

    async with _LIQ_START_LOCK:
        if _LIQ_TASK is not None and (not _LIQ_TASK.done()):
            return
        _LIQ_TASK = asyncio.create_task(_liq_worker(), name="inst_liq_ws_worker")


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

    product_type = _product_type_param()
    warnings: List[str] = []
    sources: Dict[str, str] = {}

    use_liq = bool(INST_INCLUDE_LIQUIDATIONS or include_liquidations)

    if _is_bitget_soft_blocked():
        warnings.append(f"bitget_soft_cooldown_until_ms={_BITGET_SOFT_UNTIL_MS}")

    oi_value: Optional[float] = await _fetch_open_interest(symbol, product_type)
    if oi_value is None:
        warnings.append("no_oi")
        sources["oi"] = "none"
    else:
        sources["oi"] = "bitget_rest"

    oi_slope: Optional[float] = None
    try:
        if oi_value is not None:
            prev = _OI_HISTORY.get(symbol)
            if prev and float(prev[1]) > 0:
                oi_slope = float((float(oi_value) - float(prev[1])) / float(prev[1]))
            else:
                oi_slope = 0.0
            _OI_HISTORY[symbol] = (time.time(), float(oi_value))
    except Exception:
        oi_slope = None

    funding_rate: Optional[float] = await _fetch_current_funding(symbol, product_type)
    if funding_rate is None:
        warnings.append("no_funding")
        sources["funding_rate"] = "none"
    else:
        sources["funding_rate"] = "bitget_rest"

    last_price, index_price, mark_price = await _fetch_symbol_price(symbol, product_type)
    basis_pct: Optional[float] = None
    try:
        if index_price is not None and mark_price is not None and float(index_price) > 0:
            basis_pct = float((float(mark_price) - float(index_price)) / float(index_price))
            sources["basis_pct"] = "bitget_rest"
        else:
            sources["basis_pct"] = "none"
    except Exception:
        basis_pct = None
        sources["basis_pct"] = "none"

    tape_5m: Optional[float] = None
    tape_buy_vol: Optional[float] = None
    tape_sell_vol: Optional[float] = None
    ob_10: Optional[float] = None
    ob_25: Optional[float] = None
    cvd_slope: Optional[float] = None

    taker_series: Optional[List[Dict[str, Any]]] = None

    if eff_mode in ("NORMAL", "FULL"):
        taker_series = await _fetch_taker_buy_sell(symbol, product_type, period="5m", limit=60)
        if taker_series:
            b, s, dr = _compute_tape_from_taker(taker_series)
            tape_buy_vol, tape_sell_vol, tape_5m = b, s, dr
            sources["tape"] = "bitget_rest"
        else:
            warnings.append("no_taker_buy_sell")
            sources["tape"] = "none"

        depth = await _fetch_merge_depth(symbol, product_type, limit=100)
        if depth:
            ob_10 = _compute_orderbook_imbalance(depth, band_bps=10.0)
            ob_25 = _compute_orderbook_imbalance(depth, band_bps=25.0)
            sources["orderbook"] = "bitget_rest"
        else:
            warnings.append("no_depth")
            sources["orderbook"] = "none"

    if eff_mode == "FULL":
        if taker_series:
            cvd_slope = _compute_cvd_slope_from_taker(taker_series, window=30)
            sources["cvd_slope"] = "bitget_rest" if cvd_slope is not None else "none"
        else:
            sources["cvd_slope"] = "none"

    funding_mean = funding_std = funding_z = None
    if eff_mode == "FULL":
        sources["funding_hist"] = "none"

    binance_symbol: Optional[str] = None
    liq_buy_usd_5m: Optional[float] = None
    liq_sell_usd_5m: Optional[float] = None
    liq_total_usd_5m: Optional[float] = None
    liq_delta_ratio_5m: Optional[float] = None
    liq_regime: str = "unknown"

    if use_liq:
        try:
            bs = await _get_binance_symbols()
            binance_symbol = _map_symbol_to_binance(symbol, bs)
            if binance_symbol is None:
                warnings.append("liq_symbol_not_mapped_to_binance")
            else:
                await _ensure_liq_stream()
                b, s, t, d = await _liq_metrics(binance_symbol, window_sec=_LIQ_WINDOW_SEC)
                liq_buy_usd_5m, liq_sell_usd_5m, liq_total_usd_5m, liq_delta_ratio_5m = b, s, t, d
                liq_regime = _classify_liq(liq_delta_ratio_5m, liq_total_usd_5m)
                sources["liquidations"] = "binance_ws"
        except Exception:
            warnings.append("liq_metrics_error")
            sources["liquidations"] = "none"

    liquidation_intensity: Optional[float] = None
    try:
        if liq_total_usd_5m is not None and float(liq_total_usd_5m) > 0.0:
            liquidation_intensity = float(liq_total_usd_5m)
    except Exception:
        liquidation_intensity = None

    funding_regime = _classify_funding(funding_rate, z=funding_z)
    basis_regime = _classify_basis(basis_pct)
    crowding_regime = _classify_crowding(bias, funding_rate, basis_pct, funding_z)
    flow_regime = _classify_flow(cvd_slope, tape_5m)
    ob_regime = _classify_orderbook(ob_25)

    inst_score, components, score_meta = _score_institutional(
        bias,
        oi_slope=oi_slope,
        cvd_slope=cvd_slope,
        tape_5m=tape_5m,
        funding_rate=funding_rate,
        funding_z=funding_z,
        basis_pct=basis_pct,
        ob_25bps=ob_25,
        liq_delta_ratio_5m=liq_delta_ratio_5m,
        liq_total_usd_5m=liq_total_usd_5m,
    )

    ok_count = _components_ok_count(components)
    score_meta = dict(score_meta or {})
    score_meta["mode"] = eff_mode
    score_meta["ok_count"] = int(ok_count)
    score_meta["liq_window_sec"] = int(_LIQ_WINDOW_SEC)
    score_meta["liq_min_notional_usd"] = float(_LIQ_MIN_NOTIONAL_USD)
    score_meta["provider"] = "bitget"
    score_meta["bitget_product_type"] = product_type

    available = any(
        [
            oi_value is not None,
            funding_rate is not None,
            basis_pct is not None,
            tape_5m is not None,
            ob_25 is not None,
            cvd_slope is not None,
            (liq_total_usd_5m is not None and liq_total_usd_5m > 0.0),
        ]
    )

    payload: Dict[str, Any] = {
        "institutional_score": int(inst_score),
        "binance_symbol": binance_symbol,
        "available": bool(available),
        "warnings": warnings,

        "oi": oi_value,
        "oi_slope": oi_slope,

        "funding_rate": funding_rate,
        "funding_regime": funding_regime,

        "basis_pct": basis_pct,
        "basis_regime": basis_regime,

        "tape_delta_5m": tape_5m,
        "tape_buy_vol_5m": tape_buy_vol,
        "tape_sell_vol_5m": tape_sell_vol,
        "tape_regime": _classify_tape(tape_5m),

        "orderbook_imb_10bps": ob_10,
        "orderbook_imb_25bps": ob_25,
        "orderbook_regime": ob_regime,

        "cvd_slope": cvd_slope,
        "flow_regime": flow_regime,
        "crowding_regime": crowding_regime,

        "funding_mean": funding_mean,
        "funding_std": funding_std,
        "funding_z": funding_z,

        "liq_buy_usd_5m": liq_buy_usd_5m,
        "liq_sell_usd_5m": liq_sell_usd_5m,
        "liq_total_usd_5m": liq_total_usd_5m,
        "liq_delta_ratio_5m": liq_delta_ratio_5m,
        "liq_regime": liq_regime,
        "liquidation_intensity": liquidation_intensity,

        "score_components": components,
        "score_meta": score_meta,

        "available_components": [],
        "available_components_count": 0,
        "data_sources": sources,
        "bitget_soft_until_ms": int(_BITGET_SOFT_UNTIL_MS),
        "binance_ban": {"hard_until_ms": int(_BINANCE_HARD_BAN_UNTIL_MS), "soft_until_ms": int(_BINANCE_SOFT_UNTIL_MS)},

        "openInterest": oi_value,
        "fundingRate": funding_rate,
        "basisPct": basis_pct,
        "tapeDelta5m": tape_5m,
        "orderbookImb25bps": ob_25,
        "cvdSlope": cvd_slope,
    }

    comps = _available_components_list(payload)
    payload["available_components"] = comps
    payload["available_components_count"] = int(len(comps))

    st = _get_sym_state(symbol)
    if st is not None:
        payload["symbol_cooldown_until_ms"] = int(st.until_ms)
        payload["symbol_errors"] = int(st.errors)

    return payload


async def compute_institutional(
    symbol: str,
    bias: str,
    *,
    mode: Optional[str] = None,
    include_liquidations: bool = False,
) -> Dict[str, Any]:
    return await compute_full_institutional_analysis(symbol, bias, include_liquidations=include_liquidations, mode=mode)


def get_ban_state() -> Dict[str, int]:
    return {
        "bitget_soft_until_ms": int(_BITGET_SOFT_UNTIL_MS),
        "binance_hard_until_ms": int(_BINANCE_HARD_BAN_UNTIL_MS),
        "binance_soft_until_ms": int(_BINANCE_SOFT_UNTIL_MS),
    }
