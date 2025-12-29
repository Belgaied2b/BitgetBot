# =====================================================================
# institutional_data.py — Ultra Desk OI + CVD Engine (Binance Futures) (Desk v4)
# =====================================================================
# ✅ Session aiohttp réutilisée (moins lourd)
# ✅ Cache exchangeInfo + klines + premiumIndex
# ✅ CVD slope robuste (linear regression) + normalisation
# ✅ OI slope: snapshot history (simple & efficace)
# ✅ Score insti plus "desk" (moins de 0 partout, mais reste exigeant)
# =====================================================================

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Set

import aiohttp
import numpy as np

LOGGER = logging.getLogger(__name__)
BINANCE_FAPI_BASE = "https://fapi.binance.com"

# ---------------------------------------------------------------------
# Light caches
# ---------------------------------------------------------------------
_KLINES_CACHE: Dict[Tuple[str, str], Tuple[float, List[List[Any]]]] = {}
_FUNDING_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_OI_HISTORY: Dict[str, Tuple[float, float]] = {}

_BINANCE_SYMBOLS: Optional[Set[str]] = None
_BINANCE_SYMBOLS_TS: float = 0.0

KLINES_TTL = 60.0
FUNDING_TTL = 60.0
BINANCE_SYMBOLS_TTL = 900.0

# ---------------------------------------------------------------------
# Shared session (reuse)
# ---------------------------------------------------------------------
_SESSION: Optional[aiohttp.ClientSession] = None
_SESSION_LOCK = asyncio.Lock()


async def _get_session() -> aiohttp.ClientSession:
    global _SESSION
    async with _SESSION_LOCK:
        if _SESSION is not None and not _SESSION.closed:
            return _SESSION
        timeout = aiohttp.ClientTimeout(total=12)
        connector = aiohttp.TCPConnector(limit=50, ttl_dns_cache=300)
        _SESSION = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return _SESSION


async def _http_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = BINANCE_FAPI_BASE + path
    session = await _get_session()
    try:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                txt = await resp.text()
                LOGGER.warning("[INST] HTTP %s GET %s params=%s resp=%s", resp.status, path, params, txt[:200])
                return None
            return await resp.json()
    except asyncio.TimeoutError:
        LOGGER.error("[INST] Timeout GET %s params=%s", path, params)
        return None
    except Exception as e:
        LOGGER.error("[INST] Exception GET %s params=%s: %s", path, params, e)
        return None


# =====================================================================
# Binance symbols (exchangeInfo) cache
# =====================================================================

async def _get_binance_symbols() -> Set[str]:
    global _BINANCE_SYMBOLS, _BINANCE_SYMBOLS_TS
    now = time.time()

    if _BINANCE_SYMBOLS is not None and (now - _BINANCE_SYMBOLS_TS) < BINANCE_SYMBOLS_TTL:
        return _BINANCE_SYMBOLS

    data = await _http_get("/fapi/v1/exchangeInfo")
    symbols: Set[str] = set()

    if not isinstance(data, dict) or "symbols" not in data:
        LOGGER.warning("[INST] Unable to fetch exchangeInfo, keeping old cache")
        if _BINANCE_SYMBOLS is None:
            _BINANCE_SYMBOLS = set()
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
    LOGGER.info("[INST] Binance futures symbols loaded: %d", len(symbols))
    return _BINANCE_SYMBOLS


# =====================================================================
# Symbol mapping Bitget -> Binance
# =====================================================================

def _map_symbol_to_binance(symbol: str, binance_symbols: Set[str]) -> Optional[str]:
    s = str(symbol).upper()

    if s in binance_symbols:
        return s

    if s.startswith("1000"):
        alt = s[4:]
        if alt in binance_symbols:
            return alt

    # sometimes exchanges use suffixes or variants; keep it conservative
    return None


# =====================================================================
# Klines 1h (CVD)
# =====================================================================

async def _fetch_klines_1h(binance_symbol: str, limit: int = 120) -> Optional[List[List[Any]]]:
    cache_key = (binance_symbol, "1h")
    now = time.time()

    cached = _KLINES_CACHE.get(cache_key)
    if cached is not None:
        ts, data = cached
        if now - ts < KLINES_TTL:
            return data

    params = {"symbol": binance_symbol, "interval": "1h", "limit": int(limit)}
    data = await _http_get("/fapi/v1/klines", params=params)
    if not isinstance(data, list) or len(data) == 0:
        return None

    _KLINES_CACHE[cache_key] = (now, data)
    return data


# =====================================================================
# Funding / premiumIndex
# =====================================================================

async def _fetch_funding(binance_symbol: str) -> Optional[Dict[str, Any]]:
    now = time.time()
    cached = _FUNDING_CACHE.get(binance_symbol)
    if cached is not None:
        ts, data = cached
        if now - ts < FUNDING_TTL:
            return data

    params = {"symbol": binance_symbol}
    data = await _http_get("/fapi/v1/premiumIndex", params=params)
    if not isinstance(data, dict) or "symbol" not in data:
        return None

    _FUNDING_CACHE[binance_symbol] = (now, data)
    return data


# =====================================================================
# Open interest snapshot
# =====================================================================

async def _fetch_open_interest(binance_symbol: str) -> Optional[float]:
    params = {"symbol": binance_symbol}
    data = await _http_get("/fapi/v1/openInterest", params=params)
    if not isinstance(data, dict) or "openInterest" not in data:
        return None
    try:
        oi = float(data["openInterest"])
        return oi if np.isfinite(oi) else None
    except Exception:
        return None


def _compute_oi_slope(binance_symbol: str, new_oi: Optional[float]) -> Optional[float]:
    if new_oi is None:
        return None

    prev = _OI_HISTORY.get(binance_symbol)
    if prev is None:
        return 0.0

    _, old_oi = prev
    if old_oi <= 0:
        return 0.0

    slope = (float(new_oi) - float(old_oi)) / float(old_oi)
    return float(slope) if np.isfinite(slope) else 0.0


# =====================================================================
# CVD slope robust
# =====================================================================

def _compute_cvd_slope_from_klines(klines: List[List[Any]], window: int = 48) -> Optional[float]:
    """
    Build CVD from:
      delta = (takerBuyBase - (vol - takerBuyBase)) = 2*takerBuyBase - vol
    Then compute linear-regression slope over last `window`.
    Normalize slope by mean(abs(cvd)) to keep scale stable.
    """
    try:
        if not klines or len(klines) < window + 10:
            return None

        sub = klines[-(window + 10):]
        cvd_vals: List[float] = []
        cvd = 0.0

        for item in sub:
            try:
                vol = float(item[5])
                taker_buy = float(item[9])
                if not (np.isfinite(vol) and np.isfinite(taker_buy)):
                    continue
            except Exception:
                continue

            delta = 2.0 * taker_buy - vol
            cvd += delta
            cvd_vals.append(float(cvd))

        if len(cvd_vals) < window:
            return None

        y = np.array(cvd_vals[-window:], dtype=float)
        x = np.arange(len(y), dtype=float)

        # linear regression slope
        x_mean = x.mean()
        y_mean = y.mean()
        denom = np.sum((x - x_mean) ** 2)
        if denom <= 1e-12:
            return 0.0

        slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)

        scale = float(np.mean(np.abs(y))) + 1e-8
        norm = slope / scale  # typical scale ~ [-0.2, 0.2] in practice
        if not np.isfinite(norm):
            return 0.0
        return float(norm)
    except Exception:
        return None


# =====================================================================
# Funding / crowding / flow regimes
# =====================================================================

def _classify_funding(funding_rate: Optional[float]) -> str:
    if funding_rate is None:
        return "unknown"
    fr = float(funding_rate)
    if fr <= -0.0015:
        return "very_negative"
    if fr <= -0.0005:
        return "negative"
    if fr < 0.0005:
        return "neutral"
    if fr < 0.0015:
        return "positive"
    return "very_positive"


def _classify_crowding(bias: str, funding_rate: Optional[float]) -> str:
    if funding_rate is None:
        return "unknown"
    fr = float(funding_rate)
    b = str(bias).upper()
    if b == "LONG":
        if fr >= 0.001:
            return "long_crowded_risky"
        if fr <= -0.001:
            return "short_crowded_favorable"
        return "balanced"
    if b == "SHORT":
        if fr <= -0.001:
            return "short_crowded_risky"
        if fr >= 0.001:
            return "long_crowded_favorable"
        return "balanced"
    return "unknown"


def _classify_flow(cvd_slope: Optional[float]) -> str:
    if cvd_slope is None:
        return "unknown"
    x = float(cvd_slope)
    # with our normalization, typical strong move ~ 0.12+
    if x >= 0.14:
        return "strong_buy"
    if x >= 0.06:
        return "buy"
    if x <= -0.14:
        return "strong_sell"
    if x <= -0.06:
        return "sell"
    return "neutral"


# =====================================================================
# Institutional score (0..4) — more desk-friendly distribution
# =====================================================================

def _score_institutional(
    bias: str,
    oi_slope: Optional[float],
    cvd_slope: Optional[float],
    funding_rate: Optional[float],
) -> int:
    """
    Score in [0..4]
      - CVD (main): up to +2
      - OI confirm: +1
      - Funding contrarian: +1
    """
    b = str(bias).upper()
    score = 0

    # CVD direction
    if cvd_slope is not None:
        x = float(cvd_slope)
        if b == "LONG":
            if x >= 0.14:
                score += 2
            elif x >= 0.06:
                score += 1
        elif b == "SHORT":
            if x <= -0.14:
                score += 2
            elif x <= -0.06:
                score += 1

    # OI confirm
    if oi_slope is not None:
        y = float(oi_slope)
        if b == "LONG" and y >= 0.004:
            score += 1
        elif b == "SHORT" and y <= -0.004:
            score += 1

    # Funding contrarian (when it helps the bias)
    if funding_rate is not None:
        fr = float(funding_rate)
        if b == "LONG" and fr <= -0.0005:
            score += 1
        elif b == "SHORT" and fr >= 0.0005:
            score += 1

    score = max(0, min(4, score))
    return int(score)


# =====================================================================
# MAIN API
# =====================================================================

async def compute_full_institutional_analysis(symbol: str, bias: str) -> Dict[str, Any]:
    """
    Binance Futures USDT-M institutional snapshot (free endpoints).
    Returns dict with stable keys used by analyze_signal.py.
    """
    bias = str(bias).upper()
    warnings: List[str] = []

    oi_value: Optional[float] = None
    oi_slope: Optional[float] = None
    cvd_slope: Optional[float] = None
    funding_rate: Optional[float] = None

    # symbol mapping
    binance_symbols = await _get_binance_symbols()
    binance_symbol = _map_symbol_to_binance(symbol, binance_symbols)

    if binance_symbol is None:
        return {
            "institutional_score": 0,
            "binance_symbol": None,
            "available": False,
            "oi": None,
            "oi_slope": None,
            "cvd_slope": None,
            "funding_rate": None,
            "funding_regime": "unknown",
            "crowding_regime": "unknown",
            "flow_regime": "unknown",
            "warnings": ["symbol_not_mapped_to_binance"],
        }

    # 1) Klines -> CVD
    klines = await _fetch_klines_1h(binance_symbol, limit=140)
    if not klines:
        warnings.append("no_klines")
        cvd_slope = None
    else:
        cvd_slope = _compute_cvd_slope_from_klines(klines, window=48)

    # 2) OI snapshot + slope
    oi_value = await _fetch_open_interest(binance_symbol)
    if oi_value is None:
        warnings.append("no_oi")
        oi_slope = None
    else:
        oi_slope = _compute_oi_slope(binance_symbol, oi_value)
        _OI_HISTORY[binance_symbol] = (time.time(), float(oi_value))

    # 3) Funding
    funding_data = await _fetch_funding(binance_symbol)
    if funding_data is None:
        warnings.append("no_funding")
        funding_rate = None
    else:
        try:
            funding_rate = float(funding_data.get("lastFundingRate", "0"))
            if not np.isfinite(funding_rate):
                funding_rate = None
                warnings.append("funding_nan")
        except Exception:
            funding_rate = None
            warnings.append("funding_parse_error")

    funding_regime = _classify_funding(funding_rate)
    crowding_regime = _classify_crowding(bias, funding_rate)
    flow_regime = _classify_flow(cvd_slope)

    inst_score = _score_institutional(
        bias=bias,
        oi_slope=oi_slope,
        cvd_slope=cvd_slope,
        funding_rate=funding_rate,
    )

    available = any([oi_value is not None, cvd_slope is not None, funding_rate is not None])

    return {
        "institutional_score": int(inst_score),
        "binance_symbol": binance_symbol,
        "available": bool(available),
        "oi": oi_value,
        "oi_slope": oi_slope,
        "cvd_slope": cvd_slope,
        "funding_rate": funding_rate,
        "funding_regime": funding_regime,
        "crowding_regime": crowding_regime,
        "flow_regime": flow_regime,
        "warnings": warnings,
    }
