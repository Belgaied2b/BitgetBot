# =====================================================================
# institutional_data.py — Ultra Desk OI + CVD Engine (Binance Futures)
# =====================================================================
# Objectif :
#   - Score institutionnel pour TOUTES les cryptos (quand Binance couvre)
#   - Basé sur:
#       * Open Interest snapshot + slope (delta vs last snapshot)
#       * CVD proxy à partir des klines (taker buy volume) => ratio stable [-1..+1]
#       * Funding (premiumIndex.lastFundingRate)
#   - Cache TTL pour limiter l'API
#   - Fail-safe: jamais crash, retourne un dict exploitable
#
# API:
#   async def compute_full_institutional_analysis(symbol: str, bias: str) -> dict
# =====================================================================

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Set

import aiohttp

LOGGER = logging.getLogger(__name__)

BINANCE_FAPI_BASE = "https://fapi.binance.com"

# ---------------------------------------------------------------------
# Cache léger
# ---------------------------------------------------------------------

# Cache klines: (symbol, interval) -> (ts_sec, data)
_KLINES_CACHE: Dict[Tuple[str, str], Tuple[float, List[List[Any]]]] = {}

# Cache funding / premiumIndex: symbol -> (ts_sec, data)
_FUNDING_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}

# Historique OI: symbol -> (ts_sec, oi_value)
_OI_HISTORY: Dict[str, Tuple[float, float]] = {}

# Cache symboles Binance Futures USDT-perp
_BINANCE_SYMBOLS: Optional[Set[str]] = None
_BINANCE_SYMBOLS_TS: float = 0.0

# TTLs (sec)
KLINES_TTL = 60.0
FUNDING_TTL = 60.0
BINANCE_SYMBOLS_TTL = 900.0  # 15 min

# Limiteur global (évite bursts)
_HTTP_SEM = asyncio.Semaphore(20)

# Session HTTP globale (réutilisée)
_SESSION: Optional[aiohttp.ClientSession] = None
_SESSION_LOCK = asyncio.Lock()


# =====================================================================
# Session
# =====================================================================

async def _get_session() -> aiohttp.ClientSession:
    global _SESSION
    async with _SESSION_LOCK:
        if _SESSION is None or _SESSION.closed:
            timeout = aiohttp.ClientTimeout(total=10)
            connector = aiohttp.TCPConnector(limit=50, ttl_dns_cache=300)
            _SESSION = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return _SESSION


# =====================================================================
# Helpers HTTP
# =====================================================================

async def _http_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = BINANCE_FAPI_BASE + path
    session = await _get_session()
    async with _HTTP_SEM:
        try:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    LOGGER.warning("[INST] HTTP %s GET %s params=%s resp=%s", resp.status, path, params, txt[:250])
                    return None
                return await resp.json()
        except asyncio.TimeoutError:
            LOGGER.error("[INST] Timeout GET %s params=%s", path, params)
            return None
        except Exception as e:
            LOGGER.error("[INST] Exception GET %s params=%s: %s", path, params, e)
            return None


# =====================================================================
# Binance symbols (exchangeInfo) cached
# =====================================================================

async def _get_binance_symbols() -> Set[str]:
    global _BINANCE_SYMBOLS, _BINANCE_SYMBOLS_TS

    now = time.time()
    if _BINANCE_SYMBOLS is not None and (now - _BINANCE_SYMBOLS_TS) < BINANCE_SYMBOLS_TTL:
        return _BINANCE_SYMBOLS

    data = await _http_get("/fapi/v1/exchangeInfo", params=None)
    symbols: Set[str] = set()

    if not isinstance(data, dict) or "symbols" not in data:
        LOGGER.warning("[INST] Unable to fetch Binance exchangeInfo, keeping old symbols cache")
        if _BINANCE_SYMBOLS is None:
            _BINANCE_SYMBOLS = set()
        return _BINANCE_SYMBOLS

    for s in data.get("symbols", []) or []:
        try:
            if s.get("status") != "TRADING":
                continue
            if s.get("contractType") != "PERPETUAL":
                continue
            if s.get("quoteAsset") != "USDT":
                continue
            sym = str(s.get("symbol", "")).upper().strip()
            if sym:
                symbols.add(sym)
        except Exception:
            continue

    _BINANCE_SYMBOLS = symbols
    _BINANCE_SYMBOLS_TS = now
    LOGGER.info("[INST] Binance futures symbols loaded: %d", len(symbols))
    return _BINANCE_SYMBOLS


# =====================================================================
# Symbol mapping (Bitget -> Binance)
# =====================================================================

def _normalize_symbol(symbol: str) -> str:
    s = (symbol or "").upper().strip()
    # remove separators sometimes used
    s = s.replace("-", "").replace("_", "").replace("/", "")
    # common suffix artifacts (if any)
    s = s.replace("PERP", "")
    return s


def _map_symbol_to_binance(symbol: str, binance_symbols: Set[str]) -> Optional[str]:
    s = _normalize_symbol(symbol)

    # direct
    if s in binance_symbols:
        return s

    # 1000TOKENUSDT -> TOKENUSDT
    if s.startswith("1000"):
        alt = s[4:]
        if alt in binance_symbols:
            return alt

    # sometimes exchanges include "USDT" twice or weird forms
    # keep it minimal: return None if not found
    return None


# =====================================================================
# Fetch endpoints
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


async def _fetch_funding(binance_symbol: str) -> Optional[Dict[str, Any]]:
    now = time.time()
    cached = _FUNDING_CACHE.get(binance_symbol)
    if cached is not None:
        ts, data = cached
        if now - ts < FUNDING_TTL:
            return data

    data = await _http_get("/fapi/v1/premiumIndex", params={"symbol": binance_symbol})
    if not isinstance(data, dict) or "symbol" not in data:
        return None

    _FUNDING_CACHE[binance_symbol] = (now, data)
    return data


async def _fetch_open_interest(binance_symbol: str) -> Optional[float]:
    data = await _http_get("/fapi/v1/openInterest", params={"symbol": binance_symbol})
    if not isinstance(data, dict) or "openInterest" not in data:
        return None
    try:
        return float(data["openInterest"])
    except Exception:
        return None


def _compute_oi_slope(binance_symbol: str, new_oi: Optional[float]) -> Optional[float]:
    if new_oi is None:
        return None

    prev = _OI_HISTORY.get(binance_symbol)
    if prev is None:
        return 0.0  # neutral but valid

    _, old_oi = prev
    if old_oi <= 0:
        return 0.0

    slope = (float(new_oi) - float(old_oi)) / float(old_oi)
    return float(slope)


# =====================================================================
# CVD proxy (stable): ratio = sum(delta)/sum(volume) over window => [-1..+1]
# =====================================================================

def _compute_cvd_ratio_from_klines(klines: List[List[Any]], window: int = 40) -> Optional[float]:
    """
    delta per candle = 2*takerBuyBase - totalVolume
    ratio over window = sum(delta) / sum(volume)  -> stable [-1..+1]
    """
    try:
        if not klines or len(klines) < window + 5:
            return None

        sub = klines[-window:]
        sum_delta = 0.0
        sum_vol = 0.0

        for item in sub:
            try:
                vol = float(item[5])
                taker_buy = float(item[9])
            except Exception:
                continue

            if not (vol >= 0.0):
                continue

            delta = 2.0 * taker_buy - vol
            sum_delta += delta
            sum_vol += max(vol, 0.0)

        if sum_vol <= 1e-12:
            return None

        ratio = sum_delta / sum_vol
        # clamp
        if ratio > 1.0:
            ratio = 1.0
        if ratio < -1.0:
            ratio = -1.0
        return float(ratio)
    except Exception:
        return None


# =====================================================================
# Funding regimes / crowding / flow
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
    b = (bias or "").upper()

    # Desk logic:
    # - if you're LONG and funding is very positive -> longs crowded -> risky
    # - if you're LONG and funding is negative -> shorts crowded -> favorable
    if b == "LONG":
        if fr >= 0.0012:
            return "long_crowded_risky"
        if fr <= -0.0006:
            return "short_crowded_favorable"
        return "balanced"
    if b == "SHORT":
        if fr <= -0.0012:
            return "short_crowded_risky"
        if fr >= 0.0006:
            return "long_crowded_favorable"
        return "balanced"
    return "unknown"


def _classify_flow(cvd_ratio: Optional[float]) -> str:
    if cvd_ratio is None:
        return "unknown"
    x = float(cvd_ratio)
    if x >= 0.25:
        return "strong_buy"
    if x >= 0.10:
        return "buy"
    if x <= -0.25:
        return "strong_sell"
    if x <= -0.10:
        return "sell"
    return "neutral"


# =====================================================================
# Institutional score (desk): more usable, less "always 0"
# =====================================================================

def _score_institutional(
    bias: str,
    oi_slope: Optional[float],
    cvd_ratio: Optional[float],
    funding_rate: Optional[float],
    crowding_regime: str,
) -> int:
    """
    Score integer [0..4], desk-friendly.
    - Main driver: CVD ratio (stable)
    - OI slope: confirmation (even small)
    - Funding: contrarian / crowding-aware
    """
    b = (bias or "").upper()
    score = 0.0

    # ---- CVD ratio (stable) ----
    if cvd_ratio is not None:
        x = float(cvd_ratio)

        if b == "LONG":
            if x >= 0.25:
                score += 2.0
            elif x >= 0.10:
                score += 1.0
            elif x <= -0.25:
                score -= 1.0
            elif x <= -0.10:
                score -= 0.5

        elif b == "SHORT":
            if x <= -0.25:
                score += 2.0
            elif x <= -0.10:
                score += 1.0
            elif x >= 0.25:
                score -= 1.0
            elif x >= 0.10:
                score -= 0.5

    # ---- OI slope (confirmation) ----
    if oi_slope is not None:
        o = float(oi_slope)

        # thresholds are SMALL on purpose (OI moves slowly on 1h)
        strong_thr = 0.008   # 0.8%
        mild_thr = 0.002     # 0.2%
        contra_thr = 0.004

        if b == "LONG":
            if o >= strong_thr:
                score += 1.0
            elif o >= mild_thr:
                score += 0.5
            elif o <= -contra_thr:
                score -= 0.5

        elif b == "SHORT":
            if o <= -strong_thr:
                score += 1.0
            elif o <= -mild_thr:
                score += 0.5
            elif o >= contra_thr:
                score -= 0.5

    # ---- Funding / crowding ----
    if funding_rate is not None:
        fr = float(funding_rate)

        # contrarian bonus small
        if b == "LONG":
            if fr <= -0.0005:
                score += 0.8
            elif fr >= 0.0015:
                score -= 0.5
        elif b == "SHORT":
            if fr >= 0.0005:
                score += 0.8
            elif fr <= -0.0015:
                score -= 0.5

    # If clearly crowded risky, dampen a bit (avoid chasing crowded side)
    cr = (crowding_regime or "").lower()
    if "risky" in cr:
        score -= 0.4

    # clamp + convert to int (round)
    score_i = int(round(score))
    if score_i < 0:
        score_i = 0
    if score_i > 4:
        score_i = 4
    return score_i


# =====================================================================
# API PRINCIPALE
# =====================================================================

async def compute_full_institutional_analysis(symbol: str, bias: str) -> Dict[str, Any]:
    """
    Retourne un dict stable. Jamais crash.

    Important:
      - "available" = True si on a au moins 1 source exploitable (klines OR oi OR funding)
      - "institutional_score" = [0..4]
    """
    bias = (bias or "").upper()
    warnings: List[str] = []

    oi_value: Optional[float] = None
    oi_slope: Optional[float] = None
    cvd_ratio: Optional[float] = None
    funding_rate: Optional[float] = None

    # 0) symbol list + mapping
    try:
        binance_symbols = await _get_binance_symbols()
        binance_symbol = _map_symbol_to_binance(symbol, binance_symbols)
    except Exception as e:
        binance_symbol = None
        warnings.append(f"exchangeInfo_error:{e}")

    if binance_symbol is None:
        return {
            "institutional_score": 0,
            "binance_symbol": None,
            "available": False,
            "oi": None,
            "oi_slope": None,
            "cvd_slope": None,          # kept key for compatibility (we store ratio here)
            "cvd_ratio": None,          # new explicit key
            "funding_rate": None,
            "funding_regime": "unknown",
            "crowding_regime": "unknown",
            "flow_regime": "unknown",
            "warnings": warnings or ["symbol_not_mapped_to_binance"],
        }

    # 1) klines -> CVD ratio
    try:
        klines = await _fetch_klines_1h(binance_symbol, limit=140)
        if not klines:
            warnings.append("no_klines")
            cvd_ratio = None
        else:
            cvd_ratio = _compute_cvd_ratio_from_klines(klines, window=40)
            if cvd_ratio is None:
                warnings.append("cvd_ratio_none")
    except Exception as e:
        warnings.append(f"klines_error:{e}")
        cvd_ratio = None

    # 2) OI snapshot + slope (vs last snapshot)
    try:
        oi_value = await _fetch_open_interest(binance_symbol)
        if oi_value is None:
            warnings.append("no_oi")
            oi_slope = None
        else:
            oi_slope = _compute_oi_slope(binance_symbol, oi_value)
            _OI_HISTORY[binance_symbol] = (time.time(), float(oi_value))
    except Exception as e:
        warnings.append(f"oi_error:{e}")
        oi_value = None
        oi_slope = None

    # 3) funding
    try:
        funding_data = await _fetch_funding(binance_symbol)
        if funding_data is None:
            warnings.append("no_funding")
            funding_rate = None
        else:
            try:
                funding_rate = float(funding_data.get("lastFundingRate", "0") or 0.0)
            except Exception:
                funding_rate = None
                warnings.append("funding_parse_error")
    except Exception as e:
        warnings.append(f"funding_error:{e}")
        funding_rate = None

    funding_regime = _classify_funding(funding_rate)
    crowding_regime = _classify_crowding(bias, funding_rate)
    flow_regime = _classify_flow(cvd_ratio)

    # Score
    inst_score = _score_institutional(
        bias=bias,
        oi_slope=oi_slope,
        cvd_ratio=cvd_ratio,
        funding_rate=funding_rate,
        crowding_regime=crowding_regime,
    )

    available = any([
        (oi_value is not None),
        (cvd_ratio is not None),
        (funding_rate is not None),
    ])

    return {
        "institutional_score": int(inst_score),
        "binance_symbol": binance_symbol,
        "available": bool(available),
        "oi": oi_value,
        "oi_slope": oi_slope,
        # compatibility: old code reads cvd_slope -> we store ratio here
        "cvd_slope": cvd_ratio,
        "cvd_ratio": cvd_ratio,
        "funding_rate": funding_rate,
        "funding_regime": funding_regime,
        "crowding_regime": crowding_regime,
        "flow_regime": flow_regime,
        "warnings": warnings,
    }
