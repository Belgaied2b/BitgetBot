# =====================================================================
# institutional_data.py — Ultra Desk OI + CVD Engine (Binance Futures)
# =====================================================================
# Objectif :
#   - Fournir un score institutionnel pour TOUTES les cryptos, basé sur :
#       * Open Interest (OI)
#       * CVD (Cumulative Volume Delta) via taker buy volume
#       * Funding (premiumIndex)
#       * (Optionnel) Orderbook imbalance (depth)
#   - Endpoints gratuits Binance USDT-M futures.
#   - Léger : appelé seulement si structure/BOS OK.
#
# API principale :
#   async def compute_full_institutional_analysis(symbol: str, bias: str) -> dict
# =====================================================================

from __future__ import annotations

import asyncio
import logging
import os  # ✅ FIX 1
import time
from typing import Any, Dict, List, Optional, Tuple, Set

import aiohttp

LOGGER = logging.getLogger(__name__)

BINANCE_FAPI_BASE = "https://fapi.binance.com"

# Order book (depth) microstructure (lightweight, optional)
# Docs: GET /fapi/v1/depth (USDS-M futures)
def _safe_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _safe_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

ORDERBOOK_LIMIT = _safe_int_env("BINANCE_ORDERBOOK_LIMIT", 100)  # valid: 5..1000
ORDERBOOK_LIMIT = max(5, min(1000, ORDERBOOK_LIMIT))
ORDERBOOK_TTL_S = _safe_float_env("BINANCE_ORDERBOOK_TTL_S", 12.0)

# threshold fixed (as you had): avoid noise
ORDERBOOK_IMB_THRESHOLD = _safe_float_env("ORDERBOOK_IMB_THRESHOLD", 0.08)

_ORDERBOOK_CACHE: Dict[str, Tuple[float, Optional[float], str]] = {}  # sym -> (ts, imbalance, regime)
_ORDERBOOK_LOCK = asyncio.Lock()

# ---------------------------------------------------------------------
# Cache léger pour éviter de spammer l'API
# ---------------------------------------------------------------------

# Cache klines: (symbol, interval) -> (timestamp_sec, data)
_KLINES_CACHE: Dict[Tuple[str, str], Tuple[float, List[List[Any]]]] = {}

# Cache funding / premiumIndex: symbol -> (timestamp_sec, data)
_FUNDING_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}

# Historique OI: symbol -> (timestamp_sec, oi_value)
_OI_HISTORY: Dict[str, Tuple[float, float]] = {}

# Cache symboles Binance Futures USDT-perp
_BINANCE_SYMBOLS: Optional[Set[str]] = None
_BINANCE_SYMBOLS_TS: float = 0.0

# TTLs (en secondes)
KLINES_TTL = 60.0
FUNDING_TTL = 60.0
OI_TTL = 60.0
BINANCE_SYMBOLS_TTL = 900.0  # 15 minutes


# =====================================================================
# Helpers HTTP
# =====================================================================

async def _http_get(
    session: aiohttp.ClientSession,
    path: str,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    url = BINANCE_FAPI_BASE + path
    try:
        async with session.get(url, params=params, timeout=10) as resp:
            if resp.status != 200:
                txt = await resp.text()
                LOGGER.warning("[INST] HTTP %s GET %s params=%s resp=%s", resp.status, path, params, (txt or "")[:250])
                return None
            return await resp.json()
    except asyncio.TimeoutError:
        LOGGER.warning("[INST] Timeout GET %s params=%s", path, params)
        return None
    except Exception as e:
        LOGGER.warning("[INST] Exception GET %s params=%s err=%s", path, params, e)
        return None


async def _fetch_orderbook_imbalance(session: aiohttp.ClientSession, binance_symbol: str) -> Tuple[Optional[float], str]:
    """
    Returns (imbalance, regime).
    imbalance in [-1..+1] computed from summed qty of bids/asks.
    regime: strong_buy / strong_sell / neutral / unavailable
    """
    sym = (binance_symbol or "").upper().strip()
    if not sym:
        return None, "unavailable"

    now = time.time()
    async with _ORDERBOOK_LOCK:
        cached = _ORDERBOOK_CACHE.get(sym)
        if cached and (now - float(cached[0])) < ORDERBOOK_TTL_S:
            return cached[1], cached[2]

    params = {"symbol": sym, "limit": int(ORDERBOOK_LIMIT)}
    data = await _http_get(session, "/fapi/v1/depth", params=params)

    imb: Optional[float] = None
    regime = "unavailable"
    try:
        if isinstance(data, dict) and "bids" in data and "asks" in data:
            bids = data.get("bids") or []
            asks = data.get("asks") or []
            bid_qty = 0.0
            ask_qty = 0.0

            for p, q in bids:
                bid_qty += float(q)
            for p, q in asks:
                ask_qty += float(q)

            den = bid_qty + ask_qty
            if den > 0:
                imb = (bid_qty - ask_qty) / den
                if imb >= ORDERBOOK_IMB_THRESHOLD:
                    regime = "strong_buy"
                elif imb <= -ORDERBOOK_IMB_THRESHOLD:
                    regime = "strong_sell"
                else:
                    regime = "neutral"
    except Exception:
        imb = None
        regime = "unavailable"

    async with _ORDERBOOK_LOCK:
        _ORDERBOOK_CACHE[sym] = (now, imb, regime)

    return imb, regime


# =====================================================================
# Symboles Binance Futures (cache exchangeInfo)
# =====================================================================

async def _get_binance_symbols(session: aiohttp.ClientSession) -> Set[str]:
    """
    Charge / met à jour la liste des symboles Binance Futures USDT-perp (PERPETUAL, quoteAsset=USDT).
    Résultat mis en cache pour BINANCE_SYMBOLS_TTL secondes.
    """
    global _BINANCE_SYMBOLS, _BINANCE_SYMBOLS_TS

    now = time.time()
    if _BINANCE_SYMBOLS is not None and (now - _BINANCE_SYMBOLS_TS) < BINANCE_SYMBOLS_TTL:
        return _BINANCE_SYMBOLS

    data = await _http_get(session, "/fapi/v1/exchangeInfo", params=None)
    symbols: Set[str] = set()

    if not isinstance(data, dict) or "symbols" not in data:
        LOGGER.warning("[INST] Unable to fetch Binance exchangeInfo, keeping old symbols cache")
        if _BINANCE_SYMBOLS is None:
            _BINANCE_SYMBOLS = set()
        return _BINANCE_SYMBOLS

    for s in data["symbols"]:
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
    """
    Map symbol Bitget (ex: 'BTCUSDT') vers Binance USDT-M futures symbol.
    """
    s = str(symbol or "").upper()

    if s in binance_symbols:
        return s

    if s.startswith("1000"):
        alt = s[4:]
        if alt in binance_symbols:
            return alt

    return None


# =====================================================================
# Fetch klines 1h Binance (pour CVD)
# =====================================================================

async def _fetch_klines_1h(session: aiohttp.ClientSession, binance_symbol: str, limit: int = 120) -> Optional[List[List[Any]]]:
    cache_key = (binance_symbol, "1h")
    now = time.time()

    cached = _KLINES_CACHE.get(cache_key)
    if cached is not None:
        ts, data = cached
        if now - ts < KLINES_TTL:
            return data

    params = {"symbol": binance_symbol, "interval": "1h", "limit": limit}
    data = await _http_get(session, "/fapi/v1/klines", params=params)
    if not isinstance(data, list) or len(data) == 0:
        return None

    _KLINES_CACHE[cache_key] = (now, data)
    return data


# =====================================================================
# Fetch funding (premiumIndex)
# =====================================================================

async def _fetch_funding(session: aiohttp.ClientSession, binance_symbol: str) -> Optional[Dict[str, Any]]:
    now = time.time()
    cached = _FUNDING_CACHE.get(binance_symbol)
    if cached is not None:
        ts, data = cached
        if now - ts < FUNDING_TTL:
            return data

    data = await _http_get(session, "/fapi/v1/premiumIndex", params={"symbol": binance_symbol})
    if not isinstance(data, dict) or "symbol" not in data:
        return None

    _FUNDING_CACHE[binance_symbol] = (now, data)
    return data


# =====================================================================
# Fetch open interest snapshot
# =====================================================================

async def _fetch_open_interest(session: aiohttp.ClientSession, binance_symbol: str) -> Optional[float]:
    data = await _http_get(session, "/fapi/v1/openInterest", params={"symbol": binance_symbol})
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
        return 0.0

    _, old_oi = prev
    if old_oi <= 0:
        return 0.0

    return float((new_oi - old_oi) / old_oi)


# =====================================================================
# CVD à partir des klines
# =====================================================================

def _compute_cvd_slope_from_klines(klines: List[List[Any]], window: int = 40) -> Optional[float]:
    if not klines or len(klines) < window + 5:
        return None

    sub = klines[-(window + 5):]
    cvs: List[float] = []
    cvd = 0.0

    for item in sub:
        try:
            vol = float(item[5])
            taker_buy = float(item[9])
        except Exception:
            continue
        delta = 2.0 * taker_buy - vol
        cvd += delta
        cvs.append(cvd)

    if len(cvs) < window:
        return None

    last_segment = cvs[-window:]
    start = last_segment[0]
    end = last_segment[-1]

    if abs(start) < 1e-8:
        denom = max(abs(end), 1e-8)
        slope = (end - start) / denom
    else:
        slope = (end - start) / abs(start)

    return float(slope)


# =====================================================================
# Funding / crowding regimes
# =====================================================================

def _classify_funding(funding_rate: Optional[float]) -> str:
    if funding_rate is None:
        return "unknown"
    fr = funding_rate
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

    fr = funding_rate
    b = bias.upper()
    if b == "LONG":
        if fr <= -0.001:
            return "short_crowded_favorable"
        if fr >= 0.001:
            return "long_crowded_risky"
        return "balanced"
    if b == "SHORT":
        if fr >= 0.001:
            return "long_crowded_favorable"
        if fr <= -0.001:
            return "short_crowded_risky"
        return "balanced"
    return "unknown"


def _classify_flow(cvd_slope: Optional[float]) -> str:
    if cvd_slope is None:
        return "unknown"

    x = cvd_slope
    if x >= 1.0:
        return "strong_buy"
    if x >= 0.2:
        return "buy"
    if x <= -1.0:
        return "strong_sell"
    if x <= -0.2:
        return "sell"
    return "neutral"


# =====================================================================
# Score institutionnel global
# =====================================================================

def _score_institutional(
    bias: str,
    oi_slope: Optional[float],
    cvd_slope: Optional[float],
    funding_rate: Optional[float],
    ob_regime: Optional[str] = None,
) -> int:
    b = bias.upper()
    score = 0

    # CVD directionnel
    if cvd_slope is not None:
        if b == "LONG":
            if cvd_slope >= 1.0:
                score += 2
            elif cvd_slope >= 0.2:
                score += 1
        elif b == "SHORT":
            if cvd_slope <= -1.0:
                score += 2
            elif cvd_slope <= -0.2:
                score += 1

    # OI slope : renfort
    if oi_slope is not None:
        if b == "LONG" and oi_slope > 0.01:
            score += 1
        elif b == "SHORT" and oi_slope < -0.01:
            score += 1

    # Funding contrarian
    if funding_rate is not None:
        if b == "LONG" and funding_rate < -0.0005:
            score += 1
        elif b == "SHORT" and funding_rate > 0.0005:
            score += 1

    # Orderbook bonus (optional)
    if ob_regime:
        ob = str(ob_regime)
        if b == "LONG" and ob == "strong_buy":
            score += 1
        elif b == "SHORT" and ob == "strong_sell":
            score += 1

    return int(max(0, min(5, score)))


# =====================================================================
# API PRINCIPALE
# =====================================================================

async def compute_full_institutional_analysis(symbol: str, bias: str) -> Dict[str, Any]:
    """
    Calcule un score institutionnel pour un symbol Bitget donné, en utilisant
    les données Binance USDT-M Futures (klines, OI, funding, orderbook).

    - Si Binance ne connaît pas le symbole -> score 0, available=False.
    """
    bias_u = str(bias or "").upper()
    if bias_u not in ("LONG", "SHORT"):
        bias_u = "LONG"

    warnings: List[str] = []
    oi_value: Optional[float] = None
    oi_slope: Optional[float] = None
    cvd_slope: Optional[float] = None
    funding_rate: Optional[float] = None
    ob_imbalance: Optional[float] = None
    ob_regime: str = "unavailable"

    binance_symbol: Optional[str] = None

    async with aiohttp.ClientSession() as session:
        # 0) Liste des symboles Binance Futures
        binance_symbols = await _get_binance_symbols(session)
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
                "ob_imbalance": None,
                "ob_regime": "unavailable",
                "funding_regime": "unknown",
                "crowding_regime": "unknown",
                "flow_regime": "unknown",
                "warnings": ["symbol_not_mapped_to_binance"],
            }

        # 1) Fetch concurrent (plus "desk")
        t_kl = asyncio.create_task(_fetch_klines_1h(session, binance_symbol, limit=120))
        t_oi = asyncio.create_task(_fetch_open_interest(session, binance_symbol))
        t_fu = asyncio.create_task(_fetch_funding(session, binance_symbol))
        t_ob = asyncio.create_task(_fetch_orderbook_imbalance(session, binance_symbol))

        klines = await t_kl
        oi_value = await t_oi
        funding_data = await t_fu
        ob_imbalance, ob_regime = await t_ob

        # 1) CVD
        if not klines:
            warnings.append("no_klines")
            cvd_slope = None
        else:
            cvd_slope = _compute_cvd_slope_from_klines(klines, window=40)

        # 2) OI + slope
        if oi_value is None:
            warnings.append("no_oi")
            oi_slope = None
        else:
            oi_slope = _compute_oi_slope(binance_symbol, oi_value)
            _OI_HISTORY[binance_symbol] = (time.time(), oi_value)

        # 3) Funding
        if funding_data is None:
            warnings.append("no_funding")
            funding_rate = None
        else:
            try:
                funding_rate = float(funding_data.get("lastFundingRate", "0"))
            except Exception:
                funding_rate = None
                warnings.append("funding_parse_error")

    # Regimes
    funding_regime = _classify_funding(funding_rate)
    crowding_regime = _classify_crowding(bias_u, funding_rate)
    flow_regime = _classify_flow(cvd_slope)

    # available = on a AU MOINS une info exploitable
    available = any([
        oi_value is not None,
        cvd_slope is not None,
        funding_rate is not None,
        ob_imbalance is not None,
    ])

    inst_score = _score_institutional(
        bias=bias_u,
        oi_slope=oi_slope,
        cvd_slope=cvd_slope,
        funding_rate=funding_rate,
        ob_regime=ob_regime,
    )

    return {
        "institutional_score": inst_score,
        "binance_symbol": binance_symbol,
        "available": available,
        "oi": oi_value,
        "oi_slope": oi_slope,
        "cvd_slope": cvd_slope,
        "funding_rate": funding_rate,
        "ob_imbalance": ob_imbalance,
        "ob_regime": ob_regime,
        "funding_regime": funding_regime,
        "crowding_regime": crowding_regime,
        "flow_regime": flow_regime,
        "warnings": warnings,
    }
