# =====================================================================
# institutional_data.py — Ultra Desk OI + CVD Engine (Binance Futures)
# =====================================================================
# Fournit un score institutionnel basé sur:
#   - Open Interest (OI) via /fapi/v1/openInterest
#   - CVD proxy via klines 1h (takerBuyBaseAssetVolume) /fapi/v1/klines
#   - Funding via premiumIndex /fapi/v1/premiumIndex
#
# API principale:
#   async def compute_full_institutional_analysis(symbol: str, bias: str) -> dict
#
# Retour:
#   {
#       "institutional_score": int (0..4),
#       "binance_symbol": str | None,
#       "available": bool,
#       "oi": float | None,
#       "oi_slope": float | None,
#       "cvd_slope": float | None,          # scale ~ [-3..+3] (compatible seuils 0.2 / 1.0 / 1.5)
#       "funding_rate": float | None,
#       "funding_regime": str,
#       "crowding_regime": str,
#       "flow_regime": str,
#       "warnings": list[str],
#   }
# =====================================================================

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Set

import aiohttp

LOGGER = logging.getLogger(__name__)

BINANCE_FAPI_BASE = "https://fapi.binance.com"

# ---------------------------------------------------------------------
# Cache léger pour éviter de spammer l'API
# ---------------------------------------------------------------------

_KLINES_CACHE: Dict[Tuple[str, str], Tuple[float, List[List[Any]]]] = {}
_FUNDING_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}

# Historique OI: symbol -> (timestamp_sec, oi_value)
_OI_HISTORY: Dict[str, Tuple[float, float]] = {}

_BINANCE_SYMBOLS: Optional[Set[str]] = None
_BINANCE_SYMBOLS_TS: float = 0.0

# TTLs (secondes)
KLINES_TTL = 60.0
FUNDING_TTL = 60.0
BINANCE_SYMBOLS_TTL = 900.0  # 15 minutes

# Rate limit soft (binance)
_HTTP_SEM = asyncio.Semaphore(8)

# Global HTTP session (réutilisée)
_SESSION: Optional[aiohttp.ClientSession] = None
_SESSION_LOCK = asyncio.Lock()

_UA = "desk-bot/inst/1.0"
_SYMBOL_SAN_RE = re.compile(r"[^A-Z0-9]+")


async def _get_session() -> aiohttp.ClientSession:
    global _SESSION
    async with _SESSION_LOCK:
        if _SESSION is not None and (not _SESSION.closed):
            return _SESSION
        timeout = aiohttp.ClientTimeout(total=10)
        connector = aiohttp.TCPConnector(limit=50, limit_per_host=20, ttl_dns_cache=300)
        _SESSION = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={"User-Agent": _UA, "Accept": "application/json"},
        )
        return _SESSION


async def close_http_session() -> None:
    """Optionnel: à appeler si tu veux fermer proprement la session à l'arrêt."""
    global _SESSION
    try:
        if _SESSION is not None and (not _SESSION.closed):
            await _SESSION.close()
    finally:
        _SESSION = None


# =====================================================================
# Helpers HTTP (avec retries soft)
# =====================================================================

async def _http_get(
    session: aiohttp.ClientSession,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    retries: int = 2,
) -> Any:
    url = BINANCE_FAPI_BASE + path
    params = params or {}

    for k in range(retries + 1):
        try:
            async with _HTTP_SEM:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        return await resp.json()

                    txt = await resp.text()
                    # 429 / 5xx -> retry soft
                    if resp.status in (418, 429, 500, 502, 503, 504) and k < retries:
                        ra = resp.headers.get("Retry-After")
                        sleep_s = float(ra) if ra and ra.replace(".", "", 1).isdigit() else (0.25 * (2 ** k))
                        sleep_s = min(3.0, max(0.15, sleep_s))
                        LOGGER.warning("[INST] HTTP %s GET %s retry=%s sleep=%.2fs params=%s resp=%s",
                                       resp.status, path, k + 1, sleep_s, params, txt[:220])
                        await asyncio.sleep(sleep_s)
                        continue

                    LOGGER.warning("[INST] HTTP %s GET %s params=%s resp=%s", resp.status, path, params, txt[:240])
                    return None

        except asyncio.TimeoutError:
            if k < retries:
                await asyncio.sleep(0.2 * (2 ** k))
                continue
            LOGGER.error("[INST] Timeout GET %s params=%s", path, params)
            return None
        except Exception as e:
            if k < retries:
                await asyncio.sleep(0.2 * (2 ** k))
                continue
            LOGGER.error("[INST] Exception GET %s params=%s: %s", path, params, e)
            return None

    return None


# =====================================================================
# Symboles Binance Futures (cache exchangeInfo)
# =====================================================================

async def _get_binance_symbols(session: aiohttp.ClientSession) -> Set[str]:
    global _BINANCE_SYMBOLS, _BINANCE_SYMBOLS_TS

    now = time.time()
    if _BINANCE_SYMBOLS is not None and (now - _BINANCE_SYMBOLS_TS) < BINANCE_SYMBOLS_TTL:
        return _BINANCE_SYMBOLS

    data = await _http_get(session, "/fapi/v1/exchangeInfo", params=None, retries=2)
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

def _normalize_bitget_symbol(symbol: str) -> str:
    """
    Rend le symbol "propre" pour tenter un mapping Binance:
    - retire suffixes type _UMCBL, _DMCBL, -PERP, -SWAP, etc.
    - garde uniquement A-Z0-9
    """
    s = str(symbol or "").upper().strip()
    if not s:
        return s

    # split sur séparateurs fréquents
    for sep in ("_", "-", ":"):
        if sep in s:
            s = s.split(sep)[0]

    # nettoyage
    s = _SYMBOL_SAN_RE.sub("", s)

    # certains brokers ajoutent USDTM / USDTPERP etc
    s = s.replace("USDTM", "USDT").replace("USDTPERP", "USDT").replace("USDT_PERP", "USDT")
    return s


def _map_symbol_to_binance(symbol: str, binance_symbols: Set[str]) -> Optional[str]:
    """
    Map Bitget -> Binance futures USDT perp.
    - direct: BTCUSDT -> BTCUSDT
    - 1000TOKENUSDT -> TOKENUSDT
    - normalisation suffixes
    """
    s = _normalize_bitget_symbol(symbol)
    if not s:
        return None

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

    params = {"symbol": binance_symbol, "interval": "1h", "limit": int(limit)}
    data = await _http_get(session, "/fapi/v1/klines", params=params, retries=2)
    if not isinstance(data, list) or len(data) == 0:
        return None

    _KLINES_CACHE[cache_key] = (now, data)
    return data


# =====================================================================
# Funding (premiumIndex)
# =====================================================================

async def _fetch_funding(session: aiohttp.ClientSession, binance_symbol: str) -> Optional[Dict[str, Any]]:
    now = time.time()
    cached = _FUNDING_CACHE.get(binance_symbol)
    if cached is not None:
        ts, data = cached
        if now - ts < FUNDING_TTL:
            return data

    params = {"symbol": binance_symbol}
    data = await _http_get(session, "/fapi/v1/premiumIndex", params=params, retries=2)
    if not isinstance(data, dict) or "symbol" not in data:
        return None

    _FUNDING_CACHE[binance_symbol] = (now, data)
    return data


# =====================================================================
# Open Interest snapshot
# =====================================================================

async def _fetch_open_interest(session: aiohttp.ClientSession, binance_symbol: str) -> Optional[float]:
    params = {"symbol": binance_symbol}
    data = await _http_get(session, "/fapi/v1/openInterest", params=params, retries=2)
    if not isinstance(data, dict) or "openInterest" not in data:
        return None
    try:
        return float(data["openInterest"])
    except Exception:
        return None


def _compute_oi_slope(binance_symbol: str, new_oi: Optional[float]) -> Optional[float]:
    """
    Slope simple vs dernier snapshot:
      (new - old) / old
    Si pas d'historique => 0.0
    """
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
# CVD proxy depuis klines
# =====================================================================

def _compute_cvd_slope_from_klines(klines: List[List[Any]], window: int = 40) -> Optional[float]:
    """
    Proxy flow: delta = 2*takerBuyBase - totalVol  (base asset units)
    On calcule:
      ratio = sum(delta) / sum(vol)  ∈ [-1..+1]
    Puis on scale pour coller à tes seuils:
      cvd_slope = ratio * 3.0  ∈ [-3..+3]
    """
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

        if vol <= 0:
            continue

        delta = 2.0 * taker_buy - vol
        sum_delta += delta
        sum_vol += vol

    if sum_vol <= 1e-12:
        return None

    ratio = sum_delta / sum_vol  # [-1..1]
    return float(ratio * 3.0)


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
    b = (bias or "").upper()
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


# =====================================================================
# Score institutionnel global (0..4)
# =====================================================================

def _score_institutional(
    bias: str,
    oi_slope: Optional[float],
    cvd_slope: Optional[float],
    funding_rate: Optional[float],
) -> int:
    """
    Score 0..4:
      - CVD directionnel (0..2)
      - OI slope (0..1)
      - Funding contrarian (0..1)
    """
    b = (bias or "").upper()
    score = 0

    # CVD directionnel
    if cvd_slope is not None:
        x = float(cvd_slope)
        if b == "LONG":
            if x >= 1.0:
                score += 2
            elif x >= 0.2:
                score += 1
        elif b == "SHORT":
            if x <= -1.0:
                score += 2
            elif x <= -0.2:
                score += 1

    # OI slope
    if oi_slope is not None:
        x = float(oi_slope)
        if b == "LONG" and x > 0.01:
            score += 1
        elif b == "SHORT" and x < -0.01:
            score += 1

    # Funding contrarian
    if funding_rate is not None:
        fr = float(funding_rate)
        if b == "LONG" and fr < -0.0005:
            score += 1
        elif b == "SHORT" and fr > 0.0005:
            score += 1

    return int(max(0, min(4, score)))


# =====================================================================
# API PRINCIPALE
# =====================================================================

async def compute_full_institutional_analysis(symbol: str, bias: str) -> Dict[str, Any]:
    """
    Score institutionnel via Binance USDT-M futures (free endpoints).
    Si symbole non mappé -> available=False + score=0 (le reste de ta pipeline peut bypass).
    """
    bias = (bias or "").upper()

    warnings: List[str] = []
    oi_value: Optional[float] = None
    oi_slope: Optional[float] = None
    cvd_slope: Optional[float] = None
    funding_rate: Optional[float] = None

    session = await _get_session()

    # 0) symbol list + mapping
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
            "funding_regime": "unknown",
            "crowding_regime": "unknown",
            "flow_regime": "unknown",
            "warnings": ["symbol_not_mapped_to_binance"],
        }

    # 1) Fetch en parallèle
    kl_task = _fetch_klines_1h(session, binance_symbol, limit=120)
    oi_task = _fetch_open_interest(session, binance_symbol)
    fu_task = _fetch_funding(session, binance_symbol)

    klines, oi_value, funding_data = await asyncio.gather(kl_task, oi_task, fu_task, return_exceptions=False)

    # CVD slope
    if not klines:
        warnings.append("no_klines")
        cvd_slope = None
    else:
        cvd_slope = _compute_cvd_slope_from_klines(klines, window=40)
        if cvd_slope is None:
            warnings.append("cvd_unavailable")

    # OI slope
    if oi_value is None:
        warnings.append("no_oi")
        oi_slope = None
    else:
        oi_slope = _compute_oi_slope(binance_symbol, oi_value)
        _OI_HISTORY[binance_symbol] = (time.time(), float(oi_value))

    # Funding
    if not isinstance(funding_data, dict):
        warnings.append("no_funding")
        funding_rate = None
    else:
        try:
            funding_rate = float(funding_data.get("lastFundingRate", "0"))
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

    # available = on a au moins une info exploitable
    available = any([
        oi_value is not None,
        cvd_slope is not None,
        funding_rate is not None,
    ])

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
