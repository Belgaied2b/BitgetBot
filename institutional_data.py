# =====================================================================
# institutional_data.py — Ultra Desk OI + CVD Engine (Binance Futures)
# =====================================================================
# Objectif :
#   - Fournir un score institutionnel pour TOUTES les cryptos, basé sur :
#       * Open Interest (OI)
#       * CVD (Cumulative Volume Delta) via taker buy volume
#       * Funding (premiumIndex)
#   - Utiliser exclusivement des endpoints gratuits Binance USDT-M futures.
#   - Rester léger : appelé uniquement quand la structure / BOS est OK,
#     donc pas 543 symboles à chaque scan.
#
# API principale :
#
#   async def compute_full_institutional_analysis(symbol: str, bias: str) -> dict:
#       bias: "LONG" ou "SHORT"
#
# Retourne un dict du type :
#   {
#       "institutional_score": int,
#       "binance_symbol": str | None,
#       "available": bool,
#       "oi": float | None,
#       "oi_slope": float | None,
#       "cvd_slope": float | None,
#       "funding_rate": float | None,
#       "funding_regime": str,
#       "crowding_regime": str,
#       "flow_regime": str,
#       "warnings": list[str],
#   }
#
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
# Cache léger pour éviter de spammer l'API
# ---------------------------------------------------------------------

# Cache klines: (symbol, interval) -> (timestamp_sec, data)
_KLINES_CACHE: Dict[Tuple[str, str], Tuple[float, List[List[Any]]]] = {}

# Cache funding / premiumIndex: symbol -> (timestamp_sec, data)
_FUNDING_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}

# Historique OI: symbol -> (timestamp_sec, oi_value)
# (on garde le dernier snapshot pour calculer un slope simple)
_OI_HISTORY: Dict[str, Tuple[float, float]] = {}

# Cache symboles Binance Futures USDT-perp
_BINANCE_SYMBOLS: Optional[Set[str]] = None
_BINANCE_SYMBOLS_TS: float = 0.0

# TTLs (en secondes)
KLINES_TTL = 60.0
FUNDING_TTL = 60.0
OI_TTL = 60.0  # si on veut un jour filtrer par temps, déjà prêt
BINANCE_SYMBOLS_TTL = 900.0  # 15 minutes pour la liste des symboles


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
                LOGGER.warning(f"[INST] HTTP {resp.status} GET {path} params={params} resp={txt}")
                return None
            return await resp.json()
    except asyncio.TimeoutError:
        LOGGER.error(f"[INST] Timeout GET {path} params={params}")
        return None
    except Exception as e:
        LOGGER.error(f"[INST] Exception GET {path} params={params}: {e}")
        return None


# =====================================================================
# Symboles Binance Futures (cache exchangeInfo)
# =====================================================================

async def _get_binance_symbols(
    session: aiohttp.ClientSession,
) -> Set[str]:
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

    - Cas direct : BTCUSDT -> BTCUSDT si présent dans binance_symbols.
    - Cas 1000TOKENUSDT -> TOKENUSDT si TOKENUSDT est présent.
    - Sinon -> None (pas de couverture insti Binance).
    """
    s = symbol.upper()

    # Mapping direct
    if s in binance_symbols:
        return s

    # Cas 1000TOKENUSDT -> TOKENUSDT
    if s.startswith("1000"):
        alt = s[4:]
        if alt in binance_symbols:
            return alt

    # Ici tu peux ajouter d'autres règles si besoin (tokens exotiques)
    return None


# =====================================================================
# Fetch klines 1h Binance (pour CVD)
# =====================================================================

async def _fetch_klines_1h(
    session: aiohttp.ClientSession,
    binance_symbol: str,
    limit: int = 100,
) -> Optional[List[List[Any]]]:
    """
    Récupère les klines 1h Binance Futures USDT-M :
    /fapi/v1/klines?symbol=...&interval=1h&limit=...

    Format klines Binance :
      [
        [
          0  openTime,
          1  open,
          2  high,
          3  low,
          4  close,
          5  volume,
          6  closeTime,
          7  quoteAssetVolume,
          8  numberOfTrades,
          9  takerBuyBaseAssetVolume,
          10 takerBuyQuoteAssetVolume,
          11 ignore
        ],
        ...
      ]
    """
    cache_key = (binance_symbol, "1h")
    now = time.time()

    cached = _KLINES_CACHE.get(cache_key)
    if cached is not None:
        ts, data = cached
        if now - ts < KLINES_TTL:
            return data

    params = {
        "symbol": binance_symbol,
        "interval": "1h",
        "limit": limit,
    }
    data = await _http_get(session, "/fapi/v1/klines", params=params)
    if not isinstance(data, list) or len(data) == 0:
        return None

    _KLINES_CACHE[cache_key] = (now, data)
    return data


# =====================================================================
# Fetch funding (premiumIndex)
# =====================================================================

async def _fetch_funding(
    session: aiohttp.ClientSession,
    binance_symbol: str,
) -> Optional[Dict[str, Any]]:
    """
    Récupère premiumIndex / funding actuel :
    /fapi/v1/premiumIndex?symbol=...

    Retour type:
    {
      "symbol": "BTCUSDT",
      "markPrice": "87879.59652239",
      "indexPrice": "87859.22134211",
      "lastFundingRate": "0.00010000",
      ...
    }
    """
    now = time.time()
    cached = _FUNDING_CACHE.get(binance_symbol)
    if cached is not None:
        ts, data = cached
        if now - ts < FUNDING_TTL:
            return data

    params = {"symbol": binance_symbol}
    data = await _http_get(session, "/fapi/v1/premiumIndex", params=params)
    if not isinstance(data, dict) or "symbol" not in data:
        return None

    _FUNDING_CACHE[binance_symbol] = (now, data)
    return data


# =====================================================================
# Fetch open interest snapshot
# =====================================================================

async def _fetch_open_interest(
    session: aiohttp.ClientSession,
    binance_symbol: str,
) -> Optional[float]:
    """
    Récupère le snapshot d'open interest :
    /fapi/v1/openInterest?symbol=...

    Retour type:
    {
      "symbol": "BTCUSDT",
      "openInterest": "12345.678",
      ...
    }

    NOTE IMPORTANTE :
      - Ici, on NE met PAS à jour _OI_HISTORY.
        On laisse _compute_oi_slope() utiliser l'ancien snapshot,
        puis compute_full_institutional_analysis() mettra à jour l'historique
        après calcul du slope. Sinon on aurait toujours slope = 0.
    """
    params = {"symbol": binance_symbol}
    data = await _http_get(session, "/fapi/v1/openInterest", params=params)
    if not isinstance(data, dict) or "openInterest" not in data:
        return None

    try:
        oi = float(data["openInterest"])
    except Exception:
        return None

    return oi


def _compute_oi_slope(binance_symbol: str, new_oi: Optional[float]) -> Optional[float]:
    """
    Calcule une pente simple d'OI par rapport au dernier snapshot,
    si disponible. Retourne un ratio approximatif.

    - new_oi: valeur courante
    - historique dans _OI_HISTORY (timestamp, old_oi)

    Slope approximatif :
      (new_oi - old_oi) / max(old_oi, 1e-8)
    """
    if new_oi is None:
        return None

    prev = _OI_HISTORY.get(binance_symbol)
    if prev is None:
        # pas d'historique -> slope neutre mais valide
        return 0.0

    _, old_oi = prev
    if old_oi <= 0:
        return 0.0

    slope = (new_oi - old_oi) / old_oi
    return float(slope)


# =====================================================================
# CVD à partir des klines
# =====================================================================

def _compute_cvd_slope_from_klines(
    klines: List[List[Any]],
    window: int = 40,
) -> Optional[float]:
    """
    Calcule un CVD (Cumulative Volume Delta) approximatif à partir
    des klines Binance.

    Pour chaque kline :
      delta = 2 * takerBuyBase - totalVolume
    (si takerBuyBase = totalVolume -> delta = +totalVolume)
    (si takerBuyBase = 0          -> delta = -totalVolume)

    On cumule le delta, puis on prend la pente sur les N dernières barres.
    """
    if not klines or len(klines) < window + 5:
        return None

    sub = klines[-(window + 5) :]
    cvs: List[float] = []
    cvd = 0.0

    for item in sub:
        try:
            vol = float(item[5])       # total volume
            taker_buy = float(item[9]) # taker buy base volume
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
    """
    Classe le funding en régimes :
      - "very_negative" / "negative" / "neutral" / "positive" / "very_positive"
    """
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
    """
    Crowding prédateur, version très simple :

      - Pour un LONG :
          * funding très négatif => shorts crowdés (bon pour LONG)
          * funding très positif => longs crowdés (risque de squeeze down)

      - Pour un SHORT :
          * funding très positif => longs crowdés (bon pour SHORT)
          * funding très négatif => shorts crowdés (risque de squeeze up)
    """
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
    elif b == "SHORT":
        if fr >= 0.001:
            return "long_crowded_favorable"
        if fr <= -0.001:
            return "short_crowded_risky"
        return "balanced"
    else:
        return "unknown"


def _classify_flow(cvd_slope: Optional[float]) -> str:
    """
    Simple classification du flux depuis la pente de CVD :
      - "strong_buy" / "buy" / "neutral" / "sell" / "strong_sell"
    """
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
) -> int:
    """
    Construit un score institutionnel simple dans [0, 4].

    Logique :
      - CVD directionnel = facteur principal (jusqu'à +2)
      - OI slope = renfort (jusqu'à +1)
      - Funding contrarian = bonus (jusqu'à +1)
    """
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

    score = max(0, min(4, score))
    return int(score)


# =====================================================================
# API PRINCIPALE
# =====================================================================

async def compute_full_institutional_analysis(symbol: str, bias: str) -> Dict[str, Any]:
    """
    Calcule un score institutionnel pour un symbol Bitget donné, en utilisant
    les données Binance USDT-M Futures (klines, OI, funding).

    - Si Binance ne connaît pas le symbole -> score neutre (0), available=False,
      mais le reste de la pipeline peut continuer (structure/momentum/RR).
    """
    bias = bias.upper()

    warnings: List[str] = []
    oi_value: Optional[float] = None
    oi_slope: Optional[float] = None
    cvd_slope: Optional[float] = None
    funding_rate: Optional[float] = None

    async with aiohttp.ClientSession() as session:
        # 0) Récupérer / rafraîchir la liste des symboles Binance Futures
        binance_symbols = await _get_binance_symbols(session)
        binance_symbol = _map_symbol_to_binance(symbol, binance_symbols)

        # Si aucun mapping viable -> pas de couverture insti Binance
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
        klines = await _fetch_klines_1h(session, binance_symbol, limit=120)
        if not klines:
            warnings.append("no_klines")
            cvd_slope = None
        else:
            cvd_slope = _compute_cvd_slope_from_klines(klines, window=40)

        # 2) OI snapshot + slope (utilise l'historique _OI_HISTORY)
        oi_value = await _fetch_open_interest(session, binance_symbol)
        if oi_value is None:
            warnings.append("no_oi")
            oi_slope = None
        else:
            oi_slope = _compute_oi_slope(binance_symbol, oi_value)
            # met à jour l'historique APRÈS calcul du slope
            _OI_HISTORY[binance_symbol] = (time.time(), oi_value)

        # 3) Funding / premiumIndex
        funding_data = await _fetch_funding(session, binance_symbol)
        if funding_data is None:
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

    # available = on a AU MOINS une info exploitable (klines, oi ou funding)
    available = any([
        oi_value is not None,
        cvd_slope is not None,
        funding_rate is not None,
    ])

    return {
        "institutional_score": inst_score,
        "binance_symbol": binance_symbol,
        "available": available,
        "oi": oi_value,
        "oi_slope": oi_slope,
        "cvd_slope": cvd_slope,
        "funding_rate": funding_rate,
        "funding_regime": funding_regime,
        "crowding_regime": crowding_regime,
        "flow_regime": flow_regime,
        "warnings": warnings,
    }
