# =====================================================================
# institutional_data.py — Ultra Desk OI + CVD + Tape + Orderbook Engine
#   (Binance USDT-M Futures, free endpoints only)
# =====================================================================
# Objectif :
#   - Fournir un score institutionnel (0..4) pour TOUTES les cryptos
#     couvertes par Binance USDT-M Perpetual, basé sur :
#       * Open Interest (snapshot + hist slope)
#       * CVD (klines 1h via takerBuyBaseAssetVolume)
#       * Tape (aggTrades, delta 1m/5m)
#       * Funding (premiumIndex + fundingRate history)
#       * Basis (mark-index spread)
#       * Orderbook imbalance (depth)
#       * (Optionnel) Liquidations (allForceOrders) — best effort
#       * (Bonus) Long/Short ratios (global/top/taker) — best effort
#
# API principale :
#   async def compute_full_institutional_analysis(symbol: str, bias: str) -> dict
#       bias: "LONG" ou "SHORT"
#
# Retour minimal compatible :
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
#       # + aliases pour compat analyze_signal.py :
#       "tape_delta": float | None,
#       "orderbook_imbalance": float | None,
#       "components": {"oi_ok": bool, "funding_ok": bool, "cvd_ok": bool}
#   }
# =====================================================================

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
import numpy as np

LOGGER = logging.getLogger(__name__)

BINANCE_FAPI_BASE = "https://fapi.binance.com"

# ---------------------------------------------------------------------
# Cache léger pour éviter de spammer l'API
# ---------------------------------------------------------------------

# Cache generic: (key) -> (timestamp, data)
_KLINES_CACHE: Dict[Tuple[str, str, int], Tuple[float, Any]] = {}
_DEPTH_CACHE: Dict[Tuple[str, int], Tuple[float, Any]] = {}
_TRADES_CACHE: Dict[Tuple[str, int], Tuple[float, Any]] = {}
_FUNDING_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_FUNDING_HIST_CACHE: Dict[Tuple[str, int], Tuple[float, Any]] = {}
_OI_CACHE: Dict[str, Tuple[float, Any]] = {}
_OI_HIST_CACHE: Dict[Tuple[str, str, int], Tuple[float, Any]] = {}
_FORCE_CACHE: Dict[Tuple[str, int], Tuple[float, Any]] = {}
_LSR_CACHE: Dict[Tuple[str, str, str, int], Tuple[float, Any]] = {}

# Last OI snapshot for slope (in-memory)
_OI_HISTORY: Dict[str, Tuple[float, float]] = {}

# Cache symboles Binance Futures USDT-perp
_BINANCE_SYMBOLS: Optional[Set[str]] = None
_BINANCE_SYMBOLS_TS: float = 0.0

# TTLs (seconds)
KLINES_TTL = 60.0
DEPTH_TTL = 6.0
TRADES_TTL = 6.0
FUNDING_TTL = 60.0
FUNDING_HIST_TTL = 300.0
OI_TTL = 60.0
OI_HIST_TTL = 300.0
FORCE_TTL = 120.0
LSR_TTL = 300.0
BINANCE_SYMBOLS_TTL = 900.0


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
                LOGGER.warning(
                    "[INST] HTTP %s GET %s params=%s resp=%s",
                    resp.status,
                    path,
                    params,
                    (txt or "")[:200],
                )
                return None
            return await resp.json()
    except asyncio.TimeoutError:
        LOGGER.error("[INST] Timeout GET %s params=%s", path, params)
        return None
    except Exception as e:
        LOGGER.error("[INST] Exception GET %s params=%s: %s", path, params, e)
        return None


# =====================================================================
# Symboles Binance Futures (cache exchangeInfo)
# =====================================================================

async def _get_binance_symbols(session: aiohttp.ClientSession) -> Set[str]:
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
# Symbol mapping KuCoin/Bitget -> Binance
# =====================================================================

def _map_symbol_to_binance(symbol: str, binance_symbols: Set[str]) -> Optional[str]:
    """
    Map symbol to Binance USDT-M perp.
    Handles:
      - direct "BTCUSDT"
      - "1000PEPEUSDT" -> "PEPEUSDT"
      - "BTC-USDT", "BTCUSDTM", "BTCUSDT_UMCBL"
    """
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
# Fetch klines 1h Binance (for CVD)
# =====================================================================

async def _fetch_klines_1h(
    session: aiohttp.ClientSession,
    binance_symbol: str,
    limit: int = 120,
) -> Optional[List[List[Any]]]:
    cache_key = (binance_symbol, "1h", int(limit))
    now = time.time()

    cached = _KLINES_CACHE.get(cache_key)
    if cached is not None:
        ts, data = cached
        if now - ts < KLINES_TTL:
            return data  # type: ignore

    params = {"symbol": binance_symbol, "interval": "1h", "limit": int(limit)}
    data = await _http_get(session, "/fapi/v1/klines", params=params)
    if not isinstance(data, list) or not data:
        return None

    _KLINES_CACHE[cache_key] = (now, data)
    return data


# =====================================================================
# Fetch orderbook depth (imbalance)
# =====================================================================

async def _fetch_depth(
    session: aiohttp.ClientSession,
    binance_symbol: str,
    limit: int = 100,
) -> Optional[Dict[str, Any]]:
    cache_key = (binance_symbol, int(limit))
    now = time.time()

    cached = _DEPTH_CACHE.get(cache_key)
    if cached is not None:
        ts, data = cached
        if now - ts < DEPTH_TTL:
            return data  # type: ignore

    params = {"symbol": binance_symbol, "limit": int(limit)}
    data = await _http_get(session, "/fapi/v1/depth", params=params)
    if not isinstance(data, dict) or "bids" not in data or "asks" not in data:
        return None

    _DEPTH_CACHE[cache_key] = (now, data)
    return data


def _compute_orderbook_imbalance(depth: Dict[str, Any], band_bps: float = 25.0) -> Optional[float]:
    """
    Imbalance in [-1,+1] computed within +/- band_bps around mid.
    band_bps=25 => 0.25%
    """
    try:
        bids = depth.get("bids") or []
        asks = depth.get("asks") or []
        if not bids or not asks:
            return None

        b0p = float(bids[0][0])
        a0p = float(asks[0][0])
        if not (np.isfinite(b0p) and np.isfinite(a0p)) or a0p <= 0 or b0p <= 0:
            return None
        mid = (b0p + a0p) / 2.0
        if mid <= 0:
            return None

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


# =====================================================================
# Fetch aggTrades (tape delta)
# =====================================================================

async def _fetch_agg_trades(
    session: aiohttp.ClientSession,
    binance_symbol: str,
    limit: int = 1000,
) -> Optional[List[Dict[str, Any]]]:
    cache_key = (binance_symbol, int(limit))
    now = time.time()

    cached = _TRADES_CACHE.get(cache_key)
    if cached is not None:
        ts, data = cached
        if now - ts < TRADES_TTL:
            return data  # type: ignore

    params = {"symbol": binance_symbol, "limit": int(limit)}
    data = await _http_get(session, "/fapi/v1/aggTrades", params=params)
    if not isinstance(data, list) or not data:
        return None

    _TRADES_CACHE[cache_key] = (now, data)
    return data


def _compute_tape_delta(trades: List[Dict[str, Any]], window_sec: int = 300) -> Optional[float]:
    """
    Returns normalized taker delta in [-1,+1] over last window_sec.
    For aggTrades:
      - "m" (isBuyerMaker) == True  => seller initiated (taker sells)
      - "m" == False               => buyer initiated (taker buys)
    """
    try:
        if not trades:
            return None

        now_ms = int(time.time() * 1000)
        cutoff = now_ms - int(window_sec) * 1000

        buy = 0.0
        sell = 0.0
        for t in reversed(trades):
            ts = int(t.get("T") or 0)
            if ts < cutoff:
                break
            qty = float(t.get("q") or 0.0)
            if qty <= 0:
                continue
            is_buyer_maker = bool(t.get("m", False))
            if is_buyer_maker:
                sell += qty
            else:
                buy += qty

        den = buy + sell
        if den <= 0:
            return None
        return float((buy - sell) / den)
    except Exception:
        return None


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


# =====================================================================
# Funding / premiumIndex + funding history
# =====================================================================

async def _fetch_premium_index(session: aiohttp.ClientSession, binance_symbol: str) -> Optional[Dict[str, Any]]:
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


async def _fetch_funding_history(
    session: aiohttp.ClientSession,
    binance_symbol: str,
    limit: int = 30,
) -> Optional[List[Dict[str, Any]]]:
    key = (binance_symbol, int(limit))
    now = time.time()
    cached = _FUNDING_HIST_CACHE.get(key)
    if cached is not None:
        ts, data = cached
        if now - ts < FUNDING_HIST_TTL:
            return data  # type: ignore

    params = {"symbol": binance_symbol, "limit": int(limit)}
    data = await _http_get(session, "/fapi/v1/fundingRate", params=params)
    if not isinstance(data, list) or not data:
        return None

    _FUNDING_HIST_CACHE[key] = (now, data)
    return data


def _compute_funding_stats(
    funding_hist: Optional[List[Dict[str, Any]]],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns: (mean, std, zscore_last)
    """
    try:
        if not funding_hist:
            return None, None, None
        rates: List[float] = []
        for x in funding_hist[-24:]:
            try:
                rates.append(float(x.get("fundingRate")))
            except Exception:
                continue
        if len(rates) < 5:
            return None, None, None
        mean = float(np.mean(rates))
        std0 = float(np.std(rates))
        std = std0 if std0 > 1e-12 else 0.0
        last = float(rates[-1])
        z = float((last - mean) / std) if std > 0 else None
        return mean, (std if std > 0 else None), z
    except Exception:
        return None, None, None


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


def _classify_crowding(
    bias: str,
    funding_rate: Optional[float],
    basis_pct: Optional[float],
    funding_z: Optional[float],
    *,
    lsr_hint: Optional[float] = None,
) -> str:
    """
    Crowding regime:
      - primary: funding + basis (+ optional zscore)
      - fallback: lsr_hint (long/short ratio) if funding/basis missing
    """
    b = (bias or "").upper()

    if funding_rate is None and basis_pct is None:
        # fallback on LSR
        if lsr_hint is None:
            return "unknown"
        x = float(lsr_hint)
        # x > 1 => more longs than shorts
        if b == "LONG":
            if x >= 2.2:
                return "long_crowded_risky"
            if x <= 0.7:
                return "short_crowded_favorable"
            return "balanced"
        if b == "SHORT":
            if x <= 0.45:
                return "short_crowded_risky"
            if x >= 1.6:
                return "long_crowded_favorable"
            return "balanced"
        return "unknown"

    fr = float(funding_rate) if funding_rate is not None else 0.0
    bs = float(basis_pct) if basis_pct is not None else 0.0

    crowded_long = (fr >= 0.001) or (bs >= 0.0015) or (funding_z is not None and float(funding_z) >= 2.0)
    crowded_short = (fr <= -0.001) or (bs <= -0.0015) or (funding_z is not None and float(funding_z) <= -2.0)

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


# =====================================================================
# Open Interest (snapshot + hist)
# =====================================================================

async def _fetch_open_interest(session: aiohttp.ClientSession, binance_symbol: str) -> Optional[float]:
    now = time.time()
    cached = _OI_CACHE.get(binance_symbol)
    if cached is not None:
        ts, data = cached
        if now - ts < OI_TTL:
            return data  # type: ignore

    params = {"symbol": binance_symbol}
    data = await _http_get(session, "/fapi/v1/openInterest", params=params)
    if not isinstance(data, dict) or "openInterest" not in data:
        return None
    try:
        oi = float(data["openInterest"])
    except Exception:
        return None

    _OI_CACHE[binance_symbol] = (now, oi)
    return oi


async def _fetch_open_interest_hist(
    session: aiohttp.ClientSession,
    binance_symbol: str,
    period: str = "5m",
    limit: int = 30,
) -> Optional[List[Dict[str, Any]]]:
    now = time.time()
    key = (binance_symbol, str(period), int(limit))
    cached = _OI_HIST_CACHE.get(key)
    if cached is not None:
        ts, data = cached
        if now - ts < OI_HIST_TTL:
            return data  # type: ignore

    params = {"symbol": binance_symbol, "period": period, "limit": int(limit)}
    data = await _http_get(session, "/futures/data/openInterestHist", params=params)
    if not isinstance(data, list) or not data:
        return None

    _OI_HIST_CACHE[key] = (now, data)
    return data


def _compute_oi_slope(binance_symbol: str, new_oi: Optional[float]) -> Optional[float]:
    if new_oi is None:
        return None
    prev = _OI_HISTORY.get(binance_symbol)
    if prev is None:
        return 0.0
    _, old_oi = prev
    if old_oi <= 0:
        return 0.0
    return float((float(new_oi) - float(old_oi)) / float(old_oi))


def _compute_oi_hist_slope(oi_hist: Optional[List[Dict[str, Any]]]) -> Optional[float]:
    """
    Smoother slope over last points:
      (last - first) / |first|
    """
    try:
        if not oi_hist or len(oi_hist) < 8:
            return None
        xs: List[float] = []
        for x in oi_hist[-20:]:
            try:
                xs.append(float(x.get("sumOpenInterest") or x.get("openInterest") or x.get("sumOpenInterestValue")))
            except Exception:
                continue
        if len(xs) < 6:
            return None
        a = float(xs[0])
        b = float(xs[-1])
        den = abs(a) if abs(a) > 1e-12 else max(abs(b), 1e-12)
        return float((b - a) / den)
    except Exception:
        return None


# =====================================================================
# CVD from klines (taker buy base)
# =====================================================================

def _compute_cvd_slope_from_klines(klines: List[List[Any]], window: int = 40) -> Optional[float]:
    """
    delta = 2 * takerBuyBase - totalVolume
    cumulate delta; slope on last window:
      (end - start) / |start|
    """
    if not klines or len(klines) < window + 6:
        return None

    sub = klines[-(window + 6):]
    cvd = 0.0
    cvs: List[float] = []

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

    seg = cvs[-window:]
    start = float(seg[0])
    end = float(seg[-1])
    den = abs(start) if abs(start) > 1e-12 else max(abs(end), 1e-12)
    return float((end - start) / den)


# =====================================================================
# Liquidations (best effort)
# =====================================================================

async def _fetch_force_orders(
    session: aiohttp.ClientSession,
    binance_symbol: str,
    limit: int = 50,
) -> Optional[List[Dict[str, Any]]]:
    """
    Public endpoint but sometimes restricted. Best-effort.
    """
    now = time.time()
    key = (binance_symbol, int(limit))
    cached = _FORCE_CACHE.get(key)
    if cached is not None:
        ts, data = cached
        if now - ts < FORCE_TTL:
            return data  # type: ignore

    params = {"symbol": binance_symbol, "limit": int(limit)}
    data = await _http_get(session, "/fapi/v1/allForceOrders", params=params)
    if not isinstance(data, list) or not data:
        return None

    _FORCE_CACHE[key] = (now, data)
    return data


def _compute_liquidation_intensity(
    force_orders: Optional[List[Dict[str, Any]]],
    window_sec: int = 900,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Returns: (intensity, bias) where bias in {"buy_liq","sell_liq","mixed"}
    """
    try:
        if not force_orders:
            return None, None

        now_ms = int(time.time() * 1000)
        cutoff = now_ms - int(window_sec) * 1000

        buy_qty = 0.0
        sell_qty = 0.0
        for x in reversed(force_orders):
            ts = int(x.get("time") or x.get("T") or 0)
            if ts < cutoff:
                break
            qty = float(x.get("origQty") or x.get("q") or 0.0)
            side = str(x.get("side") or "").upper()
            if qty <= 0:
                continue
            if side == "BUY":
                buy_qty += qty
            elif side == "SELL":
                sell_qty += qty

        total = buy_qty + sell_qty
        if total <= 0:
            return None, None

        bias = "mixed"
        if buy_qty > 1.6 * sell_qty:
            bias = "buy_liq"
        elif sell_qty > 1.6 * buy_qty:
            bias = "sell_liq"

        return float(total), bias
    except Exception:
        return None, None


# =====================================================================
# Long/Short Ratios (best effort)
# =====================================================================

async def _fetch_lsr(
    session: aiohttp.ClientSession,
    endpoint: str,
    binance_symbol: str,
    *,
    period: str = "1h",
    limit: int = 30,
) -> Optional[List[Dict[str, Any]]]:
    """
    Endpoints:
      - /futures/data/globalLongShortAccountRatio
      - /futures/data/topLongShortAccountRatio
      - /futures/data/takerlongshortRatio
    """
    now = time.time()
    key = (endpoint, binance_symbol, str(period), int(limit))
    cached = _LSR_CACHE.get(key)
    if cached is not None:
        ts, data = cached
        if now - ts < LSR_TTL:
            return data  # type: ignore

    params = {"symbol": binance_symbol, "period": str(period), "limit": int(limit)}
    data = await _http_get(session, endpoint, params=params)
    if not isinstance(data, list) or not data:
        return None

    _LSR_CACHE[key] = (now, data)
    return data


def _extract_lsr_stats(lsr: Optional[List[Dict[str, Any]]]) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns: (last_ratio, slope_ratio)
      slope_ratio ~ (last - first)/|first| over last points
    """
    try:
        if not lsr or len(lsr) < 6:
            return None, None

        vals: List[float] = []
        for x in lsr[-20:]:
            # ratio key differs by endpoint
            v = x.get("longShortRatio")
            if v is None:
                v = x.get("buySellRatio")
            if v is None:
                # fallback from longAccount/shortAccount
                try:
                    la = float(x.get("longAccount"))
                    sa = float(x.get("shortAccount"))
                    if sa > 0:
                        v = la / sa
                except Exception:
                    v = None
            try:
                if v is None:
                    continue
                vals.append(float(v))
            except Exception:
                continue

        if len(vals) < 6:
            return None, None

        first = float(vals[0])
        last = float(vals[-1])
        den = abs(first) if abs(first) > 1e-12 else max(abs(last), 1e-12)
        slope = float((last - first) / den)
        return last, slope
    except Exception:
        return None, None


# =====================================================================
# Institutional scoring
# =====================================================================

def _score_institutional(
    bias: str,
    *,
    oi_slope: Optional[float],
    oi_hist_slope: Optional[float],
    cvd_slope: Optional[float],
    tape_5m: Optional[float],
    funding_rate: Optional[float],
    funding_z: Optional[float],
    basis_pct: Optional[float],
    ob_25bps: Optional[float],
) -> Tuple[int, Dict[str, int], Dict[str, Any], Dict[str, bool]]:
    """
    Score in [0..4] (compatible with MIN_INST_SCORE=2).
    Components:
      - Flow (CVD + Tape) = 0..2
      - OI trend          = 0..1
      - Crowding          = 0..1 (contrarian funding/basis)
      - Orderbook         = 0..1
    """
    b = (bias or "").upper()
    comp: Dict[str, int] = {"flow": 0, "oi": 0, "crowding": 0, "orderbook": 0}
    meta: Dict[str, Any] = {}

    # --- Flow: combine tape + cvd ---
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

    if flow_points >= 3:
        comp["flow"] = 2
    elif flow_points >= 1:
        comp["flow"] = 1
    else:
        comp["flow"] = 0

    # --- OI: snapshot slope OR hist slope aligned ---
    oi_ok = False
    if oi_slope is not None:
        x = float(oi_slope)
        if b == "LONG" and x >= 0.008:
            oi_ok = True
        if b == "SHORT" and x <= -0.008:
            oi_ok = True
    if (not oi_ok) and oi_hist_slope is not None:
        x = float(oi_hist_slope)
        if b == "LONG" and x >= 0.012:
            oi_ok = True
        if b == "SHORT" and x <= -0.012:
            oi_ok = True
    if oi_ok:
        comp["oi"] = 1

    # --- Crowding (contrarian): funding/basis extreme opposite side helps ---
    if funding_rate is not None:
        fr = float(funding_rate)
        # For LONG: slightly negative funding is favorable
        if b == "LONG" and fr < -0.0005:
            comp["crowding"] = 1
        # For SHORT: positive funding is favorable
        if b == "SHORT" and fr > 0.0005:
            comp["crowding"] = 1

    if funding_z is not None:
        z = float(funding_z)
        if b == "LONG" and z <= -1.6:
            comp["crowding"] = 1
        if b == "SHORT" and z >= 1.6:
            comp["crowding"] = 1

    if basis_pct is not None:
        bs = float(basis_pct)
        if b == "LONG" and bs < -0.0006:
            comp["crowding"] = 1
        if b == "SHORT" and bs > 0.0006:
            comp["crowding"] = 1

    # --- Orderbook: aligned imbalance ---
    if ob_25bps is not None:
        x = float(ob_25bps)
        if b == "LONG" and x >= 0.12:
            comp["orderbook"] = 1
        if b == "SHORT" and x <= -0.12:
            comp["orderbook"] = 1

    total_raw = int(comp["flow"] + comp["oi"] + comp["crowding"] + comp["orderbook"])
    total = max(0, min(4, total_raw))
    meta["raw_components_sum"] = int(total_raw)

    # -----------------------------------------------------------------
    # Flags "2/3" (OI, Funding, CVD) pour analyze_signal
    # -----------------------------------------------------------------
    cvd_ok = False
    if cvd_slope is not None:
        x = float(cvd_slope)
        if b == "LONG" and x >= 0.2:
            cvd_ok = True
        if b == "SHORT" and x <= -0.2:
            cvd_ok = True

    # Funding ok = pas crowded contre nous (ou crowding contrarian favorable)
    # On considère "ok" si funding n'est pas défavorable pour le biais.
    funding_ok = False
    if funding_rate is not None:
        fr = float(funding_rate)
        if b == "LONG":
            funding_ok = (fr <= 0.0008)  # évite longs quand funding trop positif
        else:
            funding_ok = (fr >= -0.0008)  # évite shorts quand funding trop négatif
        # contrarian bonus
        if b == "LONG" and fr < -0.0002:
            funding_ok = True
        if b == "SHORT" and fr > 0.0002:
            funding_ok = True

    flags = {"oi_ok": bool(oi_ok), "funding_ok": bool(funding_ok), "cvd_ok": bool(cvd_ok)}
    return total, comp, meta, flags


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


# =====================================================================
# API PRINCIPALE
# =====================================================================

async def compute_full_institutional_analysis(
    symbol: str,
    bias: str,
    *,
    include_liquidations: bool = False,
) -> Dict[str, Any]:
    """
    Institutional analysis for a symbol (KuCoin/Bitget format) using Binance USDT-M Futures.

    Performance note (important for scanning):
      - Stage 1 (always): openInterest + premiumIndex + aggTrades + depth + LSR (best effort)
      - Stage 2 (only if needed): klines (CVD) + oiHist + fundingHist
      - Stage 3 (optional): liquidations (allForceOrders) if include_liquidations=True

    If not mapped => available=False, score=0 (pipeline continues).
    """
    bias = (bias or "").upper()
    warnings: List[str] = []

    binance_symbol: Optional[str] = None

    oi_value: Optional[float] = None
    oi_slope: Optional[float] = None
    oi_hist_slope: Optional[float] = None

    cvd_slope: Optional[float] = None

    funding_rate: Optional[float] = None
    funding_mean: Optional[float] = None
    funding_std: Optional[float] = None
    funding_z: Optional[float] = None

    basis_pct: Optional[float] = None

    tape_1m: Optional[float] = None
    tape_5m: Optional[float] = None

    ob_10: Optional[float] = None
    ob_25: Optional[float] = None

    liq_intensity: Optional[float] = None
    liq_bias: Optional[str] = None

    # LSR
    lsr_global_last: Optional[float] = None
    lsr_global_slope: Optional[float] = None
    lsr_top_last: Optional[float] = None
    lsr_top_slope: Optional[float] = None
    taker_ls_last: Optional[float] = None
    taker_ls_slope: Optional[float] = None

    async with aiohttp.ClientSession() as session:
        # 0) symbols list
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
                "tape_delta": None,
                "orderbook_imbalance": None,
                "components": {"oi_ok": False, "funding_ok": False, "cvd_ok": False},
            }

        # -------------------------
        # Stage 1 (cheap, always)
        # -------------------------
        tasks1 = [
            _fetch_open_interest(session, binance_symbol),
            _fetch_premium_index(session, binance_symbol),
            _fetch_agg_trades(session, binance_symbol, limit=1000),
            _fetch_depth(session, binance_symbol, limit=100),
            _fetch_lsr(session, "/futures/data/globalLongShortAccountRatio", binance_symbol, period="1h", limit=30),
            _fetch_lsr(session, "/futures/data/topLongShortAccountRatio", binance_symbol, period="1h", limit=30),
            _fetch_lsr(session, "/futures/data/takerlongshortRatio", binance_symbol, period="1h", limit=30),
        ]
        r = await asyncio.gather(*tasks1, return_exceptions=True)
        oi_value, prem, trades, depth, lsr_g, lsr_t, lsr_k = r

        # unwrap exceptions
        if isinstance(oi_value, Exception):
            warnings.append(f"oi_exc:{type(oi_value).__name__}")
            oi_value = None
        if isinstance(prem, Exception):
            warnings.append(f"premium_exc:{type(prem).__name__}")
            prem = None
        if isinstance(trades, Exception):
            warnings.append(f"trades_exc:{type(trades).__name__}")
            trades = None
        if isinstance(depth, Exception):
            warnings.append(f"depth_exc:{type(depth).__name__}")
            depth = None
        if isinstance(lsr_g, Exception):
            warnings.append(f"lsr_global_exc:{type(lsr_g).__name__}")
            lsr_g = None
        if isinstance(lsr_t, Exception):
            warnings.append(f"lsr_top_exc:{type(lsr_t).__name__}")
            lsr_t = None
        if isinstance(lsr_k, Exception):
            warnings.append(f"lsr_taker_exc:{type(lsr_k).__name__}")
            lsr_k = None

        # OI slope
        if oi_value is None:
            warnings.append("no_oi")
        else:
            oi_slope = _compute_oi_slope(binance_symbol, oi_value)
            _OI_HISTORY[binance_symbol] = (time.time(), float(oi_value))

        # PremiumIndex => funding + basis
        if isinstance(prem, dict):
            try:
                funding_rate = float(prem.get("lastFundingRate", "0"))
            except Exception:
                funding_rate = None
                warnings.append("funding_parse_error")

            try:
                mark = float(prem.get("markPrice", "0"))
                index = float(prem.get("indexPrice", "0"))
                if index > 0:
                    basis_pct = (mark - index) / index
            except Exception:
                basis_pct = None
                warnings.append("basis_parse_error")
        else:
            warnings.append("no_premiumIndex")

        # Tape delta
        if isinstance(trades, list) and trades:
            tape_1m = _compute_tape_delta(trades, window_sec=60)
            tape_5m = _compute_tape_delta(trades, window_sec=300)
        else:
            warnings.append("no_trades")

        # Orderbook imbalance
        if isinstance(depth, dict):
            ob_10 = _compute_orderbook_imbalance(depth, band_bps=10.0)
            ob_25 = _compute_orderbook_imbalance(depth, band_bps=25.0)
        else:
            warnings.append("no_depth")

        # LSR stats
        lsr_global_last, lsr_global_slope = _extract_lsr_stats(lsr_g if isinstance(lsr_g, list) else None)
        lsr_top_last, lsr_top_slope = _extract_lsr_stats(lsr_t if isinstance(lsr_t, list) else None)
        taker_ls_last, taker_ls_slope = _extract_lsr_stats(lsr_k if isinstance(lsr_k, list) else None)

        # Early scoring (without hist/cvd)
        inst_score_early, comp_early, meta_early, flags_early = _score_institutional(
            bias,
            oi_slope=oi_slope,
            oi_hist_slope=None,
            cvd_slope=None,
            tape_5m=tape_5m,
            funding_rate=funding_rate,
            funding_z=None,
            basis_pct=basis_pct,
            ob_25bps=ob_25,
        )

        # Decide if we need Stage 2:
        # - if early score already >=2
        # - OR tape is strong + OB supports (microstructure confirmation)
        strong_tape = tape_5m is not None and abs(float(tape_5m)) >= 0.35
        strong_ob_support = ob_25 is not None and ((float(ob_25) >= 0.12) if bias == "LONG" else (float(ob_25) <= -0.12))
        need_stage2 = not (inst_score_early >= 2 or (strong_tape and strong_ob_support))

        klines = None
        oi_hist = None
        funding_hist = None
        force_orders = None

        if need_stage2:
            tasks2 = [
                _fetch_klines_1h(session, binance_symbol, limit=120),
                _fetch_open_interest_hist(session, binance_symbol, period="5m", limit=30),
                _fetch_funding_history(session, binance_symbol, limit=30),
            ]
            klines, oi_hist, funding_hist = await asyncio.gather(*tasks2, return_exceptions=True)

            if isinstance(klines, Exception):
                warnings.append(f"klines_exc:{type(klines).__name__}")
                klines = None
            if isinstance(oi_hist, Exception):
                warnings.append(f"oi_hist_exc:{type(oi_hist).__name__}")
                oi_hist = None
            if isinstance(funding_hist, Exception):
                warnings.append(f"funding_hist_exc:{type(funding_hist).__name__}")
                funding_hist = None

            if isinstance(klines, list) and klines:
                cvd_slope = _compute_cvd_slope_from_klines(klines, window=40)
            else:
                warnings.append("no_klines")

            if isinstance(oi_hist, list) and oi_hist:
                oi_hist_slope = _compute_oi_hist_slope(oi_hist)
            else:
                warnings.append("no_oi_hist")

            if isinstance(funding_hist, list) and funding_hist:
                funding_mean, funding_std, funding_z = _compute_funding_stats(funding_hist)
            else:
                warnings.append("no_funding_hist")

        # Stage 3 (optional): liquidations
        if include_liquidations:
            try:
                force_orders = await _fetch_force_orders(session, binance_symbol, limit=50)
            except Exception:
                force_orders = None

            if isinstance(force_orders, list) and force_orders:
                liq_intensity, liq_bias = _compute_liquidation_intensity(force_orders, window_sec=900)
            else:
                warnings.append("no_force_orders")

    # Regimes
    funding_regime = _classify_funding(funding_rate, z=funding_z)
    basis_regime = _classify_basis(basis_pct)

    # LSR hint for crowding fallback (prefer top, else global)
    lsr_hint = lsr_top_last if lsr_top_last is not None else lsr_global_last
    crowding_regime = _classify_crowding(bias, funding_rate, basis_pct, funding_z, lsr_hint=lsr_hint)

    flow_regime = _classify_flow(cvd_slope, tape_5m)
    ob_regime = _classify_orderbook(ob_25)

    # Final score
    inst_score, components, score_meta, flags = _score_institutional(
        bias,
        oi_slope=oi_slope,
        oi_hist_slope=oi_hist_slope,
        cvd_slope=cvd_slope,
        tape_5m=tape_5m,
        funding_rate=funding_rate,
        funding_z=funding_z,
        basis_pct=basis_pct,
        ob_25bps=ob_25,
    )

    # Debug meta includes early stage
    score_meta = dict(score_meta or {})
    score_meta["early_score"] = int(inst_score_early)
    score_meta["early_components"] = comp_early
    score_meta["early_meta"] = meta_early
    score_meta["early_flags"] = flags_early

    available = any(
        [
            oi_value is not None,
            oi_hist_slope is not None,
            cvd_slope is not None,
            funding_rate is not None,
            tape_5m is not None,
            ob_25 is not None,
        ]
    )

    # Aliases used in analyze_signal.py (breakout check)
    tape_delta_alias = tape_5m
    orderbook_imb_alias = ob_25

    return {
        # --- required ---
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

        # --- compatibility flags for "2/3" gate ---
        "components": dict(flags),
        "oi_ok": bool(flags.get("oi_ok")),
        "funding_ok": bool(flags.get("funding_ok")),
        "cvd_ok": bool(flags.get("cvd_ok")),

        # --- extra (desk) ---
        "oi_hist_slope": oi_hist_slope,
        "tape_delta_1m": tape_1m,
        "tape_delta_5m": tape_5m,
        "tape_delta": tape_delta_alias,  # alias
        "tape_regime": _classify_tape(tape_5m),
        "basis_pct": basis_pct,
        "basis_regime": basis_regime,
        "orderbook_imb_10bps": ob_10,
        "orderbook_imb_25bps": ob_25,
        "orderbook_imbalance": orderbook_imb_alias,  # alias
        "orderbook_regime": ob_regime,
        "funding_mean": funding_mean,
        "funding_std": funding_std,
        "funding_z": funding_z,
        "liquidation_intensity": liq_intensity,
        "liquidation_bias": liq_bias,

        # LSR
        "lsr_global_last": lsr_global_last,
        "lsr_global_slope": lsr_global_slope,
        "lsr_top_last": lsr_top_last,
        "lsr_top_slope": lsr_top_slope,
        "taker_ls_last": taker_ls_last,
        "taker_ls_slope": taker_ls_slope,

        # Score decomposition
        "score_components": components,
        "score_meta": score_meta,
    }
