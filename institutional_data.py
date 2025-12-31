# =====================================================================
# institutional_data.py — Ultra Desk OI + Funding/Basis + (optional) Tape/OB/CVD + Liquidations (WS)
# Binance USDT-M Futures (public endpoints), rate-limited + circuit-breaker
#
# Robustesse scan multi-coins :
# - Rate limiter global (semaphore + pacing)
# - Circuit breaker "hard ban" (418 / -1003 avec ban-until) + "soft cooldown" (429 / -1003 / 5xx)
# - Backoff / cooldown par symbole (évite de marteler le même coin)
# - Shared aiohttp session (pas de session par call)
# - Modes LIGHT/NORMAL/FULL + override par paramètre (scanner pass1/pass2)
# - Sortie enrichie : available_components_count + available_components + ban info + mode effectif
#
# Liquidations:
# - REST "all market force orders" n'est pas utilisé ici.
# - On consomme le flux WebSocket "All Market Liquidation Order Streams": !forceOrder@arr
#   (1 seule connexion, agrégation en mémoire, puis lecture par symbole dans compute_*).
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
BINANCE_FAPI_BASE = "https://fapi.binance.com"
BINANCE_FSTREAM_WS_BASE = "wss://fstream.binance.com/ws"

# ---------------------------------------------------------------------
# Modes (env) + overrides
# ---------------------------------------------------------------------
# LIGHT: OI + premiumIndex (safe)
# NORMAL: LIGHT + aggTrades + depth
# FULL: NORMAL + klines + oiHist + fundingHist (+ LSR optional)
INST_MODE = str(os.getenv("INST_MODE", "LIGHT")).upper().strip()
if INST_MODE not in ("LIGHT", "NORMAL", "FULL"):
    INST_MODE = "LIGHT"

# Optional add-ons (disabled by default to save calls)
INCLUDE_LSR = str(os.getenv("INST_INCLUDE_LSR", "0")).strip() == "1"

# Liquidations (WebSocket)
INST_INCLUDE_LIQUIDATIONS = str(os.getenv("INST_INCLUDE_LIQUIDATIONS", "0")).strip() == "1"
_LIQ_WINDOW_SEC = int(float(os.getenv("INST_LIQ_WINDOW_SEC", "300")))      # metrics window (default 5m)
_LIQ_STORE_SEC = int(float(os.getenv("INST_LIQ_STORE_SEC", "900")))        # store depth (default 15m)
_LIQ_MIN_NOTIONAL_USD = float(os.getenv("INST_LIQ_MIN_NOTIONAL_USD", "50000"))  # ignore tiny flows by default

# ---------------------------------------------------------------------
# Global rate limiting + circuit breaker
# ---------------------------------------------------------------------
_BINANCE_CONCURRENCY = max(1, int(os.getenv("BINANCE_HTTP_CONCURRENCY", "3")))
_BINANCE_MIN_INTERVAL_SEC = float(os.getenv("BINANCE_MIN_INTERVAL_SEC", "0.12"))  # ~8.3 req/s total

# Soft cooldown (ms) after 429/5xx/-1003 (sans ban-until)
_SOFT_COOLDOWN_MS_DEFAULT = int(float(os.getenv("BINANCE_SOFT_COOLDOWN_SEC", "20")) * 1000)

# Hard ban fallback if we cannot parse "banned until"
_HARD_BAN_FALLBACK_MS = int(float(os.getenv("BINANCE_HARD_BAN_FALLBACK_MIN", "15")) * 60_000)

_HTTP_SEM = asyncio.Semaphore(_BINANCE_CONCURRENCY)
_PACE_LOCK = asyncio.Lock()
_LAST_REQ_TS = 0.0

# Circuit breaker times (ms timestamps)
_BINANCE_HARD_BAN_UNTIL_MS = 0
_BINANCE_SOFT_UNTIL_MS = 0

# Per-symbol backoff
_SYM_STATE: Dict[str, "SymbolBackoff"] = {}

# Regex to extract ban-until ms from Binance messages
_RE_BAN_UNTIL = re.compile(r"banned until (\d+)", re.IGNORECASE)

# ---------------------------------------------------------------------
# Shared session
# ---------------------------------------------------------------------
_SESSION: Optional[aiohttp.ClientSession] = None
_SESSION_LOCK = asyncio.Lock()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _is_hard_banned() -> bool:
    return _now_ms() < int(_BINANCE_HARD_BAN_UNTIL_MS)


def _is_soft_blocked() -> bool:
    return _now_ms() < int(_BINANCE_SOFT_UNTIL_MS)


def _set_hard_ban_until(ms: int, reason: str) -> None:
    global _BINANCE_HARD_BAN_UNTIL_MS
    ms = int(ms)
    if ms > _BINANCE_HARD_BAN_UNTIL_MS:
        _BINANCE_HARD_BAN_UNTIL_MS = ms
    LOGGER.error("[INST] BINANCE HARD BAN until_ms=%s reason=%s", ms, reason)


def _set_soft_cooldown(ms_from_now: int, reason: str) -> None:
    global _BINANCE_SOFT_UNTIL_MS
    until = _now_ms() + int(ms_from_now)
    if until > _BINANCE_SOFT_UNTIL_MS:
        _BINANCE_SOFT_UNTIL_MS = until
    LOGGER.warning("[INST] BINANCE SOFT COOLDOWN until_ms=%s reason=%s", _BINANCE_SOFT_UNTIL_MS, reason)


async def _get_session() -> aiohttp.ClientSession:
    global _SESSION
    if _SESSION is not None and not _SESSION.closed:
        return _SESSION

    async with _SESSION_LOCK:
        if _SESSION is not None and not _SESSION.closed:
            return _SESSION
        timeout = aiohttp.ClientTimeout(total=10)
        connector = aiohttp.TCPConnector(limit=50, ttl_dns_cache=300)
        _SESSION = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return _SESSION


# ---------------------------------------------------------------------
# Liquidations WebSocket (single shared worker)
# ---------------------------------------------------------------------
_LIQ_TASK: Optional[asyncio.Task] = None
_LIQ_START_LOCK = asyncio.Lock()
_LIQ_STOP: Optional[asyncio.Event] = None
_LIQ_LOCK = asyncio.Lock()
# symbol -> deque[(ts_ms, side, notional_usd)]
_LIQ_EVENTS: Dict[str, Deque[Tuple[int, str, float]]] = {}


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
    """
    Returns: (buy_usd, sell_usd, total_usd, delta_ratio) over last window_sec.
      delta_ratio = (buy - sell) / (buy + sell)
    """
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

            # iterate from newest backwards until cutoff
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

                    # handle combined-stream envelope: {"stream": "...", "data": {...}}
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

                            try:
                                qf = float(qty)
                                pf = float(price)
                            except Exception:
                                continue
                            notional = qf * pf
                            await _liq_add_event(str(sym), int(ts), str(side), float(notional))
                        except Exception:
                            continue

        except asyncio.CancelledError:
            break
        except Exception as e:
            LOGGER.warning("[INST_LIQ] WS error: %s", e)

        # reconnect backoff
        if _LIQ_STOP is not None and _LIQ_STOP.is_set():
            break
        await asyncio.sleep(min(60.0, backoff))
        backoff = min(60.0, backoff * 2.0)

    LOGGER.info("[INST_LIQ] WS worker stopped")


async def _ensure_liq_stream() -> None:
    global _LIQ_TASK, _LIQ_STOP
    # start only if enabled
    if not INST_INCLUDE_LIQUIDATIONS:
        return

    if _LIQ_STOP is None or _LIQ_STOP.is_set():
        _LIQ_STOP = asyncio.Event()

    if _LIQ_TASK is not None and (not _LIQ_TASK.done()):
        return

    async with _LIQ_START_LOCK:
        if _LIQ_TASK is not None and (not _LIQ_TASK.done()):
            return
        _LIQ_TASK = asyncio.create_task(_liq_worker())


async def close_institutional_session() -> None:
    """Optionnel: à appeler proprement au shutdown."""
    global _SESSION, _LIQ_TASK, _LIQ_STOP

    # stop ws worker
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

    # close http session
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
        wait = _BINANCE_MIN_INTERVAL_SEC - (now - _LAST_REQ_TS)
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
        # backoff exponentiel raisonnable
        self.errors += 1
        mult = 1.7 ** min(self.errors, 8)
        cd = int(min(cap_ms, base_ms * mult))
        self.until_ms = max(self.until_ms, _now_ms() + cd)


def _sym_key(symbol: Optional[str]) -> Optional[str]:
    if not symbol:
        return None
    return str(symbol).upper()


def _get_sym_state(symbol: Optional[str]) -> Optional[SymbolBackoff]:
    k = _sym_key(symbol)
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
    - hard/soft circuit breaker
    - concurrency semaphore
    - pacing
    - per-symbol backoff
    - ban parsing
    """
    if _is_hard_banned():
        return None
    if _is_soft_blocked():
        return None

    st = _get_sym_state(symbol)
    if st is not None and st.blocked():
        return None

    url = BINANCE_FAPI_BASE + path
    session = await _get_session()

    async with _HTTP_SEM:
        await _pace()
        try:
            async with session.get(url, params=params) as resp:
                status = resp.status
                raw = await resp.read()
                txt = ""
                try:
                    txt = raw.decode("utf-8", errors="ignore")
                except Exception:
                    txt = str(raw)[:500]

                data: Any = None
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

                    # -1003 : rate limit (parfois avec banned until)
                    if isinstance(code, int) and code == -1003:
                        low = msg.lower()
                        if "banned until" in low:
                            m = _RE_BAN_UNTIL.search(low)
                            if m:
                                _set_hard_ban_until(int(m.group(1)), reason=f"{path} -1003 {msg[:140]}")
                            else:
                                _set_hard_ban_until(_now_ms() + _HARD_BAN_FALLBACK_MS, reason=f"{path} -1003 no_ts")
                        else:
                            _set_soft_cooldown(_SOFT_COOLDOWN_MS_DEFAULT, reason=f"{path} -1003 {msg[:140]}")
                        if st is not None:
                            st.mark_err(base_ms=3_000)
                        LOGGER.warning("[INST] HTTP %s GET %s params=%s msg=%s", status, path, params, (msg or txt)[:200])
                        return None

                    # 418 : ban IP en pratique -> hard ban
                    if status == 418:
                        raw_msg = (msg or txt or "")
                        m = _RE_BAN_UNTIL.search(raw_msg)
                        if m:
                            _set_hard_ban_until(int(m.group(1)), reason=f"{path} 418 {raw_msg[:140]}")
                        else:
                            _set_hard_ban_until(_now_ms() + _HARD_BAN_FALLBACK_MS, reason=f"{path} 418 no_ts")
                        if st is not None:
                            st.mark_err(base_ms=5_000)
                        LOGGER.warning("[INST] HTTP 418 GET %s params=%s resp=%s", path, params, (raw_msg or "")[:200])
                        return None

                    # 429 : trop de requêtes -> soft cooldown
                    if status == 429:
                        _set_soft_cooldown(_SOFT_COOLDOWN_MS_DEFAULT, reason=f"{path} 429")
                        if st is not None:
                            st.mark_err(base_ms=2_500)
                        LOGGER.warning("[INST] HTTP 429 GET %s params=%s", path, params)
                        return None

                    # 5xx : soft cooldown court
                    if 500 <= status <= 599:
                        _set_soft_cooldown(5_000, reason=f"{path} {status}")
                        if st is not None:
                            st.mark_err(base_ms=1_800)
                        LOGGER.warning("[INST] HTTP %s GET %s params=%s", status, path, params)
                        return None

                    if st is not None:
                        st.mark_err(base_ms=1_500)
                    LOGGER.warning("[INST] HTTP %s GET %s params=%s resp=%s", status, path, params, (txt or "")[:200])
                    return None

                # OK
                if st is not None:
                    st.mark_ok()
                return data

        except asyncio.TimeoutError:
            if st is not None:
                st.mark_err(base_ms=1_600)
            LOGGER.error("[INST] Timeout GET %s params=%s", path, params)
            return None
        except Exception as e:
            if st is not None:
                st.mark_err(base_ms=2_000)
            LOGGER.error("[INST] Exception GET %s params=%s: %s", path, params, e)
            return None


# ---------------------------------------------------------------------
# Light caches
# ---------------------------------------------------------------------
# Cache: key -> (ts, data)
_KLINES_CACHE: Dict[Tuple[str, str, int], Tuple[float, Any]] = {}
_DEPTH_CACHE: Dict[Tuple[str, int], Tuple[float, Any]] = {}
_TRADES_CACHE: Dict[Tuple[str, int], Tuple[float, Any]] = {}
_FUNDING_CACHE: Dict[str, Tuple[float, Any]] = {}
_FUNDING_HIST_CACHE: Dict[Tuple[str, int], Tuple[float, Any]] = {}
_OI_CACHE: Dict[str, Tuple[float, Any]] = {}
_OI_HIST_CACHE: Dict[Tuple[str, str, int], Tuple[float, Any]] = {}
_LSR_CACHE: Dict[Tuple[str, str, str, int], Tuple[float, Any]] = {}

# OI slope memory: symbol -> (ts, oi)
_OI_HISTORY: Dict[str, Tuple[float, float]] = {}

# Binance symbols cache
_BINANCE_SYMBOLS: Optional[Set[str]] = None
_BINANCE_SYMBOLS_TS: float = 0.0

# TTLs (seconds)
KLINES_TTL = float(os.getenv("INST_KLINES_TTL", "120"))
DEPTH_TTL = float(os.getenv("INST_DEPTH_TTL", "10"))
TRADES_TTL = float(os.getenv("INST_TRADES_TTL", "10"))
FUNDING_TTL = float(os.getenv("INST_FUNDING_TTL", "60"))
FUNDING_HIST_TTL = float(os.getenv("INST_FUNDING_HIST_TTL", "300"))
OI_TTL = float(os.getenv("INST_OI_TTL", "60"))
OI_HIST_TTL = float(os.getenv("INST_OI_HIST_TTL", "300"))
BINANCE_SYMBOLS_TTL = float(os.getenv("INST_EXCHANGEINFO_TTL", "900"))
LSR_TTL = float(os.getenv("INST_LSR_TTL", "300"))


# =====================================================================
# Binance symbols (exchangeInfo)
# =====================================================================
async def _get_binance_symbols() -> Set[str]:
    global _BINANCE_SYMBOLS, _BINANCE_SYMBOLS_TS
    now = time.time()
    if _BINANCE_SYMBOLS is not None and (now - _BINANCE_SYMBOLS_TS) < BINANCE_SYMBOLS_TTL:
        return _BINANCE_SYMBOLS

    data = await _http_get("/fapi/v1/exchangeInfo", params=None, symbol=None)
    symbols: Set[str] = set()

    if not isinstance(data, dict) or "symbols" not in data:
        LOGGER.warning("[INST] Unable to fetch Binance exchangeInfo, keeping old symbols cache")
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
    LOGGER.info("[INST] Binance futures symbols loaded: %d", len(symbols))
    return _BINANCE_SYMBOLS


def _map_symbol_to_binance(symbol: str, binance_symbols: Set[str]) -> Optional[str]:
    """
    Map KuCoin/Bitget symbols to Binance USDT-M perp symbol.
    Handles:
      - direct
      - 1000TOKENUSDT -> TOKENUSDT
      - TOKENUSDT_UMCBL / BTCUSDTM / BTC-USDT -> BTCUSDT
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
# Endpoints
# =====================================================================
async def _fetch_open_interest(binance_symbol: str) -> Optional[float]:
    now = time.time()
    cached = _OI_CACHE.get(binance_symbol)
    if cached is not None and (now - cached[0]) < OI_TTL:
        return cached[1]  # type: ignore

    data = await _http_get("/fapi/v1/openInterest", params={"symbol": binance_symbol}, symbol=binance_symbol)
    if not isinstance(data, dict) or "openInterest" not in data:
        return None
    try:
        oi = float(data["openInterest"])
    except Exception:
        return None

    _OI_CACHE[binance_symbol] = (now, oi)
    return oi


async def _fetch_premium_index(binance_symbol: str) -> Optional[Dict[str, Any]]:
    """
    Binance doc: Mark Price endpoint is GET /fapi/v1/premiumIndex and returns
    markPrice, indexPrice, lastFundingRate, etc.
    """
    now = time.time()
    cached = _FUNDING_CACHE.get(binance_symbol)
    if cached is not None and (now - cached[0]) < FUNDING_TTL:
        return cached[1]  # type: ignore

    data = await _http_get("/fapi/v1/premiumIndex", params={"symbol": binance_symbol}, symbol=binance_symbol)
    if not isinstance(data, dict) or "symbol" not in data:
        return None

    _FUNDING_CACHE[binance_symbol] = (now, data)
    return data


async def _fetch_agg_trades(binance_symbol: str, limit: int = 1000) -> Optional[List[Dict[str, Any]]]:
    cache_key = (binance_symbol, int(limit))
    now = time.time()
    cached = _TRADES_CACHE.get(cache_key)
    if cached is not None and (now - cached[0]) < TRADES_TTL:
        return cached[1]  # type: ignore

    data = await _http_get("/fapi/v1/aggTrades", params={"symbol": binance_symbol, "limit": int(limit)}, symbol=binance_symbol)
    if not isinstance(data, list) or not data:
        return None

    _TRADES_CACHE[cache_key] = (now, data)
    return data


async def _fetch_depth(binance_symbol: str, limit: int = 100) -> Optional[Dict[str, Any]]:
    cache_key = (binance_symbol, int(limit))
    now = time.time()
    cached = _DEPTH_CACHE.get(cache_key)
    if cached is not None and (now - cached[0]) < DEPTH_TTL:
        return cached[1]  # type: ignore

    data = await _http_get("/fapi/v1/depth", params={"symbol": binance_symbol, "limit": int(limit)}, symbol=binance_symbol)
    if not isinstance(data, dict) or "bids" not in data or "asks" not in data:
        return None

    _DEPTH_CACHE[cache_key] = (now, data)
    return data


async def _fetch_klines_1h(binance_symbol: str, limit: int = 120) -> Optional[List[List[Any]]]:
    cache_key = (binance_symbol, "1h", int(limit))
    now = time.time()
    cached = _KLINES_CACHE.get(cache_key)
    if cached is not None and (now - cached[0]) < KLINES_TTL:
        return cached[1]  # type: ignore

    data = await _http_get(
        "/fapi/v1/klines",
        params={"symbol": binance_symbol, "interval": "1h", "limit": int(limit)},
        symbol=binance_symbol,
    )
    if not isinstance(data, list) or not data:
        return None

    _KLINES_CACHE[cache_key] = (now, data)
    return data


async def _fetch_open_interest_hist(binance_symbol: str, period: str = "5m", limit: int = 30) -> Optional[List[Dict[str, Any]]]:
    cache_key = (binance_symbol, str(period), int(limit))
    now = time.time()
    cached = _OI_HIST_CACHE.get(cache_key)
    if cached is not None and (now - cached[0]) < OI_HIST_TTL:
        return cached[1]  # type: ignore

    data = await _http_get(
        "/futures/data/openInterestHist",
        params={"symbol": binance_symbol, "period": period, "limit": int(limit)},
        symbol=binance_symbol,
    )
    if not isinstance(data, list) or not data:
        return None

    _OI_HIST_CACHE[cache_key] = (now, data)
    return data


async def _fetch_funding_history(binance_symbol: str, limit: int = 30) -> Optional[List[Dict[str, Any]]]:
    cache_key = (binance_symbol, int(limit))
    now = time.time()
    cached = _FUNDING_HIST_CACHE.get(cache_key)
    if cached is not None and (now - cached[0]) < FUNDING_HIST_TTL:
        return cached[1]  # type: ignore

    data = await _http_get("/fapi/v1/fundingRate", params={"symbol": binance_symbol, "limit": int(limit)}, symbol=binance_symbol)
    if not isinstance(data, list) or not data:
        return None

    _FUNDING_HIST_CACHE[cache_key] = (now, data)
    return data


# =====================================================================
# LSR endpoints (optional)
# =====================================================================
async def _fetch_lsr(path: str, binance_symbol: str, period: str = "1h", limit: int = 30) -> Optional[List[Dict[str, Any]]]:
    cache_key = (path, binance_symbol, period, int(limit))
    now = time.time()
    cached = _LSR_CACHE.get(cache_key)
    if cached is not None and (now - cached[0]) < LSR_TTL:
        return cached[1]  # type: ignore

    data = await _http_get(path, params={"symbol": binance_symbol, "period": period, "limit": int(limit)}, symbol=binance_symbol)
    if not isinstance(data, list) or not data:
        return None

    _LSR_CACHE[cache_key] = (now, data)
    return data


def _extract_lsr_stats(lsr: Optional[List[Dict[str, Any]]]) -> Tuple[Optional[float], Optional[float]]:
    """Returns: (last_ratio, slope)"""
    try:
        if not lsr or len(lsr) < 6:
            return None, None
        vals: List[float] = []
        for x in lsr[-20:]:
            v = x.get("longShortRatio")
            if v is None:
                v = x.get("longAccount")
            try:
                vals.append(float(v))
            except Exception:
                continue
        if len(vals) < 6:
            return None, None
        a = float(vals[0])
        b = float(vals[-1])
        den = abs(a) if abs(a) > 1e-12 else max(abs(b), 1e-12)
        slope = float((b - a) / den)
        return float(vals[-1]), slope
    except Exception:
        return None, None


# =====================================================================
# Metrics helpers
# =====================================================================
def _compute_orderbook_imbalance(depth: Dict[str, Any], band_bps: float = 25.0) -> Optional[float]:
    """Imbalance in [-1,+1] computed within +/- band_bps around mid."""
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


def _compute_tape_delta(trades: List[Dict[str, Any]], window_sec: int = 300) -> Optional[float]:
    """
    Normalized taker delta in [-1,+1] over last window_sec.

    Binance aggTrades:
      - "m" == True means buyer is the maker -> trade was seller-initiated (taker sells)
      - "m" == False -> buyer-initiated (taker buys)
    """
    try:
        if not trades:
            return None
        now_ms = _now_ms()
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


def _compute_funding_stats(funding_hist: Optional[List[Dict[str, Any]]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Returns: (mean, std, zscore_last)"""
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

        mean, stdv = _mean_std(rates)
        if mean is None or stdv is None:
            return None, None, None

        std = stdv if stdv > 1e-12 else 0.0
        last = float(rates[-1])
        z = float((last - mean) / std) if std > 0 else None
        return float(mean), (float(std) if std > 0 else None), z
    except Exception:
        return None, None, None


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


def _compute_cvd_slope_from_klines(klines: List[List[Any]], window: int = 40) -> Optional[float]:
    """
    Approx CVD from futures klines:
    delta = 2 * takerBuyBase - totalVolume
    cumulate delta; slope on last window.
    """
    try:
        if not klines or len(klines) < window + 6:
            return None
        sub = klines[-(window + 6):]
        cvd = 0.0
        cvs: List[float] = []
        for item in sub:
            try:
                vol = float(item[5])        # Volume
                taker_buy = float(item[9])  # Taker buy base asset volume
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
    except Exception:
        return None


# =====================================================================
# Regimes + scoring
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
    oi_hist_slope: Optional[float],
    cvd_slope: Optional[float],
    tape_5m: Optional[float],
    funding_rate: Optional[float],
    funding_z: Optional[float],
    basis_pct: Optional[float],
    ob_25bps: Optional[float],
    liq_delta_ratio_5m: Optional[float],
    liq_total_usd_5m: Optional[float],
) -> Tuple[int, Dict[str, int], Dict[str, Any]]:
    """
    Score in [0..4].
    Components:
      - flow: 0..2 (tape/cvd/liquidations)
      - oi: 0..1
      - crowding: 0..1 (contrarian funding/basis)
      - orderbook: 0..1
    """
    b = (bias or "").upper()
    comp: Dict[str, int] = {"flow": 0, "oi": 0, "crowding": 0, "orderbook": 0}
    meta: Dict[str, Any] = {}

    # Flow
    flow_points = 0

    # Tape
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

    # CVD
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

    # Liquidations (WS): treated as additional flow confirmation, only if notional is meaningful.
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

    # OI
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

    # Crowding (contrarian)
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

    # Orderbook
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
    if payload.get("tape_delta_5m") is not None:
        out.append("tape")
    if payload.get("orderbook_imb_25bps") is not None:
        out.append("orderbook")
    if payload.get("cvd_slope") is not None:
        out.append("cvd")
    if payload.get("oi_hist_slope") is not None:
        out.append("oi_hist")
    if payload.get("funding_z") is not None:
        out.append("funding_hist")
    if payload.get("liq_total_usd_5m") is not None:
        out.append("liquidations")
    return out


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
    """
    Institutional analysis for a given symbol (KuCoin/Bitget format) using Binance USDT-M Futures.

    - Uses circuit breaker if banned.
    - Uses mode (override) or INST_MODE env to control endpoint cost.
      mode: "LIGHT" | "NORMAL" | "FULL"
    - Liquidations: consumed via WebSocket stream !forceOrder@arr when enabled.
      Enable with INST_INCLUDE_LIQUIDATIONS=1 or include_liquidations=True.
    """
    bias = (bias or "").upper().strip()
    eff_mode = (mode or INST_MODE).upper().strip()
    if eff_mode not in ("LIGHT", "NORMAL", "FULL"):
        eff_mode = "LIGHT"

    warnings: List[str] = []

    # liquidations enable (env OR param)
    use_liq = bool(INST_INCLUDE_LIQUIDATIONS or include_liquidations)
    if use_liq:
        # start WS worker lazily
        try:
            await _ensure_liq_stream()
        except Exception:
            warnings.append("liq_ws_start_error")

    if _is_hard_banned():
        payload = {
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
            "warnings": [f"binance_hard_ban_until_ms={_BINANCE_HARD_BAN_UNTIL_MS}"],
            "score_components": {"flow": 0, "oi": 0, "crowding": 0, "orderbook": 0},
            "score_meta": {"mode": eff_mode},
            "available_components": [],
            "available_components_count": 0,
            "ban": {"hard_until_ms": int(_BINANCE_HARD_BAN_UNTIL_MS), "soft_until_ms": int(_BINANCE_SOFT_UNTIL_MS)},
            # liq fields
            "liq_buy_usd_5m": None,
            "liq_sell_usd_5m": None,
            "liq_total_usd_5m": None,
            "liq_delta_ratio_5m": None,
            "liq_regime": "unknown",
        }
        return payload

    # Resolve Binance symbol
    binance_symbols = await _get_binance_symbols()
    binance_symbol = _map_symbol_to_binance(symbol, binance_symbols)
    if binance_symbol is None:
        payload = {
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
            "score_components": {"flow": 0, "oi": 0, "crowding": 0, "orderbook": 0},
            "score_meta": {"mode": eff_mode},
            "available_components": [],
            "available_components_count": 0,
            "ban": {"hard_until_ms": int(_BINANCE_HARD_BAN_UNTIL_MS), "soft_until_ms": int(_BINANCE_SOFT_UNTIL_MS)},
            # liq fields
            "liq_buy_usd_5m": None,
            "liq_sell_usd_5m": None,
            "liq_total_usd_5m": None,
            "liq_delta_ratio_5m": None,
            "liq_regime": "unknown",
        }
        return payload

    # Fields
    oi_value: Optional[float] = None
    oi_slope: Optional[float] = None
    oi_hist_slope: Optional[float] = None

    funding_rate: Optional[float] = None
    funding_mean: Optional[float] = None
    funding_std: Optional[float] = None
    funding_z: Optional[float] = None

    basis_pct: Optional[float] = None

    tape_1m: Optional[float] = None
    tape_5m: Optional[float] = None

    ob_10: Optional[float] = None
    ob_25: Optional[float] = None

    cvd_slope: Optional[float] = None

    # Liquidations (WS metrics)
    liq_buy_usd_5m: Optional[float] = None
    liq_sell_usd_5m: Optional[float] = None
    liq_total_usd_5m: Optional[float] = None
    liq_delta_ratio_5m: Optional[float] = None
    liq_regime: str = "unknown"

    # Optional LSR
    lsr_global_last = lsr_global_slope = None
    lsr_top_last = lsr_top_slope = None
    taker_ls_last = taker_ls_slope = None

    # -------------------------
    # Stage 1 (always): OI + premiumIndex
    # -------------------------
    oi_value = await _fetch_open_interest(binance_symbol)
    prem = await _fetch_premium_index(binance_symbol)

    if oi_value is None:
        warnings.append("no_oi")
    else:
        oi_slope = _compute_oi_slope(binance_symbol, oi_value)
        _OI_HISTORY[binance_symbol] = (time.time(), float(oi_value))

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

    # -------------------------
    # Liquidations window (WS) — independent of mode
    # -------------------------
    if use_liq:
        try:
            b, s, t, d = await _liq_metrics(binance_symbol, window_sec=_LIQ_WINDOW_SEC)
            liq_buy_usd_5m, liq_sell_usd_5m, liq_total_usd_5m, liq_delta_ratio_5m = b, s, t, d
            liq_regime = _classify_liq(liq_delta_ratio_5m, liq_total_usd_5m)
        except Exception:
            warnings.append("liq_metrics_error")

    # -------------------------
    # Mode NORMAL/FULL: add Tape + Orderbook (parallel)
    # -------------------------
    trades = None
    depth = None
    if eff_mode in ("NORMAL", "FULL"):
        trades, depth = await asyncio.gather(
            _fetch_agg_trades(binance_symbol, limit=1000),
            _fetch_depth(binance_symbol, limit=100),
        )

        if isinstance(trades, list) and trades:
            tape_1m = _compute_tape_delta(trades, window_sec=60)
            tape_5m = _compute_tape_delta(trades, window_sec=300)
        else:
            warnings.append("no_trades")

        if isinstance(depth, dict):
            ob_10 = _compute_orderbook_imbalance(depth, band_bps=10.0)
            ob_25 = _compute_orderbook_imbalance(depth, band_bps=25.0)
        else:
            warnings.append("no_depth")

    # -------------------------
    # Mode FULL: add CVD + OI hist + funding hist (parallel)
    # -------------------------
    if eff_mode == "FULL":
        klines, oi_hist, funding_hist = await asyncio.gather(
            _fetch_klines_1h(binance_symbol, limit=120),
            _fetch_open_interest_hist(binance_symbol, period="5m", limit=30),
            _fetch_funding_history(binance_symbol, limit=30),
        )

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

        # Optional LSR (heavier, off by default)
        if INCLUDE_LSR:
            try:
                lsr_g, lsr_t, tk = await asyncio.gather(
                    _fetch_lsr("/futures/data/globalLongShortAccountRatio", binance_symbol, period="1h", limit=30),
                    _fetch_lsr("/futures/data/topLongShortAccountRatio", binance_symbol, period="1h", limit=30),
                    _fetch_lsr("/futures/data/takerlongshortRatio", binance_symbol, period="1h", limit=30),
                )
                lsr_global_last, lsr_global_slope = _extract_lsr_stats(lsr_g)
                lsr_top_last, lsr_top_slope = _extract_lsr_stats(lsr_t)
                taker_ls_last, taker_ls_slope = _extract_lsr_stats(tk)
            except Exception:
                warnings.append("lsr_error")

    funding_regime = _classify_funding(funding_rate, z=funding_z)
    basis_regime = _classify_basis(basis_pct)
    crowding_regime = _classify_crowding(bias, funding_rate, basis_pct, funding_z)
    flow_regime = _classify_flow(cvd_slope, tape_5m)
    ob_regime = _classify_orderbook(ob_25)

    inst_score, components, score_meta = _score_institutional(
        bias,
        oi_slope=oi_slope,
        oi_hist_slope=oi_hist_slope,
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

    available = any(
        [
            oi_value is not None,
            funding_rate is not None,
            tape_5m is not None,
            ob_25 is not None,
            cvd_slope is not None,
            oi_hist_slope is not None,
            (liq_total_usd_5m is not None and liq_total_usd_5m > 0.0),
        ]
    )

    payload: Dict[str, Any] = {
        # required (compat)
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
        # extras
        "oi_hist_slope": oi_hist_slope,
        "tape_delta_1m": tape_1m,
        "tape_delta_5m": tape_5m,
        "tape_regime": _classify_tape(tape_5m),
        "basis_pct": basis_pct,
        "basis_regime": basis_regime,
        "orderbook_imb_10bps": ob_10,
        "orderbook_imb_25bps": ob_25,
        "orderbook_regime": ob_regime,
        "funding_mean": funding_mean,
        "funding_std": funding_std,
        "funding_z": funding_z,
        # liquidations (WS)
        "liq_buy_usd_5m": liq_buy_usd_5m,
        "liq_sell_usd_5m": liq_sell_usd_5m,
        "liq_total_usd_5m": liq_total_usd_5m,
        "liq_delta_ratio_5m": liq_delta_ratio_5m,
        "liq_regime": liq_regime,
        # scoring debug
        "score_components": components,
        "score_meta": score_meta,
        # LSR debug (only in FULL + INCLUDE_LSR)
        "lsr_global_last": lsr_global_last,
        "lsr_global_slope": lsr_global_slope,
        "lsr_top_last": lsr_top_last,
        "lsr_top_slope": lsr_top_slope,
        "taker_ls_last": taker_ls_last,
        "taker_ls_slope": taker_ls_slope,
        # new: availability & bans
        "available_components": [],
        "available_components_count": 0,
        "ban": {"hard_until_ms": int(_BINANCE_HARD_BAN_UNTIL_MS), "soft_until_ms": int(_BINANCE_SOFT_UNTIL_MS)},
    }

    comps = _available_components_list(payload)
    payload["available_components"] = comps
    payload["available_components_count"] = int(len(comps))

    # ajoute la backoff symbole (utile scanner)
    st = _get_sym_state(binance_symbol)
    if st is not None:
        payload["symbol_cooldown_until_ms"] = int(st.until_ms)
        payload["symbol_errors"] = int(st.errors)

    return payload


# Alias (si tu utilises déjà un autre nom ailleurs)
async def compute_institutional(symbol: str, bias: str, *, mode: Optional[str] = None, include_liquidations: bool = False) -> Dict[str, Any]:
    return await compute_full_institutional_analysis(symbol, bias, include_liquidations=include_liquidations, mode=mode)


def get_ban_state() -> Dict[str, int]:
    """Helper sync pour logs."""
    return {"hard_until_ms": int(_BINANCE_HARD_BAN_UNTIL_MS), "soft_until_ms": int(_BINANCE_SOFT_UNTIL_MS)}
