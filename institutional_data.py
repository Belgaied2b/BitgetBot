# =====================================================================
# institutional_data.py — Ultra Desk 2.0
# Binance USDT-M Futures (public endpoints + optional WebSocket orderflow)
#
# Ajouts majeurs :
# - WS orderflow natif Binance (fallback) : aggTrade + bookTicker + markPrice@1s
#   -> réduit fortement les appels REST (tape/ob/funding/basis en WS)
# - Normalisation (rolling z-scores) pour tape / OI slope / OB / basis / liq total
# - Scoring v2 “regime-aware” (tout en gardant un score v1 raw pour debug)
#
# Robustesse scan multi-coins :
# - Rate limiter global (semaphore + pacing)
# - Circuit breaker "hard ban" (418 / -1003 avec ban-until) + "soft cooldown" (429 / -1003 / 5xx)
# - Backoff / cooldown par symbole (évite de marteler le même coin)
# - Shared aiohttp session (pas de session par call)
# - Modes LIGHT/NORMAL/FULL + override par paramètre (scanner pass1/pass2)
# - Sortie enrichie : available_components_count + available_components + ban info + mode effectif
#
# WS integration (preferred):
# - Si institutional_ws_hub.HUB tourne : on l’utilise en priorité (read-only)
# - Sinon : hub WS interne (single connection + subscribe on-demand)
#
# Liquidations:
# - Si HUB externe fournit les métriques liq -> on les utilise
# - Sinon fallback : stream all-market !forceOrder@arr (single connection, agrégation mémoire)
#
# Compat:
# - Ajoute/maintient les clés legacy (openInterest, fundingRate, ...) pour éviter KeyError
# =====================================================================

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
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
# Optional external WS hub (read-only)
# ---------------------------------------------------------------------
_WS_HUB = None
try:
    from institutional_ws_hub import HUB as _WS_HUB  # type: ignore
except Exception:
    _WS_HUB = None

INST_USE_WS_HUB = str(os.getenv("INST_USE_WS_HUB", "1")).strip() == "1"
WS_STALE_SEC = float(os.getenv("INST_WS_STALE_SEC", "15"))

# ---------------------------------------------------------------------
# Internal WS orderflow hub (fallback) — single connection + on-demand subscribe
# ---------------------------------------------------------------------
INST_USE_INTERNAL_ORDERFLOW_WS = str(os.getenv("INST_USE_INTERNAL_ORDERFLOW_WS", "1")).strip() == "1"
INST_WS_INCLUDE_MARKPRICE = str(os.getenv("INST_WS_INCLUDE_MARKPRICE", "1")).strip() == "1"
INST_WS_INCLUDE_BOOKTICKER = str(os.getenv("INST_WS_INCLUDE_BOOKTICKER", "1")).strip() == "1"
INST_WS_TRADE_STORE_SEC = int(float(os.getenv("INST_WS_TRADE_STORE_SEC", "900")))  # 15m
INST_WS_TRADE_MAXLEN = int(float(os.getenv("INST_WS_TRADE_MAXLEN", "12000")))

# ---------------------------------------------------------------------
# Normalisation (rolling z-scores)
# ---------------------------------------------------------------------
INST_NORM_ENABLED = str(os.getenv("INST_NORM_ENABLED", "1")).strip() == "1"
INST_NORM_MIN_POINTS = int(float(os.getenv("INST_NORM_MIN_POINTS", "20")))
INST_NORM_WINDOW = int(float(os.getenv("INST_NORM_WINDOW", "120")))

# ---------------------------------------------------------------------
# Modes (env) + overrides
# ---------------------------------------------------------------------
INST_MODE = str(os.getenv("INST_MODE", "LIGHT")).upper().strip()
if INST_MODE not in ("LIGHT", "NORMAL", "FULL"):
    INST_MODE = "LIGHT"

INCLUDE_LSR = str(os.getenv("INST_INCLUDE_LSR", "0")).strip() == "1"

# ---------------------------------------------------------------------
# Liquidations (all-market WebSocket fallback)
# ---------------------------------------------------------------------
INST_INCLUDE_LIQUIDATIONS = str(os.getenv("INST_INCLUDE_LIQUIDATIONS", "0")).strip() == "1"
_LIQ_WINDOW_SEC = int(float(os.getenv("INST_LIQ_WINDOW_SEC", "300")))      # metrics window (default 5m)
_LIQ_STORE_SEC = int(float(os.getenv("INST_LIQ_STORE_SEC", "900")))        # store depth (default 15m)
_LIQ_MIN_NOTIONAL_USD = float(os.getenv("INST_LIQ_MIN_NOTIONAL_USD", "50000"))

# ---------------------------------------------------------------------
# Global rate limiting + circuit breaker
# ---------------------------------------------------------------------
_BINANCE_CONCURRENCY = max(1, int(os.getenv("BINANCE_HTTP_CONCURRENCY", "3")))
_BINANCE_MIN_INTERVAL_SEC = float(os.getenv("BINANCE_MIN_INTERVAL_S", os.getenv("BINANCE_MIN_INTERVAL_SEC", "0.12")))
_BINANCE_HTTP_TIMEOUT_S = float(os.getenv("BINANCE_HTTP_TIMEOUT_S", os.getenv("BINANCE_HTTP_TIMEOUT", "10")))
_BINANCE_HTTP_RETRIES = max(0, int(os.getenv("BINANCE_HTTP_RETRIES", "2")))

_SOFT_COOLDOWN_MS_DEFAULT = int(float(os.getenv("BINANCE_SOFT_COOLDOWN_SEC", "20")) * 1000)
_HARD_BAN_FALLBACK_MS = int(float(os.getenv("BINANCE_HARD_BAN_FALLBACK_MIN", "15")) * 60_000)

_HTTP_SEM = asyncio.Semaphore(_BINANCE_CONCURRENCY)
_PACE_LOCK = asyncio.Lock()
_LAST_REQ_TS = 0.0

_BINANCE_HARD_BAN_UNTIL_MS = 0
_BINANCE_SOFT_UNTIL_MS = 0

_RE_BAN_UNTIL = re.compile(r"banned until (\d+)", re.IGNORECASE)

# Per-symbol backoff
_SYM_STATE: Dict[str, "SymbolBackoff"] = {}

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

        timeout = aiohttp.ClientTimeout(total=float(_BINANCE_HTTP_TIMEOUT_S))
        connector = aiohttp.TCPConnector(limit=50, ttl_dns_cache=300, enable_cleanup_closed=True)
        _SESSION = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return _SESSION


# =====================================================================
# Internal WS Orderflow Hub (single connection + subscribe on-demand)
# =====================================================================

@dataclass
class _TradeEvent:
    ts_ms: int
    side: str  # "BUY" or "SELL" (taker direction)
    qty: float
    notional: float


@dataclass
class _BookTicker:
    ts_ms: int = 0
    bid_p: float = 0.0
    bid_q: float = 0.0
    ask_p: float = 0.0
    ask_q: float = 0.0


@dataclass
class _MarkPrice:
    ts_ms: int = 0
    mark_price: Optional[float] = None
    index_price: Optional[float] = None
    funding_rate: Optional[float] = None


@dataclass
class _SymbolOFState:
    trades: Deque[_TradeEvent] = field(default_factory=lambda: deque(maxlen=INST_WS_TRADE_MAXLEN))
    book: _BookTicker = field(default_factory=_BookTicker)
    mark: _MarkPrice = field(default_factory=_MarkPrice)


class _BinanceOrderflowHub:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._ws_task: Optional[asyncio.Task] = None
        self._stop: Optional[asyncio.Event] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._id = 1

        self._subs: Set[str] = set()  # stream names (lowercase)
        self._state: Dict[str, _SymbolOFState] = {}
        self._last_msg_ts: float = 0.0

    def is_running(self) -> bool:
        return self._ws_task is not None and not self._ws_task.done()

    async def ensure_running(self) -> None:
        if self.is_running():
            return
        async with self._lock:
            if self.is_running():
                return
            if self._stop is None or self._stop.is_set():
                self._stop = asyncio.Event()
            self._ws_task = asyncio.create_task(self._run(), name="inst_orderflow_ws_hub")

    async def watch(self, binance_symbol: str) -> None:
        sym = (binance_symbol or "").upper().strip()
        if not sym:
            return
        await self.ensure_running()

        streams: List[str] = []
        s = sym.lower()
        streams.append(f"{s}@aggTrade")
        if INST_WS_INCLUDE_BOOKTICKER:
            streams.append(f"{s}@bookTicker")
        if INST_WS_INCLUDE_MARKPRICE:
            streams.append(f"{s}@markPrice@1s")

        async with self._lock:
            if sym not in self._state:
                self._state[sym] = _SymbolOFState()
            to_add = [x for x in streams if x not in self._subs]
            if not to_add:
                return
            self._subs.update(to_add)
            await self._send({"method": "SUBSCRIBE", "params": to_add, "id": self._next_id()})

    def get_snapshot(self, binance_symbol: str) -> Optional[Dict[str, Any]]:
        sym = (binance_symbol or "").upper().strip()
        if not sym:
            return None
        st = self._state.get(sym)
        if st is None:
            return None

        now = time.time()
        if (now - self._last_msg_ts) > float(WS_STALE_SEC):
            return None

        tape_1m = self._tape_delta(st.trades, window_sec=60)
        tape_5m = self._tape_delta(st.trades, window_sec=300)
        cvd_notional_5m = self._cvd_notional(st.trades, window_sec=300)
        ob_imb = self._tob_imbalance(st.book)

        return {
            "available": True,
            "ts": now,
            "tape_delta_1m": tape_1m,
            "tape_delta_5m": tape_5m,
            "cvd_notional_5m": cvd_notional_5m,
            "orderbook_imbalance": ob_imb,
            "book_ts_ms": int(st.book.ts_ms or 0),
            "mark_price": st.mark.mark_price,
            "index_price": st.mark.index_price,
            "funding_rate": st.mark.funding_rate,
            "mark_ts_ms": int(st.mark.ts_ms or 0),
        }

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    async def _send(self, payload: Dict[str, Any]) -> None:
        ws = self._ws
        if ws is None or ws.closed:
            return
        try:
            await ws.send_str(json.dumps(payload))
        except Exception:
            return

    @staticmethod
    def _prune_trades(trades: Deque[_TradeEvent], cutoff_ms: int) -> None:
        while trades and trades[0].ts_ms < cutoff_ms:
            trades.popleft()

    @staticmethod
    def _tape_delta(trades: Deque[_TradeEvent], window_sec: int) -> Optional[float]:
        if not trades:
            return None
        cutoff = _now_ms() - int(window_sec) * 1000
        buy = 0.0
        sell = 0.0
        for ev in reversed(trades):
            if ev.ts_ms < cutoff:
                break
            if ev.side == "BUY":
                buy += ev.qty
            else:
                sell += ev.qty
        den = buy + sell
        if den <= 0:
            return None
        return float((buy - sell) / den)

    @staticmethod
    def _cvd_notional(trades: Deque[_TradeEvent], window_sec: int) -> Optional[float]:
        if not trades:
            return None
        cutoff = _now_ms() - int(window_sec) * 1000
        acc = 0.0
        for ev in reversed(trades):
            if ev.ts_ms < cutoff:
                break
            acc += ev.notional if ev.side == "BUY" else -ev.notional
        return float(acc)

    @staticmethod
    def _tob_imbalance(book: _BookTicker) -> Optional[float]:
        try:
            if book.bid_p <= 0 or book.ask_p <= 0:
                return None
            bid_val = float(book.bid_p) * float(book.bid_q)
            ask_val = float(book.ask_p) * float(book.ask_q)
            den = bid_val + ask_val
            if den <= 0:
                return None
            return float((bid_val - ask_val) / den)
        except Exception:
            return None

    async def _run(self) -> None:
        self._last_msg_ts = time.time()
        backoff = 1.0
        session = await _get_session()
        url = f"{BINANCE_FSTREAM_WS_BASE}"
        LOGGER.info("[INST_OF] WS hub start url=%s", url)

        while self._stop is not None and (not self._stop.is_set()):
            try:
                async with session.ws_connect(url, heartbeat=30, autoping=True) as ws:
                    self._ws = ws
                    backoff = 1.0

                    # Try to enable combined (safe if unsupported)
                    try:
                        await self._send({"method": "SET_PROPERTY", "params": ["combined", True], "id": self._next_id()})
                    except Exception:
                        pass

                    # Re-subscribe after reconnect
                    async with self._lock:
                        if self._subs:
                            await self._send({"method": "SUBSCRIBE", "params": sorted(self._subs), "id": self._next_id()})

                    async for msg in ws:
                        if self._stop is not None and self._stop.is_set():
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

                        self._last_msg_ts = time.time()

                        try:
                            payload = json.loads(raw) if raw else None
                        except Exception:
                            continue

                        # Combined wrapper: {"stream":"...","data":{...}}
                        if isinstance(payload, dict) and "data" in payload and isinstance(payload.get("data"), dict):
                            payload = payload.get("data")

                        if not isinstance(payload, dict):
                            continue

                        # ACK replies {"result": null, "id": ...}
                        if "result" in payload and "id" in payload and "e" not in payload:
                            continue

                        ev_type = payload.get("e")

                        # aggTrade
                        if ev_type == "aggTrade":
                            sym = str(payload.get("s") or "").upper()
                            if not sym:
                                continue
                            try:
                                price = float(payload.get("p") or 0.0)
                                qty = float(payload.get("q") or 0.0)
                                ts_ms = int(payload.get("T") or payload.get("E") or _now_ms())
                                is_buyer_maker = bool(payload.get("m", False))  # True => taker sells
                                side = "SELL" if is_buyer_maker else "BUY"
                                notional = price * qty
                            except Exception:
                                continue

                            st = self._state.get(sym)
                            if st is None:
                                st = _SymbolOFState()
                                self._state[sym] = st

                            st.trades.append(_TradeEvent(ts_ms=ts_ms, side=side, qty=qty, notional=notional))
                            cutoff_ms = _now_ms() - int(INST_WS_TRADE_STORE_SEC) * 1000
                            self._prune_trades(st.trades, cutoff_ms)
                            continue

                        # bookTicker
                        if ev_type == "bookTicker":
                            sym = str(payload.get("s") or "").upper()
                            if not sym:
                                continue
                            try:
                                st = self._state.get(sym)
                                if st is None:
                                    st = _SymbolOFState()
                                    self._state[sym] = st
                                st.book.ts_ms = int(payload.get("T") or payload.get("E") or _now_ms())
                                st.book.bid_p = float(payload.get("b") or 0.0)
                                st.book.bid_q = float(payload.get("B") or 0.0)
                                st.book.ask_p = float(payload.get("a") or 0.0)
                                st.book.ask_q = float(payload.get("A") or 0.0)
                            except Exception:
                                continue
                            continue

                        # markPrice
                        if ev_type == "markPriceUpdate":
                            sym = str(payload.get("s") or "").upper()
                            if not sym:
                                continue
                            try:
                                st = self._state.get(sym)
                                if st is None:
                                    st = _SymbolOFState()
                                    self._state[sym] = st
                                st.mark.ts_ms = int(payload.get("T") or payload.get("E") or _now_ms())
                                mp = payload.get("p")
                                ip = payload.get("i")
                                fr = payload.get("r")
                                st.mark.mark_price = float(mp) if mp is not None else None
                                st.mark.index_price = float(ip) if ip is not None else None
                                st.mark.funding_rate = float(fr) if fr is not None else None
                            except Exception:
                                continue
                            continue

            except asyncio.CancelledError:
                break
            except Exception as e:
                LOGGER.warning("[INST_OF] WS hub error: %s", e)

            self._ws = None
            if self._stop is not None and self._stop.is_set():
                break
            await asyncio.sleep(min(60.0, backoff))
            backoff = min(60.0, backoff * 2.0)

        self._ws = None
        LOGGER.info("[INST_OF] WS hub stopped")

    async def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._ws_task is not None:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except Exception:
                pass
        self._ws_task = None
        self._ws = None


_OF_HUB: Optional[_BinanceOrderflowHub] = None
_OF_LOCK = asyncio.Lock()


async def _get_of_hub() -> Optional[_BinanceOrderflowHub]:
    global _OF_HUB
    if not INST_USE_INTERNAL_ORDERFLOW_WS:
        return None
    if _OF_HUB is not None and _OF_HUB.is_running():
        return _OF_HUB
    async with _OF_LOCK:
        if _OF_HUB is None:
            _OF_HUB = _BinanceOrderflowHub()
        await _OF_HUB.ensure_running()
        return _OF_HUB


def _of_snapshot(binance_symbol: str) -> Optional[Dict[str, Any]]:
    if not INST_USE_INTERNAL_ORDERFLOW_WS:
        return None
    if _OF_HUB is None:
        return None
    try:
        snap = _OF_HUB.get_snapshot(binance_symbol)
        if not isinstance(snap, dict) or not snap.get("available"):
            return None
        ts = snap.get("ts")
        if ts is None:
            return None
        if (time.time() - float(ts)) > float(WS_STALE_SEC):
            return None
        return snap
    except Exception:
        return None


# =====================================================================
# Liquidations WebSocket (single shared worker) — fallback if hub not used
# =====================================================================
_LIQ_TASK: Optional[asyncio.Task] = None
_LIQ_START_LOCK = asyncio.Lock()
_LIQ_STOP: Optional[asyncio.Event] = None
_LIQ_LOCK = asyncio.Lock()
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
            async with session.ws_connect(url, heartbeat=30, autoping=True) as ws:
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


async def close_institutional_session() -> None:
    """Optionnel: à appeler proprement au shutdown."""
    global _SESSION, _LIQ_TASK, _LIQ_STOP, _OF_HUB

    # stop internal orderflow hub
    try:
        if _OF_HUB is not None:
            await _OF_HUB.stop()
    except Exception:
        pass
    _OF_HUB = None

    # stop ws liquidation worker
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
        wait = float(_BINANCE_MIN_INTERVAL_SEC) - (now - float(_LAST_REQ_TS))
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
    - small retries on transient errors only (timeouts / 5xx)
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
        for attempt in range(0, _BINANCE_HTTP_RETRIES + 1):
            await _pace()
            try:
                async with session.get(url, params=params) as resp:
                    status = resp.status
                    raw = await resp.read()
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

                        if status == 429:
                            _set_soft_cooldown(_SOFT_COOLDOWN_MS_DEFAULT, reason=f"{path} 429")
                            if st is not None:
                                st.mark_err(base_ms=2_500)
                            LOGGER.warning("[INST] HTTP 429 GET %s params=%s", path, params)
                            return None

                        if 500 <= status <= 599:
                            _set_soft_cooldown(5_000, reason=f"{path} {status}")
                            if st is not None:
                                st.mark_err(base_ms=1_800)
                            if attempt < _BINANCE_HTTP_RETRIES:
                                await asyncio.sleep(min(2.5, 0.6 * (1.8 ** attempt)))
                                continue
                            LOGGER.warning("[INST] HTTP %s GET %s params=%s", status, path, params)
                            return None

                        if st is not None:
                            st.mark_err(base_ms=1_500)
                        LOGGER.warning("[INST] HTTP %s GET %s params=%s resp=%s", status, path, params, (txt or "")[:200])
                        return None

                    if st is not None:
                        st.mark_ok()
                    return data

            except asyncio.TimeoutError:
                if st is not None:
                    st.mark_err(base_ms=1_600)
                if attempt < _BINANCE_HTTP_RETRIES:
                    await asyncio.sleep(min(2.5, 0.6 * (1.8 ** attempt)))
                    continue
                LOGGER.error("[INST] Timeout GET %s params=%s", path, params)
                return None
            except Exception as e:
                if st is not None:
                    st.mark_err(base_ms=2_000)
                if attempt < _BINANCE_HTTP_RETRIES:
                    await asyncio.sleep(min(2.5, 0.6 * (1.8 ** attempt)))
                    continue
                LOGGER.error("[INST] Exception GET %s params=%s: %s", path, params, e)
                return None

    return None


# ---------------------------------------------------------------------
# Light caches
# ---------------------------------------------------------------------
_KLINES_CACHE: Dict[Tuple[str, str, int], Tuple[float, Any]] = {}
_DEPTH_CACHE: Dict[Tuple[str, int], Tuple[float, Any]] = {}
_TRADES_CACHE: Dict[Tuple[str, int], Tuple[float, Any]] = {}
_FUNDING_CACHE: Dict[str, Tuple[float, Any]] = {}
_FUNDING_HIST_CACHE: Dict[Tuple[str, int], Tuple[float, Any]] = {}
_OI_CACHE: Dict[str, Tuple[float, Any]] = {}
_OI_HIST_CACHE: Dict[Tuple[str, str, int], Tuple[float, Any]] = {}
_LSR_CACHE: Dict[Tuple[str, str, str, int], Tuple[float, Any]] = {}

_OI_HISTORY: Dict[str, Tuple[float, float]] = {}

_BINANCE_SYMBOLS: Optional[Set[str]] = None
_BINANCE_SYMBOLS_TS: float = 0.0

KLINES_TTL = float(os.getenv("INST_KLINES_TTL", "120"))
DEPTH_TTL = float(os.getenv("INST_DEPTH_TTL", "10"))
TRADES_TTL = float(os.getenv("INST_TRADES_TTL", "10"))
FUNDING_TTL = float(os.getenv("INST_FUNDING_TTL", "60"))
FUNDING_HIST_TTL = float(os.getenv("INST_FUNDING_HIST_TTL", "300"))
OI_TTL = float(os.getenv("INST_OI_TTL", "60"))
OI_HIST_TTL = float(os.getenv("INST_OI_HIST_TTL", "300"))
BINANCE_SYMBOLS_TTL = float(os.getenv("INST_EXCHANGEINFO_TTL", os.getenv("BINANCE_SYMBOLS_TTL_S", "900")))
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
# External WS snapshot reader
# =====================================================================
def _ws_hub_running() -> bool:
    """HUB.is_running can be bool, property, or method."""
    if _WS_HUB is None:
        return False
    try:
        v = getattr(_WS_HUB, "is_running", False)
        if callable(v):
            return bool(v())
        return bool(v)
    except Exception:
        return False


def _ws_snapshot(binance_symbol: str) -> Optional[Dict[str, Any]]:
    if not INST_USE_WS_HUB or _WS_HUB is None:
        return None
    try:
        if not _ws_hub_running():
            return None
        snap = _WS_HUB.get_snapshot(binance_symbol)
        if not isinstance(snap, dict) or not snap.get("available"):
            return None
        ts = snap.get("ts")
        if ts is None:
            return None
        if (time.time() - float(ts)) > float(WS_STALE_SEC):
            return None
        return snap
    except Exception:
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
# Normalization (rolling z-scores)
# =====================================================================

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


def _score_institutional_v1(
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
    Score "legacy" en [0..4] (conservé pour debug).
    """
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
    if (not oi_ok) and oi_hist_slope is not None:
        x = float(oi_hist_slope)
        if b == "LONG" and x >= 0.012:
            oi_ok = True
        if b == "SHORT" and x <= -0.012:
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


def _score_institutional_v2_regime(
    bias: str,
    *,
    crowding_regime: str,
    tape_5m: Optional[float],
    ob_imb: Optional[float],
    oi_slope: Optional[float],
    funding_rate: Optional[float],
    funding_z: Optional[float],
    basis_pct: Optional[float],
    liq_delta_ratio_5m: Optional[float],
    liq_total_usd_5m: Optional[float],
    tape_5m_z: Optional[float],
    ob_imb_z: Optional[float],
    oi_slope_z: Optional[float],
    basis_pct_z: Optional[float],
    liq_total_z: Optional[float],
) -> Tuple[int, Dict[str, int], Dict[str, Any]]:
    """
    Score regime-aware en [0..4]. Utilise z-scores si dispo, sinon seuils raw.
    """
    b = (bias or "").upper().strip()
    comp: Dict[str, int] = {"flow": 0, "oi": 0, "crowding": 0, "orderbook": 0}
    meta: Dict[str, Any] = {"used_z": {}, "penalties": []}

    # FLOW
    flow_points = 0
    used_z_flow = False

    if tape_5m_z is not None:
        used_z_flow = True
        if b == "LONG":
            if tape_5m_z >= 1.5:
                flow_points += 2
            elif tape_5m_z >= 0.7:
                flow_points += 1
        elif b == "SHORT":
            if tape_5m_z <= -1.5:
                flow_points += 2
            elif tape_5m_z <= -0.7:
                flow_points += 1
    elif tape_5m is not None:
        x = float(tape_5m)
        if b == "LONG":
            if x >= 0.35:
                flow_points += 2
            elif x >= 0.12:
                flow_points += 1
        elif b == "SHORT":
            if x <= -0.35:
                flow_points += 2
            elif x <= -0.12:
                flow_points += 1

    liq_used = False
    if liq_delta_ratio_5m is not None and liq_total_usd_5m is not None and float(liq_total_usd_5m) >= float(_LIQ_MIN_NOTIONAL_USD):
        liq_used = True
        boost_ok = True
        if liq_total_z is not None and liq_total_z < 0.7:
            boost_ok = False

        if boost_ok:
            x = float(liq_delta_ratio_5m)
            if b == "LONG" and x >= 0.35:
                flow_points += 1
            if b == "SHORT" and x <= -0.35:
                flow_points += 1

    if flow_points >= 2:
        comp["flow"] = 2
    elif flow_points >= 1:
        comp["flow"] = 1

    meta["used_z"]["flow"] = bool(used_z_flow)
    meta["liq_used"] = bool(liq_used)

    # OI
    used_z_oi = False
    oi_ok = False
    if oi_slope_z is not None:
        used_z_oi = True
        if b == "LONG" and oi_slope_z >= 0.8:
            oi_ok = True
        elif b == "SHORT" and oi_slope_z <= -0.8:
            oi_ok = True
    elif oi_slope is not None:
        x = float(oi_slope)
        if b == "LONG" and x >= 0.008:
            oi_ok = True
        elif b == "SHORT" and x <= -0.008:
            oi_ok = True
    if oi_ok:
        comp["oi"] = 1
    meta["used_z"]["oi"] = bool(used_z_oi)

    # CROWDING (contrarian)
    crowding_ok = False
    if funding_z is not None:
        z = float(funding_z)
        if b == "LONG" and z <= -1.6:
            crowding_ok = True
        if b == "SHORT" and z >= 1.6:
            crowding_ok = True
    if (not crowding_ok) and funding_rate is not None:
        fr = float(funding_rate)
        if b == "LONG" and fr < -0.0005:
            crowding_ok = True
        if b == "SHORT" and fr > 0.0005:
            crowding_ok = True
    if (not crowding_ok) and basis_pct is not None and basis_pct_z is not None:
        if b == "LONG" and float(basis_pct) < -0.0006 and float(basis_pct_z) <= -0.7:
            crowding_ok = True
        if b == "SHORT" and float(basis_pct) > 0.0006 and float(basis_pct_z) >= 0.7:
            crowding_ok = True

    if crowding_ok:
        comp["crowding"] = 1

    penalty = 0
    if isinstance(crowding_regime, str) and crowding_regime.endswith("_risky"):
        if comp["flow"] < 2:
            meta["penalties"].append("crowding_risky_without_strong_flow")
            penalty = 1

    # ORDERBOOK
    used_z_ob = False
    ob_ok = False
    if ob_imb_z is not None:
        used_z_ob = True
        if b == "LONG" and ob_imb_z >= 0.7:
            ob_ok = True
        elif b == "SHORT" and ob_imb_z <= -0.7:
            ob_ok = True
    elif ob_imb is not None:
        x = float(ob_imb)
        if b == "LONG" and x >= 0.12:
            ob_ok = True
        elif b == "SHORT" and x <= -0.12:
            ob_ok = True
    if ob_ok:
        comp["orderbook"] = 1
    meta["used_z"]["orderbook"] = bool(used_z_ob)

    total_raw = int(comp["flow"] + comp["oi"] + comp["crowding"] + comp["orderbook"])
    total = max(0, min(4, total_raw - int(penalty)))
    meta["penalty_points"] = int(penalty)
    meta["raw_sum"] = int(total_raw)
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
    if payload.get("cvd_slope") is not None or payload.get("cvd_notional_5m") is not None:
        out.append("cvd")
    if payload.get("oi_hist_slope") is not None:
        out.append("oi_hist")
    if payload.get("funding_z") is not None:
        out.append("funding_hist")
    if payload.get("liq_total_usd_5m") is not None:
        out.append("liquidations")
    if payload.get("ws_snapshot_used"):
        out.append("ws_hub")
    if payload.get("orderflow_ws_used"):
        out.append("orderflow_ws")
    if payload.get("normalization_enabled"):
        out.append("norm")
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
    Institutional analysis for a symbol (KuCoin/Bitget format) using Binance USDT-M Futures.
    """
    bias = (bias or "").upper().strip()
    eff_mode = (mode or INST_MODE).upper().strip()
    if eff_mode not in ("LIGHT", "NORMAL", "FULL"):
        eff_mode = "LIGHT"

    warnings: List[str] = []
    sources: Dict[str, str] = {}

    use_liq = bool(INST_INCLUDE_LIQUIDATIONS or include_liquidations)

    if _is_hard_banned():
        return {
            "institutional_score": 0,
            "institutional_score_raw": 0,
            "binance_symbol": None,
            "available": False,
            "oi": None,
            "oi_slope": None,
            "cvd_slope": None,
            "cvd_notional_5m": None,
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
            "liq_buy_usd_5m": None,
            "liq_sell_usd_5m": None,
            "liq_total_usd_5m": None,
            "liq_delta_ratio_5m": None,
            "liq_regime": "unknown",
            "liquidation_intensity": None,
            "ws_snapshot_used": False,
            "orderflow_ws_used": False,
            "normalization_enabled": bool(INST_NORM_ENABLED),
            "data_sources": {},
            "openInterest": None,
            "fundingRate": None,
        }

    if _is_soft_blocked():
        warnings.append(f"binance_soft_cooldown_until_ms={_BINANCE_SOFT_UNTIL_MS}")

    # Resolve Binance symbol
    binance_symbols = await _get_binance_symbols()
    binance_symbol = _map_symbol_to_binance(symbol, binance_symbols)
    if binance_symbol is None:
        return {
            "institutional_score": 0,
            "institutional_score_raw": 0,
            "binance_symbol": None,
            "available": False,
            "oi": None,
            "oi_slope": None,
            "cvd_slope": None,
            "cvd_notional_5m": None,
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
            "liq_buy_usd_5m": None,
            "liq_sell_usd_5m": None,
            "liq_total_usd_5m": None,
            "liq_delta_ratio_5m": None,
            "liq_regime": "unknown",
            "liquidation_intensity": None,
            "ws_snapshot_used": False,
            "orderflow_ws_used": False,
            "normalization_enabled": bool(INST_NORM_ENABLED),
            "data_sources": {},
            "openInterest": None,
            "fundingRate": None,
        }

    # Start internal orderflow hub early if relevant (NORMAL/FULL) or LIGHT if markPrice enabled
    orderflow_ws_used = False
    need_of = False
    if INST_USE_INTERNAL_ORDERFLOW_WS:
        if eff_mode in ("NORMAL", "FULL"):
            need_of = True
        elif eff_mode == "LIGHT" and INST_WS_INCLUDE_MARKPRICE:
            need_of = True

    if need_of:
        try:
            hub = await _get_of_hub()
            if hub is not None:
                await hub.watch(binance_symbol)
        except Exception:
            pass

    # Try WS snapshots (external hub first, then internal hub)
    ws_snap = _ws_snapshot(binance_symbol)
    ws_used = bool(ws_snap is not None)

    of_snap = None if ws_used else _of_snapshot(binance_symbol)
    if isinstance(of_snap, dict) and of_snap.get("available"):
        orderflow_ws_used = True

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

    cvd_slope: Optional[float] = None  # FULL fallback (kline-based)
    cvd_notional_5m: Optional[float] = None  # WS micro-CVD

    # Liquidations
    liq_buy_usd_5m: Optional[float] = None
    liq_sell_usd_5m: Optional[float] = None
    liq_total_usd_5m: Optional[float] = None
    liq_delta_ratio_5m: Optional[float] = None
    liq_regime: str = "unknown"

    # Optional LSR
    lsr_global_last = lsr_global_slope = None
    lsr_top_last = lsr_top_slope = None
    taker_ls_last = taker_ls_slope = None

    # Stage 1: OI (REST)
    oi_value = await _fetch_open_interest(binance_symbol)
    if oi_value is None:
        warnings.append("no_oi")
        sources["oi"] = "none"
    else:
        oi_slope = _compute_oi_slope(binance_symbol, oi_value)
        _OI_HISTORY[binance_symbol] = (time.time(), float(oi_value))
        sources["oi"] = "rest"

    # Funding/Basis: external hub > internal markPrice > REST premiumIndex
    if ws_snap is not None:
        try:
            fr = ws_snap.get("funding_rate")
            mp = ws_snap.get("mark_price")
            ip = ws_snap.get("index_price")
            funding_rate = float(fr) if fr is not None else None
            if mp is not None and ip is not None:
                ipf = float(ip)
                mpf = float(mp)
                if ipf > 0:
                    basis_pct = (mpf - ipf) / ipf
            sources["funding_rate"] = "ws" if funding_rate is not None else "none"
            sources["basis_pct"] = "ws" if basis_pct is not None else "none"
        except Exception:
            warnings.append("ws_funding_parse_error")

    if (funding_rate is None or basis_pct is None) and of_snap is not None:
        try:
            if of_snap.get("funding_rate") is not None:
                funding_rate = float(of_snap.get("funding_rate"))
                sources["funding_rate"] = "ws_orderflow"
            mp = of_snap.get("mark_price")
            ip = of_snap.get("index_price")
            if basis_pct is None and mp is not None and ip is not None:
                ipf = float(ip)
                mpf = float(mp)
                if ipf > 0:
                    basis_pct = (mpf - ipf) / ipf
                    sources["basis_pct"] = "ws_orderflow"
        except Exception:
            warnings.append("orderflow_markprice_parse_error")

    if funding_rate is None or basis_pct is None:
        prem = await _fetch_premium_index(binance_symbol)
        if isinstance(prem, dict):
            try:
                if funding_rate is None:
                    funding_rate = float(prem.get("lastFundingRate", "0"))
                sources["funding_rate"] = sources.get("funding_rate", "rest")
            except Exception:
                funding_rate = None
                warnings.append("funding_parse_error")
                sources["funding_rate"] = sources.get("funding_rate", "none")

            try:
                if basis_pct is None:
                    mark = float(prem.get("markPrice", "0"))
                    index = float(prem.get("indexPrice", "0"))
                    if index > 0:
                        basis_pct = (mark - index) / index
                sources["basis_pct"] = sources.get("basis_pct", "rest")
            except Exception:
                basis_pct = None
                warnings.append("basis_parse_error")
                sources["basis_pct"] = sources.get("basis_pct", "none")
        else:
            warnings.append("no_premiumIndex")

    # Liquidations: external hub else all-market WS
    if use_liq and ws_snap is not None:
        try:
            liq_total_usd_5m = float(ws_snap.get("liq_notional_5m") or 0.0)
            liq_buy_usd_5m = ws_snap.get("liq_buy_usd_5m")
            liq_sell_usd_5m = ws_snap.get("liq_sell_usd_5m")
            liq_delta_ratio_5m = ws_snap.get("liq_delta_ratio_5m")
            if liq_buy_usd_5m is not None:
                liq_buy_usd_5m = float(liq_buy_usd_5m)
            if liq_sell_usd_5m is not None:
                liq_sell_usd_5m = float(liq_sell_usd_5m)
            if liq_delta_ratio_5m is not None:
                liq_delta_ratio_5m = float(liq_delta_ratio_5m)
            liq_regime = _classify_liq(liq_delta_ratio_5m, liq_total_usd_5m)
            sources["liquidations"] = "ws"
        except Exception:
            warnings.append("ws_liq_parse_error")
    elif use_liq:
        try:
            await _ensure_liq_stream()
            b, s, t, d = await _liq_metrics(binance_symbol, window_sec=_LIQ_WINDOW_SEC)
            liq_buy_usd_5m, liq_sell_usd_5m, liq_total_usd_5m, liq_delta_ratio_5m = b, s, t, d
            liq_regime = _classify_liq(liq_delta_ratio_5m, liq_total_usd_5m)
            sources["liquidations"] = "ws_all_market"
        except Exception:
            warnings.append("liq_metrics_error")
            sources["liquidations"] = "none"

    # Mode NORMAL/FULL: Tape + Orderbook (ws hub > internal ws > rest)
    if eff_mode in ("NORMAL", "FULL"):
        if ws_snap is not None:
            try:
                tape_1m = ws_snap.get("tape_delta_1m")
                tape_5m = ws_snap.get("tape_delta_5m")
                if tape_1m is not None:
                    tape_1m = float(tape_1m)
                if tape_5m is not None:
                    tape_5m = float(tape_5m)
                sources["tape"] = "ws" if tape_5m is not None else "none"

                ob_25 = ws_snap.get("orderbook_imbalance")
                if ob_25 is not None:
                    ob_25 = float(ob_25)
                    ob_10 = float(ob_25)
                    sources["orderbook"] = "ws"

                cvd_notional_5m = ws_snap.get("cvd_notional_5m")
                if cvd_notional_5m is not None:
                    cvd_notional_5m = float(cvd_notional_5m)
                    sources["cvd_notional_5m"] = "ws"
            except Exception:
                warnings.append("ws_micro_parse_error")

        if (tape_5m is None or ob_25 is None) and of_snap is not None:
            try:
                if tape_1m is None and of_snap.get("tape_delta_1m") is not None:
                    tape_1m = float(of_snap.get("tape_delta_1m"))
                if tape_5m is None and of_snap.get("tape_delta_5m") is not None:
                    tape_5m = float(of_snap.get("tape_delta_5m"))
                    sources["tape"] = sources.get("tape", "ws_orderflow")

                if ob_25 is None and of_snap.get("orderbook_imbalance") is not None:
                    ob_25 = float(of_snap.get("orderbook_imbalance"))
                    ob_10 = float(ob_25)
                    sources["orderbook"] = sources.get("orderbook", "ws_orderflow")

                if cvd_notional_5m is None and of_snap.get("cvd_notional_5m") is not None:
                    cvd_notional_5m = float(of_snap.get("cvd_notional_5m"))
                    sources["cvd_notional_5m"] = sources.get("cvd_notional_5m", "ws_orderflow")
            except Exception:
                warnings.append("orderflow_micro_parse_error")

        if tape_5m is None or ob_25 is None:
            trades, depth = await asyncio.gather(
                _fetch_agg_trades(binance_symbol, limit=1000),
                _fetch_depth(binance_symbol, limit=100),
            )

            if isinstance(trades, list) and trades:
                if tape_1m is None:
                    tape_1m = _compute_tape_delta(trades, window_sec=60)
                if tape_5m is None:
                    tape_5m = _compute_tape_delta(trades, window_sec=300)
                sources["tape"] = sources.get("tape", "rest")
            else:
                warnings.append("no_trades")

            if isinstance(depth, dict):
                if ob_10 is None:
                    ob_10 = _compute_orderbook_imbalance(depth, band_bps=10.0)
                if ob_25 is None:
                    ob_25 = _compute_orderbook_imbalance(depth, band_bps=25.0)
                sources["orderbook"] = sources.get("orderbook", "rest")
            else:
                warnings.append("no_depth")

    # Mode FULL: add kline CVD + OI hist + funding hist (+ optional LSR)
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

    # Normalization (rolling z)
    oi_slope_z = _norm_update(binance_symbol, "oi_slope", oi_slope)
    tape_5m_z = _norm_update(binance_symbol, "tape_5m", tape_5m)
    ob_imb_z = _norm_update(binance_symbol, "ob_imb", ob_25)
    basis_pct_z = _norm_update(binance_symbol, "basis_pct", basis_pct)
    liq_total_z = _norm_update(binance_symbol, "liq_total", liq_total_usd_5m)

    # Regimes
    funding_regime = _classify_funding(funding_rate, z=funding_z)
    basis_regime = _classify_basis(basis_pct)
    crowding_regime = _classify_crowding(bias, funding_rate, basis_pct, funding_z)
    flow_regime = _classify_flow(cvd_slope, tape_5m)
    ob_regime = _classify_orderbook(ob_25)

    # Scoring v1 + v2
    inst_score_raw, components_raw, meta_raw = _score_institutional_v1(
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

    inst_score, components, meta_v2 = _score_institutional_v2_regime(
        bias,
        crowding_regime=crowding_regime,
        tape_5m=tape_5m,
        ob_imb=ob_25,
        oi_slope=oi_slope,
        funding_rate=funding_rate,
        funding_z=funding_z,
        basis_pct=basis_pct,
        liq_delta_ratio_5m=liq_delta_ratio_5m,
        liq_total_usd_5m=liq_total_usd_5m,
        tape_5m_z=tape_5m_z,
        ob_imb_z=ob_imb_z,
        oi_slope_z=oi_slope_z,
        basis_pct_z=basis_pct_z,
        liq_total_z=liq_total_z,
    )

    ok_count = _components_ok_count(components)

    score_meta: Dict[str, Any] = {
        "mode": eff_mode,
        "ok_count": int(ok_count),
        "liq_window_sec": int(_LIQ_WINDOW_SEC),
        "liq_min_notional_usd": float(_LIQ_MIN_NOTIONAL_USD),
        "ws_snapshot_used": bool(ws_used),
        "orderflow_ws_used": bool(orderflow_ws_used),
        "scoring": "v2_regime",
        "v2": meta_v2,
        "v1_raw": meta_raw,
        "norm_enabled": bool(INST_NORM_ENABLED),
        "norm_min_points": int(INST_NORM_MIN_POINTS),
        "norm_window": int(INST_NORM_WINDOW),
    }

    available = any(
        [
            oi_value is not None,
            funding_rate is not None,
            tape_5m is not None,
            ob_25 is not None,
            cvd_slope is not None,
            cvd_notional_5m is not None,
            oi_hist_slope is not None,
            (liq_total_usd_5m is not None and liq_total_usd_5m > 0.0),
        ]
    )

    liquidation_intensity: Optional[float] = None
    try:
        if liq_total_usd_5m is not None:
            liquidation_intensity = float(liq_total_usd_5m)
    except Exception:
        liquidation_intensity = None

    payload: Dict[str, Any] = {
        "institutional_score": int(inst_score),
        "institutional_score_raw": int(inst_score_raw),
        "binance_symbol": binance_symbol,
        "available": bool(available),

        "oi": oi_value,
        "oi_slope": oi_slope,
        "oi_slope_z": oi_slope_z,

        "cvd_slope": cvd_slope,
        "cvd_notional_5m": cvd_notional_5m,

        "funding_rate": funding_rate,
        "funding_regime": funding_regime,
        "funding_mean": funding_mean,
        "funding_std": funding_std,
        "funding_z": funding_z,

        "basis_pct": basis_pct,
        "basis_pct_z": basis_pct_z,
        "basis_regime": basis_regime,

        "tape_delta_1m": tape_1m,
        "tape_delta_5m": tape_5m,
        "tape_delta_5m_z": tape_5m_z,
        "tape_regime": _classify_tape(tape_5m),

        "orderbook_imb_10bps": ob_10,
        "orderbook_imb_25bps": ob_25,
        "orderbook_imb_25bps_z": ob_imb_z,
        "orderbook_regime": ob_regime,

        "crowding_regime": crowding_regime,
        "flow_regime": flow_regime,

        "oi_hist_slope": oi_hist_slope,

        "liq_buy_usd_5m": liq_buy_usd_5m,
        "liq_sell_usd_5m": liq_sell_usd_5m,
        "liq_total_usd_5m": liq_total_usd_5m,
        "liq_total_usd_5m_z": liq_total_z,
        "liq_delta_ratio_5m": liq_delta_ratio_5m,
        "liq_regime": liq_regime,
        "liquidation_intensity": liquidation_intensity,

        "warnings": warnings,

        "score_components": components,
        "score_components_raw": components_raw,
        "score_meta": score_meta,

        "lsr_global_last": lsr_global_last,
        "lsr_global_slope": lsr_global_slope,
        "lsr_top_last": lsr_top_last,
        "lsr_top_slope": lsr_top_slope,
        "taker_ls_last": taker_ls_last,
        "taker_ls_slope": taker_ls_slope,

        "available_components": [],
        "available_components_count": 0,
        "ban": {"hard_until_ms": int(_BINANCE_HARD_BAN_UNTIL_MS), "soft_until_ms": int(_BINANCE_SOFT_UNTIL_MS)},

        "ws_snapshot_used": bool(ws_used),
        "orderflow_ws_used": bool(orderflow_ws_used),
        "normalization_enabled": bool(INST_NORM_ENABLED),
        "data_sources": sources,

        # Legacy/compat keys
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

    st = _get_sym_state(binance_symbol)
    if st is not None:
        payload["symbol_cooldown_until_ms"] = int(st.until_ms)
        payload["symbol_errors"] = int(st.errors)

    return payload


# Alias
async def compute_institutional(
    symbol: str,
    bias: str,
    *,
    mode: Optional[str] = None,
    include_liquidations: bool = False,
) -> Dict[str, Any]:
    return await compute_full_institutional_analysis(symbol, bias, include_liquidations=include_liquidations, mode=mode)


def get_ban_state() -> Dict[str, int]:
    return {"hard_until_ms": int(_BINANCE_HARD_BAN_UNTIL_MS), "soft_until_ms": int(_BINANCE_SOFT_UNTIL_MS)}
