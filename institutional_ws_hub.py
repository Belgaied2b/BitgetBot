# =====================================================================
# institutional_ws_hub.py — Binance USD-M Futures WebSocket hub + cache
# =====================================================================
# Goal:
#   - Avoid REST spam (418 / -1003) during multi-symbol scans.
#   - Provide a real-time cache that analyze/institutional_data can read
#     without making per-symbol REST calls.
#
# WS docs (USD-M Futures):
#   - Base: wss://fstream.binance.com/ws (live subscriptions)
#   - Max 1024 streams per connection (combined/subscribe limits)
#
# Dependency:
#   pip install websockets
# =====================================================================

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

LOGGER = logging.getLogger(__name__)

BINANCE_FUTURES_WS = "wss://fstream.binance.com/ws"

# ---- internal knobs ----
_DEFAULT_MARK_SPEED = "1s"   # markPrice update speed (1s/3s depending on Binance)
_MAX_STREAMS_PER_CONN = 1024

# rolling windows
_TAPE_WIN_1M = 60.0
_TAPE_WIN_5M = 300.0
_LIQ_WIN_5M = 300.0


def _now() -> float:
    return time.time()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        return v if v == v else default
    except Exception:
        return default


def _norm_symbol(sym: str) -> str:
    return str(sym or "").upper().strip()


def _stream_symbol(stream: str) -> str:
    # examples: btcusdt@aggTrade, ethusdt@markPrice@1s, btcusdt@bookTicker, btcusdt@forceOrder
    try:
        base = stream.split("@", 1)[0]
        return base.upper()
    except Exception:
        return ""


@dataclass
class _TapeRec:
    ts: float
    delta_notional: float
    notional: float


@dataclass
class _LiqRec:
    ts: float
    notional: float
    side: str  # BUY/SELL in liquidation message meaning? we store raw


@dataclass
class _SymbolState:
    # last market data
    mark_price: float = 0.0
    index_price: float = 0.0
    funding_rate: float = 0.0
    next_funding_time: Optional[int] = None

    best_bid: float = 0.0
    best_ask: float = 0.0
    bid_qty: float = 0.0
    ask_qty: float = 0.0

    # rolling microstructure
    tape: Deque[_TapeRec] = field(default_factory=lambda: deque(maxlen=5000))
    liquidations: Deque[_LiqRec] = field(default_factory=lambda: deque(maxlen=2000))

    last_update_ts: float = 0.0


class InstitutionalWSHub:
    """
    One WS connection, live SUBSCRIBE/UNSUBSCRIBE, per-symbol cache.

    Public API:
      - await hub.start(symbols)
      - await hub.stop()
      - await hub.set_symbols(symbols)
      - hub.get_snapshot(symbol) -> dict
      - hub.is_running
    """

    def __init__(self, *, mark_speed: str = _DEFAULT_MARK_SPEED) -> None:
        self._mark_speed = (mark_speed or _DEFAULT_MARK_SPEED).strip()
        self._desired_symbols: Set[str] = set()
        self._subscribed_streams: Set[str] = set()

        self._state: Dict[str, _SymbolState] = {}
        self._lock = asyncio.Lock()

        self._ws_task: Optional[asyncio.Task] = None
        self._stop_evt = asyncio.Event()

        self._id_counter = 1

    @property
    def is_running(self) -> bool:
        return self._ws_task is not None and not self._ws_task.done()

    # -----------------------------
    # stream builder
    # -----------------------------
    def _streams_for_symbol(self, sym: str) -> List[str]:
        s = sym.lower()
        out = [
            f"{s}@aggTrade",
            f"{s}@bookTicker",
            f"{s}@markPrice@{self._mark_speed}",
            f"{s}@forceOrder",
        ]
        return out

    def _streams_for_symbols(self, symbols: Set[str]) -> Set[str]:
        streams: Set[str] = set()
        for sym in symbols:
            streams.update(self._streams_for_symbol(sym))
        return streams

    # -----------------------------
    # control
    # -----------------------------
    async def start(self, symbols: Optional[List[str]] = None) -> None:
        if symbols:
            await self.set_symbols(symbols)

        if self.is_running:
            return

        self._stop_evt.clear()
        self._ws_task = asyncio.create_task(self._run_loop(), name="institutional_ws_hub")

    async def stop(self) -> None:
        self._stop_evt.set()
        if self._ws_task:
            try:
                await asyncio.wait_for(self._ws_task, timeout=8.0)
            except Exception:
                pass
        self._ws_task = None

    async def set_symbols(self, symbols: List[str]) -> None:
        desired = {_norm_symbol(s) for s in (symbols or []) if _norm_symbol(s)}
        async with self._lock:
            self._desired_symbols = desired
            # ensure state entries exist
            for sym in desired:
                self._state.setdefault(sym, _SymbolState())

        # if running: resubscribe dynamically
        # (we don't block; the ws loop reads desired state and updates subs)
        return

    # -----------------------------
    # read API
    # -----------------------------
    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        sym = _norm_symbol(symbol)
        st = self._state.get(sym)
        if not st:
            return {"available": False, "symbol": sym, "reason": "no_state"}

        tape_1m, tape_5m, tape_1m_abs, tape_5m_abs = self._compute_tape(st)
        liq_5m = self._compute_liq_5m(st)
        ob_imb = self._compute_book_imbalance(st)

        return {
            "available": True,
            "symbol": sym,
            "ts": st.last_update_ts,
            "mark_price": st.mark_price,
            "index_price": st.index_price,
            "funding_rate": st.funding_rate,
            "next_funding_time": st.next_funding_time,
            "best_bid": st.best_bid,
            "best_ask": st.best_ask,
            "bid_qty": st.bid_qty,
            "ask_qty": st.ask_qty,
            "orderbook_imbalance": ob_imb,
            "tape_delta_1m": tape_1m,
            "tape_delta_5m": tape_5m,
            "tape_abs_1m": tape_1m_abs,
            "tape_abs_5m": tape_5m_abs,
            "liq_notional_5m": liq_5m,
        }

    # -----------------------------
    # internals: rolling metrics
    # -----------------------------
    def _prune_left(self, dq: Deque, cutoff_ts: float) -> None:
        try:
            while dq and getattr(dq[0], "ts", 0.0) < cutoff_ts:
                dq.popleft()
        except Exception:
            return

    def _compute_tape(self, st: _SymbolState) -> Tuple[float, float, float, float]:
        now = _now()
        # prune to 5m
        self._prune_left(st.tape, now - (_TAPE_WIN_5M + 2.0))

        d1 = 0.0
        n1 = 0.0
        d5 = 0.0
        n5 = 0.0
        for rec in st.tape:
            if rec.ts >= now - _TAPE_WIN_5M:
                d5 += rec.delta_notional
                n5 += rec.notional
            if rec.ts >= now - _TAPE_WIN_1M:
                d1 += rec.delta_notional
                n1 += rec.notional

        # normalize to [-1..+1] if possible
        tape_1m = (d1 / n1) if n1 > 1e-9 else 0.0
        tape_5m = (d5 / n5) if n5 > 1e-9 else 0.0
        return float(tape_1m), float(tape_5m), float(n1), float(n5)

    def _compute_liq_5m(self, st: _SymbolState) -> float:
        now = _now()
        self._prune_left(st.liquidations, now - (_LIQ_WIN_5M + 2.0))
        total = 0.0
        for rec in st.liquidations:
            if rec.ts >= now - _LIQ_WIN_5M:
                total += abs(rec.notional)
        return float(total)

    def _compute_book_imbalance(self, st: _SymbolState) -> float:
        denom = (abs(st.bid_qty) + abs(st.ask_qty))
        if denom <= 1e-12:
            return 0.0
        return float((st.bid_qty - st.ask_qty) / denom)

    # -----------------------------
    # WS loop
    # -----------------------------
    async def _send(self, ws, payload: Dict[str, Any]) -> None:
        try:
            await ws.send(json.dumps(payload))
        except Exception:
            pass

    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    async def _sync_subscriptions(self, ws) -> None:
        async with self._lock:
            desired = set(self._desired_symbols)

        desired_streams = self._streams_for_symbols(desired)

        # safety: cap (rough guard)
        if len(desired_streams) > _MAX_STREAMS_PER_CONN:
            # keep deterministic order
            desired_list = sorted(list(desired))
            keep: Set[str] = set()
            for sym in desired_list:
                streams = set(self._streams_for_symbol(sym))
                if len(keep) + len(streams) > _MAX_STREAMS_PER_CONN:
                    break
                keep |= streams
            desired_streams = keep

        to_add = sorted(list(desired_streams - self._subscribed_streams))
        to_remove = sorted(list(self._subscribed_streams - desired_streams))

        if to_remove:
            await self._send(ws, {"method": "UNSUBSCRIBE", "params": to_remove, "id": self._next_id()})
            for s in to_remove:
                self._subscribed_streams.discard(s)

        if to_add:
            await self._send(ws, {"method": "SUBSCRIBE", "params": to_add, "id": self._next_id()})
            for s in to_add:
                self._subscribed_streams.add(s)

    async def _handle_data(self, stream: str, data: Dict[str, Any]) -> None:
        sym = _stream_symbol(stream)
        if not sym:
            return

        async with self._lock:
            st = self._state.setdefault(sym, _SymbolState())

        # ---- markPrice stream ----
        if "@markprice" in stream.lower():
            st.mark_price = _safe_float(data.get("p"), st.mark_price)
            st.index_price = _safe_float(data.get("i"), st.index_price)
            st.funding_rate = _safe_float(data.get("r"), st.funding_rate)
            nft = data.get("T")
            st.next_funding_time = int(nft) if nft is not None else st.next_funding_time
            st.last_update_ts = _now()
            return

        # ---- bookTicker ----
        if "@bookticker" in stream.lower():
            st.best_bid = _safe_float(data.get("b"), st.best_bid)
            st.best_ask = _safe_float(data.get("a"), st.best_ask)
            st.bid_qty = _safe_float(data.get("B"), st.bid_qty)
            st.ask_qty = _safe_float(data.get("A"), st.ask_qty)
            st.last_update_ts = _now()
            return

        # ---- aggTrade ----
        if "@aggtrade" in stream.lower():
            # Binance aggTrade:
            #   p price, q qty, m: is buyer maker?
            # If buyer is maker -> seller is taker (aggressive sell) => delta negative.
            price = _safe_float(data.get("p"), 0.0)
            qty = _safe_float(data.get("q"), 0.0)
            notional = abs(price * qty)
            m = bool(data.get("m", False))
            delta = (-notional) if m else (+notional)

            st.tape.append(_TapeRec(ts=_now(), delta_notional=delta, notional=notional))
            st.last_update_ts = _now()
            return

        # ---- forceOrder (liquidation) ----
        if "@forceorder" in stream.lower():
            o = data.get("o") if isinstance(data.get("o"), dict) else {}
            side = str(o.get("S") or "").upper()  # BUY/SELL
            price = _safe_float(o.get("p"), 0.0)
            qty = _safe_float(o.get("q"), 0.0)
            notional = abs(price * qty)
            if notional > 0:
                st.liquidations.append(_LiqRec(ts=_now(), notional=notional, side=side))
                st.last_update_ts = _now()
            return

    async def _run_loop(self) -> None:
        backoff = 1.0
        try:
            import websockets  # type: ignore
        except Exception as e:
            LOGGER.error("institutional_ws_hub requires `websockets` package: %s", e)
            return

        while not self._stop_evt.is_set():
            try:
                async with websockets.connect(
                    BINANCE_FUTURES_WS,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
                    max_queue=1024,
                ) as ws:
                    LOGGER.info("[WS_HUB] connected %s", BINANCE_FUTURES_WS)
                    backoff = 1.0
                    self._subscribed_streams = set()

                    # initial subscribe
                    await self._sync_subscriptions(ws)

                    last_sync = _now()

                    while not self._stop_evt.is_set():
                        # periodic subscription sync (in case set_symbols() changed)
                        if _now() - last_sync >= 2.0:
                            await self._sync_subscriptions(ws)
                            last_sync = _now()

                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        except asyncio.TimeoutError:
                            continue

                        if not msg:
                            continue

                        try:
                            payload = json.loads(msg)
                        except Exception:
                            continue

                        # subscription acks (result: null)
                        if isinstance(payload, dict) and "result" in payload and "id" in payload:
                            continue

                        # combined format: {"stream":"btcusdt@aggTrade","data":{...}}
                        if isinstance(payload, dict) and isinstance(payload.get("stream"), str) and isinstance(payload.get("data"), dict):
                            await self._handle_data(payload["stream"], payload["data"])
                            continue

                        # sometimes Binance can send raw event (rare when using /ws)
                        if isinstance(payload, dict) and isinstance(payload.get("e"), str):
                            # try to infer by event type
                            # we ignore for now
                            continue

            except Exception as e:
                LOGGER.warning("[WS_HUB] disconnected: %s", e)

            # backoff reconnect
            await asyncio.sleep(min(20.0, backoff))
            backoff = min(20.0, backoff * 1.8)

        LOGGER.info("[WS_HUB] stopped")


# Singleton (simple import + use)
HUB = InstitutionalWSHub()
