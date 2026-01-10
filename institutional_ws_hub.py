# =====================================================================
# institutional_ws_hub.py — Binance USD-M Futures WebSocket hub + cache
# =====================================================================
# Goal:
#   - Avoid REST spam (418 / -1003) during multi-symbol scans.
#   - Provide a real-time cache that analyze/institutional_data can read
#     without making per-symbol REST calls.
#
# WS docs (USD-M Futures):
#   - Base: wss://fstream.binance.com/ws (live subscribe)
#   - Combined payloads can be enabled via SET_PROPERTY ["combined", true]
#   - Max 1024 streams per connection (combined/subscribe limits)
#
# Dependency:
#   pip install websockets
#
# ✅ UPDATED for your autonomous institutional watcher:
#   - start(..., **kwargs) accepts product_type safely (scanner compatibility)
#   - snapshot exposes watcher-friendly aliases:
#       price / last / markPrice / mark_price
#       ts / timestamp / ts_ms
#   - adds age_s (freshness) + ws_ok (freshness gate)
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

# rolling windows (seconds)
_TAPE_WIN_1M = 60.0
_TAPE_WIN_5M = 300.0
_TAPE_WIN_15M = 900.0
_LIQ_WIN_5M = 300.0

# snapshot freshness: watcher can trust WS price if age <= this
_WS_FRESH_MAX_AGE_S = float(4.0)


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
    delta_notional: float   # signed notional (buy - sell)
    notional: float         # absolute notional


@dataclass
class _LiqRec:
    ts: float
    notional: float
    side: str  # BUY/SELL as received in Binance forceOrder payload ("S")


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
    tape: Deque[_TapeRec] = field(default_factory=lambda: deque(maxlen=12000))
    liquidations: Deque[_LiqRec] = field(default_factory=lambda: deque(maxlen=4000))

    last_update_ts: float = 0.0


class InstitutionalWSHub:
    """
    One WS connection, live SUBSCRIBE/UNSUBSCRIBE, per-symbol cache.

    Public API:
      - await hub.start(symbols, **kwargs)
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
        return [
            f"{s}@aggTrade",
            f"{s}@bookTicker",
            f"{s}@markPrice@{self._mark_speed}",
            f"{s}@forceOrder",
        ]

    def _streams_for_symbols(self, symbols: Set[str]) -> Set[str]:
        streams: Set[str] = set()
        for sym in symbols:
            streams.update(self._streams_for_symbol(sym))
        return streams

    # -----------------------------
    # control
    # -----------------------------
    async def start(self, symbols: Optional[List[str]] = None, **kwargs: Any) -> None:
        """
        kwargs accepted for compatibility (ex: product_type=... in scanner),
        ignored safely.
        """
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
            for sym in desired:
                self._state.setdefault(sym, _SymbolState())
        return

    # -----------------------------
    # read API (SYNC)
    # -----------------------------
    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        sym = _norm_symbol(symbol)
        st = self._state.get(sym)
        if not st:
            return {"available": False, "symbol": sym, "reason": "no_state"}

        # tape / cvd
        t = self._compute_tape(st)
        # liquidations
        liq = self._compute_liq_5m_breakdown(st)
        # orderbook
        ob_imb = self._compute_book_imbalance(st)
        micro = self._compute_microprice(st)
        spread_bps = self._compute_spread_bps(st)

        # watcher-friendly price (mid if possible, else mark)
        px_mid = (st.best_bid + st.best_ask) / 2.0 if (st.best_bid > 0 and st.best_ask > 0) else 0.0
        price = float(px_mid or st.mark_price or 0.0)

        ts = float(st.last_update_ts or 0.0)
        age_s = float(_now() - ts) if ts > 0 else 1e9
        ws_ok = bool(age_s <= _WS_FRESH_MAX_AGE_S)

        return {
            "available": True,
            "symbol": sym,

            # timestamps / freshness
            "ts": ts,                      # seconds epoch
            "timestamp": ts,               # alias
            "ts_ms": int(ts * 1000) if ts > 0 else 0,
            "age_s": age_s,
            "ws_ok": ws_ok,

            # prices
            "mark_price": st.mark_price,
            "markPrice": st.mark_price,    # alias for watcher patch
            "index_price": st.index_price,
            "funding_rate": st.funding_rate,
            "next_funding_time": st.next_funding_time,
            "best_bid": st.best_bid,
            "best_ask": st.best_ask,

            # helpful extras
            "microprice": micro,
            "spread_bps": spread_bps,

            # top-of-book qty
            "bid_qty": st.bid_qty,
            "ask_qty": st.ask_qty,
            "orderbook_imbalance": ob_imb,

            # tape ratios (buy-sell / total)
            "tape_delta_1m": t["ratio_1m"],
            "tape_delta_5m": t["ratio_5m"],
            "tape_delta_15m": t["ratio_15m"],

            # tape signed notionals (what institutional_data expects)
            "tape_delta_1m_notional": t["delta_1m"],
            "tape_delta_5m_notional": t["delta_5m"],
            "tape_delta_15m_notional": t["delta_15m"],

            # tape total notionals (useful for gating)
            "tape_abs_1m": t["abs_1m"],
            "tape_abs_5m": t["abs_5m"],
            "tape_abs_15m": t["abs_15m"],

            # Backward-compatible field
            "liq_notional_5m": liq["total"],

            # NEW: liquidation breakdown
            "liq_buy_usd_5m": liq["buy_abs"],
            "liq_sell_usd_5m": liq["sell_abs"],
            "liq_delta_ratio_5m": liq["delta_ratio"],
            "liq_count_5m": liq["count"],

            # aliases for watcher usage
            "cvd_notional_5m": t["delta_5m"],
            "cvd_notional_15m": t["delta_15m"],
            "cvd_5m": t["delta_5m"],
            "cvd_15m": t["delta_15m"],

            # watcher patch expects these sometimes
            "price": price,
            "last": price,                 # alias
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

    def _compute_tape(self, st: _SymbolState) -> Dict[str, float]:
        now = _now()
        # keep some margin so prune doesn't lag due to timings
        self._prune_left(st.tape, now - (_TAPE_WIN_15M + 5.0))

        d1 = n1 = 0.0
        d5 = n5 = 0.0
        d15 = n15 = 0.0

        for rec in st.tape:
            if rec.ts >= now - _TAPE_WIN_15M:
                d15 += rec.delta_notional
                n15 += rec.notional
            if rec.ts >= now - _TAPE_WIN_5M:
                d5 += rec.delta_notional
                n5 += rec.notional
            if rec.ts >= now - _TAPE_WIN_1M:
                d1 += rec.delta_notional
                n1 += rec.notional

        r1 = (d1 / n1) if n1 > 1e-9 else 0.0
        r5 = (d5 / n5) if n5 > 1e-9 else 0.0
        r15 = (d15 / n15) if n15 > 1e-9 else 0.0

        return {
            "delta_1m": float(d1),
            "delta_5m": float(d5),
            "delta_15m": float(d15),
            "abs_1m": float(n1),
            "abs_5m": float(n5),
            "abs_15m": float(n15),
            "ratio_1m": float(r1),
            "ratio_5m": float(r5),
            "ratio_15m": float(r15),
        }

    def _compute_liq_5m_breakdown(self, st: _SymbolState) -> Dict[str, Any]:
        now = _now()
        self._prune_left(st.liquidations, now - (_LIQ_WIN_5M + 5.0))

        buy_abs = 0.0
        sell_abs = 0.0
        cnt = 0

        for rec in st.liquidations:
            if rec.ts < now - _LIQ_WIN_5M:
                continue
            n = abs(float(rec.notional))
            if n <= 0:
                continue
            cnt += 1
            s = str(rec.side or "").upper()
            if s == "BUY":
                buy_abs += n
            elif s == "SELL":
                sell_abs += n

        total = float(buy_abs + sell_abs)
        delta_ratio = float((buy_abs - sell_abs) / total) if total > 1e-12 else 0.0

        return {
            "total": float(total),
            "buy_abs": float(buy_abs),
            "sell_abs": float(sell_abs),
            "delta_ratio": float(delta_ratio),
            "count": int(cnt),
        }

    def _compute_book_imbalance(self, st: _SymbolState) -> float:
        denom = (abs(st.bid_qty) + abs(st.ask_qty))
        if denom <= 1e-12:
            return 0.0
        return float((st.bid_qty - st.ask_qty) / denom)

    def _compute_microprice(self, st: _SymbolState) -> float:
        try:
            b = float(st.best_bid or 0.0)
            a = float(st.best_ask or 0.0)
            B = float(st.bid_qty or 0.0)
            A = float(st.ask_qty or 0.0)
            if b <= 0 or a <= 0:
                return 0.0
            den = (A + B)
            if den <= 1e-12:
                return (a + b) / 2.0
            # classic microprice
            return float((b * A + a * B) / den)
        except Exception:
            return 0.0

    def _compute_spread_bps(self, st: _SymbolState) -> float:
        try:
            b = float(st.best_bid or 0.0)
            a = float(st.best_ask or 0.0)
            if b <= 0 or a <= 0:
                return 0.0
            mid = (a + b) / 2.0
            if mid <= 0:
                return 0.0
            return float((a - b) / mid * 10000.0)
        except Exception:
            return 0.0

    # -----------------------------
    # WS loop + handlers
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

        # safety: cap
        if len(desired_streams) > _MAX_STREAMS_PER_CONN:
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

    async def _ensure_state(self, sym: str) -> _SymbolState:
        sym = _norm_symbol(sym)
        async with self._lock:
            return self._state.setdefault(sym, _SymbolState())

    async def _handle_event_mark(self, sym: str, data: Dict[str, Any]) -> None:
        st = await self._ensure_state(sym)
        st.mark_price = _safe_float(data.get("p"), st.mark_price)
        st.index_price = _safe_float(data.get("i"), st.index_price)
        st.funding_rate = _safe_float(data.get("r"), st.funding_rate)
        nft = data.get("T")
        st.next_funding_time = int(nft) if nft is not None else st.next_funding_time
        st.last_update_ts = _now()

    async def _handle_event_book(self, sym: str, data: Dict[str, Any]) -> None:
        st = await self._ensure_state(sym)
        st.best_bid = _safe_float(data.get("b"), st.best_bid)
        st.best_ask = _safe_float(data.get("a"), st.best_ask)
        st.bid_qty = _safe_float(data.get("B"), st.bid_qty)
        st.ask_qty = _safe_float(data.get("A"), st.ask_qty)
        st.last_update_ts = _now()

        # optional: if mark is missing, approximate from mid
        if st.mark_price <= 0 and st.best_bid > 0 and st.best_ask > 0:
            st.mark_price = (st.best_bid + st.best_ask) / 2.0

    async def _handle_event_aggtrade(self, sym: str, data: Dict[str, Any]) -> None:
        st = await self._ensure_state(sym)
        price = _safe_float(data.get("p"), 0.0)
        qty = _safe_float(data.get("q"), 0.0)
        notional = abs(price * qty)

        # Binance aggTrade: m=True means "buyer is the market maker" => net sell pressure
        m = bool(data.get("m", False))
        delta = (-notional) if m else (+notional)

        if notional > 0:
            st.tape.append(_TapeRec(ts=_now(), delta_notional=delta, notional=notional))
            st.last_update_ts = _now()

    async def _handle_event_forceorder(self, sym: str, data: Dict[str, Any]) -> None:
        st = await self._ensure_state(sym)
        o = data.get("o") if isinstance(data.get("o"), dict) else {}
        side = str(o.get("S") or "").upper()
        if side not in ("BUY", "SELL"):
            side = "UNKNOWN"
        price = _safe_float(o.get("p"), 0.0)
        qty = _safe_float(o.get("q"), 0.0)
        notional = abs(price * qty)
        if notional > 0:
            st.liquidations.append(_LiqRec(ts=_now(), notional=notional, side=side))
            st.last_update_ts = _now()

    async def _handle_combined(self, stream: str, data: Dict[str, Any]) -> None:
        # combined format: {"stream":"btcusdt@aggTrade","data":{...}}
        sym = _stream_symbol(stream) or _norm_symbol(str(data.get("s") or ""))
        if not sym:
            return

        sl = (stream or "").lower()
        et = str(data.get("e") or "").lower()

        if ("@markprice" in sl) or (et == "markpriceupdate"):
            await self._handle_event_mark(sym, data)
            return
        if ("@bookticker" in sl) or (et == "bookticker"):
            await self._handle_event_book(sym, data)
            return
        if ("@aggtrade" in sl) or (et == "aggtrade"):
            await self._handle_event_aggtrade(sym, data)
            return
        if ("@forceorder" in sl) or (et == "forceorder"):
            await self._handle_event_forceorder(sym, data)
            return

    async def _handle_raw(self, payload: Dict[str, Any]) -> None:
        # raw format: {"e":"aggTrade","s":"BTCUSDT", ...}
        et = str(payload.get("e") or "").lower()
        sym = _norm_symbol(str(payload.get("s") or payload.get("symbol") or ""))
        if not et or not sym:
            return

        if et == "markpriceupdate":
            await self._handle_event_mark(sym, payload)
            return
        if et == "bookticker":
            await self._handle_event_book(sym, payload)
            return
        if et == "aggtrade":
            await self._handle_event_aggtrade(sym, payload)
            return
        if et == "forceorder":
            await self._handle_event_forceorder(sym, payload)
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
                    max_queue=2048,
                ) as ws:
                    LOGGER.info("[WS_HUB] connected %s", BINANCE_FUTURES_WS)
                    backoff = 1.0
                    self._subscribed_streams = set()

                    # Enable combined payloads: {"stream":"...","data":{...}}
                    await self._send(ws, {"method": "SET_PROPERTY", "params": ["combined", True], "id": self._next_id()})

                    # initial subscribe
                    await self._sync_subscriptions(ws)

                    last_sync = _now()

                    while not self._stop_evt.is_set():
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

                        # acks / property responses
                        if isinstance(payload, dict) and "result" in payload and "id" in payload:
                            continue

                        # combined format
                        if (
                            isinstance(payload, dict)
                            and isinstance(payload.get("stream"), str)
                            and isinstance(payload.get("data"), dict)
                        ):
                            await self._handle_combined(payload["stream"], payload["data"])
                            continue

                        # raw events
                        if isinstance(payload, dict) and isinstance(payload.get("e"), str):
                            await self._handle_raw(payload)
                            continue

            except Exception as e:
                LOGGER.warning("[WS_HUB] disconnected: %s", e)

            await asyncio.sleep(min(20.0, backoff))
            backoff = min(20.0, backoff * 1.8)

        LOGGER.info("[WS_HUB] stopped")


# Singleton (simple import + use)
HUB = InstitutionalWSHub()
