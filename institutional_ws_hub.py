# =====================================================================
# institutional_ws_hub.py — Binance USD-M Futures WebSocket hub (sharded) + cache
# =====================================================================
# Desk upgrades (max impact):
# 1) Anti-1008 "Payload too long": chunk SUBSCRIBE payloads + pacing
# 2) Stream cap safe: enforce 1024-stream limit PER CONNECTION
# 3) SHARDING: distribute symbols across N websocket connections (INST_WS_HUB_SHARDS)
# 4) Dirty-flag sync + periodic refresh (prevents resubscribe spam)
# 5) Warning anti-spam for caps/disconnects
#
# Public API (kept compatible with your scanner/institutional_data):
# - hub = InstitutionalWSHub()
# - await hub.start(symbols)
# - await hub.set_symbols(symbols)
# - await hub.stop()
# - hub.get_snapshot(symbol) -> dict
# - hub.is_running
#
# Note: if `websockets` package is missing, this module degrades gracefully
#       (hub stays disabled, no crashes).
# =====================================================================

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

LOGGER = logging.getLogger(__name__)

try:
    import websockets  # type: ignore
except Exception as e:  # pragma: no cover
    websockets = None  # type: ignore
    LOGGER.error("institutional_ws_hub requires `websockets` package: %s", e)

BINANCE_FUTURES_WS = "wss://fstream.binance.com/ws"

# -----------------------------
# Env tunables
# -----------------------------
_DEFAULT_MARK_SPEED = str(os.getenv("INST_WS_MARK_SPEED", "1s")).strip()  # 1s / 3s
_ENABLE_COMBINED = str(os.getenv("INST_WS_ENABLE_COMBINED", "1")).strip() == "1"

# Binance commonly supports up to ~1024 streams per connection.
_MAX_STREAMS_PER_CONN = int(os.getenv("INST_WS_MAX_STREAMS_PER_CONN", "1024"))

# TOTAL symbol cap across all shards.
# Each symbol consumes 4 streams by default (aggTrade, bookTicker, markPrice, forceOrder).
_MAX_SYMBOLS_TOTAL = int(os.getenv("INST_WS_HUB_MAX_SYMBOLS", "220") or "220")

# Number of websocket connections to shard across.
_SHARDS = int(os.getenv("INST_WS_HUB_SHARDS", "2") or "2")
_SHARDS = max(1, min(_SHARDS, 6))

# Payload-too-long mitigation (keep SUBSCRIBE JSON small)
_SUBSCRIBE_BATCH_SIZE = int(os.getenv("INST_WS_SUBSCRIBE_BATCH_SIZE", "90") or "90")

# Control msg pacing (safe default <5 msg/s)
_CTRL_MSG_DELAY_S = float(os.getenv("INST_WS_CTRL_MSG_DELAY_S", "0.25") or "0.25")

# Freshness gate for watcher
_WS_FRESH_MAX_AGE_S = float(os.getenv("INST_WS_FRESH_MAX_AGE_S", "4.0") or "4.0")

# Periodic resync safety (seconds) even if no change (handles silent WS drift)
_FORCE_REFRESH_S = float(os.getenv("INST_WS_FORCE_REFRESH_S", "60.0") or "60.0")

# rolling windows
_TAPE_WIN_1M = 60.0
_TAPE_WIN_5M = 300.0
_TAPE_WIN_15M = 900.0
_LIQ_WIN_5M = 300.0

# reconnect backoff
_RECONNECT_MIN_S = float(os.getenv("INST_WS_RECONNECT_MIN_S", "1.0") or "1.0")
_RECONNECT_MAX_S = float(os.getenv("INST_WS_RECONNECT_MAX_S", "20.0") or "20.0")

# streams per symbol (must match _streams_for_symbol)
_STREAMS_PER_SYMBOL = 4

# per-connection max symbols implied by stream limit
_MAX_SYMBOLS_PER_CONN_BY_STREAMS = max(1, int(_MAX_STREAMS_PER_CONN // max(1, _STREAMS_PER_SYMBOL)))

# If user sets a too-large total cap, we still keep per-connection safe.
# Total max = shards * per-conn max.
_MAX_SYMBOLS_HARD_TOTAL = int(_SHARDS * _MAX_SYMBOLS_PER_CONN_BY_STREAMS)


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
    try:
        base = stream.split("@", 1)[0]
        return base.upper()
    except Exception:
        return ""


def _stable_hash32(s: str) -> int:
    import zlib
    return int(zlib.crc32(str(s).encode("utf-8")) & 0xFFFFFFFF)


@dataclass
class _TapeRec:
    ts: float
    delta_notional: float
    notional: float


@dataclass
class _LiqRec:
    ts: float
    notional: float
    side: str


@dataclass
class _SymbolState:
    mark_price: float = 0.0
    index_price: float = 0.0
    funding_rate: float = 0.0
    next_funding_time: Optional[int] = None

    best_bid: float = 0.0
    best_ask: float = 0.0
    bid_qty: float = 0.0
    ask_qty: float = 0.0

    tape: Deque[_TapeRec] = field(default_factory=lambda: deque(maxlen=12000))
    liquidations: Deque[_LiqRec] = field(default_factory=lambda: deque(maxlen=4000))

    last_update_ts: float = 0.0


class _SingleInstitutionalWSHub:
    """
    Single Binance WS connection + local rolling cache.
    """
    def __init__(self, *, name: str = "hub0", mark_speed: str = _DEFAULT_MARK_SPEED, max_symbols: int = 220) -> None:
        self._name = str(name)
        self._mark_speed = (mark_speed or _DEFAULT_MARK_SPEED).strip()

        self._max_symbols = int(max_symbols)

        self._desired_symbols: List[str] = []
        self._desired_set: Set[str] = set()

        self._subscribed_streams: Set[str] = set()
        self._state: Dict[str, _SymbolState] = {}

        self._lock = asyncio.Lock()
        self._stop_evt = asyncio.Event()
        self._ws_task: Optional[asyncio.Task] = None
        self._ws = None

        self._id_counter = 1
        self._subscribe_batch_size = max(10, min(int(_SUBSCRIBE_BATCH_SIZE), 300))

        # dirty + versioning to avoid resync spam
        self._dirty = True
        self._desired_version = 0
        self._last_synced_version = -1
        self._last_force_refresh = 0.0

        # cap warning anti-spam
        self._last_cap_warn_key: Optional[Tuple[int, int]] = None  # (desired_symbols, effective_symbols)

        # disconnect spam limiter
        self._last_disconnect_reason: Optional[str] = None
        self._last_disconnect_ts: float = 0.0

    @property
    def is_running(self) -> bool:
        return self._ws_task is not None and not self._ws_task.done()

    # -----------------------------
    # streams
    # -----------------------------
    def _streams_for_symbol(self, sym: str) -> List[str]:
        s = sym.lower()
        return [
            f"{s}@aggTrade",
            f"{s}@bookTicker",
            f"{s}@markPrice@{self._mark_speed}",
            f"{s}@forceOrder",
        ]

    def _streams_for_symbols(self, symbols: List[str]) -> Set[str]:
        out: Set[str] = set()
        for sym in symbols:
            out.update(self._streams_for_symbol(sym))
        return out

    # -----------------------------
    # public API
    # -----------------------------
    async def start(self, symbols: Optional[List[str]] = None, **_kwargs: Any) -> None:
        if websockets is None:
            return

        if symbols:
            await self.set_symbols(symbols)

        if self.is_running:
            return

        self._stop_evt.clear()
        self._ws_task = asyncio.create_task(self._run_loop(), name=f"institutional_ws_{self._name}")
        LOGGER.info("[WS_HUB:%s] started", self._name)

    async def stop(self) -> None:
        self._stop_evt.set()
        if self._ws_task:
            try:
                await asyncio.wait_for(self._ws_task, timeout=8.0)
            except Exception:
                pass
        self._ws_task = None
        LOGGER.info("[WS_HUB:%s] stopped", self._name)

    async def set_symbols(self, symbols: List[str]) -> None:
        cleaned = [_norm_symbol(s) for s in (symbols or [])]
        cleaned = [s for s in cleaned if s]

        desired_count = len(cleaned)

        # enforce per-connection cap deterministically (keep first)
        effective = cleaned[: max(0, int(self._max_symbols))] if int(self._max_symbols) > 0 else cleaned
        effective_count = len(effective)

        cap_key = (desired_count, effective_count)
        if desired_count > effective_count and cap_key != self._last_cap_warn_key:
            self._last_cap_warn_key = cap_key
            LOGGER.warning(
                "[WS_HUB:%s] symbols capped (per-conn) max=%s: desired=%d effective=%d",
                self._name, str(self._max_symbols), desired_count, effective_count
            )

        async with self._lock:
            new_set = set(effective)
            if new_set == self._desired_set:
                return
            self._desired_symbols = effective
            self._desired_set = new_set
            self._desired_version += 1
            self._dirty = True

    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        sym = _norm_symbol(symbol)
        st = self._state.get(sym)
        if st is None:
            return {"available": False, "symbol": sym, "reason": "no_state"}

        age = _now() - float(st.last_update_ts or 0.0)
        if age > _WS_FRESH_MAX_AGE_S:
            return {"available": False, "symbol": sym, "reason": "stale", "age_s": age}

        # spread / depth
        bid = float(st.best_bid or 0.0)
        ask = float(st.best_ask or 0.0)
        spread = (ask - bid) if (bid > 0 and ask > 0) else 0.0
        mid = (ask + bid) / 2.0 if (bid > 0 and ask > 0) else 0.0
        spread_bps = (spread / mid * 10000.0) if mid > 0 and spread > 0 else 0.0

        # tape metrics (notional delta)
        tape_1m, tape_5m, tape_15m = self._tape_sums(st.tape)

        # liquidation sums
        liq_5m = self._liq_sum(st.liquidations, _LIQ_WIN_5M)

        out = {
            "available": True,
            "symbol": sym,
            "age_s": age,
            "mark_price": float(st.mark_price or 0.0),
            "index_price": float(st.index_price or 0.0),
            "funding_rate": float(st.funding_rate or 0.0),
            "next_funding_time": st.next_funding_time,

            "best_bid": bid,
            "best_ask": ask,
            "bid_qty": float(st.bid_qty or 0.0),
            "ask_qty": float(st.ask_qty or 0.0),
            "spread": spread,
            "spread_bps": spread_bps,

            "tape_1m": tape_1m,
            "tape_5m": tape_5m,
            "tape_15m": tape_15m,
            "liq_5m": liq_5m,
        }
        return out

    # -----------------------------
    # rolling windows helpers
    # -----------------------------
    def _tape_sums(self, tape: Deque[_TapeRec]) -> Tuple[float, float, float]:
        now = _now()
        s1 = s5 = s15 = 0.0
        for rec in reversed(tape):
            age = now - rec.ts
            if age <= _TAPE_WIN_1M:
                s1 += rec.delta_notional
                s5 += rec.delta_notional
                s15 += rec.delta_notional
            elif age <= _TAPE_WIN_5M:
                s5 += rec.delta_notional
                s15 += rec.delta_notional
            elif age <= _TAPE_WIN_15M:
                s15 += rec.delta_notional
            else:
                break
        return float(s1), float(s5), float(s15)

    def _liq_sum(self, liqs: Deque[_LiqRec], win_s: float) -> float:
        now = _now()
        s = 0.0
        for rec in reversed(liqs):
            if (now - rec.ts) <= win_s:
                s += rec.notional
            else:
                break
        return float(s)

    # -----------------------------
    # websocket core
    # -----------------------------
    async def _run_loop(self) -> None:
        if websockets is None:
            return

        backoff = float(_RECONNECT_MIN_S)

        while not self._stop_evt.is_set():
            try:
                await self._connect_and_run()
                backoff = float(_RECONNECT_MIN_S)
            except asyncio.CancelledError:
                break
            except Exception as e:
                # throttle repeated disconnect logs
                msg = str(e)
                now = _now()
                if msg != self._last_disconnect_reason or (now - self._last_disconnect_ts) > 10.0:
                    self._last_disconnect_reason = msg
                    self._last_disconnect_ts = now
                    LOGGER.warning("[WS_HUB:%s] disconnected: %s", self._name, e)

                await asyncio.sleep(backoff)
                backoff = min(float(_RECONNECT_MAX_S), max(float(_RECONNECT_MIN_S), backoff * 1.7))

        # ensure close
        try:
            if self._ws:
                await self._ws.close()
        except Exception:
            pass
        self._ws = None

    async def _connect_and_run(self) -> None:
        assert websockets is not None

        async with websockets.connect(
            BINANCE_FUTURES_WS,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=8,
            max_size=None,  # allow big frames from Binance
        ) as ws:
            self._ws = ws
            LOGGER.info("[WS_HUB:%s] connected %s", self._name, BINANCE_FUTURES_WS)

            # initial sync
            await self._sync_subscriptions(force=True)

            # main loop
            while not self._stop_evt.is_set():
                # periodic resync safety
                if (_now() - self._last_force_refresh) >= float(_FORCE_REFRESH_S):
                    await self._sync_subscriptions(force=True)

                # sync only when dirty
                await self._sync_subscriptions(force=False)

                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if not msg:
                    continue

                try:
                    js = json.loads(msg)
                except Exception:
                    continue

                await self._handle_message(js)

    async def _sync_subscriptions(self, *, force: bool) -> None:
        # do not resync too often
        async with self._lock:
            desired_version = self._desired_version
            dirty = self._dirty

        if not force and (not dirty or desired_version == self._last_synced_version):
            return

        now = _now()
        self._last_force_refresh = now

        async with self._lock:
            desired = list(self._desired_symbols)
            self._dirty = False

        desired_streams = self._streams_for_symbols(desired)

        # hard cap streams to avoid connection limits
        if len(desired_streams) > int(_MAX_STREAMS_PER_CONN):
            # keep deterministic order: symbols list order
            max_symbols = max(1, int(_MAX_STREAMS_PER_CONN // _STREAMS_PER_SYMBOL))
            desired = desired[:max_symbols]
            desired_streams = self._streams_for_symbols(desired)
            LOGGER.warning("[WS_HUB:%s] desired streams capped to %d", self._name, int(_MAX_STREAMS_PER_CONN))

        to_sub = sorted(list(desired_streams - self._subscribed_streams))
        to_unsub = sorted(list(self._subscribed_streams - desired_streams))

        if to_unsub:
            await self._send_unsubscribe(to_unsub)
            self._subscribed_streams.difference_update(to_unsub)

        if to_sub:
            await self._send_subscribe(to_sub)
            self._subscribed_streams.update(to_sub)

        self._last_synced_version = desired_version

    async def _send_subscribe(self, streams: List[str]) -> None:
        if not streams or not self._ws:
            return
        # chunk requests
        for chunk in _chunks(streams, self._subscribe_batch_size):
            await self._send_json({"method": "SUBSCRIBE", "params": chunk, "id": self._next_id()})
            await asyncio.sleep(float(_CTRL_MSG_DELAY_S))

    async def _send_unsubscribe(self, streams: List[str]) -> None:
        if not streams or not self._ws:
            return
        for chunk in _chunks(streams, self._subscribe_batch_size):
            await self._send_json({"method": "UNSUBSCRIBE", "params": chunk, "id": self._next_id()})
            await asyncio.sleep(float(_CTRL_MSG_DELAY_S))

    async def _send_json(self, js: Dict[str, Any]) -> None:
        try:
            await self._ws.send(json.dumps(js, separators=(",", ":")))
        except Exception:
            pass

    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    async def _handle_message(self, js: Dict[str, Any]) -> None:
        stream = js.get("stream")
        data = js.get("data")

        # combined stream format {stream, data}
        if stream and data:
            await self._handle_stream(stream, data)
            return

        # single stream: data itself includes event type (e)
        if isinstance(js, dict) and "e" in js:
            ev = js.get("e")
            sym = str(js.get("s") or "").upper()
            if not sym:
                return
            await self._handle_event(ev, sym, js)

    async def _handle_stream(self, stream: str, data: Dict[str, Any]) -> None:
        sym = _stream_symbol(stream)
        if not sym:
            return
        ev = data.get("e")
        if not ev:
            # markPrice combined doesn't always have "e" in data; infer from stream
            if "@markPrice" in stream:
                ev = "markPriceUpdate"
            elif "@bookTicker" in stream:
                ev = "bookTicker"
            elif "@aggTrade" in stream:
                ev = "aggTrade"
            elif "@forceOrder" in stream:
                ev = "forceOrder"
        await self._handle_event(ev, sym, data)

    async def _handle_event(self, ev: str, sym: str, data: Dict[str, Any]) -> None:
        if not sym:
            return
        st = self._state.get(sym)
        if st is None:
            st = _SymbolState()
            self._state[sym] = st

        now = _now()
        st.last_update_ts = now

        try:
            if ev in ("markPriceUpdate", "markPrice"):
                st.mark_price = _safe_float(data.get("p"))
                st.index_price = _safe_float(data.get("i"))
                st.funding_rate = _safe_float(data.get("r"))
                nft = data.get("T")
                st.next_funding_time = int(nft) if nft is not None else st.next_funding_time

            elif ev == "bookTicker":
                st.best_bid = _safe_float(data.get("b"))
                st.best_ask = _safe_float(data.get("a"))
                st.bid_qty = _safe_float(data.get("B"))
                st.ask_qty = _safe_float(data.get("A"))

            elif ev == "aggTrade":
                # delta notional: positive = aggressive buys, negative = aggressive sells
                price = _safe_float(data.get("p"))
                qty = _safe_float(data.get("q"))
                is_buyer_maker = bool(data.get("m"))  # True => seller initiated
                notional = abs(price * qty)
                delta = -notional if is_buyer_maker else notional
                st.tape.append(_TapeRec(ts=now, delta_notional=delta, notional=notional))

            elif ev == "forceOrder":
                o = data.get("o") or data
                price = _safe_float(o.get("p"))
                qty = _safe_float(o.get("q"))
                side = str(o.get("S") or "").upper() or "UNKNOWN"
                notional = abs(price * qty)
                st.liquidations.append(_LiqRec(ts=now, notional=notional, side=side))
        except Exception:
            return


def _chunks(xs: List[str], n: int) -> List[List[str]]:
    if n <= 0:
        return [xs]
    out: List[List[str]] = []
    cur: List[str] = []
    for x in xs:
        cur.append(x)
        if len(cur) >= n:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


class InstitutionalWSHub:
    """
    Sharded hub wrapper.
    Creates N independent websocket connections and distributes symbols deterministically.
    """
    def __init__(self, *, mark_speed: str = _DEFAULT_MARK_SPEED, shards: int = _SHARDS) -> None:
        self._mark_speed = (mark_speed or _DEFAULT_MARK_SPEED).strip()
        self._shards_n = max(1, min(int(shards), 6))
        self._lock = asyncio.Lock()

        # If websockets missing, keep disabled
        if websockets is None:
            self._shards_n = 0
            self._shards: List[_SingleInstitutionalWSHub] = []
        else:
            # effective total cap
            self._total_cap = int(_MAX_SYMBOLS_TOTAL) if int(_MAX_SYMBOLS_TOTAL) > 0 else 0
            # hard cap by stream limits
            if self._total_cap <= 0:
                self._total_cap = int(_MAX_SYMBOLS_HARD_TOTAL)
            self._total_cap = min(int(self._total_cap), int(_MAX_SYMBOLS_HARD_TOTAL))

            # per-shard cap: at most per-conn stream cap
            per_conn_cap = int(_MAX_SYMBOLS_PER_CONN_BY_STREAMS)
            self._shards = [
                _SingleInstitutionalWSHub(
                    name=f"hub{i}",
                    mark_speed=self._mark_speed,
                    max_symbols=per_conn_cap,
                )
                for i in range(self._shards_n)
            ]

        self._sym_to_shard: Dict[str, int] = {}
        self._last_cap_warn_key: Optional[Tuple[int, int, int]] = None  # desired, effective, shards

    @property
    def is_running(self) -> bool:
        if not self._shards:
            return False
        return any(h.is_running for h in self._shards)

    async def start(self, symbols: Optional[List[str]] = None, **kwargs: Any) -> None:
        if websockets is None or not self._shards:
            return
        if symbols:
            await self.set_symbols(symbols)
        # start all
        await asyncio.gather(*[h.start(None) for h in self._shards])
        LOGGER.info("[WS_HUB] sharded started shards=%d total_cap=%d per_conn_cap=%d", self._shards_n, getattr(self, "_total_cap", 0), _MAX_SYMBOLS_PER_CONN_BY_STREAMS)

    async def stop(self) -> None:
        if websockets is None or not self._shards:
            return
        await asyncio.gather(*[h.stop() for h in self._shards])
        LOGGER.info("[WS_HUB] sharded stopped")

    async def set_symbols(self, symbols: List[str]) -> None:
        if websockets is None or not self._shards:
            return

        cleaned = [_norm_symbol(s) for s in (symbols or [])]
        cleaned = [s for s in cleaned if s]
        desired_count = len(cleaned)

        # total cap across shards (deterministic: keep first)
        effective = cleaned[: max(0, int(self._total_cap))] if int(getattr(self, "_total_cap", 0)) > 0 else cleaned
        effective_count = len(effective)

        cap_key = (desired_count, effective_count, self._shards_n)
        if desired_count > effective_count and cap_key != self._last_cap_warn_key:
            self._last_cap_warn_key = cap_key
            LOGGER.warning(
                "[WS_HUB] symbols capped total=%d (hard=%d): desired=%d effective=%d shards=%d",
                int(self._total_cap),
                int(_MAX_SYMBOLS_HARD_TOTAL),
                desired_count,
                effective_count,
                self._shards_n,
            )

        # distribute deterministically by hash modulo shards
        buckets: List[List[str]] = [[] for _ in range(self._shards_n)]
        sym_to_shard: Dict[str, int] = {}

        for sym in effective:
            idx = int(_stable_hash32(sym) % max(1, self._shards_n))
            buckets[idx].append(sym)
            sym_to_shard[sym] = idx

        async with self._lock:
            self._sym_to_shard = sym_to_shard

        await asyncio.gather(*[self._shards[i].set_symbols(buckets[i]) for i in range(self._shards_n)])

    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        sym = _norm_symbol(symbol)
        if not self._shards:
            return {"available": False, "symbol": sym, "reason": "disabled"}

        idx = None
        try:
            idx = self._sym_to_shard.get(sym)
        except Exception:
            idx = None
        if idx is None:
            idx = int(_stable_hash32(sym) % max(1, self._shards_n))

        try:
            return self._shards[int(idx)].get_snapshot(sym)
        except Exception:
            return {"available": False, "symbol": sym, "reason": "shard_error"}

# Backward compatible global instance
HUB = InstitutionalWSHub()
