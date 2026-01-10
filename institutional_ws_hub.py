# =====================================================================
# institutional_ws_hub.py — Binance USD-M Futures WebSocket hub + cache
# =====================================================================
# FIXES (Desk Lead):
# 1) Avoid close 1008 "Payload too long": chunk SUBSCRIBE payloads
# 2) Respect control msg pacing: delay between SUBSCRIBE/UNSUBSCRIBE
# 3) Enforce MAX SYMBOLS (INST_WS_HUB_MAX_SYMBOLS) to avoid >1024 streams
# 4) Prevent warning spam: warn only when cap state changes
# 5) Sync subscriptions only when symbols changed (dirty flag) + periodic refresh
#
# Public API:
# - await HUB.start(symbols, **kwargs)
# - await HUB.stop()
# - await HUB.set_symbols(symbols)
# - HUB.get_snapshot(symbol) -> dict
# - HUB.is_running
# =====================================================================

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

LOGGER = logging.getLogger(__name__)

BINANCE_FUTURES_WS = "wss://fstream.binance.com/ws"

# -----------------------------
# Env tunables
# -----------------------------
_DEFAULT_MARK_SPEED = str(os.getenv("INST_WS_MARK_SPEED", "1s")).strip()  # 1s / 3s
_ENABLE_COMBINED = str(os.getenv("INST_WS_ENABLE_COMBINED", "1")).strip() == "1"

# Binance commonly supports up to ~1024 streams per connection (general limit).
_MAX_STREAMS_PER_CONN = int(os.getenv("INST_WS_MAX_STREAMS_PER_CONN", "1024"))

# IMPORTANT: cap symbols to avoid >1024 streams.
# Each symbol uses 4 streams by default => 1024/4 = 256 symbols max in theory.
_MAX_SYMBOLS = int(os.getenv("INST_WS_HUB_MAX_SYMBOLS", os.getenv("INST_WS_HUB_MAX_SYMBOLS", "220")) or "220")

# Payload-too-long mitigation (keep SUBSCRIBE JSON small)
_SUBSCRIBE_BATCH_SIZE = int(os.getenv("INST_WS_SUBSCRIBE_BATCH_SIZE", "90"))

# Control msg pacing (safe default <5 msg/s)
_CTRL_MSG_DELAY_S = float(os.getenv("INST_WS_CTRL_MSG_DELAY_S", "0.25"))

# Freshness gate for watcher
_WS_FRESH_MAX_AGE_S = float(os.getenv("INST_WS_FRESH_MAX_AGE_S", "4.0"))

# Periodic resync safety (seconds) even if no change (handles silent WS drift)
_FORCE_REFRESH_S = float(os.getenv("INST_WS_FORCE_REFRESH_S", "60.0"))

# rolling windows
_TAPE_WIN_1M = 60.0
_TAPE_WIN_5M = 300.0
_TAPE_WIN_15M = 900.0
_LIQ_WIN_5M = 300.0

# reconnect backoff
_RECONNECT_MIN_S = float(os.getenv("INST_WS_RECONNECT_MIN_S", "1.0"))
_RECONNECT_MAX_S = float(os.getenv("INST_WS_RECONNECT_MAX_S", "20.0"))


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


class InstitutionalWSHub:
    def __init__(self, *, mark_speed: str = _DEFAULT_MARK_SPEED) -> None:
        self._mark_speed = (mark_speed or _DEFAULT_MARK_SPEED).strip()

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
    async def start(self, symbols: Optional[List[str]] = None, **kwargs: Any) -> None:
        if symbols:
            await self.set_symbols(symbols)

        if self.is_running:
            return

        self._stop_evt.clear()
        self._ws_task = asyncio.create_task(self._run_loop(), name="institutional_ws_hub")
        LOGGER.info("[WS_HUB] started")

    async def stop(self) -> None:
        self._stop_evt.set()
        if self._ws_task:
            try:
                await asyncio.wait_for(self._ws_task, timeout=8.0)
            except Exception:
                pass
        self._ws_task = None
        LOGGER.info("[WS_HUB] stopped")

    async def set_symbols(self, symbols: List[str]) -> None:
        cleaned = [_norm_symbol(s) for s in (symbols or [])]
        cleaned = [s for s in cleaned if s]

        # enforce max symbols deterministically (keep first)
        desired_count = len(cleaned)
        effective = cleaned[: max(0, int(_MAX_SYMBOLS))] if int(_MAX_SYMBOLS) > 0 else cleaned
        effective_count = len(effective)

        # cap warning only when it changes (anti-spam)
        cap_key = (desired_count, effective_count)
        if desired_count > effective_count and cap_key != self._last_cap_warn_key:
            self._last_cap_warn_key = cap_key
            LOGGER.warning(
                "[WS_HUB] symbols capped by INST_WS_HUB_MAX_SYMBOLS=%s: desired=%d effective=%d",
                str(_MAX_SYMBOLS), desired_count, effective_count
            )

        async with self._lock:
            new_set = set(effective)
            if new_set == self._desired_set:
                return

            self._desired_symbols = effective
            self._desired_set = new_set

            for sym in new_set:
                self._state.setdefault(sym, _SymbolState())

            self._dirty = True
            self._desired_version += 1

        # apply live if connected
        if self._ws is not None:
            await self._sync_subscriptions(self._ws)

    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        sym = _norm_symbol(symbol)
        st = self._state.get(sym)
        if not st:
            return {"available": False, "symbol": sym, "reason": "no_state"}

        tape = self._tape_stats(st)
        liq = self._liq_stats_5m(st)

        spread_bps = self._spread_bps(st)
        ob_imb = self._book_imbalance(st)

        ts = float(st.last_update_ts or 0.0)
        age_s = float(_now() - ts) if ts > 0 else 1e9
        ws_ok = bool(age_s <= _WS_FRESH_MAX_AGE_S)

        # mid as best "tradable last"
        mid = (st.best_bid + st.best_ask) / 2.0 if (st.best_bid > 0 and st.best_ask > 0) else 0.0
        last = float(mid or st.mark_price or 0.0)

        return {
            "available": True,
            "symbol": sym,

            "ts": ts,
            "timestamp": ts,
            "ts_ms": int(ts * 1000) if ts > 0 else 0,
            "age_s": age_s,
            "ws_ok": ws_ok,

            "mark_price": st.mark_price,
            "markPrice": st.mark_price,
            "index_price": st.index_price,
            "funding_rate": st.funding_rate,
            "fundingRate": st.funding_rate,
            "next_funding_time": st.next_funding_time,
            "nextFundingTime": st.next_funding_time,

            "best_bid": st.best_bid,
            "best_ask": st.best_ask,
            "bid_qty": st.bid_qty,
            "ask_qty": st.ask_qty,

            "spread_bps": spread_bps,
            "ob_imbalance": ob_imb,

            # tape
            "tape_delta_ratio_1m": tape["r1"],
            "tape_delta_ratio_5m": tape["r5"],
            "tape_delta_ratio_15m": tape["r15"],
            "tape_delta_notional_1m": tape["d1"],
            "tape_delta_notional_5m": tape["d5"],
            "tape_delta_notional_15m": tape["d15"],
            "tape_notional_1m": tape["n1"],
            "tape_notional_5m": tape["n5"],
            "tape_notional_15m": tape["n15"],

            # liquidation
            "liq_notional_5m": liq["tot"],
            "liq_buy_5m": liq["buy"],
            "liq_sell_5m": liq["sell"],
            "liq_delta_ratio_5m": liq["ratio"],
            "liq_count_5m": liq["cnt"],

            # watcher aliases
            "orderflow_ws": tape["r5"],      # ratio
            "cvd_notional_5m": tape["d5"],   # signed
            "cvd_notional_15m": tape["d15"],

            "price": last,
            "last": last,
        }

    # -----------------------------
    # metrics
    # -----------------------------
    def _prune(self, dq: Deque, cutoff: float) -> None:
        while dq and getattr(dq[0], "ts", 0.0) < cutoff:
            dq.popleft()

    def _tape_stats(self, st: _SymbolState) -> Dict[str, float]:
        now = _now()
        self._prune(st.tape, now - (_TAPE_WIN_15M + 5.0))

        d1 = n1 = 0.0
        d5 = n5 = 0.0
        d15 = n15 = 0.0

        for r in st.tape:
            if r.ts >= now - _TAPE_WIN_15M:
                d15 += r.delta_notional
                n15 += r.notional
            if r.ts >= now - _TAPE_WIN_5M:
                d5 += r.delta_notional
                n5 += r.notional
            if r.ts >= now - _TAPE_WIN_1M:
                d1 += r.delta_notional
                n1 += r.notional

        r1 = (d1 / n1) if n1 > 1e-9 else 0.0
        r5 = (d5 / n5) if n5 > 1e-9 else 0.0
        r15 = (d15 / n15) if n15 > 1e-9 else 0.0

        return {"d1": d1, "n1": n1, "r1": r1, "d5": d5, "n5": n5, "r5": r5, "d15": d15, "n15": n15, "r15": r15}

    def _liq_stats_5m(self, st: _SymbolState) -> Dict[str, Any]:
        now = _now()
        self._prune(st.liquidations, now - (_LIQ_WIN_5M + 5.0))

        buy = sell = 0.0
        cnt = 0

        for r in st.liquidations:
            if r.ts < now - _LIQ_WIN_5M:
                continue
            n = abs(float(r.notional))
            if n <= 0:
                continue
            cnt += 1
            s = str(r.side or "").upper()
            if s == "BUY":
                buy += n
            elif s == "SELL":
                sell += n

        tot = buy + sell
        ratio = ((buy - sell) / tot) if tot > 1e-12 else 0.0
        return {"tot": tot, "buy": buy, "sell": sell, "ratio": ratio, "cnt": cnt}

    def _book_imbalance(self, st: _SymbolState) -> float:
        den = abs(st.bid_qty) + abs(st.ask_qty)
        if den <= 1e-12:
            return 0.0
        return (st.bid_qty - st.ask_qty) / den

    def _spread_bps(self, st: _SymbolState) -> float:
        b = float(st.best_bid or 0.0)
        a = float(st.best_ask or 0.0)
        if b <= 0 or a <= 0:
            return 0.0
        mid = (a + b) / 2.0
        if mid <= 0:
            return 0.0
        return (a - b) / mid * 10000.0

    # -----------------------------
    # WS helpers
    # -----------------------------
    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    async def _send(self, ws, payload: Dict[str, Any]) -> None:
        await ws.send(json.dumps(payload, separators=(",", ":")))

    async def _send_ctrl_chunked(self, ws, method: str, streams: List[str]) -> None:
        if not streams:
            return
        bs = self._subscribe_batch_size
        for i in range(0, len(streams), bs):
            batch = streams[i : i + bs]
            await self._send(ws, {"method": method, "params": batch, "id": self._next_id()})
            await asyncio.sleep(_CTRL_MSG_DELAY_S)

    async def _sync_subscriptions(self, ws) -> None:
        """
        Only sync when dirty OR on periodic forced refresh.
        Avoids spam and reduces ws control traffic.
        """
        async with self._lock:
            desired_symbols = list(self._desired_symbols)
            desired_version = self._desired_version
            dirty = self._dirty

        force_refresh = (_now() - self._last_force_refresh) >= _FORCE_REFRESH_S
        if (not dirty) and (desired_version == self._last_synced_version) and (not force_refresh):
            return

        desired_streams = self._streams_for_symbols(desired_symbols)

        # hard cap by streams (defensive)
        if len(desired_streams) > _MAX_STREAMS_PER_CONN:
            # shrink symbols until streams fit
            kept: List[str] = []
            sset: Set[str] = set()
            for sym in desired_symbols:
                add = set(self._streams_for_symbol(sym))
                if len(sset) + len(add) > _MAX_STREAMS_PER_CONN:
                    break
                kept.append(sym)
                sset |= add
            desired_symbols = kept
            desired_streams = sset

        to_add = sorted(list(desired_streams - self._subscribed_streams))
        to_remove = sorted(list(self._subscribed_streams - desired_streams))

        if to_remove:
            await self._send_ctrl_chunked(ws, "UNSUBSCRIBE", to_remove)
            for s in to_remove:
                self._subscribed_streams.discard(s)

        if to_add:
            await self._send_ctrl_chunked(ws, "SUBSCRIBE", to_add)
            for s in to_add:
                self._subscribed_streams.add(s)

        async with self._lock:
            self._dirty = False
        self._last_synced_version = desired_version
        self._last_force_refresh = _now()

    async def _ensure_state(self, sym: str) -> _SymbolState:
        sym = _norm_symbol(sym)
        async with self._lock:
            return self._state.setdefault(sym, _SymbolState())

    # -----------------------------
    # WS event handlers
    # -----------------------------
    async def _handle_mark(self, sym: str, d: Dict[str, Any]) -> None:
        st = await self._ensure_state(sym)
        st.mark_price = _safe_float(d.get("p"), st.mark_price)
        st.index_price = _safe_float(d.get("i"), st.index_price)
        st.funding_rate = _safe_float(d.get("r"), st.funding_rate)
        nft = d.get("T")
        try:
            st.next_funding_time = int(nft) if nft is not None else st.next_funding_time
        except Exception:
            pass
        st.last_update_ts = _now()

    async def _handle_book(self, sym: str, d: Dict[str, Any]) -> None:
        st = await self._ensure_state(sym)
        st.best_bid = _safe_float(d.get("b"), st.best_bid)
        st.best_ask = _safe_float(d.get("a"), st.best_ask)
        st.bid_qty = _safe_float(d.get("B"), st.bid_qty)
        st.ask_qty = _safe_float(d.get("A"), st.ask_qty)
        st.last_update_ts = _now()
        if st.mark_price <= 0 and st.best_bid > 0 and st.best_ask > 0:
            st.mark_price = (st.best_bid + st.best_ask) / 2.0

    async def _handle_trade(self, sym: str, d: Dict[str, Any]) -> None:
        st = await self._ensure_state(sym)
        p = _safe_float(d.get("p"), 0.0)
        q = _safe_float(d.get("q"), 0.0)
        n = abs(p * q)
        m = bool(d.get("m", False))  # True => seller initiated
        delta = (-n) if m else (+n)
        if n > 0:
            st.tape.append(_TapeRec(ts=_now(), delta_notional=delta, notional=n))
        st.last_update_ts = _now()

    async def _handle_liq(self, sym: str, d: Dict[str, Any]) -> None:
        st = await self._ensure_state(sym)
        o = d.get("o") if isinstance(d.get("o"), dict) else {}
        side = str(o.get("S") or "").upper()
        if side not in ("BUY", "SELL"):
            side = "UNKNOWN"
        p = _safe_float(o.get("p"), 0.0)
        q = _safe_float(o.get("q"), 0.0)
        n = abs(p * q)
        if n > 0:
            st.liquidations.append(_LiqRec(ts=_now(), notional=n, side=side))
        st.last_update_ts = _now()

    async def _handle_combined(self, stream: str, data: Dict[str, Any]) -> None:
        sym = _stream_symbol(stream) or _norm_symbol(str(data.get("s") or ""))
        if not sym:
            return

        sl = (stream or "").lower()
        et = str(data.get("e") or "").lower()

        if ("@markprice" in sl) or (et == "markpriceupdate"):
            await self._handle_mark(sym, data)
        elif ("@bookticker" in sl) or (et == "bookticker"):
            await self._handle_book(sym, data)
        elif ("@aggtrade" in sl) or (et == "aggtrade"):
            await self._handle_trade(sym, data)
        elif ("@forceorder" in sl) or (et == "forceorder"):
            await self._handle_liq(sym, data)

    async def _handle_raw(self, payload: Dict[str, Any]) -> None:
        et = str(payload.get("e") or "").lower()
        sym = _norm_symbol(str(payload.get("s") or payload.get("symbol") or ""))
        if not et or not sym:
            return

        if et == "markpriceupdate":
            await self._handle_mark(sym, payload)
        elif et == "bookticker":
            await self._handle_book(sym, payload)
        elif et == "aggtrade":
            await self._handle_trade(sym, payload)
        elif et == "forceorder":
            await self._handle_liq(sym, payload)

    # -----------------------------
    # WS main loop
    # -----------------------------
    async def _run_loop(self) -> None:
        try:
            import websockets  # type: ignore
        except Exception as e:
            LOGGER.error("institutional_ws_hub requires `websockets` package: %s", e)
            return

        backoff = _RECONNECT_MIN_S

        while not self._stop_evt.is_set():
            try:
                async with websockets.connect(
                    BINANCE_FUTURES_WS,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
                    max_queue=2048,
                ) as ws:
                    self._ws = ws
                    LOGGER.info("[WS_HUB] connected %s", BINANCE_FUTURES_WS)

                    # reset per connection
                    self._subscribed_streams = set()
                    self._last_force_refresh = 0.0
                    backoff = _RECONNECT_MIN_S

                    if _ENABLE_COMBINED:
                        await self._send(ws, {"method": "SET_PROPERTY", "params": ["combined", True], "id": self._next_id()})
                        await asyncio.sleep(_CTRL_MSG_DELAY_S)

                    # first sync
                    await self._sync_subscriptions(ws)

                    while not self._stop_evt.is_set():
                        # sync only when needed
                        await self._sync_subscriptions(ws)

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

                        # ack
                        if isinstance(payload, dict) and "result" in payload and "id" in payload:
                            continue

                        # combined
                        if isinstance(payload, dict) and isinstance(payload.get("stream"), str) and isinstance(payload.get("data"), dict):
                            await self._handle_combined(payload["stream"], payload["data"])
                            continue

                        # raw
                        if isinstance(payload, dict) and isinstance(payload.get("e"), str):
                            await self._handle_raw(payload)
                            continue

            except Exception as e:
                self._ws = None
                s = str(e)
                LOGGER.warning("[WS_HUB] disconnected: %s", s)

                # if still hitting payload-too-long, auto-reduce batch size
                if "1008" in s and "Payload too long" in s:
                    old = self._subscribe_batch_size
                    self._subscribe_batch_size = max(10, int(self._subscribe_batch_size * 0.6))
                    if self._subscribe_batch_size != old:
                        LOGGER.warning("[WS_HUB] auto-reducing subscribe batch size: %d -> %d", old, self._subscribe_batch_size)

                await asyncio.sleep(backoff)
                backoff = min(_RECONNECT_MAX_S, max(_RECONNECT_MIN_S, backoff * 1.6))

        self._ws = None
        LOGGER.info("[WS_HUB] stopped")


# Singleton
HUB = InstitutionalWSHub()
