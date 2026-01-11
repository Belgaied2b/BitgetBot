# =====================================================================
# institutional_ws_hub.py — Bitget MIX (v1) WebSocket hub (sharded) + cache
# =====================================================================
# Compatibility goals:
# - Keep the same public API as the previous Binance hub:
#     hub = InstitutionalWSHub()
#     await hub.start(symbols)
#     await hub.set_symbols(symbols)
#     await hub.stop()
#     hub.get_snapshot(symbol)
# - Keep the snapshot structure/métriques identical for downstream code.
#
# Bitget WebSocket notes (from Bitget docs):
# - Heartbeat: send string "ping" every ~30s, expect "pong"; server disconnects if
#   no ping received for ~2 minutes.
# - Websocket accepts up to 10 messages per second.
# - Recommended to subscribe to < 50 channels (topics) per connection.
#
# This hub:
# - Shards across multiple connections (INST_WS_HUB_SHARDS) to stay under limits.
# - Batches subscription messages and paces them (anti-disconnect / rate limit).
# - Normalizes symbols: internal API can pass "BTCUSDT_UMCBL", WS instId is "BTCUSDT".
#   (Bitget FAQ: WebSocket instId uses base symbol, not the v1 order symbol with suffix.)
# =====================================================================

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

try:
    import websockets  # type: ignore
except Exception:  # pragma: no cover
    websockets = None  # type: ignore

LOGGER = logging.getLogger(__name__)

# -----------------------------
# Config (env)
# -----------------------------

WS_URL = str(os.getenv("BITGET_WS_BASE", "wss://ws.bitget.com/mix/v1/stream")).strip()

# Bitget docs: ping every ~30 seconds; server disconnects if no ping for 2 min.
PING_INTERVAL_SEC = float(os.getenv("BITGET_WS_PING_INTERVAL_SEC", "30"))

# If we don't see any market data for this long, snapshot is considered stale.
WS_STALE_SEC = float(os.getenv("INST_WS_STALE_SEC", "15"))

# WebSocket message rate limit: 10 msg/s max (Bitget docs).
# We'll stay well under this.
SUBSCRIBE_PACE_SEC = float(os.getenv("BITGET_WS_SUBSCRIBE_PACE_SEC", "0.12"))  # ~8 msg/s max if needed
SUBSCRIBE_BATCH_SIZE = int(os.getenv("BITGET_WS_SUBSCRIBE_BATCH_SIZE", "12"))  # topics per subscribe message

# Recommended < 50 topics per connection (Bitget docs).
MAX_TOPICS_PER_CONN = int(os.getenv("BITGET_WS_MAX_TOPICS_PER_CONN", "48"))

# Sharding
DEFAULT_SHARDS = int(os.getenv("INST_WS_HUB_SHARDS", "4"))

# InstType: docs show "mc" (perpetual contract) for MIX v1.
BITGET_INST_TYPE = str(os.getenv("BITGET_WS_INST_TYPE", "mc")).strip() or "mc"

# Channels (primary + fallback)
# The user requested "depth/trade/markPrice/forceOrders". Bitget MIX v1 docs
# commonly use "booksX" instead of "depth" for order book.
# We try the primary first, and fall back if the server returns an error.
CHANNEL_TRADES_PRIMARY = str(os.getenv("BITGET_WS_CH_TRADES", "trade")).strip() or "trade"
CHANNEL_ORDERBOOK_PRIMARY = str(os.getenv("BITGET_WS_CH_ORDERBOOK", "depth")).strip() or "depth"
CHANNEL_MARKPRICE_PRIMARY = str(os.getenv("BITGET_WS_CH_MARKPRICE", "markPrice")).strip() or "markPrice"
CHANNEL_LIQUIDATIONS_PRIMARY = str(os.getenv("BITGET_WS_CH_LIQ", "forceOrders")).strip() or "forceOrders"

# Fallback candidates (will be attempted on subscribe errors)
ORDERBOOK_FALLBACKS = [c for c in ["books5", "books15", "books"] if c != CHANNEL_ORDERBOOK_PRIMARY]
MARKPRICE_FALLBACKS = [c for c in ["mark-price", "markPrice", "mark_price"] if c != CHANNEL_MARKPRICE_PRIMARY]
LIQUIDATION_FALLBACKS = [c for c in ["forceOrders", "liquidation", "forceOrder"] if c != CHANNEL_LIQUIDATIONS_PRIMARY]


# -----------------------------
# Helpers
# -----------------------------

def _now() -> float:
    return time.time()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _norm_symbol(sym: str) -> str:
    return str(sym or "").upper().strip()


def _ws_inst_id(symbol: str) -> str:
    """
    Bitget v1 REST order symbols often look like BTCUSDT_UMCBL,
    but WebSocket instId commonly uses the base, e.g. BTCUSDT.
    """
    s = _norm_symbol(symbol)
    if not s:
        return ""
    if "_" in s:
        return s.split("_", 1)[0]
    return s


def _topic(inst_id: str, channel: str) -> str:
    return f"{inst_id}|{channel}"


def _parse_topic(t: str) -> Tuple[str, str]:
    if "|" not in t:
        return t, ""
    a, b = t.split("|", 1)
    return a, b


def _stable_hash32(s: str) -> int:
    # deterministic across runs (unlike built-in hash())
    h = 2166136261
    for ch in (s or ""):
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


# -----------------------------
# Rolling records
# -----------------------------

@dataclass
class _TapeRec:
    ts: float
    px: float
    qty: float
    side: str
    notional: float


@dataclass
class _LiqRec:
    ts: float
    px: float
    qty: float
    side: str
    notional: float


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


# -----------------------------
# Single WS connection (one shard)
# -----------------------------

class _SingleInstitutionalWSHub:
    """
    Single Bitget WebSocket connection + local rolling cache.
    """
    def __init__(self, *, name: str = "hub0", max_symbols: int = 64) -> None:
        self._name = str(name)
        self._max_symbols = int(max_symbols)

        self._desired_symbols: List[str] = []
        self._desired_set: Set[str] = set()

        self._subscribed_topics: Set[str] = set()
        self._state: Dict[str, _SymbolState] = {}

        # symbol mapping: instId -> full symbols (BTCUSDT -> {BTCUSDT_UMCBL, ...})
        self._inst_to_full: Dict[str, Set[str]] = {}

        self._lock = asyncio.Lock()
        self._dirty = True
        self._desired_version = 0
        self._last_synced_version = -1
        self._last_force_refresh = 0.0

        self._stop_evt = asyncio.Event()
        self._ws_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        self._ws = None

        self._last_msg_ts = 0.0

        # fallback channel mapping per instId
        self._resolved_channels: Dict[str, Dict[str, str]] = {}  # instId -> {"orderbook": "...", "mark": "...", "liq": "..."}

    @property
    def is_running(self) -> bool:
        return self._ws_task is not None and not self._ws_task.done()

    def _topics_for_symbol(self, full_symbol: str) -> List[str]:
        inst = _ws_inst_id(full_symbol)
        if not inst:
            return []
        # channels may have per-inst fallbacks chosen at runtime
        ch = self._resolved_channels.get(inst, {})
        ob = ch.get("orderbook", CHANNEL_ORDERBOOK_PRIMARY)
        mk = ch.get("mark", CHANNEL_MARKPRICE_PRIMARY)
        lq = ch.get("liq", CHANNEL_LIQUIDATIONS_PRIMARY)

        return [
            _topic(inst, CHANNEL_TRADES_PRIMARY),
            _topic(inst, ob),
            _topic(inst, mk),
            _topic(inst, lq),
        ]

    def _topics_for_symbols(self, symbols: List[str]) -> Set[str]:
        out: Set[str] = set()
        for s in symbols:
            out.update(self._topics_for_symbol(s))
        return out

    async def start(self, symbols: Optional[List[str]] = None, **_kwargs: Any) -> None:
        if websockets is None:
            return
        if self.is_running:
            return

        await self.set_symbols(symbols or [])
        self._stop_evt.clear()
        self._ws_task = asyncio.create_task(self._run_loop(), name=f"bitget_institutional_ws_{self._name}")
        LOGGER.info("[WS_HUB:%s] started (Bitget)", self._name)

    async def stop(self) -> None:
        self._stop_evt.set()

        if self._ping_task:
            try:
                self._ping_task.cancel()
            except Exception:
                pass
            self._ping_task = None

        if self._ws_task:
            try:
                await asyncio.wait_for(self._ws_task, timeout=8.0)
            except Exception:
                pass
            self._ws_task = None

        # ensure close
        try:
            if self._ws:
                await self._ws.close()
        except Exception:
            pass
        self._ws = None
        LOGGER.info("[WS_HUB:%s] stopped", self._name)

    async def set_symbols(self, symbols: List[str]) -> None:
        cleaned = [_norm_symbol(s) for s in (symbols or [])]
        cleaned = [s for s in cleaned if s]

        if self._max_symbols > 0 and len(cleaned) > self._max_symbols:
            cleaned = cleaned[: self._max_symbols]

        async with self._lock:
            self._desired_symbols = list(cleaned)
            self._desired_set = set(cleaned)
            self._dirty = True
            self._desired_version += 1

            # rebuild instId -> full mapping
            inst_to_full: Dict[str, Set[str]] = {}
            for fs in cleaned:
                inst = _ws_inst_id(fs)
                if not inst:
                    continue
                inst_to_full.setdefault(inst, set()).add(fs)
                self._state.setdefault(fs, _SymbolState())
            # remove state for dropped symbols
            for old in list(self._state.keys()):
                if old not in self._desired_set:
                    self._state.pop(old, None)
            self._inst_to_full = inst_to_full

    # -----------------------------
    # network / loop
    # -----------------------------

    async def _run_loop(self) -> None:
        # reconnect loop
        backoff = 1.0
        while not self._stop_evt.is_set():
            try:
                await self._connect_and_run()
                backoff = 1.0
            except asyncio.CancelledError:
                break
            except Exception as e:
                LOGGER.warning("[WS_HUB:%s] ws error: %s", self._name, e)
                await self._safe_close()
                # jittered exponential backoff
                await asyncio.sleep(min(20.0, backoff) + random.random() * 0.25)
                backoff = min(20.0, backoff * 1.8)

    async def _safe_close(self) -> None:
        try:
            if self._ws:
                await self._ws.close()
        except Exception:
            pass
        self._ws = None

    async def _connect_and_run(self) -> None:
        if websockets is None:
            return

        LOGGER.info("[WS_HUB:%s] connecting %s", self._name, WS_URL)
        async with websockets.connect(WS_URL, ping_interval=None, close_timeout=2) as ws:  # type: ignore
            self._ws = ws
            self._last_msg_ts = _now()

            # Start ping task (Bitget expects "ping" string)
            self._ping_task = asyncio.create_task(self._ping_loop(), name=f"bitget_ws_ping_{self._name}")

            # initial sync
            await self._sync_subscriptions(force=True)

            # recv loop
            while not self._stop_evt.is_set():
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=max(5.0, PING_INTERVAL_SEC))
                except asyncio.TimeoutError:
                    # no message; ping loop should keep connection alive
                    continue

                if msg is None:
                    raise RuntimeError("ws recv returned None")

                if isinstance(msg, (bytes, bytearray)):
                    try:
                        msg = msg.decode("utf-8", errors="ignore")
                    except Exception:
                        continue

                if msg == "pong":
                    self._last_msg_ts = _now()
                    continue
                if msg == "ping":
                    # some servers send ping as text; reply pong
                    try:
                        await ws.send("pong")
                    except Exception:
                        pass
                    continue

                try:
                    js = json.loads(msg)
                except Exception:
                    continue

                self._last_msg_ts = _now()
                await self._handle_message(js)

                # occasionally resync if dirty
                await self._sync_subscriptions(force=False)

    async def _ping_loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                if self._ws:
                    await self._ws.send("ping")
            except Exception:
                # connect loop will handle reconnect
                return
            await asyncio.sleep(max(5.0, PING_INTERVAL_SEC))

    # -----------------------------
    # subscribe / unsubscribe
    # -----------------------------

    async def _send_json(self, payload: Dict[str, Any]) -> None:
        if not self._ws:
            return
        await self._ws.send(json.dumps(payload, separators=(",", ":")))

    async def _sync_subscriptions(self, *, force: bool) -> None:
        async with self._lock:
            desired_version = self._desired_version
            dirty = self._dirty

        if not force and (not dirty or desired_version == self._last_synced_version):
            return

        # refresh at most every ~2s unless forced
        now = _now()
        if not force and (now - self._last_force_refresh) < 2.0:
            return
        self._last_force_refresh = now

        async with self._lock:
            desired_symbols = list(self._desired_symbols)
            self._dirty = False

        desired_topics = self._topics_for_symbols(desired_symbols)

        # enforce per-connection topic cap
        if len(desired_topics) > MAX_TOPICS_PER_CONN:
            # keep deterministic symbol order
            max_symbols = max(1, int(MAX_TOPICS_PER_CONN // 4))
            desired_symbols = desired_symbols[:max_symbols]
            desired_topics = self._topics_for_symbols(desired_symbols)

        to_add = sorted(desired_topics - self._subscribed_topics)
        to_remove = sorted(self._subscribed_topics - desired_topics)

        # Unsubscribe first (optional)
        if to_remove:
            await self._unsubscribe_topics(to_remove)

        if to_add:
            await self._subscribe_topics(to_add)

        self._subscribed_topics = desired_topics
        self._last_synced_version = desired_version

    async def _subscribe_topics(self, topics: List[str]) -> None:
        # batch topics into args messages
        for i in range(0, len(topics), max(1, SUBSCRIBE_BATCH_SIZE)):
            batch = topics[i : i + max(1, SUBSCRIBE_BATCH_SIZE)]
            args: List[Dict[str, str]] = []
            for t in batch:
                inst, ch = _parse_topic(t)
                if not inst or not ch:
                    continue
                args.append({"instType": BITGET_INST_TYPE, "channel": ch, "instId": inst})
            if not args:
                continue
            await self._send_json({"op": "subscribe", "args": args})
            await asyncio.sleep(max(0.01, SUBSCRIBE_PACE_SEC))

    async def _unsubscribe_topics(self, topics: List[str]) -> None:
        for i in range(0, len(topics), max(1, SUBSCRIBE_BATCH_SIZE)):
            batch = topics[i : i + max(1, SUBSCRIBE_BATCH_SIZE)]
            args: List[Dict[str, str]] = []
            for t in batch:
                inst, ch = _parse_topic(t)
                if not inst or not ch:
                    continue
                args.append({"instType": BITGET_INST_TYPE, "channel": ch, "instId": inst})
            if not args:
                continue
            await self._send_json({"op": "unsubscribe", "args": args})
            await asyncio.sleep(max(0.01, SUBSCRIBE_PACE_SEC))

    # -----------------------------
    # message parsing
    # -----------------------------

    async def _handle_message(self, js: Dict[str, Any]) -> None:
        # subscription acks / errors: {"event":"subscribe"|"error", ...}
        ev = js.get("event")
        if ev:
            if ev == "error":
                await self._handle_subscribe_error(js)
            return

        # data pushes: {"action":"snapshot"/"update","arg":{...},"data":[...]}
        arg = js.get("arg") or {}
        channel = str(arg.get("channel") or "")
        inst_id = str(arg.get("instId") or "")

        if not channel or not inst_id:
            return

        data = js.get("data")
        if not isinstance(data, list) or not data:
            return

        # route by channel
        if channel.lower().startswith("trade"):
            self._handle_trades(inst_id, data)
        elif channel.lower() in {CHANNEL_ORDERBOOK_PRIMARY.lower(), "depth"} or channel.lower().startswith("books"):
            self._handle_orderbook(inst_id, data)
        elif channel.lower() in {CHANNEL_MARKPRICE_PRIMARY.lower(), "mark-price", "markprice", "mark_price"}:
            self._handle_markprice(inst_id, data)
        elif channel.lower() in {CHANNEL_LIQUIDATIONS_PRIMARY.lower(), "forceorders", "forceorder", "liquidation"}:
            self._handle_liquidations(inst_id, data)
        else:
            # ignore other channels
            return

    async def _handle_subscribe_error(self, js: Dict[str, Any]) -> None:
        """
        Attempt to fall back when server rejects a channel.
        """
        arg = js.get("arg") or {}
        inst = str(arg.get("instId") or "")
        ch = str(arg.get("channel") or "")
        msg = str(js.get("msg") or "")

        if not inst or not ch:
            return

        # Decide which fallback list to use
        if ch == CHANNEL_ORDERBOOK_PRIMARY and ORDERBOOK_FALLBACKS:
            self._resolved_channels.setdefault(inst, {})["orderbook"] = ORDERBOOK_FALLBACKS[0]
            LOGGER.warning(
                "[WS_HUB:%s] %s orderbook channel '%s' rejected (%s) -> fallback '%s'",
                self._name,
                inst,
                ch,
                msg,
                ORDERBOOK_FALLBACKS[0],
            )
            # mark dirty so we resubscribe with fallback
            async with self._lock:
                self._dirty = True
                self._desired_version += 1

        elif ch == CHANNEL_MARKPRICE_PRIMARY and MARKPRICE_FALLBACKS:
            self._resolved_channels.setdefault(inst, {})["mark"] = MARKPRICE_FALLBACKS[0]
            LOGGER.warning(
                "[WS_HUB:%s] %s mark channel '%s' rejected (%s) -> fallback '%s'",
                self._name,
                inst,
                ch,
                msg,
                MARKPRICE_FALLBACKS[0],
            )
            async with self._lock:
                self._dirty = True
                self._desired_version += 1

        elif ch == CHANNEL_LIQUIDATIONS_PRIMARY and LIQUIDATION_FALLBACKS:
            self._resolved_channels.setdefault(inst, {})["liq"] = LIQUIDATION_FALLBACKS[0]
            LOGGER.warning(
                "[WS_HUB:%s] %s liq channel '%s' rejected (%s) -> fallback '%s'",
                self._name,
                inst,
                ch,
                msg,
                LIQUIDATION_FALLBACKS[0],
            )
            async with self._lock:
                self._dirty = True
                self._desired_version += 1

    def _targets_for_inst(self, inst_id: str) -> List[str]:
        inst = _norm_symbol(inst_id)
        if not inst:
            return []
        targets = self._inst_to_full.get(inst)
        if not targets:
            # in case caller passes base symbol directly
            if inst in self._desired_set:
                return [inst]
            return []
        return list(targets)

    def _touch(self, full_symbol: str) -> _SymbolState:
        st = self._state.get(full_symbol)
        if st is None:
            st = _SymbolState()
            self._state[full_symbol] = st
        st.last_update_ts = _now()
        return st

    def _handle_trades(self, inst_id: str, data: List[Any]) -> None:
        # Data records can be either:
        # - {"ts": "...", "px": "...", "sz": "...", "side": "buy"/"sell"}  (mix v1 docs)
        # - {"ts": "...", "price": "...", "size": "...", "side": "..."}   (v2 style)
        for rec in data:
            if not isinstance(rec, dict):
                continue
            ts = _safe_float(rec.get("ts"), _now() * 1000.0) / 1000.0
            px = _safe_float(rec.get("px", rec.get("price")))
            sz = _safe_float(rec.get("sz", rec.get("size")))
            side = str(rec.get("side") or "").lower() or "unknown"
            if px <= 0 or sz <= 0:
                continue
            notion = px * sz
            for full in self._targets_for_inst(inst_id):
                st = self._touch(full)
                st.tape.append(_TapeRec(ts=ts, px=px, qty=sz, side=side, notional=notion))

    def _handle_orderbook(self, inst_id: str, data: List[Any]) -> None:
        # For books/books5 etc, Bitget pushes data=[{"asks":[[p,q],...],"bids":[[p,q],...], "ts":"..."}]
        for rec in data:
            if not isinstance(rec, dict):
                continue
            bids = rec.get("bids") or []
            asks = rec.get("asks") or []
            if not isinstance(bids, list) or not isinstance(asks, list):
                continue

            best_bid, bid_qty = self._best_from_side(bids, is_bid=True)
            best_ask, ask_qty = self._best_from_side(asks, is_bid=False)

            if best_bid <= 0 or best_ask <= 0:
                continue

            for full in self._targets_for_inst(inst_id):
                st = self._touch(full)
                st.best_bid = best_bid
                st.bid_qty = bid_qty
                st.best_ask = best_ask
                st.ask_qty = ask_qty

    def _best_from_side(self, levels: List[Any], *, is_bid: bool) -> Tuple[float, float]:
        best_px = 0.0
        best_qty = 0.0
        if not levels:
            return best_px, best_qty

        # levels usually like [["27000.1","12"], ...]
        for lvl in levels:
            if not isinstance(lvl, (list, tuple)) or len(lvl) < 2:
                continue
            px = _safe_float(lvl[0])
            qty = _safe_float(lvl[1])
            if px <= 0 or qty < 0:
                continue
            if best_px == 0.0:
                best_px, best_qty = px, qty
                continue
            if is_bid:
                if px > best_px:
                    best_px, best_qty = px, qty
            else:
                if px < best_px:
                    best_px, best_qty = px, qty
        return best_px, best_qty

    def _handle_markprice(self, inst_id: str, data: List[Any]) -> None:
        # Not consistently documented across versions; parse defensively.
        for rec in data:
            if not isinstance(rec, dict):
                continue
            mp = _safe_float(rec.get("markPrice", rec.get("mark_price", rec.get("mp"))))
            ip = _safe_float(rec.get("indexPrice", rec.get("index_price", rec.get("ip"))))
            fr = _safe_float(rec.get("fundingRate", rec.get("funding_rate", rec.get("capitalRate"))))
            nft = rec.get("nextFundingTime", rec.get("nextSettleTime"))
            try:
                nft_i = int(nft) if nft is not None else None
            except Exception:
                nft_i = None
            if mp <= 0 and ip <= 0 and fr == 0.0 and nft_i is None:
                continue
            for full in self._targets_for_inst(inst_id):
                st = self._touch(full)
                if mp > 0:
                    st.mark_price = mp
                if ip > 0:
                    st.index_price = ip
                if fr != 0.0:
                    st.funding_rate = fr
                if nft_i is not None:
                    st.next_funding_time = nft_i

    def _handle_liquidations(self, inst_id: str, data: List[Any]) -> None:
        # Channel "forceOrders" is not consistently documented across all public docs.
        # Parse likely fields defensively.
        for rec in data:
            if not isinstance(rec, dict):
                continue

            ts = _safe_float(rec.get("ts", rec.get("ctime", rec.get("time"))), _now() * 1000.0) / 1000.0
            px = _safe_float(rec.get("price", rec.get("px", rec.get("fillPx"))))
            sz = _safe_float(rec.get("size", rec.get("sz", rec.get("qty"))))
            side = str(rec.get("side", rec.get("posSide", rec.get("direction"))) or "").lower() or "unknown"

            if px <= 0 or sz <= 0:
                continue
            notion = px * sz

            for full in self._targets_for_inst(inst_id):
                st = self._touch(full)
                st.liquidations.append(_LiqRec(ts=ts, px=px, qty=sz, side=side, notional=notion))

    # -----------------------------
    # snapshot
    # -----------------------------

    def get_snapshot(self, symbol: str) -> Optional[Dict[str, Any]]:
        sym = _norm_symbol(symbol)
        if not sym:
            return None

        st = self._state.get(sym)
        if st is None:
            return None

        now = _now()
        if (now - float(self._last_msg_ts)) > float(WS_STALE_SEC):
            return None

        # tape recent windows
        t_1m_notional = 0.0
        t_1m_qty = 0.0
        t_5m_notional = 0.0
        t_5m_qty = 0.0
        t_15m_notional = 0.0
        t_15m_qty = 0.0

        buy_1m = sell_1m = 0.0
        buy_5m = sell_5m = 0.0
        buy_15m = sell_15m = 0.0

        liq_1m_notional = 0.0
        liq_5m_notional = 0.0
        liq_15m_notional = 0.0
        liq_buy_1m = liq_sell_1m = 0.0
        liq_buy_5m = liq_sell_5m = 0.0
        liq_buy_15m = liq_sell_15m = 0.0

        # compute cutoffs
        cut_1m = now - 60.0
        cut_5m = now - 300.0
        cut_15m = now - 900.0

        # trades
        for tr in reversed(st.tape):
            if tr.ts < cut_15m:
                break
            if tr.ts >= cut_15m:
                t_15m_notional += tr.notional
                t_15m_qty += tr.qty
                if tr.side == "buy":
                    buy_15m += tr.notional
                elif tr.side == "sell":
                    sell_15m += tr.notional
            if tr.ts >= cut_5m:
                t_5m_notional += tr.notional
                t_5m_qty += tr.qty
                if tr.side == "buy":
                    buy_5m += tr.notional
                elif tr.side == "sell":
                    sell_5m += tr.notional
            if tr.ts >= cut_1m:
                t_1m_notional += tr.notional
                t_1m_qty += tr.qty
                if tr.side == "buy":
                    buy_1m += tr.notional
                elif tr.side == "sell":
                    sell_1m += tr.notional

        # liquidations
        for lr in reversed(st.liquidations):
            if lr.ts < cut_15m:
                break
            if lr.ts >= cut_15m:
                liq_15m_notional += lr.notional
                if lr.side == "buy":
                    liq_buy_15m += lr.notional
                elif lr.side == "sell":
                    liq_sell_15m += lr.notional
            if lr.ts >= cut_5m:
                liq_5m_notional += lr.notional
                if lr.side == "buy":
                    liq_buy_5m += lr.notional
                elif lr.side == "sell":
                    liq_sell_5m += lr.notional
            if lr.ts >= cut_1m:
                liq_1m_notional += lr.notional
                if lr.side == "buy":
                    liq_buy_1m += lr.notional
                elif lr.side == "sell":
                    liq_sell_1m += lr.notional

        return {
            "available": True,
            "symbol": sym,

            # prices
            "mark_price": float(st.mark_price or 0.0),
            "index_price": float(st.index_price or 0.0),
            "funding_rate": float(st.funding_rate or 0.0),
            "next_funding_time": st.next_funding_time,

            # best bid/ask
            "best_bid": float(st.best_bid or 0.0),
            "best_ask": float(st.best_ask or 0.0),
            "bid_qty": float(st.bid_qty or 0.0),
            "ask_qty": float(st.ask_qty or 0.0),

            # tape aggregates
            "tape_1m_notional": float(t_1m_notional),
            "tape_5m_notional": float(t_5m_notional),
            "tape_15m_notional": float(t_15m_notional),
            "tape_1m_qty": float(t_1m_qty),
            "tape_5m_qty": float(t_5m_qty),
            "tape_15m_qty": float(t_15m_qty),

            "tape_buy_1m_notional": float(buy_1m),
            "tape_sell_1m_notional": float(sell_1m),
            "tape_buy_5m_notional": float(buy_5m),
            "tape_sell_5m_notional": float(sell_5m),
            "tape_buy_15m_notional": float(buy_15m),
            "tape_sell_15m_notional": float(sell_15m),

            # liquidation aggregates
            "liq_1m_notional": float(liq_1m_notional),
            "liq_5m_notional": float(liq_5m_notional),
            "liq_15m_notional": float(liq_15m_notional),
            "liq_buy_1m_notional": float(liq_buy_1m),
            "liq_sell_1m_notional": float(liq_sell_1m),
            "liq_buy_5m_notional": float(liq_buy_5m),
            "liq_sell_5m_notional": float(liq_sell_5m),
            "liq_buy_15m_notional": float(liq_buy_15m),
            "liq_sell_15m_notional": float(liq_sell_15m),

            "ws_last_msg_ts": float(self._last_msg_ts or 0.0),
            "ws_last_update_ts": float(st.last_update_ts or 0.0),
        }


# -----------------------------
# Sharded manager
# -----------------------------

class InstitutionalWSHub:
    def __init__(self, *, shards: int = DEFAULT_SHARDS) -> None:
        self._shards_n = max(1, int(shards))
        # each shard: keep under topic cap
        self._per_shard_symbol_cap = max(1, int(MAX_TOPICS_PER_CONN // 4))
        self._shards: List[_SingleInstitutionalWSHub] = [
            _SingleInstitutionalWSHub(name=f"hub{i}", max_symbols=self._per_shard_symbol_cap)
            for i in range(self._shards_n)
        ]
        self._started = False
        self._sym_to_shard: Dict[str, int] = {}

    async def start(self, symbols: Optional[List[str]] = None, **kwargs: Any) -> None:
        if websockets is None:
            LOGGER.warning("[WS_HUB] websockets lib not available; hub disabled")
            return
        if self._started:
            return
        await self.set_symbols(symbols or [])
        for sh in self._shards:
            await sh.start([])
        self._started = True
        LOGGER.info("[WS_HUB] sharded started (Bitget) shards=%s", self._shards_n)

    async def stop(self) -> None:
        for sh in self._shards:
            await sh.stop()
        self._started = False
        LOGGER.info("[WS_HUB] sharded stopped")

    async def set_symbols(self, symbols: List[str]) -> None:
        if websockets is None:
            return

        cleaned = [_norm_symbol(s) for s in (symbols or [])]
        cleaned = [s for s in cleaned if s]

        # assign deterministic shard: stable hash
        self._sym_to_shard = {s: int(_stable_hash32(s) % self._shards_n) for s in cleaned}

        # group per shard
        per: List[List[str]] = [[] for _ in range(self._shards_n)]
        for s in cleaned:
            per[self._sym_to_shard[s]].append(s)

        # cap per shard to avoid topics > limit
        for i in range(self._shards_n):
            per[i] = per[i][: self._per_shard_symbol_cap]
            await self._shards[i].set_symbols(per[i])

    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        sym = _norm_symbol(symbol)
        if not sym:
            return {"available": False, "symbol": sym, "reason": "empty_symbol"}

        idx = self._sym_to_shard.get(sym)
        if idx is None:
            idx = int(_stable_hash32(sym) % max(1, self._shards_n))

        snap = self._shards[int(idx)].get_snapshot(sym)
        if snap is None:
            return {"available": False, "symbol": sym, "reason": "stale_or_missing"}
        return snap


# Backward compatible global instance
HUB = InstitutionalWSHub()
