# institutional_ws_hub.py
# Bitget WS Hub (institutional) — robuste + compatible institutional_data.py

import asyncio
import gzip
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple
from collections import deque

import aiohttp

from logger import get_logger

log = get_logger("institutional_ws_hub")

# ========================
# Settings (env overrides)
# ========================

# Bitget WebSocket v2 (public)
INST_WS_URL = os.getenv("INST_WS_URL", "wss://ws.bitget.com/v2/ws/public")

# Bitget futures WS v2 expects instType like: USDT-FUTURES / COIN-FUTURES / USDC-FUTURES
_INST_TYPE_RAW = os.getenv("INST_WS_INST_TYPE") or os.getenv("INST_PRODUCT_TYPE_WS") or "USDT-FUTURES"


def _normalize_inst_type(v: str) -> str:
    v0 = (v or "").strip()
    v1 = v0.upper().replace("_", "-")
    alias = {
        "MC": "USDT-FUTURES",
        "UMCBL": "USDT-FUTURES",
        "USDT": "USDT-FUTURES",
        "USDT-FUTURE": "USDT-FUTURES",
        "COIN": "COIN-FUTURES",
        "DMCBL": "COIN-FUTURES",
        "USDC": "USDC-FUTURES",
    }
    return alias.get(v1, v1)


INST_PRODUCT_TYPE = _normalize_inst_type(_INST_TYPE_RAW)

INST_WS_SHARDS = int(float(os.getenv("INST_WS_SHARDS", "4")))
INST_WS_SUB_BATCH = int(float(os.getenv("INST_WS_SUB_BATCH", "200")))

INST_WS_PING_INTERVAL_S = float(os.getenv("INST_WS_PING_INTERVAL_S", "15"))
INST_WS_RECONNECT_MIN_S = float(os.getenv("INST_WS_RECONNECT_MIN_S", "2"))
INST_WS_RECONNECT_MAX_S = float(os.getenv("INST_WS_RECONNECT_MAX_S", "25"))

# How “fresh” a snapshot must be to be considered available (seconds)
INST_WS_STALE_S = float(os.getenv("INST_WS_STALE_S", "15"))

# Tape window for delta computation
TAPE_WINDOW_S = float(os.getenv("INST_TAPE_WINDOW_S", "300"))  # 5 minutes

# Channels (futures public WS)
CHAN_BOOKS = "books5"
CHAN_TRADES = "trade"
CHAN_TICKER = "ticker"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _norm_symbol(sym: str) -> str:
    return (sym or "").strip().upper()


def _parse_ts_ms(d: Dict[str, Any]) -> int:
    """
    Bitget WS payloads often include `ts` in ms (string/int).
    Fall back to local time if absent/unparseable.
    """
    ts = d.get("ts")
    try:
        if ts is None:
            return _now_ms()
        # sometimes it's string
        return int(float(ts))
    except Exception:
        return _now_ms()


def _candidate_inst_ids(sym: str) -> List[str]:
    """
    Bitget futures WS v2 uses plain instId like 'BTCUSDT' (no _UMCBL suffix).
    """
    sym = (sym or "").upper().strip()
    return [sym] if sym else []


def _depth_usd(levels: List[Any], topn: int = 5) -> Optional[float]:
    if not levels:
        return None
    total = 0.0
    for lvl in levels[:topn]:
        try:
            px = float(lvl[0])
            sz = float(lvl[1])
            total += px * sz
        except Exception:
            continue
    return total if total > 0 else None


# ========================
# Internal states
# ========================

@dataclass
class _BookState:
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    bid_depth_usd: Optional[float] = None
    ask_depth_usd: Optional[float] = None
    spread: Optional[float] = None  # relative (not bps)
    ts_ms: int = field(default_factory=_now_ms)


@dataclass
class _TradeState:
    last_price: Optional[float] = None
    last_size: Optional[float] = None
    last_side: Optional[str] = None
    ts_ms: int = field(default_factory=_now_ms)


@dataclass
class _TickerState:
    mark_price: Optional[float] = None
    index_price: Optional[float] = None
    funding: Optional[float] = None
    holding: Optional[float] = None  # holdingAmount (open interest proxy in Bitget ticker)
    ts_ms: int = field(default_factory=_now_ms)


@dataclass
class _SymbolState:
    symbol: str
    book: _BookState = field(default_factory=_BookState)
    trade: _TradeState = field(default_factory=_TradeState)
    ticker: _TickerState = field(default_factory=_TickerState)

    # rolling signed notional for tape delta
    tape: Deque[Tuple[int, float]] = field(default_factory=lambda: deque(maxlen=8000))


# ========================
# Shard
# ========================

class InstitutionalWSHubShard:
    def __init__(self, shard_id: int, symbols: List[str]):
        self.shard_id = shard_id
        self.symbols: List[str] = [_norm_symbol(s) for s in symbols if _norm_symbol(s)]
        self._states: Dict[str, _SymbolState] = {s: _SymbolState(symbol=s) for s in self.symbols}

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None

        self._stop = asyncio.Event()
        self._started = False

        self._last_msg_monotonic = time.monotonic()
        self._send_lock = asyncio.Lock()

    def is_running(self) -> bool:
        return bool(self._started) and (self._ws is not None) and (not self._ws.closed)

    def _latest_ts_ms(self, st: _SymbolState) -> int:
        return int(max(st.book.ts_ms, st.trade.ts_ms, st.ticker.ts_ms))

    def _compute_tape_delta_5m(self, st: _SymbolState) -> Optional[float]:
        if not st.tape:
            return None

        now_ms = _now_ms()
        cutoff = now_ms - int(TAPE_WINDOW_S * 1000.0)

        # purge old (left side)
        while st.tape and st.tape[0][0] < cutoff:
            st.tape.popleft()

        if not st.tape:
            return None

        s = 0.0
        for _, v in st.tape:
            s += float(v)
        return float(s)

    def snapshot(self, symbol: str) -> Dict[str, Any]:
        sym = _norm_symbol(symbol)
        st = self._states.get(sym)
        if not st:
            return {"available": False, "ts": None, "symbol": sym, "reason": "unknown_symbol"}

        latest_ms = self._latest_ts_ms(st)
        age_s = (_now_ms() - latest_ms) / 1000.0
        available = age_s <= float(INST_WS_STALE_S)

        tape_5m = self._compute_tape_delta_5m(st)

        # IMPORTANT: format attendu par institutional_data.py:
        # - available: bool
        # - ts: float (seconds)
        return {
            "available": bool(available),
            "ts": float(latest_ms / 1000.0),
            "symbol": sym,

            # fields directly usable by institutional_data.py
            "funding_rate": st.ticker.funding,
            "open_interest": st.ticker.holding,
            "tape_delta_5m": tape_5m,

            # extra debug/info
            "best_bid": st.book.best_bid,
            "best_ask": st.book.best_ask,
            "spread": st.book.spread,
            "bid_depth_usd": st.book.bid_depth_usd,
            "ask_depth_usd": st.book.ask_depth_usd,
            "mark_price": st.ticker.mark_price,
            "index_price": st.ticker.index_price,
            "book_ts": st.book.ts_ms,
            "trade_ts": st.trade.ts_ms,
            "ticker_ts": st.ticker.ts_ms,
        }

    async def start(self) -> None:
        if self._started:
            return
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        self._started = True
        asyncio.create_task(self._run_forever(), name=f"ws_hub_shard_{self.shard_id}")
        log.info(f"[WS_HUB:hub{self.shard_id}] started (Bitget) symbols={len(self.symbols)}")

    async def stop(self) -> None:
        self._stop.set()
        await self._close_ws()
        if self._session:
            await self._session.close()
            self._session = None
        self._started = False

    async def _close_ws(self) -> None:
        ws = self._ws
        self._ws = None
        if ws and not ws.closed:
            try:
                await ws.close()
            except Exception:
                pass

    async def _safe_send(self, payload: Any) -> None:
        # avoid concurrent writes when closing/reconnecting
        async with self._send_lock:
            if not self._ws or self._ws.closed:
                return
            try:
                if isinstance(payload, str):
                    await self._ws.send_str(payload)
                else:
                    await self._ws.send_str(json.dumps(payload))
            except Exception as e:
                log.warning(f"[WS_HUB:hub{self.shard_id}] ws send error: {e}")
                await self._close_ws()

    async def _subscribe(self) -> None:
        if not self._ws or self._ws.closed:
            return

        inst_type = INST_PRODUCT_TYPE
        channels = [CHAN_BOOKS, CHAN_TRADES, CHAN_TICKER]

        args: List[Dict[str, str]] = []
        for sym in self.symbols:
            for inst_id in _candidate_inst_ids(sym):
                for ch in channels:
                    args.append({"instType": inst_type, "channel": ch, "instId": inst_id})

        if not args:
            return

        batch_size = max(50, int(INST_WS_SUB_BATCH))
        log.info(f"[WS_HUB:hub{self.shard_id}] subscribing args={len(args)} batch={batch_size} instType={inst_type}")

        for i in range(0, len(args), batch_size):
            batch = args[i:i + batch_size]
            if not self._ws or self._ws.closed:
                break
            await self._safe_send({"op": "subscribe", "args": batch})
            await asyncio.sleep(0.05)

    async def _ping_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(float(INST_WS_PING_INTERVAL_S))
            if not self._ws or self._ws.closed:
                continue
            idle = time.monotonic() - self._last_msg_monotonic
            if idle < float(INST_WS_PING_INTERVAL_S):
                continue
            # Bitget WS v2 heartbeat expects plain "ping" / "pong"
            await self._safe_send("ping")

    async def _run_forever(self) -> None:
        assert self._session is not None

        backoff = float(INST_WS_RECONNECT_MIN_S)
        backoff_max = float(INST_WS_RECONNECT_MAX_S)

        while not self._stop.is_set():
            ping_task: Optional[asyncio.Task] = None
            try:
                log.info(f"[WS_HUB:hub{self.shard_id}] connecting {INST_WS_URL}")
                self._ws = await self._session.ws_connect(
                    INST_WS_URL,
                    heartbeat=None,
                    autoping=False,
                    max_msg_size=0,
                )
                self._last_msg_monotonic = time.monotonic()

                await self._subscribe()
                ping_task = asyncio.create_task(self._ping_loop(), name=f"ws_ping_{self.shard_id}")

                backoff = float(INST_WS_RECONNECT_MIN_S)

                async for msg in self._ws:
                    self._last_msg_monotonic = time.monotonic()

                    if msg.type == aiohttp.WSMsgType.TEXT:
                        txt = msg.data
                        if txt == "pong":
                            continue
                        if txt == "ping":
                            # server heartbeat
                            await self._safe_send("pong")
                            continue
                        await self._handle_text(txt)

                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        # some deployments may gzip binary frames
                        try:
                            raw = msg.data
                            try:
                                txt = raw.decode("utf-8")
                            except Exception:
                                txt = gzip.decompress(raw).decode("utf-8")
                            if txt == "pong":
                                continue
                            if txt == "ping":
                                await self._safe_send("pong")
                                continue
                            await self._handle_text(txt)
                        except Exception:
                            continue

                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                        break

            except Exception as e:
                log.warning(f"[WS_HUB:hub{self.shard_id}] ws error: {e}")
            finally:
                if ping_task:
                    ping_task.cancel()
                    try:
                        await ping_task
                    except Exception:
                        pass
                await self._close_ws()

            if self._stop.is_set():
                break

            sleep_s = min(backoff, backoff_max)
            sleep_s *= 0.75 + random.random() * 0.5
            await asyncio.sleep(sleep_s)
            backoff = min(backoff * 2.0, backoff_max)

    async def _handle_text(self, txt: str) -> None:
        try:
            payload = json.loads(txt)
        except Exception:
            return
        if not isinstance(payload, dict):
            return

        # event frames (subscribe/unsubscribe/error)
        if payload.get("event"):
            if payload.get("event") == "error":
                log.warning(f"[WS_HUB:hub{self.shard_id}] ws event error: {payload}")
            return

        arg = payload.get("arg") or {}
        data = payload.get("data") or []
        if not arg or not data:
            return

        channel = arg.get("channel")
        inst_id = arg.get("instId")
        if not channel or not inst_id:
            return

        inst_id = _norm_symbol(inst_id)

        # states indexed by base symbol (e.g. SANDUSDT)
        st = self._states.get(inst_id)
        if st is None:
            base = inst_id.split("_")[0]
            st = self._states.get(base)
        if st is None:
            return

        # books5/ticker usually send a list with one dict; trades can send multiple
        if channel == CHAN_BOOKS:
            d0 = data[0] if data else None
            if not isinstance(d0, dict):
                return

            asks = d0.get("asks") or []
            bids = d0.get("bids") or []
            if asks:
                st.book.best_ask = _safe_float(asks[0][0])
            if bids:
                st.book.best_bid = _safe_float(bids[0][0])

            if st.book.best_bid is not None and st.book.best_ask is not None and st.book.best_ask > 0:
                st.book.spread = (st.book.best_ask - st.book.best_bid) / st.book.best_ask
            else:
                st.book.spread = None

            st.book.bid_depth_usd = _depth_usd(bids, topn=5)
            st.book.ask_depth_usd = _depth_usd(asks, topn=5)
            st.book.ts_ms = _parse_ts_ms(d0)

        elif channel == CHAN_TRADES:
            # process ALL trades in the frame (important for tape delta)
            for item in data:
                if not isinstance(item, dict):
                    continue
                px = _safe_float(item.get("price") or item.get("px"))
                sz = _safe_float(item.get("size") or item.get("sz"))
                side = item.get("side")
                ts_ms = _parse_ts_ms(item)

                if px is not None:
                    st.trade.last_price = px
                if sz is not None:
                    st.trade.last_size = sz
                if side:
                    st.trade.last_side = side
                st.trade.ts_ms = ts_ms

                # tape delta: signed notional (buy +, sell -)
                if px is not None and sz is not None and side:
                    notional = float(px) * float(sz)
                    sgn = 1.0 if str(side).lower() in ("buy", "b") else -1.0
                    st.tape.append((ts_ms, sgn * notional))

        elif channel == CHAN_TICKER:
            d0 = data[0] if data else None
            if not isinstance(d0, dict):
                return

            st.ticker.mark_price = _safe_float(d0.get("markPrice") or d0.get("mark_price"))
            st.ticker.index_price = _safe_float(d0.get("indexPrice") or d0.get("index_price"))

            st.ticker.funding = _safe_float(
                d0.get("fundingRate")
                or d0.get("capitalRate")
                or d0.get("funding_rate")
            )

            st.ticker.holding = _safe_float(
                d0.get("holdingAmount")
                or d0.get("holding")
                or d0.get("openInterest")
                or d0.get("open_interest")
            )

            st.ticker.ts_ms = _parse_ts_ms(d0)


# ========================
# Hub (sharded)
# ========================

class InstitutionalWSHub:
    def __init__(self, symbols: List[str], shards: int = INST_WS_SHARDS):
        self.symbols = [_norm_symbol(s) for s in symbols if _norm_symbol(s)]
        self.shards = max(1, int(shards))
        self._shards: List[InstitutionalWSHubShard] = []
        self._started = False

    def is_running(self) -> bool:
        if not self._started or not self._shards:
            return False
        return any(sh.is_running() for sh in self._shards)

    async def start(self) -> None:
        if self._started:
            return

        buckets: List[List[str]] = [[] for _ in range(self.shards)]
        for i, sym in enumerate(self.symbols):
            buckets[i % self.shards].append(sym)

        self._shards = [InstitutionalWSHubShard(i, buckets[i]) for i in range(self.shards)]
        for sh in self._shards:
            await sh.start()

        self._started = True
        log.info(f"[WS_HUB] sharded started (Bitget) shards={self.shards} symbols={len(self.symbols)}")

    async def stop(self) -> None:
        for sh in self._shards:
            await sh.stop()
        self._shards = []
        self._started = False

    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        sym = _norm_symbol(symbol)
        if not self._shards:
            return {"available": False, "ts": None, "symbol": sym, "reason": "not_started"}
        idx = hash(sym) % len(self._shards)
        return self._shards[idx].snapshot(sym)


# ========================
# Controller singleton exposed as HUB (what institutional_data imports)
# ========================

class _HubController:
    """
    Expose:
    - async start(symbols=[...], shards=...)
    - async stop()
    - get_snapshot(symbol) -> dict with available/ts
    - is_running() -> bool
    """

    def __init__(self):
        self._hub: Optional[InstitutionalWSHub] = None
        self._lock = asyncio.Lock()
        self._symbols_fingerprint: Optional[Tuple[str, ...]] = None

    def is_running(self) -> bool:
        return bool(self._hub is not None and self._hub.is_running())

    async def start(self, symbols: List[str], shards: int = INST_WS_SHARDS) -> None:
        syms = tuple(sorted({_norm_symbol(s) for s in (symbols or []) if _norm_symbol(s)}))
        if not syms:
            log.warning("[WS_HUB] start called with empty symbols list -> not starting")
            return

        async with self._lock:
            # already started with same universe
            if self._hub is not None and self._symbols_fingerprint == syms and self._hub.is_running():
                return

            # stop previous hub if any
            if self._hub is not None:
                try:
                    await self._hub.stop()
                except Exception:
                    pass

            self._hub = InstitutionalWSHub(list(syms), shards=shards)
            self._symbols_fingerprint = syms
            await self._hub.start()

    async def stop(self) -> None:
        async with self._lock:
            if self._hub is not None:
                try:
                    await self._hub.stop()
                except Exception:
                    pass
            self._hub = None
            self._symbols_fingerprint = None

    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        if self._hub is None:
            sym = _norm_symbol(symbol)
            return {"available": False, "ts": None, "symbol": sym, "reason": "hub_none"}
        return self._hub.get_snapshot(symbol)


# THIS is what institutional_data expects:
HUB = _HubController()
