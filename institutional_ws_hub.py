# institutional_ws_hub.py
# Bitget WS Hub (institutional) — robuste + compatible institutional_data.py
#
# Notes (Bitget WS v2):
# - keepalive: send string "ping" every ~30s, expect "pong"; server disconnects if no ping for 2 min
# - rate limit: up to 10 messages/sec; recommend <50 channels/connection for stability
# - subscription limits: 240 subscription requests/hour/connection, max 1000 channels/connection
# Docs:
# - Websocket API (common): ping/pong + limits
# - Futures public channels: Depth (books5), Trade (trade), Tickers (ticker)

import asyncio
import gzip
import json
import os
import random
import time
import threading
import zlib
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple
from collections import deque

import aiohttp

from logger import get_logger
import math
import traceback

log = get_logger("institutional_ws_hub")

# ========================
# Settings (env overrides)
# ========================

# Bitget WebSocket v2 (public)
INST_WS_URL = os.getenv("INST_WS_URL", "wss://ws.bitget.com/v2/ws/public")

# Bitget contract WS v2 expects instType like: USDT-FUTURES / COIN-FUTURES / USDC-FUTURES
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

# Sharding (multiple WS connections)
INST_WS_SHARDS = int(float(os.getenv("INST_WS_SHARDS", "4")))

# ---------------------------------------------------------------------
# Extra debug/telemetry flags (WS hub)
# ---------------------------------------------------------------------
INST_WS_AUTOSTART = str(os.getenv('INST_WS_AUTOSTART', '1')).strip() == '1'
INST_WS_AUTO_SHARDS = str(os.getenv('INST_WS_AUTO_SHARDS', '1')).strip() == '1'
INST_WS_SHARDS_FORCE = str(os.getenv('INST_WS_SHARDS_FORCE', '0')).strip() == '1'
INST_WS_LOG_STACK = str(os.getenv('INST_WS_LOG_STACK', '1')).strip() == '1'
INST_WS_HEALTH_LOG_SEC = float(os.getenv('INST_WS_HEALTH_LOG_SEC', '0'))  # 0 disables periodic health logs

# Recommended by Bitget: keep <50 channels per connection for stability
# We subscribe 3 channels per symbol -> with batch=50 args, you still may have >50 channels overall, but you control via shards.
INST_WS_SUB_BATCH = int(float(os.getenv("INST_WS_SUB_BATCH", "50")))

# Ping / reconnect
INST_WS_PING_INTERVAL_S = float(os.getenv("INST_WS_PING_INTERVAL_S", "30"))
INST_WS_RECONNECT_MIN_S = float(os.getenv("INST_WS_RECONNECT_MIN_S", "2"))
INST_WS_RECONNECT_MAX_S = float(os.getenv("INST_WS_RECONNECT_MAX_S", "25"))

# Proactive reconnect before Bitget’s 24h disconnect window (seconds)
INST_WS_MAX_LIFETIME_S = float(os.getenv("INST_WS_MAX_LIFETIME_S", str(23 * 3600)))

# How “fresh” a snapshot must be to be considered available (seconds)
INST_WS_STALE_S = float(os.getenv("INST_WS_STALE_S", "15"))

# Tape window for delta computation
TAPE_WINDOW_S = float(os.getenv("INST_TAPE_WINDOW_S", "300"))  # 5 minutes

# Outgoing message rate limit (Bitget: <=10 msg/s). Keep margin.
INST_WS_MAX_SEND_PER_SEC = float(os.getenv("INST_WS_MAX_SEND_PER_SEC", "8"))

# Maximum number of symbols allowed per shard when automatically scaling shards.
# If the environment variable is not provided, default to 30 symbols per shard.
MAX_SYMBOLS_PER_SHARD = int(os.getenv("MAX_SYMBOLS_PER_SHARD", "30"))

# Helper to safely JSON-serialize a Python object for logging without dumping
# huge payloads.  Takes the first `length` characters of the JSON dump.
def _jex(obj: Any, length: int = 500) -> str:
    try:
        s = json.dumps(obj, default=str)
        return s[:length]
    except Exception:
        return str(obj)[:length]

# Channels (Bitget Futures WS public)
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


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(float(x))
    except Exception:
        return None


def _norm_symbol(sym: str) -> str:
    # normalize for lookups: remove separators and upper-case
    return (sym or "").strip().upper().replace("-", "").replace("_", "")


def _candidate_inst_ids(sym: str, product_type: str) -> List[str]:
    # For Bitget Futures WS v2, instId is typically like "BTCUSDT"
    _ = product_type  # reserved if you want per-product variants later
    s = _norm_symbol(sym)
    return [s] if s else []


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


def _decode_ws_payload(msg: aiohttp.WSMessage) -> Optional[str]:
    """
    Best-effort decode:
    - TEXT: msg.data
    - BINARY: try gzip (magic), then zlib, then utf-8
    """
    try:
        if msg.type == aiohttp.WSMsgType.TEXT:
            return msg.data
        if msg.type == aiohttp.WSMsgType.BINARY:
            raw = msg.data
            if not raw:
                return None
            # gzip magic
            try:
                if len(raw) >= 2 and raw[:2] == b"\x1f\x8b":
                    raw = gzip.decompress(raw)
            except Exception:
                pass
            # zlib (best effort)
            if isinstance(raw, (bytes, bytearray)):
                try:
                    raw = zlib.decompress(raw)
                except Exception:
                    pass
            try:
                return raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else None
            except Exception:
                return None
    except Exception:
        return None
    return None

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
    holding: Optional[float] = None
    next_funding_time_ms: Optional[int] = None
    ts_ms: int = field(default_factory=_now_ms)


@dataclass
class _SymbolState:
    symbol: str
    book: _BookState = field(default_factory=_BookState)
    trade: _TradeState = field(default_factory=_TradeState)
    ticker: _TickerState = field(default_factory=_TickerState)
    # rolling signed notional for tape delta
    tape: Deque[Tuple[int, float]] = field(default_factory=lambda: deque(maxlen=6000))

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
        self._state_lock = threading.Lock()

        self._last_send_monotonic = 0.0
        self._conn_started_monotonic = 0.0

        # additional state for status reporting
        self._lock = threading.Lock()
        self._last_msg_ts: Optional[float] = None
        self._last_error: Optional[str] = None
        self._reconnects: int = 0
        # store current ws url for status output
        self.ws_url: str = INST_WS_URL

    def is_running(self) -> bool:
        return bool(self._started) and (self._ws is not None) and (not self._ws.closed)

    def status(self) -> Dict[str, Any]:
        """Return a summary of this shard's state for monitoring."""
        with self._lock:
            return {
                "shard_id": int(self.shard_id),
                "running": bool(self.is_running()),
                "started": bool(self._started),
                "symbols": int(len(self.symbols)),
                "ws_url": str(self.ws_url),
                "last_msg_ts": float(self._last_msg_ts) if self._last_msg_ts else None,
                "last_error": str(self._last_error) if self._last_error else None,
                "reconnects": int(self._reconnects),
                "symbol_states": int(len(self._states)),
            }

    def _latest_ts_ms(self, st: _SymbolState) -> int:
        return int(max(st.book.ts_ms, st.trade.ts_ms, st.ticker.ts_ms))

    def _compute_tape_delta_5m_locked(self, st: _SymbolState) -> Optional[float]:
        if not st.tape:
            return None
        now_ms = _now_ms()
        cutoff = now_ms - int(TAPE_WINDOW_S * 1000.0)

        # purge old
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

        with self._state_lock:
            latest_ms = self._latest_ts_ms(st)
            age_s = (_now_ms() - latest_ms) / 1000.0
            available = age_s <= float(INST_WS_STALE_S)

            tape_5m = self._compute_tape_delta_5m_locked(st)

            return {
                "available": bool(available),
                "ts": float(latest_ms / 1000.0),
                "symbol": sym,

                # fields used by institutional_data.py
                "funding_rate": st.ticker.funding,
                "open_interest": st.ticker.holding,
                "tape_delta_5m": tape_5m,
                "next_funding_time_ms": st.ticker.next_funding_time_ms,

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

    async def _rate_limited_send(self) -> None:
        # keep margin under server limit (<=10 msg/s)
        max_per_sec = float(INST_WS_MAX_SEND_PER_SEC)
        if max_per_sec <= 0:
            return
        min_gap = 1.0 / max_per_sec
        now = time.monotonic()
        gap = now - float(self._last_send_monotonic)
        if gap < min_gap:
            await asyncio.sleep(min_gap - gap)
        self._last_send_monotonic = time.monotonic()

    async def _safe_send(self, payload: Any) -> None:
        async with self._send_lock:
            if not self._ws or self._ws.closed:
                return
            try:
                await self._rate_limited_send()
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
            for inst_id in _candidate_inst_ids(sym, inst_type):
                for ch in channels:
                    args.append({"instType": inst_type, "channel": ch, "instId": inst_id})

        if not args:
            return

        batch_size = max(10, int(INST_WS_SUB_BATCH))
        log.info(f"[WS_HUB:hub{self.shard_id}] subscribing args={len(args)} batch={batch_size}")

        for i in range(0, len(args), batch_size):
            if not self._ws or self._ws.closed:
                break
            batch = args[i: i + batch_size]
            await self._safe_send({"op": "subscribe", "args": batch})

    async def _ping_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(float(INST_WS_PING_INTERVAL_S))
            if not self._ws or self._ws.closed:
                continue
            idle = time.monotonic() - self._last_msg_monotonic
            if idle < float(INST_WS_PING_INTERVAL_S):
                continue
            await self._safe_send("ping")

    async def _run_forever(self) -> None:
        assert self._session is not None

        backoff = float(INST_WS_RECONNECT_MIN_S)
        backoff_max = float(INST_WS_RECONNECT_MAX_S)

        while not self._stop.is_set():
            ping_task: Optional[asyncio.Task] = None
            try:
                log.info(f"[WS_HUB:hub{self.shard_id}] connecting {INST_WS_URL}")
                self.ws_url = INST_WS_URL
                self._ws = await self._session.ws_connect(
                    INST_WS_URL,
                    heartbeat=None,    # we manage ping manually
                    autoping=False,
                    max_msg_size=0,
                )
                self._conn_started_monotonic = time.monotonic()
                self._last_msg_monotonic = time.monotonic()
                # update last_msg_ts in real time for status
                self._last_msg_ts = float(_now_ms())

                await self._subscribe()
                ping_task = asyncio.create_task(self._ping_loop(), name=f"ws_ping_{self.shard_id}")

                backoff = float(INST_WS_RECONNECT_MIN_S)

                async for msg in self._ws:
                    self._last_msg_monotonic = time.monotonic()
                    self._last_msg_ts = float(_now_ms())

                    # proactive reconnect before max lifetime
                    if float(INST_WS_MAX_LIFETIME_S) > 0:
                        if (time.monotonic() - float(self._conn_started_monotonic)) > float(INST_WS_MAX_LIFETIME_S):
                            log.info(f"[WS_HUB:hub{self.shard_id}] proactive reconnect (lifetime)")
                            break

                    txt = _decode_ws_payload(msg)
                    if not txt:
                        continue
                    if txt in ("pong", "ping"):
                        # we ignore server ping (if any) and pong responses
                        continue
                    await self._handle_text(txt)

            except Exception as e:
                self._last_error = str(e)
                log.warning(f"[WS_HUB:hub{self.shard_id}] ws error: {e}")
            finally:
                if ping_task:
                    ping_task.cancel()
                    try:
                        await ping_task
                    except Exception:
                        pass
                await self._close_ws()

            # increase reconnection count and apply backoff
            self._reconnects += 1
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

        # event frames (subscribe/unsubscribe/errors)
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

        st = self._states.get(inst_id)
        if st is None:
            base = inst_id.split("_")[0]
            st = self._states.get(base)
        if st is None:
            return

        # Exchange timestamp if present
        # (Bitget often provides `ts` in the data objects)
        def _pick_ts_ms(d: Dict[str, Any]) -> int:
            return _safe_int(d.get("ts")) or _safe_int(payload.get("ts")) or _now_ms()

        with self._state_lock:
            if channel == CHAN_BOOKS:
                d0 = data[0] if data else None
                if not isinstance(d0, dict):
                    return
                ts_ms = _pick_ts_ms(d0)

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
                st.book.ts_ms = ts_ms
                return

            if channel == CHAN_TRADES:
                # trade channel may push multiple trades in one message
                for d in data:
                    if not isinstance(d, dict):
                        continue
                    ts_ms = _pick_ts_ms(d)

                    px = _safe_float(d.get("price") or d.get("px"))
                    sz = _safe_float(d.get("size") or d.get("sz"))
                    side = d.get("side")

                    st.trade.last_price = px
                    st.trade.last_size = sz
                    st.trade.last_side = side
                    st.trade.ts_ms = ts_ms

                    # tape delta: signed notional (buy +, sell -)
                    if px is not None and sz is not None and side:
                        notional = float(px) * float(sz)
                        sgn = 1.0 if str(side).lower() in ("buy", "b") else -1.0
                        st.tape.append((ts_ms, sgn * notional))
                return

            if channel == CHAN_TICKER:
                d0 = data[0] if data else None
                if not isinstance(d0, dict):
                    return
                ts_ms = _pick_ts_ms(d0)

                st.ticker.mark_price = _safe_float(d0.get("markPrice") or d0.get("mark_price"))
                st.ticker.index_price = _safe_float(d0.get("indexPrice") or d0.get("index_price"))

                st.ticker.funding = _safe_float(d0.get("fundingRate") or d0.get("capitalRate") or d0.get("funding_rate"))
                st.ticker.holding = _safe_float(
                    d0.get("holdingAmount")
                    or d0.get("holding")
                    or d0.get("openInterest")
                    or d0.get("open_interest")
                )
                st.ticker.next_funding_time_ms = _safe_int(
                    d0.get("nextFundingTime")
                    or d0.get("next_funding_time")
                    or d0.get("nextUpdate")
                    or d0.get("next_update")
                )

                st.ticker.ts_ms = ts_ms
                return

# ========================
# Hub (sharded)
# ========================

class InstitutionalWSHub:
    def __init__(self, symbols: List[str], shards: int = INST_WS_SHARDS):
        self.symbols = [_norm_symbol(s) for s in symbols if _norm_symbol(s)]
        self.shards = max(1, int(shards))
        self._shards: List[InstitutionalWSHubShard] = []
        self._started = False

    def status(self) -> Dict[str, Any]:
        # gather status from each shard
        shard_status = [sh.status() for sh in self._shards]
        return {
            "started": bool(self._started),
            "running": bool(self.is_running()),
            "inst_type": str(INST_PRODUCT_TYPE),
            "ws_url": str(INST_WS_URL),
            "symbols": int(len(self.symbols)),
            "shards_total": int(len(shard_status)),
            "shards_running": int(sum(1 for x in shard_status if x.get("running"))),
            "last_msg_ts": max([x.get("last_msg_ts") for x in shard_status if x.get("last_msg_ts")], default=None),
            "last_error": [x.get("last_error") for x in shard_status if x.get("last_error")][-1] if any(x.get("last_error") for x in shard_status) else None,
            "shards": shard_status,
        }

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

    def status(self) -> Dict[str, Any]:
        if self._hub is None:
            return {'available': False, 'running': False, 'reason': 'hub_none'}
        try:
            st = self._hub.status()
            st['available'] = True
            return st
        except Exception as e:
            return {'available': True, 'running': bool(self.is_running()), 'reason': f'status_error:{type(e).__name__}:{e}'}

    async def start(self, symbols: List[str], shards: int = INST_WS_SHARDS) -> None:
        syms = tuple(sorted({_norm_symbol(s) for s in (symbols or []) if _norm_symbol(s)}))
        if INST_WS_AUTO_SHARDS and not INST_WS_SHARDS_FORCE:
            required = int(max(1, math.ceil(len(syms) / float(MAX_SYMBOLS_PER_SHARD))))
            if required > int(shards):
                log.info('[WS_HUB] autoscale_shards from=%s to=%s symbols=%s max_per_shard=%s', int(shards), int(required), len(syms), int(MAX_SYMBOLS_PER_SHARD))
                shards = int(required)
        if not syms:
            log.warning("[WS_HUB] start called with empty symbols list -> not starting")
            return

        async with self._lock:
            log.info('[WS_HUB_START] symbols=%s shards=%s', len(syms), int(shards))
            # already started with same universe
            if self._hub is not None and self._symbols_fingerprint == syms:
                if self._hub.is_running():
                    return
                else:
                    log.warning('[WS_HUB_START] restart_needed reason=hub_not_running status=%s', _jex(self._hub.status(), 500))

            # stop previous hub if any
            if self._hub is not None:
                try:
                    await self._hub.stop()
                except Exception:
                    pass

            self._hub = InstitutionalWSHub(list(syms), shards=shards)
            log.info('[WS_HUB_START] hub_created status=%s', _jex(self._hub.status(), 500))
            self._symbols_fingerprint = syms
            await self._hub.start()
            log.info('[WS_HUB_START] hub_started running=%s status=%s', self._hub.is_running(), _jex(self._hub.status(), 500))

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
