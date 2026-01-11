# institutional_ws_hub.py
# =============================================================================
# Bitget Mix Institutional WebSocket Hub (100% Bitget)
#
# Docs (Bitget Mix WS):
# - WS endpoint: wss://ws.bitget.com/mix/v1/stream
# - Subscribe: {"op":"subscribe","args":[{"instType":"mc","channel":"ticker","instId":"BTCUSDT_UMCBL"}]}
# - Heartbeat: send "ping" -> receive "pong"
# =============================================================================

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp

try:
    from logger import get_logger  # type: ignore
except Exception:  # pragma: no cover
    import logging
    def get_logger(name: str):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)

log = get_logger("institutional_ws_hub")

WS_URL = "wss://ws.bitget.com/mix/v1/stream"

DEFAULT_SHARDS = 4
BOOK_CHANNEL = "books5"
TRADE_CHANNEL = "trade"
TICKER_CHANNEL = "ticker"

PING_INTERVAL_S = 20.0
WS_IDLE_TIMEOUT_S = 35.0


def _norm_symbol(s: str) -> str:
    return (s or "").upper().replace("-", "").replace("/", "")


def _candidate_inst_ids(symbol: str, product_type: Optional[str]) -> List[str]:
    """
    Bitget WS examples often use instId like BTCUSDT_UMCBL.
    But many bots manipulate symbols as BTCUSDT.
    We subscribe to both to avoid hard dependency on suffix mapping.
    """
    sym = _norm_symbol(symbol)
    if not sym:
        return []
    if "_" in sym:
        return [sym]
    # Prefer a plausible USDT-perp suffix, but keep raw fallback.
    return [sym + "_UMCBL", sym]


@dataclass
class _TradeAgg:
    window_s: int
    buys_notional: float = 0.0
    sells_notional: float = 0.0
    buys_qty: float = 0.0
    sells_qty: float = 0.0
    events: List[Tuple[int, str, float, float]] = field(default_factory=list)

    def add(self, ts_ms: int, side: str, qty: float, price: float) -> None:
        notional = qty * price
        side_l = (side or "").lower()
        if side_l == "buy":
            self.buys_notional += notional
            self.buys_qty += qty
        else:
            self.sells_notional += notional
            self.sells_qty += qty
        self.events.append((ts_ms, side_l, qty, notional))
        self.prune(ts_ms)

    def prune(self, now_ms: int) -> None:
        cutoff = now_ms - self.window_s * 1000
        i = 0
        while i < len(self.events) and self.events[i][0] < cutoff:
            ts, side, qty, notional = self.events[i]
            if side == "buy":
                self.buys_notional -= notional
                self.buys_qty -= qty
            else:
                self.sells_notional -= notional
                self.sells_qty -= qty
            i += 1
        if i:
            del self.events[:i]

    def snapshot(self, now_ms: int) -> Dict[str, float]:
        self.prune(now_ms)
        return {
            "delta_qty": self.buys_qty - self.sells_qty,
            "delta_notional": self.buys_notional - self.sells_notional,
            "buy_notional": max(self.buys_notional, 0.0),
            "sell_notional": max(self.sells_notional, 0.0),
        }


@dataclass
class _BookState:
    ts_ms: int = 0
    bids: List[Tuple[float, float]] = field(default_factory=list)
    asks: List[Tuple[float, float]] = field(default_factory=list)

    def top(self):
        bid = self.bids[0] if self.bids else (None, None)
        ask = self.asks[0] if self.asks else (None, None)
        return bid[0], bid[1], ask[0], ask[1]


@dataclass
class _TickerState:
    ts_ms: int = 0
    last: Optional[float] = None
    mark: Optional[float] = None
    index: Optional[float] = None
    funding: Optional[float] = None
    holding: Optional[float] = None
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None


@dataclass
class _SymbolState:
    base_symbol: str
    inst_id: str
    ticker: _TickerState = field(default_factory=_TickerState)
    book: _BookState = field(default_factory=_BookState)
    tape_1m: _TradeAgg = field(default_factory=lambda: _TradeAgg(60))
    tape_5m: _TradeAgg = field(default_factory=lambda: _TradeAgg(300))


def _safe_float(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except Exception:
        return None


def _safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _parse_book_side(side: Any) -> List[Tuple[float, float]]:
    out = []
    if not isinstance(side, list):
        return out
    for lvl in side:
        if isinstance(lvl, list) and len(lvl) >= 2:
            px = _safe_float(lvl[0])
            sz = _safe_float(lvl[1])
            if px is not None and sz is not None:
                out.append((px, sz))
    return out


def _compute_snapshot(st: _SymbolState) -> Dict[str, Any]:
    now_ms = int(time.time() * 1000)

    t5 = st.tape_5m.snapshot(now_ms)

    bid_px, bid_sz, ask_px, ask_sz = st.book.top()
    if bid_px is None:
        bid_px = st.ticker.best_bid
    if ask_px is None:
        ask_px = st.ticker.best_ask

    mid = None
    spread_bps = None
    if bid_px and ask_px and bid_px > 0 and ask_px > 0:
        mid = (bid_px + ask_px) / 2.0
        spread_bps = ((ask_px - bid_px) / mid) * 10_000 if mid else None

    imb = None
    microprice = None
    if st.book.bids and st.book.asks and mid:
        bid_val = sum(px * sz for px, sz in st.book.bids[:5])
        ask_val = sum(px * sz for px, sz in st.book.asks[:5])
        tot = bid_val + ask_val
        imb = ((bid_val - ask_val) / tot) if tot else None

        if bid_px and ask_px and bid_sz and ask_sz and (bid_sz + ask_sz) > 0:
            microprice = (bid_px * ask_sz + ask_px * bid_sz) / (bid_sz + ask_sz)

    return {
        "available": True,
        "ts": now_ms / 1000.0,
        "symbol": st.base_symbol,
        "inst_id": st.inst_id,
        "tape_delta_5m": t5["delta_qty"],
        "tape_delta_5m_notional": t5["delta_notional"],
        "buy_notional_5m": t5["buy_notional"],
        "sell_notional_5m": t5["sell_notional"],
        "mid": mid,
        "spread_bps": spread_bps,
        "orderbook_imbalance": imb,
        "microprice": microprice,
        "mark_price": st.ticker.mark,
        "index_price": st.ticker.index,
        "funding_rate": st.ticker.funding,
        "open_interest": st.ticker.holding,
    }


class _BitgetShard:
    def __init__(self, shard_id: int, product_type: Optional[str]):
        self.shard_id = shard_id
        self.product_type = product_type
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._task: Optional[asyncio.Task] = None
        self._running = False

        self._symbols: Set[str] = set()
        self.state: Dict[str, _SymbolState] = {}

        self._last_msg_monotonic = time.monotonic()
        self._lock = asyncio.Lock()

    def _pfx(self):
        return f"[WS_HUB:hub{self.shard_id}]"

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_forever())
        log.info("%s started (Bitget)", self._pfx())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        await self._close_ws()
        if self._session:
            await self._session.close()

    async def set_symbols(self, symbols: List[str]):
        async with self._lock:
            self._symbols = {_norm_symbol(s) for s in symbols if _norm_symbol(s)}
        await self._close_ws()

    async def _ensure_session(self):
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        return self._session

    async def _close_ws(self):
        if self._ws and not self._ws.closed:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None

    async def _connect(self):
        s = await self._ensure_session()
        log.info("%s connecting %s", self._pfx(), WS_URL)
        self._ws = await s.ws_connect(
            WS_URL, heartbeat=None, autoping=False, max_msg_size=4 * 1024 * 1024
        )
        self._last_msg_monotonic = time.monotonic()

    async def _subscribe(self):
        async with self._lock:
            symbols = list(self._symbols)

        args = []
        for base in symbols:
            for inst_id in _candidate_inst_ids(base, self.product_type):
                self.state.setdefault(inst_id, _SymbolState(base_symbol=base, inst_id=inst_id))
                args.extend([
                    {"instType": "mc", "channel": TICKER_CHANNEL, "instId": inst_id},
                    {"instType": "mc", "channel": TRADE_CHANNEL, "instId": inst_id},
                    {"instType": "mc", "channel": BOOK_CHANNEL, "instId": inst_id},
                ])

        if args and self._ws:
            await self._ws.send_str(json.dumps({"op": "subscribe", "args": args}))

    async def _ping(self):
        if self._ws and not self._ws.closed:
            try:
                await self._ws.send_str("ping")
            except Exception:
                await self._close_ws()

    async def _handle(self, payload: dict):
        self._last_msg_monotonic = time.monotonic()

        if payload.get("event") in {"subscribe", "unsubscribe"}:
            return

        arg = payload.get("arg") or {}
        inst_id = arg.get("instId")
        channel = arg.get("channel")
        data = payload.get("data")
        if not inst_id or not channel or not data:
            return

        st = self.state.get(inst_id)
        if not st:
            return

        now_ms = int(time.time() * 1000)

        if channel == TICKER_CHANNEL and isinstance(data, list) and data and isinstance(data[0], dict):
            d0 = data[0]
            st.ticker.ts_ms = now_ms
            st.ticker.last = _safe_float(d0.get("last"))
            st.ticker.mark = _safe_float(d0.get("markPrice"))
            st.ticker.index = _safe_float(d0.get("indexPrice"))
            st.ticker.funding = _safe_float(d0.get("capitalRate"))
            st.ticker.holding = _safe_float(d0.get("holding"))
            st.ticker.best_bid = _safe_float(d0.get("bestBid"))
            st.ticker.best_ask = _safe_float(d0.get("bestAsk"))

        elif channel == TRADE_CHANNEL and isinstance(data, list):
            for t in data:
                if isinstance(t, list) and len(t) >= 4:
                    ts = _safe_int(t[0], now_ms)
                    px = _safe_float(t[1])
                    qty = _safe_float(t[2])
                    side = str(t[3])
                    if px and qty:
                        st.tape_1m.add(ts, side, qty, px)
                        st.tape_5m.add(ts, side, qty, px)

        elif channel == BOOK_CHANNEL and isinstance(data, list) and data and isinstance(data[0], dict):
            d0 = data[0]
            st.book.ts_ms = now_ms
            st.book.bids = _parse_book_side(d0.get("bids"))
            st.book.asks = _parse_book_side(d0.get("asks"))

    async def _run_forever(self):
        backoff = 1.0
        while self._running:
            try:
                await self._connect()
                await self._subscribe()
                backoff = 1.0

                while self._running and self._ws and not self._ws.closed:
                    if (time.monotonic() - self._last_msg_monotonic) > PING_INTERVAL_S:
                        await self._ping()

                    try:
                        msg = await self._ws.receive(timeout=WS_IDLE_TIMEOUT_S)
                    except asyncio.TimeoutError:
                        await self._ping()
                        continue

                    if msg.type == aiohttp.WSMsgType.TEXT:
                        if msg.data == "pong":
                            continue
                        try:
                            payload = json.loads(msg.data)
                        except Exception:
                            continue
                        await self._handle(payload)
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("%s ws error: %s", self._pfx(), e)
            finally:
                await self._close_ws()

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)


class InstitutionalWSHub:
    def __init__(self, shards: int = DEFAULT_SHARDS, product_type: Optional[str] = None):
        self.shards = max(1, int(shards))
        self.product_type = product_type
        self._running = False
        self._shards = [_BitgetShard(i, product_type) for i in range(self.shards)]

    @property
    def is_running(self):
        return self._running

    async def start(self, symbols: Optional[List[str]] = None, product_type: Optional[str] = None):
        if product_type:
            self.product_type = product_type
            for sh in self._shards:
                sh.product_type = product_type

        if self._running:
            if symbols is not None:
                await self.set_symbols(symbols)
            return

        self._running = True
        for sh in self._shards:
            await sh.start()

        if symbols is not None:
            await self.set_symbols(symbols)

        log.info("[WS_HUB] sharded started (Bitget) shards=%s", self.shards)

    async def stop(self):
        self._running = False
        for sh in self._shards:
            await sh.stop()

    async def set_symbols(self, symbols: List[str]):
        buckets = [[] for _ in range(self.shards)]
        for s in symbols:
            base = _norm_symbol(s)
            buckets[hash(base) % self.shards].append(base)
        await asyncio.gather(*[self._shards[i].set_symbols(buckets[i]) for i in range(self.shards)])

    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        base = _norm_symbol(symbol)
        best = None
        for sh in self._shards:
            for inst_id, st in sh.state.items():
                if st.base_symbol == base or inst_id == base:
                    snap = _compute_snapshot(st)
                    if snap.get("available"):
                        best = snap
        return best or {"available": False, "symbol": base, "reason": "no_ws_state"}


INST_HUB = InstitutionalWSHub()
HUB = INST_HUB
