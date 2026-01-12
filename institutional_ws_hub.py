# institutional_ws_hub.py
# Bitget WS Hub (institutional) — robuste + ticker fields fallback + subscribe batching

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import aiohttp

from logger import get_logger

log = get_logger("institutional_ws_hub")

# ========================
# Settings (local defaults)
# ========================

# WS endpoint Bitget (Mix v1)
INST_WS_URL = "wss://ws.bitget.com/mix/v1/stream"

# IMPORTANT: ton code actuel utilise "mc" (et pas "USDT-FUTURES").
# Je ne change pas ça ici pour éviter de casser ta compatibilité.
INST_PRODUCT_TYPE = "mc"

# Sharding
INST_WS_SHARDS = 4

# Subscribe batching (évite les frames trop grosses)
INST_WS_SUB_BATCH = 200  # nombre d'args par message subscribe

# Ping / reconnect
INST_WS_PING_INTERVAL_S = 15.0
INST_WS_RECONNECT_MIN_S = 2.0
INST_WS_RECONNECT_MAX_S = 25.0

# Channels
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


@dataclass
class _BookState:
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    bid_depth_usd: Optional[float] = None
    ask_depth_usd: Optional[float] = None
    spread: Optional[float] = None
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
    ts_ms: int = field(default_factory=_now_ms)


@dataclass
class _SymbolState:
    symbol: str
    book: _BookState = field(default_factory=_BookState)
    trade: _TradeState = field(default_factory=_TradeState)
    ticker: _TickerState = field(default_factory=_TickerState)


def _norm_symbol(sym: str) -> str:
    return sym.strip().upper()


def _candidate_inst_ids(sym: str) -> List[str]:
    """
    Ton code original essaye plusieurs variantes (sans suffixe / _UMCBL / _DMCBL).
    Je garde cette logique pour éviter de casser des symboles,
    mais on s'assure de ne pas dupliquer inutilement.
    """
    s = _norm_symbol(sym)
    out: List[str] = []

    def add(x: str):
        x = _norm_symbol(x)
        if x not in out:
            out.append(x)

    add(s)

    # si déjà suffixé, ne rajoute pas de suffixes
    if s.endswith("_UMCBL") or s.endswith("_DMCBL") or s.endswith("_CMCBL"):
        return out

    add(f"{s}_UMCBL")
    add(f"{s}_DMCBL")
    return out


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


class InstitutionalWSHubShard:
    def __init__(self, shard_id: int, symbols: List[str]):
        self.shard_id = shard_id
        self.symbols: List[str] = [_norm_symbol(s) for s in symbols]
        self._states: Dict[str, _SymbolState] = {s: _SymbolState(symbol=s) for s in self.symbols}

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None

        self._stop = asyncio.Event()
        self._last_msg_monotonic = time.monotonic()

    def snapshot(self, symbol: str) -> Dict[str, Any]:
        sym = _norm_symbol(symbol)
        st = self._states.get(sym)
        if not st:
            return {"ok": False, "symbol": sym, "reason": "unknown_symbol"}

        return {
            "ok": True,
            "symbol": sym,
            # book
            "best_bid": st.book.best_bid,
            "best_ask": st.book.best_ask,
            "spread": st.book.spread,
            "bid_depth_usd": st.book.bid_depth_usd,
            "ask_depth_usd": st.book.ask_depth_usd,
            "book_ts": st.book.ts_ms,
            # trade
            "last_price": st.trade.last_price,
            "last_size": st.trade.last_size,
            "last_side": st.trade.last_side,
            "trade_ts": st.trade.ts_ms,
            # ticker
            "mark_price": st.ticker.mark_price,
            "index_price": st.ticker.index_price,
            "funding_rate": st.ticker.funding,
            "open_interest": st.ticker.holding,
            "ticker_ts": st.ticker.ts_ms,
        }

    async def start(self):
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        asyncio.create_task(self._run_forever(), name=f"ws_hub_shard_{self.shard_id}")
        log.info(f"[WS_HUB:hub{self.shard_id}] started (Bitget)")

    async def stop(self):
        self._stop.set()
        await self._close_ws()
        if self._session:
            await self._session.close()
            self._session = None

    async def _close_ws(self):
        ws = self._ws
        self._ws = None
        if ws and not ws.closed:
            try:
                await ws.close()
            except Exception:
                pass

    async def _subscribe(self):
        """
        Subscribe books5 + trade + ticker
        Envoie en batch pour éviter les frames trop grosses.
        """
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
        for i in range(0, len(args), batch_size):
            batch = args[i : i + batch_size]
            try:
                if not self._ws or self._ws.closed:
                    break
                await self._ws.send_str(json.dumps({"op": "subscribe", "args": batch}))
            except Exception as e:
                log.warning(f"[WS_HUB:hub{self.shard_id}] ws error: {e}")
                try:
                    await self._close_ws()
                except Exception:
                    pass
                break
            await asyncio.sleep(0.05)

    async def _ping_loop(self):
        while not self._stop.is_set():
            await asyncio.sleep(INST_WS_PING_INTERVAL_S)
            if not self._ws or self._ws.closed:
                continue
            idle = time.monotonic() - self._last_msg_monotonic
            if idle < INST_WS_PING_INTERVAL_S:
                continue
            try:
                await self._ws.send_str("ping")
            except Exception as e:
                # typiquement: "Cannot write to closing transport"
                log.warning(f"[WS_HUB:hub{self.shard_id}] ws error: {e}")
                await self._close_ws()

    async def _run_forever(self):
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
                        await self._handle_text(txt)
                    elif msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.ERROR,
                    ):
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

    async def _handle_text(self, txt: str):
        try:
            payload = json.loads(txt)
        except Exception:
            return

        if not isinstance(payload, dict):
            return

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

        # IMPORTANT:
        # Ton state est indexé par symbol "scan" (ex: SANDUSDT)
        # mais ici on reçoit possiblement "SANDUSDT_UMCBL".
        # On tente les deux: exact inst_id, puis fallback sans suffixe si besoin.
        st = self._states.get(inst_id)
        if st is None:
            base = inst_id.split("_")[0]
            st = self._states.get(base)
        if st is None:
            return

        d0 = data[0] if data else None
        if not isinstance(d0, dict):
            return

        if channel == CHAN_BOOKS:
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
            st.book.ts_ms = _now_ms()

        elif channel == CHAN_TRADES:
            st.trade.last_price = _safe_float(d0.get("price") or d0.get("px"))
            st.trade.last_size = _safe_float(d0.get("size") or d0.get("sz"))
            st.trade.last_side = d0.get("side")
            st.trade.ts_ms = _now_ms()

        elif channel == CHAN_TICKER:
            # ✅ Robust parsing: accepte plusieurs noms de champs.
            st.ticker.mark_price = _safe_float(d0.get("markPrice") or d0.get("mark_price"))
            st.ticker.index_price = _safe_float(d0.get("indexPrice") or d0.get("index_price"))

            # funding: certains flux utilisent fundingRate, d’autres capitalRate
            st.ticker.funding = _safe_float(
                d0.get("fundingRate") or d0.get("capitalRate") or d0.get("funding_rate")
            )

            # open interest / holding: holdingAmount ou holding
            st.ticker.holding = _safe_float(
                d0.get("holdingAmount")
                or d0.get("holding")
                or d0.get("openInterest")
                or d0.get("open_interest")
            )

            st.ticker.ts_ms = _now_ms()


class InstitutionalWSHub:
    def __init__(self, symbols: List[str], shards: int = INST_WS_SHARDS):
        self.symbols = [_norm_symbol(s) for s in symbols]
        self.shards = max(1, int(shards))
        self._shards: List[InstitutionalWSHubShard] = []

    async def start(self):
        buckets = [[] for _ in range(self.shards)]
        for i, sym in enumerate(self.symbols):
            buckets[i % self.shards].append(sym)

        self._shards = [InstitutionalWSHubShard(i, buckets[i]) for i in range(self.shards)]
        for sh in self._shards:
            await sh.start()

        log.info(f"[WS_HUB] sharded started (Bitget) shards={self.shards}")

    async def stop(self):
        for sh in self._shards:
            await sh.stop()
        self._shards = []

    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        sym = _norm_symbol(symbol)
        if not self._shards:
            return {"ok": False, "symbol": sym, "reason": "not_started"}
        idx = hash(sym) % len(self._shards)
        return self._shards[idx].snapshot(sym)
