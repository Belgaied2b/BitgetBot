# =====================================================================
# scanner.py — TP2 OFF (runner) + BE after TP1 + Telegram ON
# Fix Bitget min TP value (45110 "minimum amount X USDT")
# Fix SL->BE uses remaining qty after TP1
# + Governance desk (cooldown, entry timeout cancel)
# =====================================================================

from __future__ import annotations

import asyncio
import os
import logging
import time
import uuid
import re
import math
from collections import Counter
from typing import Any, Dict, Tuple, Optional

import pandas as pd

from settings import (
    API_KEY,
    API_SECRET,
    API_PASSPHRASE,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    SCAN_INTERVAL_MIN,
    TOP_N_SYMBOLS,
    MAX_ORDERS_PER_SCAN,
    DRY_RUN,
    MARGIN_USDT,
    LEVERAGE,
    STOP_TRIGGER_TYPE_SL,
    BE_FEE_BUFFER_TICKS,
)

from bitget_client import get_client
from bitget_trader import BitgetTrader
from analyze_signal import SignalAnalyzer

from duplicate_guard import DuplicateGuard, fingerprint as make_fingerprint
from risk_manager import RiskManager

logger = logging.getLogger(__name__)

DUP_GUARD = DuplicateGuard(ttl_seconds=3600)
RISK = RiskManager()

TF_H1 = "1H"
TF_H4 = "4H"
CANDLE_LIMIT = 200
MAX_CONCURRENT_FETCH = 8

TP1_CLOSE_PCT = 0.50
WATCH_INTERVAL_S = 3.0

REJECT_DEBUG_SAMPLES = 25

ARM_MAX_ATTEMPTS = 40
ARM_COOLDOWN_S = 8.0

MIN_TP_USDT_FALLBACK = 5.0
_MIN_USDT_RE = re.compile(r"minimum amount\s*([0-9]*\.?[0-9]+)\s*USDT", re.IGNORECASE)

# ===== PRICE BAND =====
_MAX_RE = re.compile(r"maximum price limit:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_MIN_RE = re.compile(r"minimum price limit:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

def _parse_band(msg: str) -> Tuple[Optional[float], Optional[float]]:
    if not msg:
        return None, None
    mmax = _MAX_RE.search(msg)
    mmin = _MIN_RE.search(msg)
    mx = float(mmax.group(1)) if mmax else None
    mn = float(mmin.group(1)) if mmin else None
    return mn, mx

def _parse_min_usdt(msg: str) -> Optional[float]:
    if not msg:
        return None
    m = _MIN_USDT_RE.search(msg)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def _clamp_and_quantize(price: float, tick: float, mn: Optional[float], mx: Optional[float]) -> Optional[float]:
    p = float(price)
    if tick <= 0:
        if mx is not None:
            p = min(p, float(mx))
        if mn is not None:
            p = max(p, float(mn))
        return p if p > 0 else None

    if mx is not None:
        p = min(p, float(mx) - 2.0 * tick)
        p = float(math.floor(p / tick) * tick)

    if mn is not None:
        p = max(p, float(mn) + 2.0 * tick)
        p = float(math.ceil(p / tick) * tick)

    if p <= 0:
        return None
    return float(p)


# =====================================================================
# Governance (desk)
# =====================================================================
# Cooldown simple par symbole (évite d'enchaîner 10 trades sur le même coin)
SYMBOL_COOLDOWN_S = float(os.getenv("SYMBOL_COOLDOWN_S", "600"))  # 10 min

# Timeout entry non fill : on annule l'ordre et on nettoie le pending
ENTRY_TIMEOUT_S = float(os.getenv("ENTRY_TIMEOUT_S", "240"))  # 4 min
ENTRY_CANCEL_ON_TIMEOUT = str(os.getenv("ENTRY_CANCEL_ON_TIMEOUT", "1")).lower() in ("1", "true", "yes", "y", "on")

# Filtre optionnel (OFF par défaut)
VETO_VOL_HIGH = str(os.getenv("VETO_VOL_HIGH", "0")).lower() in ("1", "true", "yes", "y", "on")

# Mémoire runtime (reset au restart)
LAST_TRADE_TS: Dict[str, float] = {}

def _in_symbol_cooldown(symbol: str) -> bool:
    if SYMBOL_COOLDOWN_S <= 0:
        return False
    s = (symbol or "").upper()
    last = float(LAST_TRADE_TS.get(s, 0.0) or 0.0)
    return (time.time() - last) < SYMBOL_COOLDOWN_S

async def _cancel_entry_order(trader: BitgetTrader, symbol: str, order_id: Optional[str], client_oid: Optional[str]) -> Dict[str, Any]:
    """Best-effort cancel entry. Ne casse pas le bot si l'endpoint change."""
    sym = (symbol or "").upper()
    payload: Dict[str, Any] = {
        "symbol": sym,
        "productType": getattr(trader, "product_type", "USDT-FUTURES"),
        "marginCoin": getattr(trader, "margin_coin", "USDT"),
    }
    if order_id:
        payload["orderId"] = str(order_id)
    if client_oid:
        payload["clientOid"] = str(client_oid)

    try:
        if hasattr(trader, "_request_any_status"):
            return await trader._request_any_status("POST", "/api/v2/mix/order/cancel-order", data=payload, auth=True)
        # fallback : client raw
        if hasattr(trader, "client") and hasattr(trader.client, "_request"):
            return await trader.client._request("POST", "/api/v2/mix/order/cancel-order", data=payload, auth=True)
    except Exception as e:
        return {"ok": False, "code": "CANCEL_EXC", "msg": str(e), "payload": payload}
    return {"ok": False, "code": "CANCEL_UNSUPPORTED", "msg": "no cancel method", "payload": payload}


# =====================================================================
# Desk logging
# =====================================================================

def desk_log(level: int, tag: str, symbol: str, tid: str, **kwargs):
    msg = f"[{tag}] {symbol} tid={tid}"
    if kwargs:
        msg += " " + " ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.log(level, msg)


# =====================================================================
# Telegram
# =====================================================================

def send_telegram_sync(text: str) -> None:
    try:
        import requests
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
        requests.post(url, json=payload, timeout=8)
    except Exception as e:
        logger.warning("Telegram sync error: %s", e)

async def send_telegram(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, send_telegram_sync, text)


# =====================================================================
# Utils
# =====================================================================

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _is_ok(resp: Any) -> bool:
    if not isinstance(resp, dict):
        return False
    if resp.get("ok") is True:
        return True
    return str(resp.get("code", "")) == "00000"

def _new_tid(symbol: str) -> str:
    return f"{symbol.upper()}-{uuid.uuid4().hex[:8]}"

def _direction_from_side(side: str) -> str:
    return "LONG" if side.upper() == "BUY" else "SHORT"

def _close_side_from_direction(direction: str) -> str:
    # close side for Bitget one_way : opposite
    return "sell" if direction.upper() == "LONG" else "buy"

def _trigger_type_sl() -> str:
    return STOP_TRIGGER_TYPE_SL or "mark_price"

def _oid(prefix: str, tid: str, attempt: int) -> str:
    return f"{prefix}-{tid}-{attempt}"

def _has_key_fields_for_trade(result: Dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False
    for k in ("entry", "sl", "tp1", "side"):
        if k not in result:
            return False
    return True

def _extract_reject_reason(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("reject_reason") or "reject")
    return "reject"

def _mk_signal_msg(symbol: str, tid: str, side: str, setup: str, entry: float, sl: float, tp1: float, rr: float, inst_score: int, entry_type: str, pos_mode: str) -> str:
    return (
        f"📡 *SIGNAL* `{symbol}`\n"
        f"• tid: `{tid}`\n"
        f"• side: *{side}* | setup: `{setup}` | entry_type: `{entry_type}`\n"
        f"• entry: `{entry}`\n"
        f"• sl: `{sl}`\n"
        f"• tp1: `{tp1}` (TP2 OFF)\n"
        f"• RR: `{rr}` | inst_score: `{inst_score}`\n"
        f"• posMode: `{pos_mode}`"
    )

def _mk_exec_msg(tag: str, symbol: str, tid: str, **kwargs) -> str:
    base = f"⚙️ *{tag}* `{symbol}` tid=`{tid}`"
    if kwargs:
        base += "\n" + "\n".join([f"• {k}: `{v}`" for k, v in kwargs.items()])
    return base

def _q_qty_floor(qty: float, step: float) -> float:
    if step <= 0:
        return float(qty)
    return float(math.floor(float(qty) / float(step)) * float(step))

def _q_qty_ceil(qty: float, step: float) -> float:
    if step <= 0:
        return float(qty)
    return float(math.ceil(float(qty) / float(step)) * float(step))

def _q_entry(entry: float, tick: float, side: str) -> float:
    # entry limit "desk" : small bias
    if tick <= 0:
        return float(entry)
    s = side.upper()
    if s == "BUY":
        return float(math.floor(entry / tick) * tick)
    return float(math.ceil(entry / tick) * tick)

def _q_sl(sl: float, tick: float, direction: str) -> float:
    if tick <= 0:
        return float(sl)
    d = direction.upper()
    # SL should be "worse" (further) to avoid instant triggers
    if d == "LONG":
        return float(math.floor(sl / tick) * tick)
    return float(math.ceil(sl / tick) * tick)

def _q_tp(tp: float, tick: float, direction: str) -> float:
    if tick <= 0:
        return float(tp)
    d = direction.upper()
    if d == "LONG":
        return float(math.ceil(tp / tick) * tick)
    return float(math.floor(tp / tick) * tick)

def _sanitize_tick(symbol: str, price: float, tick_meta: float, tid: str) -> float:
    # meta sometimes "suspect"
    tick = float(tick_meta or 0.0)
    p = float(price or 0.0)
    if tick <= 0:
        tick = 0.0
    if p > 0 and tick >= 1 and p < 1:
        # fallback estimated tick for micro prices
        tick = 10 ** (-6)
        desk_log(logging.WARNING, "TICK_SUSPECT_FALLBACK", symbol, tid, price=p, tick_meta=tick_meta, tick_used=tick)
    if tick <= 0:
        tick = 10 ** (-6)
    return float(tick)

# =====================================================================
# Tick cache
# =====================================================================

_TICK_CACHE: Dict[str, Tuple[float, float]] = {}  # sym -> (ts, tick)
_TICK_TTL_S = 600.0

async def _get_tick_cached(trader: BitgetTrader, symbol: str) -> float:
    sym = str(symbol).upper()
    now = time.time()
    if sym in _TICK_CACHE:
        ts, tick = _TICK_CACHE[sym]
        if now - ts < _TICK_TTL_S:
            return float(tick)
    tick = await trader.get_tick(sym)
    _TICK_CACHE[sym] = (now, float(tick))
    return float(tick)

# =====================================================================
# Position mode detection
# =====================================================================

_POS_MODE_CACHE: Tuple[float, str] = (0.0, "one_way")

async def _detect_pos_mode(trader: BitgetTrader) -> str:
    global _POS_MODE_CACHE
    now = time.time()
    ts, mode = _POS_MODE_CACHE
    if now - ts < 30:
        return mode
    try:
        if hasattr(trader, "get_position_mode"):
            pm = await trader.get_position_mode()
            if isinstance(pm, str) and pm:
                mode = pm
    except Exception:
        pass
    _POS_MODE_CACHE = (now, mode)
    return mode

# =====================================================================
# Fetch OHLCV
# =====================================================================

async def _fetch_dfs(client, symbol: str):
    df_h1 = await client.get_klines_df(symbol, tf=TF_H1, limit=CANDLE_LIMIT)
    df_h4 = await client.get_klines_df(symbol, tf=TF_H4, limit=CANDLE_LIMIT)
    return df_h1, df_h4

# =====================================================================
# Stats
# =====================================================================

class ScanStats:
    def __init__(self):
        self.total = 0
        self.valids = 0
        self.rejects = 0
        self.skips = 0
        self.duplicates = 0
        self.risk_rejects = 0
        self.exec_sent = 0
        self.exec_failed = 0
        self.reasons = Counter()
        self._rej_debug_used = 0

    async def inc(self, field: str, v: int):
        setattr(self, field, int(getattr(self, field)) + int(v))

    async def add_reason(self, reason: str):
        self.reasons[str(reason)] += 1

    async def take_reject_debug_slot(self) -> bool:
        if self._rej_debug_used >= REJECT_DEBUG_SAMPLES:
            return False
        self._rej_debug_used += 1
        return True

# =====================================================================
# PENDING state (watcher)
# =====================================================================

PENDING: Dict[str, Dict[str, Any]] = {}
PENDING_LOCK = asyncio.Lock()
WATCHER_TASK: Optional[asyncio.Task] = None

def _tp_trade_side_for_pos_mode(pos_mode: str) -> str:
    # one_way: close via opposite side + tradeSide=open
    # hedge:   close via same side + tradeSide=close + reduceOnly=YES
    pm = (pos_mode or "").lower()
    return "close" if pm == "hedge" else "open"

# =====================================================================
# Symbol processing
# =====================================================================

async def process_symbol(
    symbol: str,
    client,
    analyzer: SignalAnalyzer,
    trader: BitgetTrader,
    order_budget: asyncio.Semaphore,
    fetch_sem: asyncio.Semaphore,
    stats: ScanStats,
) -> None:
    tid = _new_tid(symbol)
    await stats.inc("total", 1)

    try:
        async with fetch_sem:
            t0 = time.time()
            df_h1, df_h4 = await _fetch_dfs(client, symbol)
            fetch_ms = int((time.time() - t0) * 1000)

        if df_h1 is None or df_h4 is None or getattr(df_h1, "empty", True) or getattr(df_h4, "empty", True):
            await stats.inc("skips", 1)
            return
        if len(df_h1) < 80 or len(df_h4) < 80:
            await stats.inc("skips", 1)
            return

        t1 = time.time()
        result = await analyzer.analyze(symbol, df_h1, df_h4, macro={})
        analyze_ms = int((time.time() - t1) * 1000)

        if not result or not isinstance(result, dict) or not result.get("valid"):
            reason = _extract_reject_reason(result)
            if await stats.take_reject_debug_slot():
                has_fields = "Y" if (isinstance(result, dict) and _has_key_fields_for_trade(result)) else "N"
                desk_log(logging.INFO, "REJ", symbol, tid, fetch_ms=fetch_ms, analyze_ms=analyze_ms, reason=reason, has_fields=has_fields)
            await stats.inc("rejects", 1)
            await stats.add_reason(reason)
            return

        side = str(result.get("side", "")).upper()
        if side not in ("BUY", "SELL"):
            await stats.inc("rejects", 1)
            await stats.add_reason("bad_side")
            return

        entry = _safe_float(result.get("entry"), 0.0)
        sl = _safe_float(result.get("sl"), 0.0)
        tp1 = _safe_float(result.get("tp1"), 0.0)
        rr = _safe_float(result.get("rr"), 0.0)
        setup = str(result.get("setup_type") or "-")
        entry_type = str(result.get("entry_type") or "MARKET")

        desk_log(logging.INFO, "EXITS", symbol, tid, side=side, entry=entry, sl=sl, tp1=tp1, rr=rr, setup=setup, entry_type=entry_type, vol_regime=str(result.get("vol_regime") or "-"), extension=str(result.get("extension") or "-"))

        if entry <= 0 or sl <= 0 or tp1 <= 0:
            await stats.inc("skips", 1)
            await stats.add_reason("missing_exits")
            return

        fp = make_fingerprint(symbol, side, entry, sl, tp1, extra=setup, precision=6)
        if DUP_GUARD.is_duplicate(fp):
            await stats.inc("duplicates", 1)
            return

        # Desk governance: cooldown symbole
        if _in_symbol_cooldown(symbol):
            await stats.inc("skips", 1)
            await stats.add_reason("cooldown")
            return

        # Filtre optionnel volatilité (si analyzer le fournit)
        vol_regime = str(result.get("vol_regime") or "")
        if VETO_VOL_HIGH and vol_regime.upper() == "HIGH":
            await stats.inc("skips", 1)
            await stats.add_reason("veto:vol_high")
            return

        direction = _direction_from_side(side)
        close_side = _close_side_from_direction(direction)
        notional = float(MARGIN_USDT) * float(LEVERAGE)

        inst = result.get("institutional") or {}
        inst_score = int(inst.get("institutional_score") or 0)

        allowed, reason = RISK.can_trade(
            symbol=symbol,
            side=direction,
            notional=notional,
            rr=rr if rr > 0 else None,
            inst_score=inst_score,
            commitment=None,
        )
        if not allowed:
            await stats.inc("risk_rejects", 1)
            await stats.add_reason(f"risk:{reason}")
            return

        pos_mode = await _detect_pos_mode(trader)

        await stats.inc("valids", 1)
        desk_log(logging.INFO, "VALID", symbol, tid, side=side, setup=setup, rr=rr, inst=inst_score, entry_type=entry_type, pos_mode=pos_mode)

        await send_telegram(_mk_signal_msg(str(symbol).upper(), tid, side, setup, entry, sl, tp1, rr, inst_score, entry_type, pos_mode))

        DUP_GUARD.mark(fp)

        if DRY_RUN:
            return

        try:
            await asyncio.wait_for(order_budget.acquire(), timeout=0.01)
        except asyncio.TimeoutError:
            await stats.add_reason("budget:max_orders_per_scan")
            return

        tick_meta = await _get_tick_cached(trader, symbol)
        tick_used = _sanitize_tick(symbol, entry, tick_meta, tid)
        q_entry = _q_entry(entry, tick_used, side)

        meta_dbg = await trader.debug_meta(symbol)

        desk_log(
            logging.INFO, "EXEC_PRE", symbol, tid,
            entry_raw=entry,
            tick_meta=tick_meta,
            tick_used=tick_used,
            q_entry=q_entry,
            direction=direction,
            pos_mode=pos_mode,
            meta_pricePlace=meta_dbg.get("pricePlace"),
            meta_priceTick=meta_dbg.get("priceTick"),
            meta_raw=meta_dbg.get("raw"),
        )

        if q_entry <= 0:
            await stats.inc("exec_failed", 1)
            await stats.add_reason("entry_q_zero")
            try:
                order_budget.release()
            except Exception:
                pass
            return

        await stats.inc("exec_sent", 1)
        desk_log(logging.INFO, "EXEC", symbol, tid, action="entry_send", entry=q_entry, notional=round(notional, 2))
        await send_telegram(_mk_exec_msg("ENTRY_SEND", str(symbol).upper(), tid, entry=q_entry, notional=round(notional, 2)))

        entry_resp = await trader.place_limit(
            symbol=symbol,
            side=side.lower(),
            price=q_entry,
            size=None,
            client_oid=f"entry-{tid}",
            trade_side="open",
            reduce_only=False,
            tick_hint=tick_used,
            debug_tag="ENTRY",
        )

        if not _is_ok(entry_resp):
            await stats.inc("exec_failed", 1)
            desk_log(logging.ERROR, "ENTRY_FAIL", symbol, tid, code=entry_resp.get("code"), msg=entry_resp.get("msg"), dbg=(entry_resp.get("_debug") or {}))
            desk_log(logging.ERROR, "META_DUMP", symbol, tid, meta=await trader.debug_meta(symbol))
            await send_telegram(_mk_exec_msg("ENTRY_FAIL", str(symbol).upper(), tid, code=entry_resp.get("code"), msg=entry_resp.get("msg")))
            try:
                order_budget.release()
            except Exception:
                pass
            return

        entry_order_id = (entry_resp.get("data") or {}).get("orderId") or entry_resp.get("orderId")
        qty_total = _safe_float(entry_resp.get("qty"), 0.0)

        desk_log(logging.INFO, "ENTRY_OK", symbol, tid, orderId=entry_order_id, qty=qty_total)

        async with PENDING_LOCK:
            PENDING[tid] = {
                "symbol": str(symbol).upper(),
                "entry_side": side.upper(),
                "direction": direction,
                "close_side": close_side,
                "pos_mode": pos_mode,
                "entry": q_entry,
                "sl": sl,
                "tp1": tp1,
                "qty_total": qty_total,
                "qty_tp1": 0.0,
                "entry_order_id": str(entry_order_id) if entry_order_id else None,
                "entry_client_oid": f"entry-{tid}",
                "sl_plan_id": None,
                "tp1_order_id": None,
                "armed": False,
                "be_done": False,
                "created_ts": time.time(),
                "arm_attempts": 0,
                "last_arm_fail_ts": 0.0,
                "sl_inflight": False,
                "tp1_inflight": False,
                "tp1_min_usdt": MIN_TP_USDT_FALLBACK,
                "qty_step": float(meta_dbg.get("qtyStep") or 1.0),
                "min_qty": float(meta_dbg.get("minQty") or 0.0),
            }

        desk_log(logging.INFO, "PENDING_NEW", symbol, tid, entry_side=side.upper(), close_side=close_side, direction=direction, pos_mode=pos_mode)

        # Cooldown starts when we successfully sent the entry
        LAST_TRADE_TS[str(symbol).upper()] = time.time()

    except Exception as e:
        desk_log(logging.ERROR, "ERR", symbol, tid, where="process_symbol", err=str(e))
        logger.exception("[%s] process_symbol error: %s", symbol, e)

# =====================================================================
# WATCHER
# =====================================================================

async def _watcher_loop(trader: BitgetTrader) -> None:
    logger.info("[WATCHER] started (interval=%.1fs)", WATCH_INTERVAL_S)

    while True:
        await asyncio.sleep(WATCH_INTERVAL_S)
        try:
            async with PENDING_LOCK:
                items = list(PENDING.items())

            if not items:
                continue

            for tid, st in items:
                sym = st["symbol"]
                direction = st["direction"]
                close_side = st["close_side"]
                pos_mode = st.get("pos_mode") or "one_way"
                entry = float(st["entry"])
                sl = float(st["sl"])
                tp1 = float(st["tp1"])
                qty_step = float(st.get("qty_step") or 1.0)
                min_qty = float(st.get("min_qty") or 0.0)

                last_fail = float(st.get("last_arm_fail_ts") or 0.0)
                if last_fail > 0 and (time.time() - last_fail) < ARM_COOLDOWN_S:
                    continue

                attempts = int(st.get("arm_attempts") or 0)
                if attempts >= ARM_MAX_ATTEMPTS:
                    desk_log(logging.ERROR, "ARM_ABORT", sym, tid, attempts=attempts)
                    async with PENDING_LOCK:
                        PENDING.pop(tid, None)
                    await send_telegram(_mk_exec_msg("ARM_ABORT", sym, tid, attempts=attempts))
                    continue

                # ----- wait fill then arm -----
                if not st["armed"]:
                    detail = await trader.get_order_detail(
                        sym,
                        order_id=st.get("entry_order_id"),
                        client_oid=st.get("entry_client_oid"),
                    )
                    if not trader.is_filled(detail):
                        # Desk governance: entry timeout -> cancel + cleanup
                        if ENTRY_TIMEOUT_S > 0 and (time.time() - float(st.get("created_ts") or time.time())) > ENTRY_TIMEOUT_S:
                            desk_log(logging.WARNING, "ENTRY_TIMEOUT", sym, tid, timeout_s=ENTRY_TIMEOUT_S)
                            if ENTRY_CANCEL_ON_TIMEOUT:
                                cresp = await _cancel_entry_order(trader, sym, st.get("entry_order_id"), st.get("entry_client_oid"))
                                desk_log(logging.INFO, "ENTRY_CANCEL", sym, tid, resp=cresp)
                            async with PENDING_LOCK:
                                PENDING.pop(tid, None)
                            await send_telegram(_mk_exec_msg("ENTRY_TIMEOUT_CANCEL", sym, tid, timeout_s=ENTRY_TIMEOUT_S))
                        continue

                    qty_total = float(st.get("qty_total") or 0.0)
                    if qty_total <= 0:
                        data = (detail.get("data") or {})
                        qty_total = float(data.get("size") or data.get("quantity") or 0.0)

                    tick_meta = await _get_tick_cached(trader, sym)
                    tick_used = _sanitize_tick(sym, entry, tick_meta, tid)

                    mn, mx = _parse_band(str(detail.get("msg") or ""))
                    q_sl = _clamp_and_quantize(_q_sl(sl, tick_used, direction), tick_used, mn, mx)
                    q_tp1 = _clamp_and_quantize(_q_tp(tp1, tick_used, direction), tick_used, mn, mx)

                    if q_sl is None or q_tp1 is None:
                        st["arm_attempts"] = attempts + 1
                        st["last_arm_fail_ts"] = time.time()
                        desk_log(logging.ERROR, "BAND_CLAMP_FAIL", sym, tid, sl=sl, tp1=tp1, q_sl=q_sl, q_tp1=q_tp1, mn=mn, mx=mx, tick=tick_used)
                        continue

                    tp_trade_side = _tp_trade_side_for_pos_mode(pos_mode)

                    min_usdt = float(st.get("tp1_min_usdt") or MIN_TP_USDT_FALLBACK)
                    if not _is_ok(detail):
                        mu = _parse_min_usdt(str(detail.get("msg") or ""))
                        if mu:
                            min_usdt = float(mu)

                    # Compute TP1 qty with min notional constraint
                    qty_tp1 = _q_qty_floor(qty_total * TP1_CLOSE_PCT, qty_step)
                    if qty_tp1 <= 0:
                        qty_tp1 = _q_qty_ceil(qty_total * TP1_CLOSE_PCT, qty_step)

                    # ensure TP1 notional >= min_usdt
                    tp1_notional = qty_tp1 * q_tp1
                    if tp1_notional < min_usdt:
                        needed_qty = _q_qty_ceil(min_usdt / max(1e-12, q_tp1), qty_step)
                        qty_tp1 = max(qty_tp1, needed_qty)

                    # Cap to total
                    qty_tp1 = min(qty_tp1, qty_total)

                    qty_rem = _q_qty_floor(max(0.0, qty_total - qty_tp1), qty_step)

                    # if runner would be dust, close full at TP1 (better than broken runner)
                    if qty_rem < (min_qty if min_qty > 0 else 0.0):
                        qty_tp1 = qty_total
                        qty_rem = 0.0

                    st["qty_total"] = qty_total
                    st["qty_tp1"] = qty_tp1
                    st["qty_rem"] = qty_rem
                    st["q_tp1"] = q_tp1
                    st["q_sl"] = q_sl

                    desk_log(
                        logging.INFO, "ARM_PRE", sym, tid,
                        tick_meta=tick_meta, tick_used=tick_used,
                        qty_total=qty_total, qty_tp1=qty_tp1, qty_rem=qty_rem,
                        sl_raw=sl, sl_q=q_sl,
                        tp1_raw=tp1, tp1_q=q_tp1,
                        close_side=close_side,
                        direction=direction,
                        pos_mode=pos_mode,
                        min_tp_usdt=min_usdt,
                        qty_step=qty_step,
                        min_qty=min_qty,
                    )

                    st["armed"] = True

                # 1) SL
                if not st.get("sl_plan_id") and (not st.get("sl_inflight", False)):
                    st["sl_inflight"] = True
                    desk_log(logging.INFO, "SL_SEND", sym, tid, close_side=close_side, trigger_type=_trigger_type_sl(), trigger_q=st["q_sl"], qty=st["qty_total"], tick=tick_used)

                    sl_resp = await trader.place_stop_market_sl(
                        symbol=sym,
                        close_side=close_side.lower(),
                        trigger_price=st["q_sl"],
                        qty=st["qty_total"],
                        client_oid=_oid("sl", tid, attempts),
                        trigger_type=_trigger_type_sl(),
                        tick_hint=tick_used,
                        debug_tag="SL",
                    )
                    st["sl_inflight"] = False

                    if not _is_ok(sl_resp):
                        st["arm_attempts"] = attempts + 1
                        st["last_arm_fail_ts"] = time.time()
                        desk_log(logging.ERROR, "SL_FAIL", sym, tid, code=sl_resp.get("code"), msg=sl_resp.get("msg"), dbg=sl_resp.get("_debug"))
                        desk_log(logging.ERROR, "META_DUMP", sym, tid, meta=await trader.debug_meta(sym))
                        await send_telegram(_mk_exec_msg("SL_FAIL", sym, tid, code=sl_resp.get("code"), msg=sl_resp.get("msg")))
                        continue

                    sl_plan_id = (sl_resp.get("data") or {}).get("orderId") or (sl_resp.get("data") or {}).get("planOrderId") or sl_resp.get("orderId")
                    async with PENDING_LOCK:
                        if tid in PENDING:
                            PENDING[tid]["sl_plan_id"] = str(sl_plan_id) if sl_plan_id else "ok"
                    desk_log(logging.INFO, "SL_OK", sym, tid, sl=st["q_sl"], planId=sl_plan_id)
                    await send_telegram(_mk_exec_msg("SL_OK", sym, tid, sl=st["q_sl"], planId=sl_plan_id))

                # 2) TP1
                if not st.get("tp1_order_id") and (not st.get("tp1_inflight", False)):
                    st["tp1_inflight"] = True
                    tp_trade_side = _tp_trade_side_for_pos_mode(pos_mode)
                    desk_log(logging.INFO, "TP1_SEND", sym, tid, close_side=close_side, price_q=st["q_tp1"], qty=st["qty_tp1"], tick=tick_used, trade_side=tp_trade_side, reduceOnly=True)

                    tp1_resp = await trader.place_limit(
                        symbol=sym,
                        side=close_side.lower(),
                        price=st["q_tp1"],
                        size=st["qty_tp1"],
                        client_oid=_oid("tp1", tid, attempts),
                        trade_side=tp_trade_side,
                        reduce_only=True,
                        tick_hint=tick_used,
                        debug_tag="TP1",
                    )
                    st["tp1_inflight"] = False

                    if not _is_ok(tp1_resp):
                        st["arm_attempts"] = attempts + 1
                        st["last_arm_fail_ts"] = time.time()
                        code = tp1_resp.get("code")
                        msg = tp1_resp.get("msg")
                        desk_log(logging.ERROR, "TP1_FAIL", sym, tid, code=code, msg=msg, dbg=tp1_resp.get("_debug"))

                        # attempt to parse min usdt
                        mu = _parse_min_usdt(str(msg or ""))
                        if mu:
                            st["tp1_min_usdt"] = float(mu)
                            desk_log(logging.WARNING, "TP1_MIN_USDT_UPDATE", sym, tid, min_usdt=mu)

                        await send_telegram(_mk_exec_msg("TP1_FAIL", sym, tid, code=code, msg=msg))
                        continue

                    tp1_order_id = (tp1_resp.get("data") or {}).get("orderId") or tp1_resp.get("orderId")
                    async with PENDING_LOCK:
                        if tid in PENDING:
                            PENDING[tid]["tp1_order_id"] = str(tp1_order_id) if tp1_order_id else "ok"
                    desk_log(logging.INFO, "TP1_OK", sym, tid, tp1=st["q_tp1"], qty=st["qty_tp1"], orderId=tp1_order_id)
                    await send_telegram(_mk_exec_msg("TP1_OK", sym, tid, tp1=st["q_tp1"], qty=st["qty_tp1"], orderId=tp1_order_id))

                # 3) TP1 filled -> SL -> BE on runner
                if st.get("tp1_order_id") and (not st.get("be_done", False)):
                    detail_tp = await trader.get_order_detail(sym, order_id=st.get("tp1_order_id"))
                    if not trader.is_filled(detail_tp):
                        continue

                    qty_total = float(st.get("qty_total") or 0.0)
                    qty_tp1 = float(st.get("qty_tp1") or 0.0)
                    qty_rem = _q_qty_floor(max(0.0, qty_total - qty_tp1), qty_step)
                    if qty_rem <= 0:
                        desk_log(logging.INFO, "BE_SKIP", sym, tid, reason="no_runner_remaining")
                        async with PENDING_LOCK:
                            PENDING.pop(tid, None)
                        continue

                    tick_meta = await _get_tick_cached(trader, sym)
                    tick_used = _sanitize_tick(sym, entry, tick_meta, tid)

                    be_ticks = int(BE_FEE_BUFFER_TICKS or 0)
                    be_delta = float(be_ticks) * float(tick_used)

                    be_raw = (entry + be_delta) if direction == "LONG" else (entry - be_delta)
                    be_q = _q_sl(be_raw, tick_used, direction)

                    desk_log(logging.INFO, "BE_SEND", sym, tid, be_raw=be_raw, be_q=be_q, tick=tick_used, be_ticks=be_ticks, qty_rem=qty_rem)

                    old_plan = st.get("sl_plan_id")
                    if old_plan:
                        try:
                            if hasattr(trader, "cancel_plan_orders"):
                                await trader.cancel_plan_orders(sym, [str(old_plan)])
                                desk_log(logging.INFO, "SL_CANCEL_OK", sym, tid, planId=old_plan)
                            else:
                                desk_log(logging.WARNING, "SL_CANCEL_SKIP", sym, tid, planId=old_plan, reason="no_cancel_plan_orders")
                        except Exception as e:
                            desk_log(logging.WARNING, "SL_CANCEL_WARN", sym, tid, planId=old_plan, err=str(e))

                    sl_be_resp = await trader.place_stop_market_sl(
                        symbol=sym,
                        close_side=close_side.lower(),
                        trigger_price=be_q,
                        qty=qty_rem,  # IMPORTANT: remaining qty
                        client_oid=_oid("slbe", tid, 0),
                        trigger_type=_trigger_type_sl(),
                        tick_hint=tick_used,
                        debug_tag="SL_BE",
                    )

                    if not _is_ok(sl_be_resp):
                        desk_log(logging.ERROR, "BE_FAIL", sym, tid, code=sl_be_resp.get("code"), msg=sl_be_resp.get("msg"))
                        await send_telegram(_mk_exec_msg("BE_FAIL", sym, tid, code=sl_be_resp.get("code"), msg=sl_be_resp.get("msg")))
                        st["last_arm_fail_ts"] = time.time()
                        continue

                    new_plan = (sl_be_resp.get("data") or {}).get("orderId") or (sl_be_resp.get("data") or {}).get("planOrderId") or sl_be_resp.get("orderId")

                    async with PENDING_LOCK:
                        if tid in PENDING:
                            PENDING[tid]["sl_plan_id"] = str(new_plan) if new_plan else st.get("sl_plan_id")
                            PENDING[tid]["be_done"] = True

                    desk_log(logging.INFO, "BE_OK", sym, tid, be=be_q, planId=new_plan, qty_rem=qty_rem)
                    await send_telegram(_mk_exec_msg("BE_OK", sym, tid, be=be_q, planId=new_plan, runner_qty=qty_rem))

                    # done
                    async with PENDING_LOCK:
                        PENDING.pop(tid, None)
                    continue

        except Exception:
            logger.exception("[WATCHER] error")

def _ensure_watcher(trader: BitgetTrader) -> None:
    global WATCHER_TASK
    if WATCHER_TASK is None or WATCHER_TASK.done():
        WATCHER_TASK = asyncio.create_task(_watcher_loop(trader))

# =====================================================================
# Scan loop
# =====================================================================

async def scan_once(client, analyzer: SignalAnalyzer, trader: BitgetTrader) -> None:
    stats = ScanStats()
    t_scan0 = time.time()

    symbols = await client.get_contracts_list()
    if not symbols:
        logger.warning("⚠️ get_contracts_list() vide")
        return

    symbols = sorted(set(map(str.upper, symbols)))[: int(TOP_N_SYMBOLS)]
    logger.info("📊 Scan %d symboles (TOP_N_SYMBOLS=%s)", len(symbols), TOP_N_SYMBOLS)

    fetch_sem = asyncio.Semaphore(MAX_CONCURRENT_FETCH)
    order_budget = asyncio.Semaphore(int(MAX_ORDERS_PER_SCAN))

    async def _worker(sym: str):
        await process_symbol(sym, client, analyzer, trader, order_budget, fetch_sem, stats)

    await asyncio.gather(*[_worker(sym) for sym in symbols])

    dt = time.time() - t_scan0
    reasons = stats.reasons.most_common(12)
    reasons_str = ", ".join([f"{k}:{v}" for k, v in reasons]) if reasons else "-"

    logger.info(
        "🧾 Scan summary: total=%s valids=%s rejects=%s skips=%s dup=%s risk_rejects=%s exec_sent=%s exec_failed=%s time=%.1fs | top_reasons=%s",
        stats.total,
        stats.valids,
        stats.rejects,
        stats.skips,
        stats.duplicates,
        stats.risk_rejects,
        stats.exec_sent,
        stats.exec_failed,
        dt,
        reasons_str,
    )

async def start_scanner() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        await send_telegram("✅ *Bot démarré* (TP2 OFF, TP1 minUSDT fix, BE after TP1 ON)")
    else:
        logger.warning("Telegram disabled: missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")

    client = await get_client(API_KEY, API_SECRET, API_PASSPHRASE)

    trader = BitgetTrader(
        client,
        margin_usdt=float(MARGIN_USDT),
        leverage=float(LEVERAGE),
        margin_mode="isolated",
    )

    analyzer = SignalAnalyzer()
    _ensure_watcher(trader)

    logger.info("🚀 Scanner started | interval=%s min | dry_run=%s", SCAN_INTERVAL_MIN, DRY_RUN)

    while True:
        t0 = time.time()
        try:
            await scan_once(client, analyzer, trader)
        except Exception:
            logger.exception("SCAN ERROR")

        dt = time.time() - t0
        sleep_s = max(1, int(float(SCAN_INTERVAL_MIN) * 60 - dt))
        await asyncio.sleep(sleep_s)

if __name__ == "__main__":
    asyncio.run(start_scanner())
