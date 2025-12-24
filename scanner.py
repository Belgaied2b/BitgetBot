# =====================================================================
# scanner.py — Desk-grade execution (ENTRY/SL/TP1 + Runner) + BE on TP1 fill
# TP2 removed (runner) + Telegram events + anti-spam arming
# =====================================================================

from __future__ import annotations

import asyncio
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

ARM_MAX_ATTEMPTS = 25
ARM_COOLDOWN_S = 8.0

# Exchange behavior seen in your logs
MIN_TP_NOTIONAL_USDT = 5.0

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

def _new_tid(symbol: str) -> str:
    return f"{str(symbol).upper()}-{uuid.uuid4().hex[:8]}"

def desk_log(level: int, tag: str, symbol: str, tid: str = "-", **kv: Any) -> None:
    parts = [f"[{tag}]", str(symbol).upper(), f"tid={tid}"]
    for k, v in kv.items():
        if v is None:
            continue
        if isinstance(v, float):
            parts.append(f"{k}={v:.10g}")
        else:
            parts.append(f"{k}={v}")
    logger.log(level, " ".join(parts))

def _oid(prefix: str, tid: str, attempt: int) -> str:
    return f"{prefix}-{tid}-{attempt}-{int(time.time()*1000)}"

# =====================================================================
# Telegram (plain text, no Markdown pitfalls)
# =====================================================================

async def send_telegram(msg: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    import requests
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "disable_web_page_preview": True,
    }

    def _do():
        try:
            r = requests.post(url, json=payload, timeout=10)
            if r.status_code >= 400:
                logger.error("Telegram HTTP %s: %s", r.status_code, r.text[:250])
        except Exception as e:
            logger.error("Telegram error: %s", e)

    await asyncio.to_thread(_do)

# =====================================================================
# Helpers
# =====================================================================

def _is_ok(resp: Any) -> bool:
    if not isinstance(resp, dict):
        return False
    if resp.get("ok") is True:
        return True
    return str(resp.get("code", "")) == "00000"

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def _side_to_direction(side: str) -> str:
    return "LONG" if (side or "").upper() == "BUY" else "SHORT"

def _close_side(entry_side: str) -> str:
    # to close LONG -> SELL, to close SHORT -> BUY
    return "SELL" if (entry_side or "").upper() == "BUY" else "BUY"

def _trigger_type_sl() -> str:
    s = (STOP_TRIGGER_TYPE_SL or "MP").upper()
    return "mark_price" if s == "MP" else "fill_price"

def _estimate_tick_from_price(price: float) -> float:
    p = abs(float(price))
    if p >= 10000:
        return 1.0
    if p >= 1000:
        return 0.1
    if p >= 100:
        return 0.01
    if p >= 10:
        return 0.001
    if p >= 1:
        return 0.0001
    if p >= 0.1:
        return 0.00001
    if p >= 0.01:
        return 0.000001
    return 0.0000001

def _sanitize_tick(symbol: str, entry: float, tick: float, tid: str) -> float:
    est = _estimate_tick_from_price(entry)
    t = float(tick or 0.0)
    if t <= 0 or t > est * 1000:
        desk_log(logging.WARNING, "TICK", symbol, tid, tick_meta=t, tick_est=est, entry=entry, action="fallback")
        return est
    return t

def _q_floor(price: float, tick: float) -> float:
    if tick <= 0:
        return float(price)
    return float(math.floor(price / tick) * tick)

def _q_ceil(price: float, tick: float) -> float:
    if tick <= 0:
        return float(price)
    return float(math.ceil(price / tick) * tick)

def _q_entry(price: float, tick: float, side: str) -> float:
    # make limit more fill-friendly:
    # BUY: floor (lower/equal), SELL: ceil (higher/equal)
    if tick <= 0:
        return float(price)
    return _q_floor(price, tick) if (side or "").upper() == "BUY" else _q_ceil(price, tick)

def _q_tp(price: float, tick: float, close_side: str) -> float:
    # fill-friendly TP:
    # close_side BUY (closing short): ceil a bit higher -> can fill earlier
    # close_side SELL (closing long): floor a bit lower -> can fill earlier
    return _q_ceil(price, tick) if (close_side or "").upper() == "BUY" else _q_floor(price, tick)

def _q_sl_trigger(price: float, tick: float, close_side: str) -> float:
    # protective rounding:
    # close_side BUY (stop for short triggers when price >= trigger): floor -> triggers earlier
    # close_side SELL (stop for long triggers when price <= trigger): ceil -> triggers earlier
    return _q_floor(price, tick) if (close_side or "").upper() == "BUY" else _q_ceil(price, tick)

def _extract_reject_reason(result: Any) -> str:
    if not isinstance(result, dict):
        return "not_valid"
    r = result.get("reject_reason") or result.get("reason") or result.get("reject")
    if r:
        return str(r)
    inst = result.get("institutional") or {}
    iscore = inst.get("institutional_score")
    if iscore is not None:
        return "inst_score_low" if int(iscore) < 2 else f"inst_score={iscore}"
    return "not_valid"

def _has_key_fields_for_trade(result: Dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False
    side = str(result.get("side", "")).upper()
    if side not in ("BUY", "SELL"):
        return False
    entry = _safe_float(result.get("entry"), 0.0)
    sl = _safe_float(result.get("sl"), 0.0)
    tp1 = _safe_float(result.get("tp1"), 0.0)
    rr = _safe_float(result.get("rr"), 0.0)
    return entry > 0 and sl > 0 and tp1 > 0 and rr > 0

def _quantize_qty(qty: float, step: float, min_qty: float) -> float:
    q = float(qty)
    if step and step > 0:
        q = math.floor(q / step) * step
    if min_qty and q < min_qty:
        return 0.0
    return float(q)

def _ceil_qty(qty: float, step: float) -> float:
    if not step or step <= 0:
        return float(qty)
    return float(math.ceil(qty / step) * step)

async def _maybe_cancel_sl(trader: BitgetTrader, symbol: str, plan_id: Optional[str]) -> bool:
    if not plan_id:
        return False
    # best-effort without requiring changes in BitgetTrader
    for fn in ("cancel_plan_order", "cancel_plan", "cancel_stop", "cancel_order"):
        if hasattr(trader, fn):
            try:
                resp = await getattr(trader, fn)(symbol, plan_id)
                ok = _is_ok(resp) if isinstance(resp, dict) else True
                return bool(ok)
            except Exception:
                return False
    return False

# =====================================================================
# Watcher state + tick cache
# =====================================================================

PENDING: Dict[str, Dict[str, Any]] = {}
PENDING_LOCK = asyncio.Lock()
WATCHER_TASK: Optional[asyncio.Task] = None

TICK_CACHE: Dict[str, float] = {}
TICK_LOCK = asyncio.Lock()

async def _get_tick_cached(trader: BitgetTrader, symbol: str) -> float:
    sym = str(symbol).upper()
    async with TICK_LOCK:
        if sym in TICK_CACHE and TICK_CACHE[sym] > 0:
            return TICK_CACHE[sym]
    t = await trader.get_tick(sym)
    t = float(t or 0.0)
    async with TICK_LOCK:
        if t > 0:
            TICK_CACHE[sym] = t
    return t

# =====================================================================
# Fetch
# =====================================================================

async def _fetch_dfs(client, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_h1 = await client.get_klines_df(symbol, TF_H1, CANDLE_LIMIT)
    df_h4 = await client.get_klines_df(symbol, TF_H4, CANDLE_LIMIT)
    return df_h1, df_h4

# =====================================================================
# Per-scan stats
# =====================================================================

class ScanStats:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.total = 0
        self.skips = 0
        self.rejects = 0
        self.valids = 0
        self.duplicates = 0
        self.risk_rejects = 0
        self.exec_sent = 0
        self.exec_failed = 0
        self.reasons = Counter()
        self.reject_debug_left = REJECT_DEBUG_SAMPLES

    async def inc(self, field: str, n: int = 1) -> None:
        async with self.lock:
            setattr(self, field, getattr(self, field) + n)

    async def add_reason(self, reason: str) -> None:
        async with self.lock:
            self.reasons[reason] += 1

    async def take_reject_debug_slot(self) -> bool:
        async with self.lock:
            if self.reject_debug_left <= 0:
                return False
            self.reject_debug_left -= 1
            return True

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
        setup = str(result.get("setup_type") or "")
        entry_type = str(result.get("entry_type") or "MARKET")
        in_zone = bool(result.get("in_zone", False))

        # EXIT DEBUG (raw)
        desk_log(
            logging.INFO, "EXITS", symbol, tid,
            side=side, entry=entry, sl=sl, tp1=tp1, rr=rr,
            setup=setup, entry_type=entry_type, in_zone=in_zone
        )

        if entry <= 0 or sl <= 0 or tp1 <= 0 or rr <= 0:
            await stats.inc("skips", 1)
            await stats.add_reason("missing_exits")
            return

        # Dup guard
        fp = make_fingerprint(symbol, side, entry, sl, tp1, extra=f"{setup}:{entry_type}", precision=6)
        if DUP_GUARD.is_duplicate(fp):
            await stats.inc("duplicates", 1)
            return

        direction = _side_to_direction(side)
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

        await stats.inc("valids", 1)
        desk_log(logging.INFO, "VALID", symbol, tid, side=side, setup=setup, rr=rr, inst=inst_score, entry_type=entry_type)

        # Telegram signal
        await send_telegram(
            f"✅ VALID {symbol} {direction}\n"
            f"setup={setup} entry_type={entry_type} in_zone={in_zone}\n"
            f"entry={entry:.6g} sl={sl:.6g} tp1={tp1:.6g} rr={rr:.3f} inst={inst_score}"
        )

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

        entry_resp = await trader.place_limit(
            symbol=symbol,
            side=side.lower(),
            price=q_entry,
            size=None,  # BitgetTrader computes size from notional/margin_usdt
            client_oid=f"entry-{tid}",
            trade_side="open",
            reduce_only=False,
            tick_hint=tick_used,
            debug_tag="ENTRY",
        )

        if not _is_ok(entry_resp):
            await stats.inc("exec_failed", 1)
            dbg = entry_resp.get("_debug") or {}
            desk_log(
                logging.ERROR, "ENTRY_FAIL", symbol, tid,
                code=entry_resp.get("code"),
                msg=entry_resp.get("msg"),
                http=entry_resp.get("_http_status"),
                dbg=dbg,
            )
            desk_log(logging.ERROR, "META_DUMP", symbol, tid, meta=await trader.debug_meta(symbol))
            await send_telegram(f"❌ ENTRY_FAIL {symbol} {direction} code={entry_resp.get('code')} msg={entry_resp.get('msg')}")
            try:
                order_budget.release()
            except Exception:
                pass
            return

        entry_order_id = (entry_resp.get("data") or {}).get("orderId") or entry_resp.get("orderId")
        qty_total = _safe_float(entry_resp.get("qty"), 0.0)

        desk_log(logging.INFO, "ENTRY_OK", symbol, tid, orderId=entry_order_id, qty=qty_total)
        await send_telegram(f"📥 ENTRY_OK {symbol} {direction} orderId={entry_order_id} qty={qty_total}")

        async with PENDING_LOCK:
            PENDING[tid] = {
                "symbol": str(symbol).upper(),
                "setup": setup,
                "entry_type": entry_type,
                "in_zone": in_zone,
                "entry_side": side.upper(),
                "close_side": _close_side(side),
                "direction": direction,
                "entry": float(q_entry),
                "sl": float(sl),
                "tp1": float(tp1),
                "tick_used": float(tick_used),

                "qty_total": float(qty_total) if qty_total > 0 else 0.0,
                "qty_tp1": 0.0,
                "qty_runner": 0.0,

                "entry_order_id": str(entry_order_id) if entry_order_id else None,
                "entry_client_oid": f"entry-{tid}",

                "sl_plan_id": None,
                "tp1_order_id": None,

                "armed": False,
                "tp1_done": False,
                "be_done": False,

                "arm_attempts": 0,
                "last_try_ts": 0.0,
                "created_ts": time.time(),
            }
        desk_log(logging.INFO, "PENDING_NEW", symbol, tid, entry_side=side.upper(), close_side=_close_side(side), direction=direction)

    except Exception as e:
        desk_log(logging.ERROR, "ERR", symbol, tid, where="process_symbol", err=str(e))
        logger.exception("[%s] process_symbol error: %s", symbol, e)

# =====================================================================
# WATCHER — waits fill then arms SL+TP1, and later moves SL to BE when TP1 filled
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

            now = time.time()

            for tid, st in items:
                sym = st["symbol"]

                # anti-spam cooldown per pending item
                last_try = float(st.get("last_try_ts") or 0.0)
                if last_try > 0 and (now - last_try) < ARM_COOLDOWN_S:
                    continue

                attempts = int(st.get("arm_attempts") or 0)
                if attempts >= ARM_MAX_ATTEMPTS:
                    desk_log(logging.ERROR, "ARM_ABORT", sym, tid, attempts=attempts)
                    await send_telegram(f"⚠️ ARM_ABORT {sym} tid={tid} attempts={attempts}")
                    async with PENDING_LOCK:
                        PENDING.pop(tid, None)
                    continue

                entry_side = st["entry_side"]
                close_side = st["close_side"]
                direction = st["direction"]
                entry = float(st["entry"])
                sl = float(st["sl"])
                tp1 = float(st["tp1"])

                # -----------------------------------------------------------------
                # Phase 1: not armed -> wait order fill, then place SL + TP1
                # -----------------------------------------------------------------
                if not st["armed"]:
                    detail = await trader.get_order_detail(
                        sym,
                        order_id=st.get("entry_order_id"),
                        client_oid=st.get("entry_client_oid"),
                    )
                    if not trader.is_filled(detail):
                        # still waiting fill
                        async with PENDING_LOCK:
                            if tid in PENDING:
                                PENDING[tid]["last_try_ts"] = now
                        continue

                    qty_total = float(st.get("qty_total") or 0.0)
                    if qty_total <= 0:
                        data = (detail.get("data") or {})
                        qty_total = _safe_float(data.get("size") or data.get("quantity"), 0.0)

                    if qty_total <= 0:
                        desk_log(logging.WARNING, "ARM", sym, tid, step="no_qty_from_fill")
                        async with PENDING_LOCK:
                            if tid in PENDING:
                                PENDING[tid]["arm_attempts"] = attempts + 1
                                PENDING[tid]["last_try_ts"] = now
                        continue

                    tick_meta = await _get_tick_cached(trader, sym)
                    tick_used = _sanitize_tick(sym, entry, tick_meta, tid)

                    # contract meta for qty rounding & minQty
                    meta = await trader.debug_meta(sym)
                    qty_step = _safe_float(meta.get("qtyStep"), 0.0) or _safe_float(meta.get("qtyStep".lower()), 0.0)
                    min_qty = _safe_float(meta.get("minQty"), 0.0)

                    # default split
                    qty_tp1_raw = qty_total * TP1_CLOSE_PCT
                    qty_tp1 = qty_tp1_raw
                    if qty_step and qty_step > 0:
                        qty_tp1 = _ceil_qty(qty_tp1, qty_step)

                    # ensure TP1 notional >= MIN_TP_NOTIONAL_USDT
                    tp1_notional = float(tp1) * float(qty_tp1)
                    if tp1_notional < MIN_TP_NOTIONAL_USDT:
                        need_qty = MIN_TP_NOTIONAL_USDT / max(1e-12, float(tp1))
                        if qty_step and qty_step > 0:
                            need_qty = _ceil_qty(need_qty, qty_step)
                        qty_tp1 = max(qty_tp1, need_qty)

                    # quantize & clamp TP1 qty to available
                    qty_tp1 = min(qty_tp1, qty_total)
                    qty_tp1 = _quantize_qty(qty_tp1, qty_step, min_qty) if (qty_step or min_qty) else float(qty_tp1)

                    qty_runner = max(0.0, qty_total - qty_tp1)
                    if qty_step and qty_step > 0:
                        qty_runner = _quantize_qty(qty_runner, qty_step, 0.0)

                    # if TP1 becomes impossible, keep 100% runner
                    if qty_tp1 <= 0 or (float(tp1) * float(qty_tp1) < MIN_TP_NOTIONAL_USDT):
                        qty_tp1 = 0.0
                        qty_runner = qty_total

                    q_sl = _q_sl_trigger(sl, tick_used, close_side)
                    q_tp1 = _q_tp(tp1, tick_used, close_side)

                    desk_log(
                        logging.INFO, "ARM_PRE", sym, tid,
                        entry_side=entry_side, close_side=close_side, direction=direction,
                        tick_meta=tick_meta, tick_used=tick_used,
                        qty_total=qty_total, qty_tp1=qty_tp1, qty_runner=qty_runner,
                        sl_raw=sl, sl_q=q_sl,
                        tp1_raw=tp1, tp1_q=q_tp1,
                        min_tp_usdt=MIN_TP_NOTIONAL_USDT,
                        qty_step=qty_step, min_qty=min_qty,
                    )

                    # SL first (full size)
                    if not st.get("sl_plan_id"):
                        desk_log(logging.INFO, "SL_SEND", sym, tid, close_side=close_side, trigger_type=_trigger_type_sl(), trigger_q=q_sl, qty=qty_total, tick=tick_used)
                        sl_resp = await trader.place_stop_market_sl(
                            symbol=sym,
                            close_side=close_side.lower(),
                            trigger_price=q_sl,
                            qty=qty_total,
                            client_oid=_oid("sl", tid, attempts),
                            trigger_type=_trigger_type_sl(),
                            tick_hint=tick_used,
                            debug_tag="SL",
                        )
                        if not _is_ok(sl_resp):
                            desk_log(logging.ERROR, "SL_FAIL", sym, tid, code=sl_resp.get("code"), msg=sl_resp.get("msg"), dbg=sl_resp.get("_debug"))
                            desk_log(logging.ERROR, "META_DUMP", sym, tid, meta=await trader.debug_meta(sym))
                            await send_telegram(f"❌ SL_FAIL {sym} code={sl_resp.get('code')} msg={sl_resp.get('msg')}")
                            async with PENDING_LOCK:
                                if tid in PENDING:
                                    PENDING[tid]["arm_attempts"] = attempts + 1
                                    PENDING[tid]["last_try_ts"] = now
                            continue

                        sl_plan_id = (sl_resp.get("data") or {}).get("orderId") or (sl_resp.get("data") or {}).get("planOrderId") or sl_resp.get("orderId")
                        async with PENDING_LOCK:
                            if tid in PENDING:
                                PENDING[tid]["sl_plan_id"] = str(sl_plan_id) if sl_plan_id else "ok"
                                PENDING[tid]["qty_total"] = float(qty_total)
                                PENDING[tid]["qty_tp1"] = float(qty_tp1)
                                PENDING[tid]["qty_runner"] = float(qty_runner)
                                PENDING[tid]["tick_used"] = float(tick_used)
                        desk_log(logging.INFO, "SL_OK", sym, tid, sl=q_sl, planId=sl_plan_id)
                        await send_telegram(f"🛡️ SL_OK {sym} trigger={q_sl:.6g} qty={qty_total:.6g} planId={sl_plan_id}")

                    # TP1 (partial close) — only if qty_tp1 > 0
                    if qty_tp1 > 0 and not st.get("tp1_order_id"):
                        # IMPORTANT: use trade_side='close' to avoid 'open' bug and 22002 spam
                        desk_log(logging.INFO, "TP1_SEND", sym, tid, close_side=close_side, price_q=q_tp1, qty=qty_tp1, tick=tick_used, trade_side="close", reduceOnly=True)
                        tp1_resp = await trader.place_limit(
                            symbol=sym,
                            side=close_side.lower(),
                            price=q_tp1,
                            size=qty_tp1,
                            client_oid=_oid("tp1", tid, attempts),
                            trade_side="close",
                            reduce_only=True,
                            tick_hint=tick_used,
                            debug_tag="TP1",
                        )

                        if not _is_ok(tp1_resp):
                            code = str(tp1_resp.get("code", ""))
                            msg = str(tp1_resp.get("msg") or "")
                            # If position not visible yet, wait (no hard fail, no spam)
                            if code == "22002" or "No position to close" in msg:
                                desk_log(logging.WARNING, "TP1_WAIT_POS", sym, tid, code=code, msg=msg)
                                async with PENDING_LOCK:
                                    if tid in PENDING:
                                        PENDING[tid]["arm_attempts"] = attempts + 1
                                        PENDING[tid]["last_try_ts"] = now
                                continue

                            desk_log(logging.ERROR, "TP1_FAIL", sym, tid, code=code, msg=msg, dbg=tp1_resp.get("_debug"))
                            desk_log(logging.ERROR, "META_DUMP", sym, tid, meta=await trader.debug_meta(sym))
                            await send_telegram(f"❌ TP1_FAIL {sym} code={code} msg={msg}")
                            async with PENDING_LOCK:
                                if tid in PENDING:
                                    PENDING[tid]["arm_attempts"] = attempts + 1
                                    PENDING[tid]["last_try_ts"] = now
                            continue

                        tp1_order_id = (tp1_resp.get("data") or {}).get("orderId") or tp1_resp.get("orderId")
                        async with PENDING_LOCK:
                            if tid in PENDING:
                                PENDING[tid]["tp1_order_id"] = str(tp1_order_id) if tp1_order_id else "ok"
                        desk_log(logging.INFO, "TP1_OK", sym, tid, tp1=q_tp1, orderId=tp1_order_id, qty_tp1=qty_tp1, qty_runner=qty_runner)
                        await send_telegram(f"🎯 TP1_OK {sym} price={q_tp1:.6g} qty={qty_tp1:.6g} orderId={tp1_order_id}")

                    # Armed (SL exists; TP1 may be 0 if impossible)
                    async with PENDING_LOCK:
                        if tid in PENDING:
                            PENDING[tid]["armed"] = True
                            PENDING[tid]["arm_attempts"] = attempts
                            PENDING[tid]["last_try_ts"] = now
                    desk_log(logging.INFO, "ARMED", sym, tid, qty_total=qty_total, close_side=close_side, direction=direction, runner_qty=qty_runner)
                    await send_telegram(f"🧩 ARMED {sym} {direction} runner_qty={qty_runner:.6g}")
                    continue

                # -----------------------------------------------------------------
                # Phase 2: armed -> if TP1 filled then move SL to BE for runner
                # -----------------------------------------------------------------
                if st["armed"] and (not st.get("be_done")) and st.get("tp1_order_id"):
                    tp1_detail = await trader.get_order_detail(sym, order_id=st.get("tp1_order_id"), client_oid=None)
                    if trader.is_filled(tp1_detail):
                        qty_total = float(st.get("qty_total") or 0.0)
                        qty_tp1 = float(st.get("qty_tp1") or 0.0)
                        qty_runner = float(st.get("qty_runner") or max(0.0, qty_total - qty_tp1))
                        tick_used = float(st.get("tick_used") or _estimate_tick_from_price(entry))

                        # compute BE trigger (fee buffer ticks)
                        buf = int(BE_FEE_BUFFER_TICKS or 0)
                        if close_side.upper() == "BUY":
                            # closing short: buy-stop triggers when price >= trigger -> set slightly BELOW entry for small profit
                            be = entry - (buf * tick_used)
                        else:
                            # closing long: sell-stop triggers when price <= trigger -> set slightly ABOVE entry for small profit
                            be = entry + (buf * tick_used)

                        be_q = _q_sl_trigger(be, tick_used, close_side)

                        # cancel old SL if possible
                        old_plan = st.get("sl_plan_id")
                        cancelled = await _maybe_cancel_sl(trader, sym, old_plan)

                        # place new SL at BE for runner only (risk-free)
                        desk_log(logging.INFO, "BE_SEND", sym, tid, old_plan=old_plan, cancelled=cancelled, be_raw=be, be_q=be_q, qty_runner=qty_runner)
                        be_resp = await trader.place_stop_market_sl(
                            symbol=sym,
                            close_side=close_side.lower(),
                            trigger_price=be_q,
                            qty=qty_runner if qty_runner > 0 else qty_total,
                            client_oid=_oid("be", tid, attempts),
                            trigger_type=_trigger_type_sl(),
                            tick_hint=tick_used,
                            debug_tag="BE",
                        )

                        if not _is_ok(be_resp):
                            desk_log(logging.ERROR, "BE_FAIL", sym, tid, code=be_resp.get("code"), msg=be_resp.get("msg"), dbg=be_resp.get("_debug"))
                            await send_telegram(f"❌ BE_FAIL {sym} code={be_resp.get('code')} msg={be_resp.get('msg')}")
                            async with PENDING_LOCK:
                                if tid in PENDING:
                                    PENDING[tid]["arm_attempts"] = attempts + 1
                                    PENDING[tid]["last_try_ts"] = now
                            continue

                        new_plan = (be_resp.get("data") or {}).get("orderId") or (be_resp.get("data") or {}).get("planOrderId") or be_resp.get("orderId")
                        async with PENDING_LOCK:
                            if tid in PENDING:
                                PENDING[tid]["tp1_done"] = True
                                PENDING[tid]["be_done"] = True
                                PENDING[tid]["sl_plan_id"] = str(new_plan) if new_plan else PENDING[tid]["sl_plan_id"]
                                PENDING[tid]["last_try_ts"] = now

                        desk_log(logging.INFO, "BE_OK", sym, tid, be=be_q, planId=new_plan, qty_runner=qty_runner)
                        await send_telegram(f"🟦 TP1_FILLED {sym}\n➡️ SL moved to BE={be_q:.6g} (runner_qty={qty_runner:.6g})")
                        # keep pending (runner managed by BE)
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
    await send_telegram(f"🤖 Bot started (Bitget Desk)\nscan_interval={SCAN_INTERVAL_MIN}m dry_run={DRY_RUN}")

    while True:
        t0 = time.time()
        try:
            await scan_once(client, analyzer, trader)
        except Exception:
            logger.exception("SCAN ERROR")
            await send_telegram("❌ SCAN ERROR (see logs)")

        dt = time.time() - t0
        sleep_s = max(1, int(float(SCAN_INTERVAL_MIN) * 60 - dt))
        await asyncio.sleep(sleep_s)

if __name__ == "__main__":
    asyncio.run(start_scanner())
