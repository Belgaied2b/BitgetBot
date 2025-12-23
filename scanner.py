# =====================================================================
# scanner.py — Telegram ON + TP2 OFF (runner) + SL -> BE après TP1
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
    BE_FEE_BUFFER_TICKS,   # nb de ticks “en faveur” pour BE
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

ARM_MAX_ATTEMPTS = 10
ARM_COOLDOWN_S = 10.0

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
# Desk logging
# =====================================================================

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
    return f"{prefix}-{tid}-{attempt}-{int(time.time() * 1000)}"


# =====================================================================
# Telegram
# =====================================================================

async def send_telegram(msg: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    import requests
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }

    def _do():
        try:
            requests.post(url, json=payload, timeout=8)
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


def _trigger_type_sl() -> str:
    s = (STOP_TRIGGER_TYPE_SL or "MP").upper()
    return "mark_price" if s == "MP" else "fill_price"


def _estimate_tick_from_price(price: float) -> float:
    p = abs(float(price))
    if p >= 10000:
        return 1.0
    elif p >= 1000:
        return 0.1
    elif p >= 100:
        return 0.01
    elif p >= 10:
        return 0.001
    elif p >= 1:
        return 0.0001
    elif p >= 0.1:
        return 0.00001
    elif p >= 0.01:
        return 0.000001
    else:
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
    # BUY -> floor, SELL -> ceil (évite “price not in step”)
    if tick <= 0:
        return float(price)
    return _q_floor(price, tick) if (side or "").upper() == "BUY" else _q_ceil(price, tick)


def _quantize_sl_tp(direction: str, sl: float, tp1: float, tick: float) -> Tuple[float, float]:
    """
    Quantize “safe” par direction (indépendant du close_side).
    LONG: SL floor (en dessous), TP1 ceil (au dessus)
    SHORT: SL ceil (au dessus), TP1 floor (en dessous)
    """
    if (direction or "").upper() == "LONG":
        q_sl = _q_floor(sl, tick)
        q_tp1 = _q_ceil(tp1, tick)
    else:
        q_sl = _q_ceil(sl, tick)
        q_tp1 = _q_floor(tp1, tick)
    return q_sl, q_tp1


def _be_trigger(entry: float, direction: str, tick: float) -> float:
    buf = max(0, int(BE_FEE_BUFFER_TICKS or 0))
    if (direction or "").upper() == "LONG":
        return float(entry + buf * tick)
    else:
        return float(entry - buf * tick)


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


async def _get_pos_mode(trader: BitgetTrader) -> str:
    # best-effort
    try:
        if hasattr(trader, "get_position_mode"):
            m = await trader.get_position_mode()
            if m:
                return str(m)
        if hasattr(trader, "position_mode"):
            return str(getattr(trader, "position_mode"))
    except Exception:
        pass
    return "one_way"


def _close_side_for_api(entry_side: str, pos_mode: str) -> str:
    """
    Important: dans tes logs actuels, Bitget “hedge” marche avec close_side = entry_side.
    En one-way, on garde l’opposé classique.
    """
    s = (entry_side or "").upper()
    pm = (pos_mode or "").lower()
    if pm == "hedge":
        return s
    return "SELL" if s == "BUY" else "BUY"


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
        setup = result.get("setup_type")

        if entry <= 0 or sl <= 0 or tp1 <= 0 or rr <= 0:
            await stats.inc("skips", 1)
            await stats.add_reason("missing_exits")
            return

        # EXIT DEBUG (raw) — TP2 OFF
        desk_log(logging.INFO, "EXITS", symbol, tid, side=side, entry=entry, sl=sl, tp1=tp1, rr=rr, setup=setup)

        fp = make_fingerprint(symbol, side, entry, sl, tp1, extra=setup, precision=6)
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

        pos_mode = await _get_pos_mode(trader)
        desk_log(logging.INFO, "VALID", symbol, tid, side=side, setup=setup, rr=rr, inst=inst_score, pos_mode=pos_mode)

        # Telegram: SIGNAL
        try:
            msg = (
                f"✅ *SIGNAL* `{str(symbol).upper()}`\n"
                f"Setup: *{setup}* | Mode: *{pos_mode}*\n"
                f"Side: *{side}* | RR: *{rr:.2f}* | Inst: *{inst_score}*\n"
                f"Entry: `{entry:.6g}`\n"
                f"SL: `{sl:.6g}`\n"
                f"TP1: `{tp1:.6g}`\n"
                f"TID: `{tid}`"
            )
            await send_telegram(msg)
        except Exception as e:
            desk_log(logging.ERROR, "TG_FAIL", symbol, tid, err=str(e))

        DUP_GUARD.mark(fp)

        if DRY_RUN:
            return

        # Budget
        try:
            await asyncio.wait_for(order_budget.acquire(), timeout=0.01)
        except asyncio.TimeoutError:
            await stats.add_reason("budget:max_orders_per_scan")
            return

        # Tick + entry quantize
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
            dbg = entry_resp.get("_debug") or {}
            desk_log(
                logging.ERROR, "ENTRY_FAIL", symbol, tid,
                code=entry_resp.get("code"),
                msg=entry_resp.get("msg"),
                http=entry_resp.get("_http_status"),
                dbg=dbg,
            )
            desk_log(logging.ERROR, "META_DUMP", symbol, tid, meta=await trader.debug_meta(symbol))
            try:
                order_budget.release()
            except Exception:
                pass
            return

        entry_order_id = (entry_resp.get("data") or {}).get("orderId") or entry_resp.get("orderId")
        qty_total = _safe_float(entry_resp.get("qty"), 0.0)

        desk_log(logging.INFO, "ENTRY_OK", symbol, tid, orderId=entry_order_id, qty=qty_total)

        close_side = _close_side_for_api(side, pos_mode)

        async with PENDING_LOCK:
            PENDING[tid] = {
                "symbol": str(symbol).upper(),
                "entry_side": side.upper(),
                "close_side": close_side.upper(),
                "direction": direction,
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
                "tp1_done": False,
                "be_done": False,
                "created_ts": time.time(),
                "arm_attempts": 0,
                "last_arm_fail_ts": 0.0,
            }

        desk_log(logging.INFO, "PENDING_NEW", symbol, tid, entry_side=side, close_side=close_side, direction=direction, pos_mode=pos_mode)

    except Exception as e:
        desk_log(logging.ERROR, "ERR", symbol, tid, where="process_symbol", err=str(e))
        logger.exception("[%s] process_symbol error: %s", symbol, e)


# =====================================================================
# WATCHER
# =====================================================================

async def _try_cancel_sl(trader: BitgetTrader, symbol: str, plan_id: str) -> bool:
    """
    Best-effort cancel plan SL. S’adapte à ton BitgetTrader si les méthodes existent.
    """
    if not plan_id:
        return False
    try:
        if hasattr(trader, "cancel_plan_order"):
            resp = await trader.cancel_plan_order(symbol=symbol, plan_id=plan_id)
            return _is_ok(resp)
        if hasattr(trader, "cancel_plan"):
            resp = await trader.cancel_plan(symbol=symbol, plan_id=plan_id)
            return _is_ok(resp)
        if hasattr(trader, "cancel_order"):
            resp = await trader.cancel_order(symbol=symbol, order_id=plan_id)
            return _is_ok(resp)
    except Exception:
        pass
    return False


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
                entry_side = st["entry_side"]
                close_side = st["close_side"]
                direction = st.get("direction", _side_to_direction(entry_side))
                pos_mode = st.get("pos_mode", "one_way")

                entry = float(st["entry"])
                sl = float(st["sl"])
                tp1 = float(st["tp1"])

                # ----- NOT ARMED: wait fill -> place SL + TP1 only -----
                if not st["armed"]:
                    last_fail = float(st.get("last_arm_fail_ts") or 0.0)
                    if last_fail > 0 and (time.time() - last_fail) < ARM_COOLDOWN_S:
                        continue

                    attempts = int(st.get("arm_attempts") or 0)
                    if attempts >= ARM_MAX_ATTEMPTS:
                        desk_log(logging.ERROR, "ARM_ABORT", sym, tid, attempts=attempts)
                        async with PENDING_LOCK:
                            PENDING.pop(tid, None)
                        continue

                    detail = await trader.get_order_detail(
                        sym,
                        order_id=st.get("entry_order_id"),
                        client_oid=st.get("entry_client_oid"),
                    )
                    if not trader.is_filled(detail):
                        continue

                    qty_total = float(st.get("qty_total") or 0.0)
                    if qty_total <= 0:
                        data = (detail.get("data") or {})
                        qty_total = float(data.get("size") or data.get("quantity") or 0.0)

                    if qty_total <= 0:
                        st["arm_attempts"] = attempts + 1
                        st["last_arm_fail_ts"] = time.time()
                        desk_log(logging.WARNING, "ARM", sym, tid, step="no_qty_from_fill")
                        continue

                    tick_meta = await _get_tick_cached(trader, sym)
                    tick_used = _sanitize_tick(sym, entry, tick_meta, tid)

                    qty_tp1 = qty_total * TP1_CLOSE_PCT

                    q_sl, q_tp1 = _quantize_sl_tp(direction, sl, tp1, tick_used)

                    desk_log(
                        logging.INFO, "ARM_PRE", sym, tid,
                        pos_mode=pos_mode,
                        entry_side=entry_side,
                        close_side=close_side,
                        direction=direction,
                        tick_meta=tick_meta,
                        tick_used=tick_used,
                        qty_total=qty_total,
                        qty_tp1=qty_tp1,
                        sl_raw=sl,
                        sl_q=q_sl,
                        tp1_raw=tp1,
                        tp1_q=q_tp1,
                    )

                    # SL first
                    if not st.get("sl_plan_id"):
                        desk_log(logging.INFO, "SL_SEND", sym, tid,
                                 close_side=close_side, trigger_type=_trigger_type_sl(),
                                 trigger_raw=sl, trigger_q=q_sl, qty=qty_total, tick=tick_used)

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
                            st["arm_attempts"] = attempts + 1
                            st["last_arm_fail_ts"] = time.time()
                            desk_log(logging.ERROR, "SL_FAIL", sym, tid, code=sl_resp.get("code"), msg=sl_resp.get("msg"), dbg=sl_resp.get("_debug"))
                            desk_log(logging.ERROR, "META_DUMP", sym, tid, meta=await trader.debug_meta(sym))
                            continue

                        sl_plan_id = (sl_resp.get("data") or {}).get("orderId") or (sl_resp.get("data") or {}).get("planOrderId") or sl_resp.get("orderId")
                        async with PENDING_LOCK:
                            if tid in PENDING:
                                PENDING[tid]["sl_plan_id"] = str(sl_plan_id) if sl_plan_id else "ok"
                                PENDING[tid]["qty_total"] = qty_total
                        desk_log(logging.INFO, "SL_OK", sym, tid, sl=q_sl, planId=sl_plan_id, dbg=sl_resp.get("_debug"))

                    # TP1 only (TP2 OFF)
                    if not st.get("tp1_order_id"):
                        desk_log(logging.INFO, "TP1_SEND", sym, tid,
                                 close_side=close_side, price_raw=tp1, price_q=q_tp1, qty=qty_tp1, tick=tick_used)

                        tp1_resp = await trader.place_reduce_limit_tp(
                            symbol=sym,
                            close_side=close_side.lower(),
                            price=q_tp1,
                            qty=qty_tp1,
                            client_oid=_oid("tp1", tid, attempts),
                            tick_hint=tick_used,
                            debug_tag="TP1",
                        )

                        # clamp band
                        if (not _is_ok(tp1_resp)) and str(tp1_resp.get("code")) == "22047":
                            mn, mx = _parse_band(str(tp1_resp.get("msg") or ""))
                            clamped = _clamp_and_quantize(q_tp1, tick_used, mn, mx)
                            desk_log(logging.WARNING, "TP1_22047", sym, tid, mn=mn, mx=mx, before=q_tp1, after=clamped, tick=tick_used)
                            if clamped is None:
                                st["arm_attempts"] = attempts + 1
                                st["last_arm_fail_ts"] = time.time()
                                continue
                            tp1_resp = await trader.place_reduce_limit_tp(
                                symbol=sym,
                                close_side=close_side.lower(),
                                price=clamped,
                                qty=qty_tp1,
                                client_oid=_oid("tp1", tid, attempts + 1),
                                tick_hint=tick_used,
                                debug_tag="TP1",
                            )

                        # “No position to close” => pas prêt côté exchange, on retente plus tard
                        if (not _is_ok(tp1_resp)) and str(tp1_resp.get("code")) == "22002":
                            st["arm_attempts"] = attempts + 1
                            st["last_arm_fail_ts"] = time.time()
                            desk_log(logging.WARNING, "TP1_WAIT_POS", sym, tid, code=tp1_resp.get("code"), msg=tp1_resp.get("msg"))
                            continue

                        if not _is_ok(tp1_resp):
                            st["arm_attempts"] = attempts + 1
                            st["last_arm_fail_ts"] = time.time()
                            desk_log(logging.ERROR, "TP1_FAIL", sym, tid, code=tp1_resp.get("code"), msg=tp1_resp.get("msg"), dbg=tp1_resp.get("_debug"))
                            desk_log(logging.ERROR, "META_DUMP", sym, tid, meta=await trader.debug_meta(sym))
                            continue

                        tp1_order_id = (tp1_resp.get("data") or {}).get("orderId") or tp1_resp.get("orderId")
                        async with PENDING_LOCK:
                            if tid in PENDING:
                                PENDING[tid]["tp1_order_id"] = str(tp1_order_id) if tp1_order_id else "ok"
                                PENDING[tid]["qty_tp1"] = float(tp1_resp.get("qty") or qty_tp1)
                        desk_log(logging.INFO, "TP1_OK", sym, tid, tp1=q_tp1, orderId=tp1_order_id, dbg=tp1_resp.get("_debug"))

                    async with PENDING_LOCK:
                        if tid in PENDING:
                            PENDING[tid]["armed"] = True

                    desk_log(logging.INFO, "ARMED", sym, tid, qty_total=qty_total, close_side=close_side, direction=direction, pos_mode=pos_mode)

                    # Telegram: ARMED
                    try:
                        msg = (
                            f"🛡️ *ARMED* `{sym}`\n"
                            f"Mode: *{pos_mode}* | Side: *{entry_side}* | Dir: *{direction}*\n"
                            f"Entry: `{entry:.6g}`\n"
                            f"SL: `{q_sl:.6g}`\n"
                            f"TP1: `{q_tp1:.6g}` (qty `{qty_tp1:.6g}`)\n"
                            f"TID: `{tid}`"
                        )
                        await send_telegram(msg)
                    except Exception as e:
                        desk_log(logging.ERROR, "TG_ARM_FAIL", sym, tid, err=str(e))

                    continue

                # ----- ARMED: monitor TP1 fill -> move SL to BE -----
                if st["armed"] and (not st.get("be_done")) and st.get("tp1_order_id"):
                    try:
                        tp1_detail = await trader.get_order_detail(sym, order_id=st.get("tp1_order_id"), client_oid=None)
                        if trader.is_filled(tp1_detail):
                            st["tp1_done"] = True

                            # compute BE trigger
                            tick_meta = await _get_tick_cached(trader, sym)
                            tick_used = _sanitize_tick(sym, entry, tick_meta, tid)

                            be_raw = _be_trigger(entry, direction, tick_used)

                            # quantize BE SL like a normal SL (LONG floor / SHORT ceil)
                            if direction == "LONG":
                                be_q = _q_floor(be_raw, tick_used)
                            else:
                                be_q = _q_ceil(be_raw, tick_used)

                            old_plan = str(st.get("sl_plan_id") or "")
                            cancelled = await _try_cancel_sl(trader, sym, old_plan)

                            desk_log(logging.INFO, "BE_PRE", sym, tid,
                                     direction=direction, entry=entry, be_raw=be_raw, be_q=be_q,
                                     old_sl_plan=old_plan, cancelled=cancelled, tick=tick_used)

                            # place new SL at BE for remaining position (Bitget stop is for full size; reduce-only is in API)
                            qty_total = float(st.get("qty_total") or 0.0)
                            be_resp = await trader.place_stop_market_sl(
                                symbol=sym,
                                close_side=close_side.lower(),
                                trigger_price=be_q,
                                qty=qty_total,
                                client_oid=_oid("be", tid, 0),
                                trigger_type=_trigger_type_sl(),
                                tick_hint=tick_used,
                                debug_tag="SL_BE",
                            )

                            if not _is_ok(be_resp):
                                desk_log(logging.ERROR, "BE_FAIL", sym, tid, code=be_resp.get("code"), msg=be_resp.get("msg"), dbg=be_resp.get("_debug"))
                                desk_log(logging.ERROR, "META_DUMP", sym, tid, meta=await trader.debug_meta(sym))
                                continue

                            new_plan_id = (be_resp.get("data") or {}).get("orderId") or (be_resp.get("data") or {}).get("planOrderId") or be_resp.get("orderId")
                            async with PENDING_LOCK:
                                if tid in PENDING:
                                    PENDING[tid]["sl_plan_id"] = str(new_plan_id) if new_plan_id else "ok"
                                    PENDING[tid]["be_done"] = True
                                    PENDING[tid]["sl"] = float(be_q)

                            desk_log(logging.INFO, "BE_OK", sym, tid, be=be_q, planId=new_plan_id, dbg=be_resp.get("_debug"))

                            # Telegram: BE
                            try:
                                msg = (
                                    f"🟢 *SL -> BE* `{sym}`\n"
                                    f"Dir: *{direction}* | Mode: *{pos_mode}*\n"
                                    f"Entry: `{entry:.6g}`\n"
                                    f"BE SL: `{be_q:.6g}`\n"
                                    f"TID: `{tid}`"
                                )
                                await send_telegram(msg)
                            except Exception as e:
                                desk_log(logging.ERROR, "TG_BE_FAIL", sym, tid, err=str(e))

                    except Exception as e:
                        desk_log(logging.ERROR, "BE_ERR", sym, tid, err=str(e))

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
