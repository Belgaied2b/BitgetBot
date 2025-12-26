# =====================================================================
# scanner.py — TP2 OFF (runner) + BE after TP1 + Telegram ON
# Desk-lead hardened runner:
# - Robust fetch (retry + timeout) + better reject stats
# - Fix SL cancel at BE (BitgetTrader.cancel_plan_orders)
# - Pending state persistence (restart-safe for arming SL/TP/BE)
# - Telegram hardened (escape markdown + non-blocking)
# - Safer watcher state machine + isolated per-tid error handling
# =====================================================================

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import uuid
from collections import Counter
from typing import Any, Dict, Tuple, Optional, List

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
from retry_utils import retry_async

logger = logging.getLogger(__name__)

# =====================================================================
# Runtime globals
# =====================================================================

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

# API timeouts (seconds)
FETCH_TIMEOUT_S = 12.0
ANALYZE_TIMEOUT_S = 20.0
META_TIMEOUT_S = 10.0
ORDER_TIMEOUT_S = 15.0
DETAIL_TIMEOUT_S = 10.0
TICK_TIMEOUT_S = 8.0

MIN_TP_USDT_FALLBACK = 5.0
_MIN_USDT_RE = re.compile(r"minimum amount\s*([0-9]*\.?[0-9]+)\s*USDT", re.IGNORECASE)

# ===== PRICE BAND =====
_MAX_RE = re.compile(r"maximum price limit:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_MIN_RE = re.compile(r"minimum price limit:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

# =====================================================================
# Pending persistence (restart-safe arming)
# =====================================================================

PENDING_STATE_FILE = os.getenv("PENDING_STATE_FILE", "pending_state.json")
PENDING_SAVE_THROTTLE_S = 4.0
_PENDING_LAST_SAVE_TS = 0.0

# Watcher state
PENDING: Dict[str, Dict[str, Any]] = {}
PENDING_LOCK = asyncio.Lock()
WATCHER_TASK: Optional[asyncio.Task] = None

# Tick cache
TICK_CACHE: Dict[str, float] = {}
TICK_LOCK = asyncio.Lock()


def _pending_serializable(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only JSON-safe keys.
    """
    keep = [
        "symbol",
        "entry_side",
        "direction",
        "close_side",
        "pos_mode",
        "entry",
        "sl",
        "tp1",
        "qty_total",
        "qty_tp1",
        "qty_rem",
        "entry_order_id",
        "entry_client_oid",
        "sl_plan_id",
        "tp1_order_id",
        "armed",
        "be_done",
        "created_ts",
        "arm_attempts",
        "last_arm_fail_ts",
        "sl_inflight",
        "tp1_inflight",
        "tp1_min_usdt",
        "qty_step",
        "min_qty",
        "q_tp1",
        "q_sl",
    ]
    out: Dict[str, Any] = {}
    for k in keep:
        if k in state:
            out[k] = state[k]
    return out


async def _pending_save(force: bool = False) -> None:
    global _PENDING_LAST_SAVE_TS
    now = time.time()
    if (not force) and (now - _PENDING_LAST_SAVE_TS) < PENDING_SAVE_THROTTLE_S:
        return
    async with PENDING_LOCK:
        snap = {tid: _pending_serializable(st) for tid, st in PENDING.items()}
    try:
        tmp = f"{PENDING_STATE_FILE}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False, indent=2)
        os.replace(tmp, PENDING_STATE_FILE)
        _PENDING_LAST_SAVE_TS = now
    except Exception as e:
        logger.warning("pending_save failed: %s", e)


async def _pending_load() -> None:
    if not os.path.exists(PENDING_STATE_FILE):
        return
    try:
        with open(PENDING_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return
        async with PENDING_LOCK:
            PENDING.clear()
            for tid, st in data.items():
                if not isinstance(st, dict):
                    continue
                # minimal sanity
                sym = str(st.get("symbol") or "").upper()
                if not sym:
                    continue
                PENDING[str(tid)] = st
        logger.info("[BOOT] restored pending=%d from %s", len(PENDING), PENDING_STATE_FILE)
    except Exception as e:
        logger.warning("pending_load failed: %s", e)


# =====================================================================
# Price band helpers
# =====================================================================

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

    # keep buffer vs band edges
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
    return f"{prefix}-{tid}-{attempt}-{int(time.time()*1000)}"


# =====================================================================
# Telegram (hardened)
# =====================================================================

_TG_ESC_RE = re.compile(r"([_*\[\]()~`>#+\-=|{}.!])")


def _tg_escape(text: str) -> str:
    try:
        return _TG_ESC_RE.sub(r"\\\1", str(text))
    except Exception:
        return str(text)


async def send_telegram(msg: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    # Use requests in a background thread (simple + reliable)
    import requests

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": "MarkdownV2",
        "disable_web_page_preview": True,
    }

    def _do():
        try:
            requests.post(url, json=payload, timeout=8)
        except Exception as e:
            logger.error("Telegram error: %s", e)

    await asyncio.to_thread(_do)


def _fmt_side(side: str) -> str:
    return "🟢 LONG" if (side or "").upper() == "BUY" else "🔴 SHORT"


def _fmt_num(x: float) -> str:
    try:
        return f"{float(x):.6g}"
    except Exception:
        return str(x)


def _mk_signal_msg(
    symbol: str,
    tid: str,
    side: str,
    setup: str,
    entry: float,
    sl: float,
    tp1: float,
    rr: float,
    inst: int,
    entry_type: str,
    pos_mode: str,
) -> str:
    # MarkdownV2 => escape everything dynamic
    return (
        f"*📌 SIGNAL*\n"
        f"*{_tg_escape(symbol)}* — {_tg_escape(_fmt_side(side))}\n"
        f"setup: `{_tg_escape(setup)}` | inst: `{inst}` | entry_type: `{_tg_escape(entry_type)}` | pos_mode: `{_tg_escape(pos_mode)}`\n"
        f"entry: `{_tg_escape(_fmt_num(entry))}`\n"
        f"SL: `{_tg_escape(_fmt_num(sl))}`\n"
        f"TP1: `{_tg_escape(_fmt_num(tp1))}` \\(runner ensuite\\)\n"
        f"RR: `{_tg_escape(_fmt_num(rr))}`\n"
        f"tid: `{_tg_escape(tid)}`"
    )


def _mk_exec_msg(tag: str, symbol: str, tid: str, **kv: Any) -> str:
    base = f"*⚙️ {_tg_escape(tag)}* — *{_tg_escape(symbol)}*\n" + f"tid: `{_tg_escape(tid)}`\n"
    lines: List[str] = []
    for k, v in kv.items():
        if v is None:
            continue
        if isinstance(v, float):
            lines.append(f"{_tg_escape(k)}: `{_tg_escape(f'{v:.6g}')}`")
        else:
            lines.append(f"{_tg_escape(k)}: `{_tg_escape(v)}`")
    return base + "\n".join(lines)


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


def _direction_from_side(side: str) -> str:
    return "LONG" if (side or "").upper() == "BUY" else "SHORT"


def _close_side_from_direction(direction: str) -> str:
    return "SELL" if direction == "LONG" else "BUY"


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
    if tick <= 0:
        return float(price)
    return _q_floor(price, tick) if (side or "").upper() == "BUY" else _q_ceil(price, tick)


def _q_sl(sl: float, tick: float, direction: str) -> float:
    return _q_floor(sl, tick) if direction == "LONG" else _q_ceil(sl, tick)


def _q_tp_limit(tp: float, tick: float, direction: str) -> float:
    return _q_floor(tp, tick) if direction == "LONG" else _q_ceil(tp, tick)


def _q_qty_floor(qty: float, step: float) -> float:
    if step <= 0:
        return float(qty)
    return float(math.floor(qty / step) * step)


def _q_qty_ceil(qty: float, step: float) -> float:
    if step <= 0:
        return float(qty)
    return float(math.ceil(qty / step) * step)


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


async def _detect_pos_mode(trader: BitgetTrader) -> str:
    # BitgetTrader in this repo doesn't expose get_position_mode; keep conservative.
    return "one_way"


def _tp_trade_side(pos_mode: str) -> str:
    pm = (pos_mode or "").lower()
    # In hedge mode: tradeSide should be close; in one-way: keep open + reduceOnly
    return "close" if "hedge" in pm else "open"


async def _get_tick_cached(trader: BitgetTrader, symbol: str) -> float:
    sym = str(symbol).upper()
    async with TICK_LOCK:
        if sym in TICK_CACHE and TICK_CACHE[sym] > 0:
            return TICK_CACHE[sym]
    try:
        t = await asyncio.wait_for(trader.get_tick(sym), timeout=TICK_TIMEOUT_S)
    except Exception:
        t = 0.0
    t = float(t or 0.0)
    async with TICK_LOCK:
        if t > 0:
            TICK_CACHE[sym] = t
    return t


def _order_state(detail: Dict[str, Any]) -> str:
    try:
        if not isinstance(detail, dict):
            return "unknown"
        data = detail.get("data") or {}
        st = str(data.get("state") or data.get("status") or "").lower()
        return st or "unknown"
    except Exception:
        return "unknown"


# =====================================================================
# Fetch
# =====================================================================

async def _fetch_dfs(client, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    async def _do():
        df_h1 = await client.get_klines_df(symbol, TF_H1, CANDLE_LIMIT)
        df_h4 = await client.get_klines_df(symbol, TF_H4, CANDLE_LIMIT)
        return df_h1, df_h4

    try:
        return await asyncio.wait_for(retry_async(_do, retries=2, base_delay=0.4), timeout=FETCH_TIMEOUT_S)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


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
        # --- fetch
        async with fetch_sem:
            t0 = time.time()
            df_h1, df_h4 = await _fetch_dfs(client, symbol)
            fetch_ms = int((time.time() - t0) * 1000)

        if df_h1 is None or df_h4 is None or getattr(df_h1, "empty", True) or getattr(df_h4, "empty", True):
            await stats.inc("skips", 1)
            await stats.add_reason("fetch_empty")
            return
        if len(df_h1) < 80 or len(df_h4) < 80:
            await stats.inc("skips", 1)
            await stats.add_reason("fetch_short")
            return

        # --- analyze
        t1 = time.time()
        try:
            result = await asyncio.wait_for(analyzer.analyze(symbol, df_h1, df_h4, macro={}), timeout=ANALYZE_TIMEOUT_S)
        except asyncio.TimeoutError:
            result = {"valid": False, "reject_reason": "analyze_timeout"}
        analyze_ms = int((time.time() - t1) * 1000)

        if not result or not isinstance(result, dict) or not result.get("valid"):
            reason = _extract_reject_reason(result)
            if await stats.take_reject_debug_slot():
                has_fields = "Y" if (isinstance(result, dict) and _has_key_fields_for_trade(result)) else "N"
                desk_log(logging.INFO, "REJ", symbol, tid, fetch_ms=fetch_ms, analyze_ms=analyze_ms, reason=reason, has_fields=has_fields)
            await stats.inc("rejects", 1)
            await stats.add_reason(reason)
            return

        # --- validate payload
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

        desk_log(logging.INFO, "EXITS", symbol, tid, side=side, entry=entry, sl=sl, tp1=tp1, rr=rr, setup=setup, entry_type=entry_type, fetch_ms=fetch_ms, analyze_ms=analyze_ms)

        if entry <= 0 or sl <= 0 or tp1 <= 0 or rr <= 0:
            await stats.inc("skips", 1)
            await stats.add_reason("missing_exits")
            return

        # --- duplicate guard
        fp = make_fingerprint(symbol, side, entry, sl, tp1, extra=setup, precision=6)
        if DUP_GUARD.is_duplicate(fp):
            await stats.inc("duplicates", 1)
            await stats.add_reason("duplicate")
            return

        direction = _direction_from_side(side)
        close_side = _close_side_from_direction(direction)
        notional = float(MARGIN_USDT) * float(LEVERAGE)

        inst = result.get("institutional") or {}
        inst_score = int(inst.get("institutional_score") or 0)

        # --- risk gate
        allowed, rreason = RISK.can_trade(
            symbol=symbol,
            side=direction,
            notional=notional,
            rr=rr if rr > 0 else None,
            inst_score=inst_score,
            commitment=None,
        )
        if not allowed:
            await stats.inc("risk_rejects", 1)
            await stats.add_reason(f"risk:{rreason}")
            return

        pos_mode = await _detect_pos_mode(trader)

        await stats.inc("valids", 1)
        desk_log(logging.INFO, "VALID", symbol, tid, side=side, setup=setup, rr=rr, inst=inst_score, entry_type=entry_type, pos_mode=pos_mode)

        # Telegram signal first (desk wants visibility)
        await send_telegram(_mk_signal_msg(str(symbol).upper(), tid, side, setup, entry, sl, tp1, rr, inst_score, entry_type, pos_mode))

        # Mark duplicate after broadcasting
        DUP_GUARD.mark(fp)

        if DRY_RUN:
            return

        # --- per-scan order budget (hard limit)
        try:
            await asyncio.wait_for(order_budget.acquire(), timeout=0.02)
        except asyncio.TimeoutError:
            await stats.add_reason("budget:max_orders_per_scan")
            await send_telegram(_mk_exec_msg("EXEC_SKIPPED", str(symbol).upper(), tid, reason="max_orders_per_scan"))
            return

        # --- meta/tick
        try:
            meta_dbg = await asyncio.wait_for(trader.debug_meta(symbol), timeout=META_TIMEOUT_S)
        except Exception:
            meta_dbg = {}

        tick_meta = await _get_tick_cached(trader, symbol)
        tick_used = _sanitize_tick(symbol, entry, tick_meta, tid)
        q_entry = _q_entry(entry, tick_used, side)

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
            meta_qtyStep=meta_dbg.get("qtyStep"),
            meta_minQty=meta_dbg.get("minQty"),
        )

        if q_entry <= 0:
            await stats.inc("exec_failed", 1)
            await stats.add_reason("entry_q_zero")
            try:
                order_budget.release()
            except Exception:
                pass
            return

        # --- place entry (idempotent clientOid)
        await stats.inc("exec_sent", 1)
        desk_log(logging.INFO, "EXEC", symbol, tid, action="entry_send", entry=q_entry, notional=round(notional, 2))
        await send_telegram(_mk_exec_msg("ENTRY_SEND", str(symbol).upper(), tid, entry=q_entry, notional=round(notional, 2)))

        entry_client_oid = f"entry-{tid}"

        async def _place_entry():
            return await trader.place_limit(
                symbol=symbol,
                side=side.lower(),
                price=q_entry,
                size=None,
                client_oid=entry_client_oid,
                trade_side="open",
                reduce_only=False,
                tick_hint=tick_used,
                debug_tag="ENTRY",
            )

        try:
            entry_resp = await asyncio.wait_for(retry_async(_place_entry, retries=2, base_delay=0.35), timeout=ORDER_TIMEOUT_S)
        except Exception as e:
            entry_resp = {"code": "EXC", "msg": str(e)}

        if not _is_ok(entry_resp):
            await stats.inc("exec_failed", 1)
            desk_log(logging.ERROR, "ENTRY_FAIL", symbol, tid, code=entry_resp.get("code"), msg=entry_resp.get("msg"), dbg=(entry_resp.get("_debug") or {}))
            try:
                md = await asyncio.wait_for(trader.debug_meta(symbol), timeout=META_TIMEOUT_S)
            except Exception:
                md = {}
            desk_log(logging.ERROR, "META_DUMP", symbol, tid, meta=md)
            await send_telegram(_mk_exec_msg("ENTRY_FAIL", str(symbol).upper(), tid, code=entry_resp.get("code"), msg=entry_resp.get("msg")))
            # budget release => allow another symbol to execute this scan
            try:
                order_budget.release()
            except Exception:
                pass
            return

        entry_order_id = (entry_resp.get("data") or {}).get("orderId") or entry_resp.get("orderId")
        qty_total = _safe_float(entry_resp.get("qty"), 0.0)

        desk_log(logging.INFO, "ENTRY_OK", symbol, tid, orderId=entry_order_id, qty=qty_total)

        # --- create pending
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
                "qty_rem": 0.0,
                "entry_order_id": str(entry_order_id) if entry_order_id else None,
                "entry_client_oid": entry_client_oid,
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
                "q_tp1": None,
                "q_sl": None,
            }

        await _pending_save(force=True)
        desk_log(logging.INFO, "PENDING_NEW", symbol, tid, entry_side=side.upper(), close_side=close_side, direction=direction, pos_mode=pos_mode)

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
        except Exception:
            continue

        if not items:
            continue

        dirty = False

        for tid, st in items:
            try:
                sym = str(st.get("symbol") or "").upper()
                if not sym:
                    continue

                direction = str(st.get("direction") or "")
                close_side = str(st.get("close_side") or "")
                pos_mode = str(st.get("pos_mode") or "one_way")

                entry = float(st.get("entry") or 0.0)
                sl = float(st.get("sl") or 0.0)
                tp1 = float(st.get("tp1") or 0.0)

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
                    dirty = True
                    await send_telegram(_mk_exec_msg("ARM_ABORT", sym, tid, attempts=attempts))
                    continue

                # ------------------------------------------------------------------
                # Phase 1: wait fill then arm SL + TP1
                # ------------------------------------------------------------------
                if not bool(st.get("armed", False)):
                    # Order detail (by orderId or clientOid)
                    try:
                        detail = await asyncio.wait_for(
                            trader.get_order_detail(sym, order_id=st.get("entry_order_id"), client_oid=st.get("entry_client_oid")),
                            timeout=DETAIL_TIMEOUT_S,
                        )
                    except Exception as e:
                        st["arm_attempts"] = attempts + 1
                        st["last_arm_fail_ts"] = time.time()
                        desk_log(logging.WARNING, "ARM", sym, tid, step="detail_exc", err=str(e))
                        dirty = True
                        continue

                    state = _order_state(detail)
                    if state in {"cancelled", "canceled", "rejected", "fail", "failed", "expired"}:
                        desk_log(logging.WARNING, "ARM_DROP", sym, tid, reason=f"entry_state={state}")
                        async with PENDING_LOCK:
                            PENDING.pop(tid, None)
                        dirty = True
                        continue

                    if not trader.is_filled(detail):
                        continue

                    qty_total = float(st.get("qty_total") or 0.0)
                    if qty_total <= 0:
                        data = (detail.get("data") or {})
                        qty_total = float(data.get("size") or data.get("quantity") or data.get("qty") or 0.0)

                    if qty_total <= 0:
                        st["arm_attempts"] = attempts + 1
                        st["last_arm_fail_ts"] = time.time()
                        desk_log(logging.WARNING, "ARM", sym, tid, step="no_qty_from_fill")
                        dirty = True
                        continue

                    tick_meta = await _get_tick_cached(trader, sym)
                    tick_used = _sanitize_tick(sym, entry, tick_meta, tid)

                    # quantized exits
                    q_sl = _q_sl(sl, tick_used, direction)
                    q_tp1 = _q_tp_limit(tp1, tick_used, direction)

                    # ===== TP1 qty sizing with MIN USDT constraint =====
                    tp_trade_side = _tp_trade_side(pos_mode)

                    base_tp1 = _q_qty_floor(qty_total * TP1_CLOSE_PCT, qty_step)
                    if min_qty > 0:
                        base_tp1 = max(base_tp1, min_qty)
                    base_tp1 = min(base_tp1, qty_total)

                    min_usdt = float(st.get("tp1_min_usdt") or MIN_TP_USDT_FALLBACK)
                    req_qty = _q_qty_ceil((min_usdt / max(q_tp1, 1e-12)), qty_step)
                    if min_qty > 0:
                        req_qty = max(req_qty, min_qty)
                    qty_tp1 = max(base_tp1, req_qty)
                    qty_tp1 = min(qty_tp1, qty_total)

                    qty_rem = _q_qty_floor(qty_total - qty_tp1, qty_step)

                    # if runner would be dust, close full at TP1
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

                    # 1) SL
                    if not st.get("sl_plan_id") and (not bool(st.get("sl_inflight", False))):
                        st["sl_inflight"] = True
                        dirty = True
                        desk_log(logging.INFO, "SL_SEND", sym, tid, close_side=close_side, trigger_type=_trigger_type_sl(), trigger_q=q_sl, qty=qty_total, tick=tick_used)

                        async def _place_sl():
                            return await trader.place_stop_market_sl(
                                symbol=sym,
                                close_side=close_side.lower(),
                                trigger_price=q_sl,
                                qty=qty_total,
                                client_oid=_oid("sl", tid, attempts),
                                trigger_type=_trigger_type_sl(),
                                tick_hint=tick_used,
                                debug_tag="SL",
                            )

                        try:
                            sl_resp = await asyncio.wait_for(retry_async(_place_sl, retries=2, base_delay=0.35), timeout=ORDER_TIMEOUT_S)
                        except Exception as e:
                            sl_resp = {"code": "EXC", "msg": str(e)}
                        st["sl_inflight"] = False
                        dirty = True

                        if not _is_ok(sl_resp):
                            st["arm_attempts"] = attempts + 1
                            st["last_arm_fail_ts"] = time.time()
                            desk_log(logging.ERROR, "SL_FAIL", sym, tid, code=sl_resp.get("code"), msg=sl_resp.get("msg"), dbg=sl_resp.get("_debug"))
                            try:
                                md = await asyncio.wait_for(trader.debug_meta(sym), timeout=META_TIMEOUT_S)
                            except Exception:
                                md = {}
                            desk_log(logging.ERROR, "META_DUMP", sym, tid, meta=md)
                            await send_telegram(_mk_exec_msg("SL_FAIL", sym, tid, code=sl_resp.get("code"), msg=sl_resp.get("msg")))
                            continue

                        sl_plan_id = (
                            (sl_resp.get("data") or {}).get("orderId")
                            or (sl_resp.get("data") or {}).get("planOrderId")
                            or sl_resp.get("orderId")
                        )
                        st["sl_plan_id"] = str(sl_plan_id) if sl_plan_id else "ok"
                        dirty = True
                        desk_log(logging.INFO, "SL_OK", sym, tid, sl=q_sl, planId=sl_plan_id)
                        await send_telegram(_mk_exec_msg("SL_OK", sym, tid, sl=q_sl, planId=sl_plan_id))

                    # 2) TP1
                    if not st.get("tp1_order_id") and (not bool(st.get("tp1_inflight", False))):
                        st["tp1_inflight"] = True
                        dirty = True
                        desk_log(logging.INFO, "TP1_SEND", sym, tid, close_side=close_side, price_q=q_tp1, qty=qty_tp1, tick=tick_used, trade_side=tp_trade_side, reduceOnly=True)

                        async def _place_tp1(price_use: float, qty_use: float, attempt_k: int):
                            return await trader.place_limit(
                                symbol=sym,
                                side=close_side.lower(),
                                price=price_use,
                                size=qty_use,
                                client_oid=_oid("tp1", tid, attempt_k),
                                trade_side=tp_trade_side,
                                reduce_only=True,
                                tick_hint=tick_used,
                                debug_tag="TP1",
                            )

                        try:
                            tp1_resp = await asyncio.wait_for(_place_tp1(q_tp1, qty_tp1, attempts), timeout=ORDER_TIMEOUT_S)
                        except Exception as e:
                            tp1_resp = {"code": "EXC", "msg": str(e)}

                        st["tp1_inflight"] = False
                        dirty = True

                        code = str(tp1_resp.get("code", ""))

                        # MIN USDT -> adjust qty_tp1 once then retry later
                        if (not _is_ok(tp1_resp)) and code == "45110":
                            msg = str(tp1_resp.get("msg") or "")
                            mmin = _parse_min_usdt(msg) or MIN_TP_USDT_FALLBACK
                            st["tp1_min_usdt"] = float(mmin)

                            req_qty2 = _q_qty_ceil((float(mmin) / max(q_tp1, 1e-12)), qty_step)
                            if min_qty > 0:
                                req_qty2 = max(req_qty2, min_qty)
                            req_qty2 = min(req_qty2, qty_total)

                            rem2 = _q_qty_floor(qty_total - req_qty2, qty_step)
                            if rem2 < (min_qty if min_qty > 0 else 0.0):
                                req_qty2 = qty_total
                                rem2 = 0.0

                            st["qty_tp1"] = req_qty2
                            st["qty_rem"] = rem2

                            desk_log(logging.WARNING, "TP1_MINUSDT_ADJUST", sym, tid, min_usdt=mmin, new_qty_tp1=req_qty2, new_qty_rem=rem2, price=q_tp1)
                            st["arm_attempts"] = attempts + 1
                            st["last_arm_fail_ts"] = time.time()
                            dirty = True
                            continue

                        # price band clamp
                        if (not _is_ok(tp1_resp)) and code == "22047":
                            mn, mx = _parse_band(str(tp1_resp.get("msg") or ""))
                            clamped = _clamp_and_quantize(q_tp1, tick_used, mn, mx)
                            desk_log(logging.WARNING, "TP1_22047", sym, tid, mn=mn, mx=mx, before=q_tp1, after=clamped, tick=tick_used)
                            if clamped is None:
                                st["arm_attempts"] = attempts + 1
                                st["last_arm_fail_ts"] = time.time()
                                dirty = True
                                continue
                            try:
                                tp1_resp = await asyncio.wait_for(_place_tp1(clamped, qty_tp1, attempts + 1), timeout=ORDER_TIMEOUT_S)
                            except Exception as e:
                                tp1_resp = {"code": "EXC", "msg": str(e)}

                        if not _is_ok(tp1_resp):
                            st["arm_attempts"] = attempts + 1
                            st["last_arm_fail_ts"] = time.time()
                            dirty = True
                            desk_log(logging.ERROR, "TP1_FAIL", sym, tid, code=tp1_resp.get("code"), msg=tp1_resp.get("msg"), dbg=tp1_resp.get("_debug"))
                            try:
                                md = await asyncio.wait_for(trader.debug_meta(sym), timeout=META_TIMEOUT_S)
                            except Exception:
                                md = {}
                            desk_log(logging.ERROR, "META_DUMP", sym, tid, meta=md)
                            await send_telegram(_mk_exec_msg("TP1_FAIL", sym, tid, code=tp1_resp.get("code"), msg=tp1_resp.get("msg")))
                            continue

                        tp1_order_id = (tp1_resp.get("data") or {}).get("orderId") or tp1_resp.get("orderId")
                        st["tp1_order_id"] = str(tp1_order_id) if tp1_order_id else "ok"
                        dirty = True
                        desk_log(logging.INFO, "TP1_OK", sym, tid, tp1=q_tp1, orderId=tp1_order_id, qty_tp1=qty_tp1, qty_rem=qty_rem)
                        await send_telegram(_mk_exec_msg("TP1_OK", sym, tid, tp1=q_tp1, orderId=tp1_order_id, qty_tp1=qty_tp1, qty_rem=qty_rem))

                    # ARMED (only after SL + TP1 exist)
                    if st.get("sl_plan_id") and st.get("tp1_order_id"):
                        st["armed"] = True
                        dirty = True
                        desk_log(logging.INFO, "ARMED", sym, tid, qty_total=st.get("qty_total"), close_side=close_side, direction=direction, pos_mode=pos_mode, runner_qty=st.get("qty_rem"))
                    continue

                # ------------------------------------------------------------------
                # Phase 2: BE after TP1 filled (and SL resized to remaining qty)
                # ------------------------------------------------------------------
                if bool(st.get("armed", False)) and (not bool(st.get("be_done", False))):
                    tp1_order_id = st.get("tp1_order_id")
                    if not tp1_order_id or tp1_order_id == "ok":
                        continue

                    try:
                        tp1_detail = await asyncio.wait_for(trader.get_order_detail(sym, order_id=tp1_order_id), timeout=DETAIL_TIMEOUT_S)
                    except Exception as e:
                        st["last_arm_fail_ts"] = time.time()
                        dirty = True
                        desk_log(logging.WARNING, "BE", sym, tid, step="tp1_detail_exc", err=str(e))
                        continue

                    if not trader.is_filled(tp1_detail):
                        continue

                    qty_total = float(st.get("qty_total") or 0.0)
                    qty_tp1 = float(st.get("qty_tp1") or 0.0)
                    qty_step = float(st.get("qty_step") or 1.0)

                    qty_rem = _q_qty_floor(max(0.0, qty_total - qty_tp1), qty_step)
                    if qty_rem <= 0:
                        desk_log(logging.INFO, "BE_SKIP", sym, tid, reason="no_runner_remaining")
                        async with PENDING_LOCK:
                            PENDING.pop(tid, None)
                        dirty = True
                        await send_telegram(_mk_exec_msg("DONE", sym, tid, reason="tp1_full_close"))
                        continue

                    tick_meta = await _get_tick_cached(trader, sym)
                    tick_used = _sanitize_tick(sym, entry, tick_meta, tid)

                    be_ticks = int(BE_FEE_BUFFER_TICKS or 0)
                    be_delta = float(be_ticks) * float(tick_used)

                    be_raw = (entry + be_delta) if direction == "LONG" else (entry - be_delta)
                    be_q = _q_sl(be_raw, tick_used, direction)

                    desk_log(logging.INFO, "BE_SEND", sym, tid, be_raw=be_raw, be_q=be_q, tick=tick_used, be_ticks=be_ticks, qty_rem=qty_rem)

                    # cancel old SL plan (FIX: BitgetTrader.cancel_plan_orders)
                    old_plan = st.get("sl_plan_id")
                    if old_plan and old_plan not in ("ok", ""):
                        try:
                            await asyncio.wait_for(trader.cancel_plan_orders(sym, [str(old_plan)]), timeout=ORDER_TIMEOUT_S)
                            desk_log(logging.INFO, "SL_CANCEL_OK", sym, tid, planId=old_plan)
                        except Exception as e:
                            desk_log(logging.WARNING, "SL_CANCEL_WARN", sym, tid, planId=old_plan, err=str(e))

                    async def _place_be():
                        return await trader.place_stop_market_sl(
                            symbol=sym,
                            close_side=close_side.lower(),
                            trigger_price=be_q,
                            qty=qty_rem,  # remaining qty only
                            client_oid=_oid("slbe", tid, 0),
                            trigger_type=_trigger_type_sl(),
                            tick_hint=tick_used,
                            debug_tag="SL_BE",
                        )

                    try:
                        sl_be_resp = await asyncio.wait_for(retry_async(_place_be, retries=2, base_delay=0.35), timeout=ORDER_TIMEOUT_S)
                    except Exception as e:
                        sl_be_resp = {"code": "EXC", "msg": str(e)}

                    if not _is_ok(sl_be_resp):
                        desk_log(logging.ERROR, "BE_FAIL", sym, tid, code=sl_be_resp.get("code"), msg=sl_be_resp.get("msg"))
                        await send_telegram(_mk_exec_msg("BE_FAIL", sym, tid, code=sl_be_resp.get("code"), msg=sl_be_resp.get("msg")))
                        st["last_arm_fail_ts"] = time.time()
                        dirty = True
                        continue

                    new_plan = (
                        (sl_be_resp.get("data") or {}).get("orderId")
                        or (sl_be_resp.get("data") or {}).get("planOrderId")
                        or sl_be_resp.get("orderId")
                    )

                    st["sl_plan_id"] = str(new_plan) if new_plan else st.get("sl_plan_id")
                    st["be_done"] = True
                    dirty = True

                    desk_log(logging.INFO, "BE_OK", sym, tid, be=be_q, planId=new_plan, qty_rem=qty_rem)
                    await send_telegram(_mk_exec_msg("BE_OK", sym, tid, be=be_q, planId=new_plan, runner_qty=qty_rem))

                    # done => remove pending
                    async with PENDING_LOCK:
                        PENDING.pop(tid, None)
                    dirty = True
                    continue

            except Exception as e:
                logger.exception("[WATCHER] tid=%s error=%s", tid, e)

        if dirty:
            await _pending_save(force=False)


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

    # gather w/ safety
    results = await asyncio.gather(*[_worker(sym) for sym in symbols], return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            logger.debug("worker exception: %s", r)

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

    await _pending_load()

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        await send_telegram("✅ *Bot démarré* \\(TP2 OFF, TP1 minUSDT fix, BE after TP1 ON\\)")
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
