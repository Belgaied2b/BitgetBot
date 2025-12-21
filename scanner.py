# =====================================================================
# scanner.py — Bitget Desk Lead Scanner (Institutionnel H1 + Validation H4)
# + Exec: ENTRY -> (TP1/TP2 + SL) after fill, SL->BE after TP1
# + Clean logs + per-scan summary + sampled reject diagnostics
# + Exchange guards:
#   - price quantization to tickSize (fix code=40020 Parameter price error)
#   - TP clamp/retry on price band (fix code=22047 max/min price limit)
#   - IMPORTANT: arm SL FIRST, then TP1/TP2 (never leave naked positions)
# =====================================================================

from __future__ import annotations

import asyncio
import logging
import time
import uuid
import re
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
    RISK_USDT,
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
# GLOBALS
# =====================================================================

DUP_GUARD = DuplicateGuard(ttl_seconds=3600)
RISK = RiskManager()

TF_H1 = "1H"
TF_H4 = "4H"
CANDLE_LIMIT = 200
MAX_CONCURRENT_FETCH = 8

TP1_CLOSE_PCT = 0.50
WATCH_INTERVAL_S = 3.0

# Reduce noisy modules
ANALYZER_LOG_LEVEL = logging.WARNING
INSTITUTIONAL_LOG_LEVEL = logging.WARNING

# Sampled reject logs (avoid flood)
REJECT_DEBUG_SAMPLES = 25

# TP behavior
ALLOW_TP2_SYNTH = True

# Soft accept (optional)
SOFT_ACCEPT_INVALID = True
SOFT_MIN_INST_SCORE = 2
SOFT_MIN_RR = 1.2

# Watcher arm attempts (avoid infinite spam)
ARM_MAX_ATTEMPTS = 8
ARM_COOLDOWN_S = 12.0  # wait between failed arm tries


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
            parts.append(f"{k}={v:.6g}")
        else:
            parts.append(f"{k}={v}")
    logger.log(level, " ".join(parts))


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
    s = (side or "").upper()
    return "LONG" if s == "BUY" else "SHORT"


def _close_side(entry_side: str) -> str:
    return "SELL" if (entry_side or "").upper() == "BUY" else "BUY"


def _trigger_type_sl() -> str:
    s = (STOP_TRIGGER_TYPE_SL or "MP").upper()
    return "mark_price" if s == "MP" else "fill_price"


def _be_price(entry: float, tick: float, side: str) -> float:
    buf = int(BE_FEE_BUFFER_TICKS or 0)
    if buf <= 0:
        return float(entry)
    if (side or "").upper() == "BUY":
        return float(entry + buf * tick)
    return float(entry - buf * tick)


def _quantize_to_tick(price: float, tick: float) -> float:
    """
    Quantize price to tick. Bitget is strict about tickSize/pricePlace.
    """
    if tick <= 0:
        return float(price)
    n = round(price / tick)
    return float(n * tick)


def _synth_tp2(entry: float, tp1: float, side: str) -> float:
    if (side or "").upper() == "BUY":
        return float(tp1 + (tp1 - entry))
    return float(tp1 - (entry - tp1))


_MAX_RE = re.compile(r"maximum price limit:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_MIN_RE = re.compile(r"minimum price limit:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def _parse_price_band(msg: str) -> Tuple[Optional[float], Optional[float]]:
    if not msg:
        return None, None
    mmax = _MAX_RE.search(msg)
    mmin = _MIN_RE.search(msg)
    mx = float(mmax.group(1)) if mmax else None
    mn = float(mmin.group(1)) if mmin else None
    return mn, mx


def _clamp_price_to_band(price: float, tick: float, mn: Optional[float], mx: Optional[float]) -> float:
    p = float(price)
    if mx is not None:
        p = min(p, float(mx) - 2.0 * tick)
    if mn is not None:
        p = max(p, float(mn) + 2.0 * tick)
    return _quantize_to_tick(p, tick)


def _extract_reject_reason(result: Any) -> str:
    if not isinstance(result, dict):
        return "not_valid"
    r = result.get("reject_reason") or result.get("reason") or result.get("reject")
    if r:
        return str(r)
    inst = result.get("institutional") or {}
    iscore = inst.get("institutional_score")
    if iscore is not None:
        return f"inst_score={iscore}"
    struct = result.get("structure") or result.get("STRUCT")
    if isinstance(struct, dict):
        tr = struct.get("trend")
        if tr:
            return f"trend={tr}"
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


def _build_signal_message(result: Dict[str, Any], tid: str, soft: bool = False, tp2_synth: bool = False) -> str:
    symbol = result.get("symbol", "?")
    side = str(result.get("side", "?")).upper()

    entry = result.get("entry")
    sl = result.get("sl")
    tp1 = result.get("tp1")
    tp2 = result.get("tp2")
    rr = result.get("rr")
    setup = result.get("setup_type") or ("SOFT_OVERRIDE" if soft else None)

    inst = result.get("institutional") or {}
    inst_score = inst.get("institutional_score")
    flow_regime = inst.get("flow_regime")
    funding = inst.get("funding_rate")
    crowd = inst.get("crowding_regime")

    msg = (
        f"🎯 *SIGNAL {symbol}* → *{side}*\n"
        f"• ID: `{tid}`\n"
        f"• Entrée: `{entry}`\n"
        f"• SL: `{sl}`\n"
    )
    if tp1 is not None:
        msg += f"• TP1: `{tp1}` (close {int(TP1_CLOSE_PCT*100)}%)\n"
    if tp2 is not None:
        msg += f"• TP2: `{tp2}` ({'runner, synth' if tp2_synth else 'runner'})\n"
    if rr is not None:
        msg += f"• RR: `{round(float(rr), 3)}`\n"
    if setup:
        msg += f"• Setup: `{setup}`\n"
    if soft:
        msg += "\n⚠️ *SOFT_ACCEPT* (valid=False mais champs trade OK + desk rules OK)\n"

    if inst_score is not None:
        msg += f"\n🏛 *Institutionnel*\n• Score: `{inst_score}`"
        if flow_regime:
            msg += f"\n• Flow: `{flow_regime}`"
        if crowd:
            msg += f"\n• Crowding: `{crowd}`"
        if funding is not None:
            msg += f"\n• Funding: `{funding}`"

    if DRY_RUN:
        msg += "\n\n🧪 *DRY_RUN=ON* (aucun ordre envoyé)"

    return msg


# =====================================================================
# Watcher state
# =====================================================================

PENDING: Dict[str, Dict[str, Any]] = {}
PENDING_LOCK = asyncio.Lock()
WATCHER_TASK: Optional[asyncio.Task] = None

# simple tick cache (avoid repeated meta refresh)
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
    async def _h1():
        return await client.get_klines_df(symbol, TF_H1, CANDLE_LIMIT)

    async def _h4():
        return await client.get_klines_df(symbol, TF_H4, CANDLE_LIMIT)

    df_h1 = await retry_async(_h1, retries=3, base_delay=0.4)
    df_h4 = await retry_async(_h4, retries=3, base_delay=0.4)
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
# Core processing
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

            if SOFT_ACCEPT_INVALID and isinstance(result, dict) and _has_key_fields_for_trade(result):
                rr_f = _safe_float(result.get("rr"), 0.0)
                inst = result.get("institutional") or {}
                inst_score = int(inst.get("institutional_score") or 0)
                if inst_score >= SOFT_MIN_INST_SCORE and rr_f >= SOFT_MIN_RR:
                    result = dict(result)
                    result["valid"] = True
                    result["setup_type"] = result.get("setup_type") or "SOFT_OVERRIDE"
                    desk_log(logging.WARNING, "SOFT", symbol, tid, rr=rr_f, inst=inst_score, reason=reason)
                else:
                    return
            else:
                return

        side = str(result.get("side", "")).upper()
        if side not in ("BUY", "SELL"):
            await stats.inc("rejects", 1)
            await stats.add_reason("bad_side")
            return

        entry = _safe_float(result.get("entry"), 0.0)
        sl = _safe_float(result.get("sl"), 0.0)
        tp1_val = _safe_float(result.get("tp1"), 0.0)
        tp2_val = _safe_float(result.get("tp2"), 0.0)
        rr_val = _safe_float(result.get("rr"), 0.0)
        setup = result.get("setup_type")

        if entry <= 0 or sl <= 0 or tp1_val <= 0:
            await stats.inc("rejects", 1)
            await stats.add_reason("bad_fields")
            return

        tp2_synth = False
        if tp2_val <= 0 and ALLOW_TP2_SYNTH:
            tp2_val = _synth_tp2(entry, tp1_val, side)
            tp2_synth = True
            result["tp2"] = tp2_val

        if tp2_val <= 0:
            await stats.inc("skips", 1)
            await stats.add_reason("missing_tp")
            return

        fp = make_fingerprint(symbol, side, entry, sl, tp1_val, extra=setup, precision=6)
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
            rr=rr_val if rr_val > 0 else None,
            inst_score=inst_score,
            commitment=None,
        )
        if not allowed:
            await stats.inc("risk_rejects", 1)
            await stats.add_reason(f"risk:{reason}")
            return

        await stats.inc("valids", 1)
        desk_log(logging.INFO, "VALID", symbol, tid, side=side, setup=setup, rr=rr_val, inst=inst_score)

        soft = (result.get("setup_type") == "SOFT_OVERRIDE")
        await send_telegram(_build_signal_message(result, tid, soft=soft, tp2_synth=tp2_synth))
        DUP_GUARD.mark(fp)

        if DRY_RUN:
            return

        try:
            await asyncio.wait_for(order_budget.acquire(), timeout=0.01)
        except asyncio.TimeoutError:
            await stats.add_reason("budget:max_orders_per_scan")
            return

        # ---- ENTRY: quantize to tick ----
        tick = await _get_tick_cached(trader, symbol)
        q_entry = _quantize_to_tick(entry, tick) if tick > 0 else entry

        oid = f"entry-{tid}"
        await stats.inc("exec_sent", 1)
        desk_log(logging.INFO, "EXEC", symbol, tid, action="entry_send", entry=q_entry, notional=round(notional, 2), oid=oid)

        entry_resp = await trader.place_limit(
            symbol=symbol,
            side=side.lower(),
            price=q_entry,
            size=None,
            client_oid=oid,
            trade_side="open",
            reduce_only=False,
        )

        if not _is_ok(entry_resp):
            await stats.inc("exec_failed", 1)
            code = entry_resp.get("code")
            msg = entry_resp.get("msg")

            # If price error, try one more time with quantized price (and slight nudge)
            if str(code) == "40020" and tick > 0:
                # nudge 1 tick to be safe
                if side == "BUY":
                    q_entry2 = _quantize_to_tick(q_entry - tick, tick)
                else:
                    q_entry2 = _quantize_to_tick(q_entry + tick, tick)

                desk_log(logging.WARNING, "EXEC", symbol, tid, action="entry_retry_40020", entry=q_entry2)
                entry_resp = await trader.place_limit(
                    symbol=symbol,
                    side=side.lower(),
                    price=q_entry2,
                    size=None,
                    client_oid=oid,
                    trade_side="open",
                    reduce_only=False,
                )

            if not _is_ok(entry_resp):
                desk_log(logging.ERROR, "EXEC", symbol, tid, action="entry_fail", code=entry_resp.get("code"), msg=entry_resp.get("msg"))
                await send_telegram(f"❌ *ENTRY FAILED* {symbol} {side} @ `{q_entry}`\nID: `{tid}`\n`{entry_resp}`")
                # release budget so other trades can pass in this scan
                try:
                    order_budget.release()
                except Exception:
                    pass
                return

        entry_order_id = (entry_resp.get("data") or {}).get("orderId") or entry_resp.get("orderId")
        qty_total = _safe_float(entry_resp.get("qty"), 0.0)

        desk_log(logging.INFO, "EXEC", symbol, tid, action="entry_ok", orderId=entry_order_id, qty=(qty_total if qty_total else None))

        async with PENDING_LOCK:
            PENDING[tid] = {
                "symbol": str(symbol).upper(),
                "entry_side": side.upper(),
                "close_side": _close_side(side),
                "entry": q_entry,
                "sl": sl,
                "tp1": tp1_val,
                "tp2": tp2_val,
                "qty_total": qty_total,
                "qty_tp1": 0.0,
                "qty_tp2": 0.0,
                "entry_order_id": str(entry_order_id) if entry_order_id else None,
                "entry_client_oid": oid,
                "tp1_order_id": None,
                "tp2_order_id": None,
                "sl_plan_id": None,
                "tp1_done": False,
                "armed": False,
                "be_done": False,
                "created_ts": time.time(),
                "arm_attempts": 0,
                "last_arm_fail_ts": 0.0,
            }

    except Exception as e:
        desk_log(logging.ERROR, "ERR", symbol, tid, where="process_symbol", err=str(e))
        logger.exception("[%s] process_symbol error: %s", symbol, e)


# =====================================================================
# WATCHER: fill -> SL first -> TP1/TP2 ; TP1 -> SL->BE ; TP2 -> cleanup
# =====================================================================

async def _watcher_loop(trader: BitgetTrader) -> None:
    logger.info("[WATCHER] started (interval=%.1fs)", WATCH_INTERVAL_S)

    while True:
        try:
            await asyncio.sleep(WATCH_INTERVAL_S)

            async with PENDING_LOCK:
                items = list(PENDING.items())

            if not items:
                continue

            for tid, st in items:
                sym = st["symbol"]
                entry_side = st["entry_side"]
                close_side = st["close_side"]
                entry = float(st["entry"])
                sl = float(st["sl"])
                tp1 = float(st["tp1"])
                tp2 = float(st["tp2"])

                # 0) Not armed: check fill and arm
                if not st["armed"]:
                    # cooldown to avoid spam
                    if st.get("arm_attempts", 0) > 0:
                        last_fail = float(st.get("last_arm_fail_ts") or 0.0)
                        if last_fail > 0 and (time.time() - last_fail) < ARM_COOLDOWN_S:
                            continue

                    if int(st.get("arm_attempts") or 0) >= ARM_MAX_ATTEMPTS:
                        desk_log(logging.ERROR, "WATCH", sym, tid, step="arm_abort_max_attempts")
                        await send_telegram(f"⚠️ *ARM ABORTED* {sym}\nID: `{tid}`\nTrop d'échecs TP/SL. Position à gérer manuellement.")
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
                        desk_log(logging.WARNING, "WATCH", sym, tid, step="entry_filled_but_no_qty")
                        st["arm_attempts"] = int(st.get("arm_attempts") or 0) + 1
                        st["last_arm_fail_ts"] = time.time()
                        continue

                    tick = await _get_tick_cached(trader, sym)
                    q_sl = _quantize_to_tick(sl, tick) if tick > 0 else sl
                    q_tp1 = _quantize_to_tick(tp1, tick) if tick > 0 else tp1
                    q_tp2 = _quantize_to_tick(tp2, tick) if tick > 0 else tp2

                    qty_tp1 = qty_total * TP1_CLOSE_PCT
                    qty_tp2 = max(0.0, qty_total - qty_tp1)

                    desk_log(logging.INFO, "WATCH", sym, tid, step="arm_start", qty_total=qty_total)

                    # ====== IMPORTANT: SL FIRST ======
                    sl_resp = await trader.place_stop_market_sl(
                        symbol=sym,
                        close_side=close_side.lower(),
                        trigger_price=q_sl,
                        qty=qty_total,
                        client_oid=f"sl-{tid}",
                        trigger_type=_trigger_type_sl(),
                    )
                    if not _is_ok(sl_resp):
                        desk_log(logging.ERROR, "WATCH", sym, tid, step="sl_fail", code=sl_resp.get("code"), msg=sl_resp.get("msg"))
                        await send_telegram(f"❌ *SL FAILED* {sym}\nID: `{tid}`\n`{sl_resp}`")
                        st["arm_attempts"] = int(st.get("arm_attempts") or 0) + 1
                        st["last_arm_fail_ts"] = time.time()
                        continue

                    sl_plan_id = (sl_resp.get("data") or {}).get("orderId") or (sl_resp.get("data") or {}).get("planOrderId") or sl_resp.get("orderId")

                    # ====== TP1 with clamp/retry on 22047 ======
                    tp1_resp = await trader.place_reduce_limit_tp(
                        symbol=sym,
                        close_side=close_side.lower(),
                        price=q_tp1,
                        qty=qty_tp1,
                        client_oid=f"tp1-{tid}",
                    )
                    if not _is_ok(tp1_resp) and str(tp1_resp.get("code")) == "22047" and tick > 0:
                        mn, mx = _parse_price_band(str(tp1_resp.get("msg") or ""))
                        q_tp1b = _clamp_price_to_band(q_tp1, tick, mn, mx)
                        desk_log(logging.WARNING, "WATCH", sym, tid, step="tp1_retry_22047", tp1=q_tp1b, band_min=mn, band_max=mx)
                        tp1_resp = await trader.place_reduce_limit_tp(
                            symbol=sym,
                            close_side=close_side.lower(),
                            price=q_tp1b,
                            qty=qty_tp1,
                            client_oid=f"tp1-{tid}",
                        )
                        q_tp1 = q_tp1b

                    if not _is_ok(tp1_resp):
                        desk_log(logging.ERROR, "WATCH", sym, tid, step="tp1_fail", code=tp1_resp.get("code"), msg=tp1_resp.get("msg"))
                        await send_telegram(f"❌ *TP1 FAILED* {sym}\nID: `{tid}`\n`{tp1_resp}`")
                        # SL is already placed => at least protected
                        st["arm_attempts"] = int(st.get("arm_attempts") or 0) + 1
                        st["last_arm_fail_ts"] = time.time()
                        continue

                    tp1_order_id = (tp1_resp.get("data") or {}).get("orderId") or tp1_resp.get("orderId")

                    # ====== TP2 with clamp/retry on 22047 ======
                    tp2_resp = await trader.place_reduce_limit_tp(
                        symbol=sym,
                        close_side=close_side.lower(),
                        price=q_tp2,
                        qty=qty_tp2,
                        client_oid=f"tp2-{tid}",
                    )
                    if not _is_ok(tp2_resp) and str(tp2_resp.get("code")) == "22047" and tick > 0:
                        mn, mx = _parse_price_band(str(tp2_resp.get("msg") or ""))
                        q_tp2b = _clamp_price_to_band(q_tp2, tick, mn, mx)
                        desk_log(logging.WARNING, "WATCH", sym, tid, step="tp2_retry_22047", tp2=q_tp2b, band_min=mn, band_max=mx)
                        tp2_resp = await trader.place_reduce_limit_tp(
                            symbol=sym,
                            close_side=close_side.lower(),
                            price=q_tp2b,
                            qty=qty_tp2,
                            client_oid=f"tp2-{tid}",
                        )
                        q_tp2 = q_tp2b

                    if not _is_ok(tp2_resp):
                        desk_log(logging.ERROR, "WATCH", sym, tid, step="tp2_fail", code=tp2_resp.get("code"), msg=tp2_resp.get("msg"))
                        await send_telegram(f"❌ *TP2 FAILED* {sym}\nID: `{tid}`\n`{tp2_resp}`")
                        st["arm_attempts"] = int(st.get("arm_attempts") or 0) + 1
                        st["last_arm_fail_ts"] = time.time()
                        continue

                    tp2_order_id = (tp2_resp.get("data") or {}).get("orderId") or tp2_resp.get("orderId")

                    # Update state as armed
                    async with PENDING_LOCK:
                        if tid in PENDING:
                            PENDING[tid]["armed"] = True
                            PENDING[tid]["qty_total"] = qty_total
                            PENDING[tid]["qty_tp1"] = float(tp1_resp.get("qty") or qty_tp1)
                            PENDING[tid]["qty_tp2"] = float(tp2_resp.get("qty") or qty_tp2)
                            PENDING[tid]["tp1_order_id"] = str(tp1_order_id) if tp1_order_id else None
                            PENDING[tid]["tp2_order_id"] = str(tp2_order_id) if tp2_order_id else None
                            PENDING[tid]["sl_plan_id"] = str(sl_plan_id) if sl_plan_id else None
                            PENDING[tid]["sl"] = q_sl
                            PENDING[tid]["tp1"] = q_tp1
                            PENDING[tid]["tp2"] = q_tp2

                    notional = float(MARGIN_USDT) * float(LEVERAGE)
                    RISK.register_open(
                        symbol=sym,
                        side=_side_to_direction(entry_side),
                        notional=notional,
                        risk=float(RISK_USDT),
                    )

                    desk_log(logging.INFO, "WATCH", sym, tid, step="armed", tp1=q_tp1, tp2=q_tp2, sl=q_sl, qty_total=qty_total)
                    await send_telegram(
                        f"🛡 *PROTECTION ARMED* {sym}\n"
                        f"• ID: `{tid}`\n"
                        f"• TP1 `{q_tp1}` ({int(TP1_CLOSE_PCT*100)}%)\n"
                        f"• TP2 `{q_tp2}` (runner)\n"
                        f"• SL `{q_sl}`"
                    )
                    continue

                # 1) TP1 filled => SL to BE
                if st["armed"] and not st["tp1_done"]:
                    tp1_detail = await trader.get_order_detail(sym, order_id=st.get("tp1_order_id"))
                    if not trader.is_filled(tp1_detail):
                        continue

                    tick = await _get_tick_cached(trader, sym)
                    new_sl = _be_price(entry, tick, entry_side)
                    new_sl = _quantize_to_tick(new_sl, tick) if tick > 0 else new_sl

                    old_sl_id = st.get("sl_plan_id")
                    if old_sl_id:
                        _ = await trader.cancel_plan_orders(sym, [str(old_sl_id)])

                    remaining = max(0.0, float(st.get("qty_total") or 0.0) - float(st.get("qty_tp1") or 0.0))
                    if remaining <= 0:
                        async with PENDING_LOCK:
                            PENDING.pop(tid, None)
                        desk_log(logging.INFO, "WATCH", sym, tid, step="tp1_closed_all")
                        await send_telegram(f"✅ *TP1 FILLED* {sym} — position fully closed (no runner)\nID: `{tid}`")
                        continue

                    sl_be_resp = await trader.place_stop_market_sl(
                        symbol=sym,
                        close_side=st["close_side"].lower(),
                        trigger_price=new_sl,
                        qty=remaining,
                        client_oid=f"be-{tid}",
                        trigger_type=_trigger_type_sl(),
                    )
                    if not _is_ok(sl_be_resp):
                        desk_log(logging.ERROR, "WATCH", sym, tid, step="be_fail", code=sl_be_resp.get("code"), msg=sl_be_resp.get("msg"))
                        await send_telegram(f"❌ *SL->BE FAILED* {sym}\nID: `{tid}`\n`{sl_be_resp}`")
                        continue

                    new_sl_id = (sl_be_resp.get("data") or {}).get("orderId") or (sl_be_resp.get("data") or {}).get("planOrderId") or sl_be_resp.get("orderId")

                    async with PENDING_LOCK:
                        if tid in PENDING:
                            PENDING[tid]["tp1_done"] = True
                            PENDING[tid]["be_done"] = True
                            PENDING[tid]["sl_plan_id"] = str(new_sl_id) if new_sl_id else None

                    desk_log(logging.INFO, "WATCH", sym, tid, step="sl_to_be", new_sl=new_sl, remaining=remaining)
                    await send_telegram(
                        f"✅ *TP1 HIT* {sym}\n"
                        f"• ID: `{tid}`\n"
                        f"🔁 SL déplacé à BE: `{new_sl}`\n"
                        f"🏃 TP2 runner reste actif: `{st.get('tp2')}`"
                    )
                    continue

                # 2) TP2 filled => cleanup
                if st["armed"]:
                    tp2_id = st.get("tp2_order_id")
                    if tp2_id:
                        tp2_detail = await trader.get_order_detail(sym, order_id=str(tp2_id))
                        if trader.is_filled(tp2_detail):
                            sl_id = st.get("sl_plan_id")
                            if sl_id:
                                _ = await trader.cancel_plan_orders(sym, [str(sl_id)])

                            async with PENDING_LOCK:
                                PENDING.pop(tid, None)

                            desk_log(logging.INFO, "WATCH", sym, tid, step="tp2_filled_done")
                            await send_telegram(f"🏁 *TP2 FILLED* {sym} — trade terminé.\nID: `{tid}`")

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
    symbols = await retry_async(client.get_contracts_list, retries=3, base_delay=0.6)
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
    reasons = stats.reasons.most_common(10)
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

    logging.getLogger("analyze_signal").setLevel(ANALYZER_LOG_LEVEL)
    logging.getLogger("institutional_data").setLevel(INSTITUTIONAL_LOG_LEVEL)

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
    try:
        asyncio.run(start_scanner())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.create_task(start_scanner())
        loop.run_forever()
