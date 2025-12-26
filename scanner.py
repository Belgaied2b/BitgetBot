# =====================================================================
# scanner.py — TP2 OFF (runner) + BE after TP1 + Telegram ON
# Fix Bitget min TP value (45110 "minimum amount X USDT")
# Fix SL->BE uses remaining qty after TP1
# =====================================================================

from __future__ import annotations

import asyncio
import logging
import time
import uuid
import re
import math
import os
from collections import Counter
from typing import Any, Dict, Tuple, Optional

import pandas as pd

from settings import (
    API_KEY,
    API_SECRET,
    API_PASSPHRASE,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TOKEN,
    CHAT_ID,
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

# Telegram behavior:
# - TELEGRAM_SIGNAL_ON_VALID: send a "setup valid" message even before execution
# - TELEGRAM_SIGNAL_ON_EXEC: send a message when an entry order is actually accepted
# - TELEGRAM_SIGNAL_ON_SKIP: send a message when a valid setup is skipped (risk/microstructure/budget)
def _env_bool(name: str, default: str = "1") -> bool:
    v = os.getenv(name, default)
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

TELEGRAM_SIGNAL_ON_VALID = _env_bool("TELEGRAM_SIGNAL_ON_VALID", "1")
TELEGRAM_SIGNAL_ON_EXEC = _env_bool("TELEGRAM_SIGNAL_ON_EXEC", "1")
TELEGRAM_SIGNAL_ON_SKIP = _env_bool("TELEGRAM_SIGNAL_ON_SKIP", "1")

# Prefer legacy env names TOKEN/CHAT_ID if you used them before.
TG_TOKEN = TOKEN or TELEGRAM_BOT_TOKEN
TG_CHAT_ID = CHAT_ID or TELEGRAM_CHAT_ID

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

# ===== DESK GOVERNANCE (microstructure + cooldown) =====
DESK_MICROSTRUCTURE = os.getenv("DESK_MICROSTRUCTURE", "1").strip() not in ("0", "false", "False")
DESK_MAX_SPREAD_PCT = float(os.getenv("DESK_MAX_SPREAD_PCT", "0.006"))  # 0.6% default  # 0.09% default
DESK_MIN_USDT_VOL_24H = float(os.getenv("DESK_MIN_USDT_VOL_24H", "250000"))  # $250k default
SYMBOL_COOLDOWN_S = float(os.getenv("SYMBOL_COOLDOWN_S", "1800"))  # 30 min
POSMODE_CACHE_TTL_S = float(os.getenv("POSMODE_CACHE_TTL_S", "60"))

SYMBOL_COOLDOWN: Dict[str, float] = {}
POSMODE_CACHE: Dict[str, Any] = {"ts": 0.0, "pos_mode": "one_way"}

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
# Telegram
# =====================================================================

async def send_telegram(msg: str) -> None:
    if not TG_TOKEN or not TG_CHAT_ID:
        return
    import requests
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
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

def _fmt_side(side: str) -> str:
    return "🟢 LONG" if (side or "").upper() == "BUY" else "🔴 SHORT"

def _fmt_num(x: float) -> str:
    try:
        return f"{float(x):.6g}"
    except Exception:
        return str(x)

def _mk_signal_msg(symbol: str, tid: str, side: str, setup: str, entry: float, sl: float, tp1: float, rr: float, inst: int, entry_type: str, pos_mode: str) -> str:
    return (
        f"*📌 SIGNAL*\n"
        f"*{symbol}* — {_fmt_side(side)}\n"
        f"setup: `{setup}` | inst: `{inst}` | entry_type: `{entry_type}` | pos_mode: `{pos_mode}`\n"
        f"entry: `{_fmt_num(entry)}`\n"
        f"SL: `{_fmt_num(sl)}`\n"
        f"TP1: `{_fmt_num(tp1)}` (runner ensuite)\n"
        f"RR: `{_fmt_num(rr)}`\n"
        f"tid: `{tid}`"
    )

def _mk_skip_msg(
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
    reason: str,
) -> str:
    return (
        f"*⚠️ SKIPPED*\n"
        f"*{symbol}* — {_fmt_side(side)}\n"
        f"setup: `{setup}` | inst: `{inst}` | entry_type: `{entry_type}`\n"
        f"reason: `{reason}`\n"
        f"entry: `{_fmt_num(entry)}` | SL: `{_fmt_num(sl)}` | TP1: `{_fmt_num(tp1)}` | RR: `{_fmt_num(rr)}`\n"
        f"tid: `{tid}`"
    )

def _mk_exec_msg(tag: str, symbol: str, tid: str, **kv: Any) -> str:
    base = f"*⚙️ {tag}* — *{symbol}*\n" + f"tid: `{tid}`\n"
    lines = []
    for k, v in kv.items():
        if v is None:
            continue
        if isinstance(v, float):
            lines.append(f"{k}: `{v:.6g}`")
        else:
            lines.append(f"{k}: `{v}`")
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

def _close_side_from_direction(direction: str, pos_mode: str = "one_way") -> str:
    """Return Bitget API 'side' field used for CLOSE orders.

    - one-way: side is order direction -> close LONG with SELL, close SHORT with BUY
    - hedge:   side encodes position direction -> close LONG with BUY, close SHORT with SELL
    """
    d = (direction or "").upper()
    pm = (pos_mode or "").lower()
    if "hedge" in pm:
        return "BUY" if d == "LONG" else "SELL"
    return "SELL" if d == "LONG" else "BUY"

def _trigger_type_sl() -> str:
    s = (STOP_TRIGGER_TYPE_SL or "MP").upper()
    return "mark_price" if s == "MP" else "fill_price"

def _estimate_tick_from_price(price: float) -> float:
    p = abs(float(price))
    if p >= 10000: return 1.0
    if p >= 1000:  return 0.1
    if p >= 100:   return 0.01
    if p >= 10:    return 0.001
    if p >= 1:     return 0.0001
    if p >= 0.1:   return 0.00001
    if p >= 0.01:  return 0.000001
    return 0.0000001

def _sanitize_tick(symbol: str, entry: float, tick: float, tid: str) -> float:
    est = _estimate_tick_from_price(entry)
    t = float(tick or 0.0)
    if t <= 0 or t > est * 1000:
        desk_log(logging.WARNING, "TICK", symbol, tid, tick_meta=t, tick_est=est, entry=entry, action="fallback")
        return est
    return t

def _q_floor(price: float, tick: float) -> float:
    if tick <= 0: return float(price)
    return float(math.floor(price / tick) * tick)

def _q_ceil(price: float, tick: float) -> float:
    if tick <= 0: return float(price)
    return float(math.ceil(price / tick) * tick)

def _q_entry(price: float, tick: float, side: str) -> float:
    if tick <= 0: return float(price)
    return _q_floor(price, tick) if (side or "").upper() == "BUY" else _q_ceil(price, tick)

def _q_sl(sl: float, tick: float, direction: str) -> float:
    return _q_floor(sl, tick) if direction == "LONG" else _q_ceil(sl, tick)

def _q_tp_limit(tp: float, tick: float, direction: str) -> float:
    return _q_floor(tp, tick) if direction == "LONG" else _q_ceil(tp, tick)

def _q_qty_floor(qty: float, step: float) -> float:
    if step <= 0: return float(qty)
    return float(math.floor(qty / step) * step)

def _q_qty_ceil(qty: float, step: float) -> float:
    if step <= 0: return float(qty)
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

async def _detect_pos_mode(trader: BitgetTrader, symbol_hint: str = "BTCUSDT") -> str:
    """Detect Bitget position mode (one-way vs hedge) via V2 account endpoint.

    Bitget: GET /api/v2/mix/account/account returns posMode=one_way_mode|hedge_mode.
    We cache the result for POSMODE_CACHE_TTL_S seconds.
    """
    now = time.time()
    if (now - float(POSMODE_CACHE.get("ts", 0.0))) < POSMODE_CACHE_TTL_S:
        return str(POSMODE_CACHE.get("pos_mode", "one_way"))

    sym = (symbol_hint or "BTCUSDT").lower()
    try:
        params = {
            "symbol": sym,
            "productType": getattr(trader, "product_type", "USDT-FUTURES"),
            "marginCoin": str(getattr(trader, "margin_coin", "USDT")).lower(),
        }
        js = await trader.client._request("GET", "/api/v2/mix/account/account", params=params, auth=True)
        data = js.get("data") if isinstance(js, dict) else None
        pm = ""
        if isinstance(data, dict):
            pm = str(data.get("posMode") or "")
        pm_l = pm.lower()
        out = "hedge" if "hedge" in pm_l else "one_way"
        POSMODE_CACHE.update({"ts": now, "pos_mode": out, "raw": pm})
        return out
    except Exception as e:
        # fallback: keep previous cached or default
        POSMODE_CACHE.update({"ts": now, "pos_mode": str(POSMODE_CACHE.get("pos_mode", "one_way")), "err": str(e)})
        return str(POSMODE_CACHE.get("pos_mode", "one_way"))


def _tp_trade_side(pos_mode: str) -> str:
    pm = (pos_mode or "").lower()
    return "close" if "hedge" in pm else "open"


def _reduce_only_for_close(pos_mode: str) -> bool:
    """Bitget reduceOnly is only applicable in one-way position mode (not hedge)."""
    return "hedge" not in (pos_mode or "").lower()

async def _get_position_total(
    trader: BitgetTrader,
    symbol: str,
    direction: str,
    pos_mode: str,
) -> float:
    """Return current position 'total' size for given symbol & direction.

    Uses: GET /api/v2/mix/position/all-position
    - hedge: filters by holdSide (long/short)
    - one-way: returns total for the symbol (no holdSide filter)
    """
    try:
        params = {
            "productType": getattr(trader, "product_type", "USDT-FUTURES"),
            "marginCoin": str(getattr(trader, "margin_coin", "USDT")).upper(),
        }
        js = await trader.client._request("GET", "/api/v2/mix/position/all-position", params=params, auth=True)
        data = js.get("data") if isinstance(js, dict) else None
        if not isinstance(data, list):
            return 0.0

        sym_u = (symbol or "").upper()
        pm = (pos_mode or "").lower()
        want_hold = "long" if (direction or "").upper() == "LONG" else "short"

        for p in data:
            if not isinstance(p, dict):
                continue
            if str(p.get("symbol") or "").upper() != sym_u:
                continue
            if "hedge" in pm:
                if str(p.get("holdSide") or "").lower() != want_hold:
                    continue
            return _safe_float(p.get("total"), 0.0)
    except Exception:
        return 0.0
    return 0.0

# =====================================================================
# Watcher state + tick cache
# =====================================================================

async def _microstructure_ok(trader: BitgetTrader, symbol: str) -> Tuple[bool, str]:
    """Desk filter: veto illiquid / wide-spread symbols using V2 ticker endpoint.

    Uses: GET /api/v2/mix/market/ticker (bidPr/askPr/usdtVolume).
    If the endpoint fails, we *don't block* (fail-open) to avoid killing the bot.
    """
    if not DESK_MICROSTRUCTURE:
        return True, "microstructure_disabled"
    sym = (symbol or "").upper()
    try:
        js = await trader.client._request(
            "GET",
            "/api/v2/mix/market/ticker",
            params={"symbol": sym, "productType": getattr(trader, "product_type", "USDT-FUTURES")},
            auth=False,
        )
        data = None
        if isinstance(js, dict):
            data = js.get("data")
        if isinstance(data, list) and data:
            data = data[0]
        if not isinstance(data, dict):
            return True, "ticker_no_data"

        bid = _safe_float(data.get("bidPr"), 0.0)
        ask = _safe_float(data.get("askPr"), 0.0)
        usdt_vol = _safe_float(data.get("usdtVolume") or data.get("quoteVolume"), 0.0)

        spread = None
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
            spread = (ask - bid) / max(1e-12, mid)
            if spread > DESK_MAX_SPREAD_PCT:
                return False, f"spread={spread:.4%} > {DESK_MAX_SPREAD_PCT:.4%} (bid={bid}, ask={ask})"

        if usdt_vol > 0 and usdt_vol < DESK_MIN_USDT_VOL_24H:
            return False, f"usdtVolume={usdt_vol:.0f} < {DESK_MIN_USDT_VOL_24H:.0f}"

        s_txt = f"spread={spread:.4%}" if spread is not None else "spread=NA"
        return True, f"{s_txt} usdtVolume={usdt_vol:.0f}"
    except Exception as e:
        return True, f"ticker_unavailable {e}"



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

    symbol = (symbol or "").upper()

    # Desk governance: cooldown per symbol (évite le spam de setups similaires)
    now = time.time()
    last = float(SYMBOL_COOLDOWN.get(symbol, 0.0))
    if (now - last) < SYMBOL_COOLDOWN_S:
        await stats.inc("skips", 1)
        return

    # Skip if déjà un trade en cours/pending sur ce symbole
    async with PENDING_LOCK:
        if any((p.get("symbol") == symbol) for p in PENDING.values()):
            await stats.inc("skips", 1)
            return

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

        desk_log(logging.INFO, "EXITS", symbol, tid, side=side, entry=entry, sl=sl, tp1=tp1, rr=rr, setup=setup, entry_type=entry_type)

        if entry <= 0 or sl <= 0 or tp1 <= 0:
            await stats.inc("skips", 1)
            await stats.add_reason("missing_exits")
            return

        fp = make_fingerprint(symbol, side, entry, sl, tp1, extra=setup, precision=6)
        if DUP_GUARD.is_duplicate(fp):
            await stats.inc("duplicates", 1)
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
            # Optional: notify Telegram that the setup was valid but skipped by risk rules
            if TELEGRAM_SIGNAL_ON_SKIP:
                await send_telegram(_mk_skip_msg(str(symbol).upper(), tid, side, setup, entry, sl, tp1, rr, inst_score, entry_type, reason=f"risk:{reason}"))
            return

        pos_mode = await _detect_pos_mode(trader, symbol)

        # close-side mapping depends on posMode (one-way vs hedge)
        close_side = _close_side_from_direction(direction, pos_mode)

        await stats.inc("valids", 1)
        desk_log(logging.INFO, "VALID", symbol, tid, side=side, setup=setup, rr=rr, inst=inst_score, entry_type=entry_type, pos_mode=pos_mode)

        if TELEGRAM_SIGNAL_ON_VALID:
            await send_telegram(_mk_signal_msg(str(symbol).upper(), tid, side, setup, entry, sl, tp1, rr, inst_score, entry_type, pos_mode))

        if DRY_RUN:
            return

        try:
            if getattr(order_budget, "_value", 0) <= 0:
                raise asyncio.TimeoutError()
            await order_budget.acquire()
        except asyncio.TimeoutError:
            await stats.add_reason("budget:max_orders_per_scan")
            if TELEGRAM_SIGNAL_ON_SKIP:
                await send_telegram(_mk_skip_msg(str(symbol).upper(), tid, side, setup, entry, sl, tp1, rr, inst_score, entry_type, reason="budget:max_orders_per_scan"))
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

        # Desk microstructure veto (spread/liquidity)
        ok_ms, ms_note = await _microstructure_ok(trader, symbol)
        # Always log; before it was often silent -> Telegram said OK but no Bitget order.
        desk_log(logging.INFO if ok_ms else logging.WARNING, "MICRO", symbol, tid, ok=ok_ms, note=ms_note, entry_type=entry_type)

        # Maker entries can tolerate wider spread (we still keep the warning)
        if (not ok_ms) and str(entry_type).upper() == "LIMIT":
            desk_log(logging.WARNING, "MICRO", symbol, tid, ok=True, note=str(ms_note) + " | limit_override", entry_type=entry_type)
            ok_ms = True

        if not ok_ms:
            desk_log(logging.WARNING, "REJ", symbol, tid, reason=f"microstructure_veto: {ms_note}")
            await stats.inc("rejects", 1)
            if TELEGRAM_SIGNAL_ON_SKIP:
                await send_telegram(_mk_skip_msg(str(symbol).upper(), tid, side, setup, entry, sl, tp1, rr, inst_score, entry_type, reason=f"microstructure_veto: {ms_note}"))
            try:
                order_budget.release()
            except Exception:
                pass
            return

        desk_log(logging.INFO, "EXEC", symbol, tid, action="entry_send", entry=q_entry, notional=round(notional, 2))

        if str(entry_type).upper() == "MARKET" and hasattr(trader, "place_market"):
            entry_resp = await trader.place_market(
                symbol=symbol,
                side=side.lower(),
                price_ref=q_entry,
                size=None,
                client_oid=f"entry-{tid}",
                trade_side="open",
                tick_hint=tick_used,
                debug_tag="ENTRY",
            )
        else:
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
        if TELEGRAM_SIGNAL_ON_EXEC:
            await send_telegram(_mk_exec_msg("ENTRY_OK", str(symbol).upper(), tid, side=side, setup=setup, entry=q_entry, qty=qty_total, pos_mode=pos_mode))
        # Mark duplicate only after Bitget accepted the entry (prevents "signal sent but no order")
        DUP_GUARD.mark(fp)
        # Send main Telegram message only after ENTRY_OK so Telegram == "order exists"
        await send_telegram(_mk_signal_msg(str(symbol).upper(), tid, side, setup, entry, sl, tp1, rr, inst_score, entry_type, pos_mode)
                          + f"\norderId: `{entry_order_id}`")

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
                "notional": float(notional),
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

        SYMBOL_COOLDOWN[symbol] = time.time()

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

            if not items:
                continue

            for tid, st in items:
                sym = st["symbol"]
                direction = st["direction"]
                pos_mode = st.get("pos_mode") or "one_way"
                close_side = _close_side_from_direction(direction, pos_mode)
                st["close_side"] = close_side
                entry = float(st["entry"])
                sl = float(st["sl"])
                tp1 = float(st["tp1"])
                qty_step = float(st.get("qty_step") or 1.0)
                min_qty = float(st.get("min_qty") or 0.0)
                # If we have already seen the position on Bitget, keep watching until it is fully closed.
                # This prevents the RiskManager from drifting (phantom open positions) and keeps desk state clean.
                if st.get("pos_seen"):
                    try:
                        pos_total_now = await _get_position_total(trader, sym, direction, pos_mode)
                    except Exception:
                        pos_total_now = 0.0
                    if float(pos_total_now) <= 0:
                        st["pos_zero_hits"] = int(st.get("pos_zero_hits") or 0) + 1
                        if st["pos_zero_hits"] >= 2:
                            desk_log(logging.INFO, "POS_CLOSED", sym, tid)
                            if st.get("risk_opened"):
                                try:
                                    open_side = "BUY" if direction == "LONG" else "SELL"
                                    RISK.register_closed(sym, open_side, pnl=0.0)
                                except Exception:
                                    logger.exception("[RISK_CLOSED_FAIL] %s %s", sym, tid)
                            async with PENDING_LOCK:
                                PENDING.pop(tid, None)
                            continue
                    else:
                        st["pos_zero_hits"] = 0


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
                        continue

                    # Wait until the position is visible before arming SL/TP1.
                    pos_total = await _get_position_total(trader, sym, direction, pos_mode)
                    if pos_total <= 0:
                        st.setdefault("pos_wait_since", time.time())
                        if (time.time() - float(st.get("pos_wait_since") or time.time())) < 12.0:
                            continue
                        desk_log(logging.WARNING, "POS_LAG", sym, tid, pos_total=pos_total)
                    else:
                        st.pop("pos_wait_since", None)
                        # Clamp qty_total to visible position (handles partial fills)
                        if float(st.get("qty_total") or 0.0) > 0:
                            st["qty_total"] = min(float(st["qty_total"]), float(pos_total))
                        st["pos_seen"] = True
                        if not st.get("risk_opened"):
                            try:
                                open_side = "BUY" if direction == "LONG" else "SELL"
                                RISK.register_open(
                                    sym,
                                    open_side,
                                    notional=float(st.get("notional") or DEFAULT_NOTIONAL_USDT),
                                    risk=RISK.risk_for_this_trade(),
                                )
                                st["risk_opened"] = True
                            except Exception:
                                logger.exception("[RISK_OPEN_FAIL] %s %s", sym, tid)

                    qty_total = float(st.get("qty_total") or 0.0)
                    if qty_total <= 0:
                        data = (detail.get("data") or {})
                        qty_total = float(data.get("size") or data.get("quantity") or 0.0)

                    if qty_total <= 0:
                        st["arm_attempts"] = attempts + 1
                        st["last_arm_fail_ts"] = time.time()
                        desk_log(logging.WARNING, "ARM", sym, tid, step="no_qty_from_fill")
                        continue

                    
                    # Gate: Bitget sometimes returns "No position to close" right after an entry fill.
                    # We only arm SL/TP when the position is visible on /position/all-position.
                    pos_total = await _get_position_total(trader, sym, direction, pos_mode)
                    if pos_total <= 0:
                        st["arm_attempts"] = attempts + 1
                        st["last_arm_fail_ts"] = time.time()
                        desk_log(logging.WARNING, "ARM", sym, tid, step="pos_not_visible", pos_total=pos_total)
                        continue

                    tick_meta = await _get_tick_cached(trader, sym)
                    tick_used = _sanitize_tick(sym, entry, tick_meta, tid)

                    # quantized exits
                    q_sl = _q_sl(sl, tick_used, direction)
                    q_tp1 = _q_tp_limit(tp1, tick_used, direction)

                    # ===== TP1 qty sizing with MIN USDT constraint =====
                    tp_trade_side = _tp_trade_side(pos_mode)

                    # base split
                    base_tp1 = _q_qty_floor(qty_total * TP1_CLOSE_PCT, qty_step)
                    base_tp1 = max(base_tp1, min_qty) if min_qty > 0 else base_tp1
                    base_tp1 = min(base_tp1, qty_total)

                    min_usdt = float(st.get("tp1_min_usdt") or MIN_TP_USDT_FALLBACK)
                    # required qty to reach min_usdt at TP1 price
                    req_qty = _q_qty_ceil((min_usdt / max(q_tp1, 1e-12)), qty_step)
                    req_qty = max(req_qty, min_qty) if min_qty > 0 else req_qty

                    qty_tp1 = max(base_tp1, req_qty)
                    qty_tp1 = min(qty_tp1, qty_total)

                    qty_rem = _q_qty_floor(qty_total - qty_tp1, qty_step)

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

                    # 1) SL
                    if not st.get("sl_plan_id") and (not st.get("sl_inflight", False)):
                        st["sl_inflight"] = True
                        desk_log(logging.INFO, "SL_SEND", sym, tid, close_side=close_side, trigger_type=_trigger_type_sl(), trigger_q=q_sl, qty=qty_total, tick=tick_used)

                        sl_resp = await trader.place_stop_market_sl(
                            symbol=sym,
                            close_side=close_side.lower(),
                            trigger_price=q_sl,
                            qty=qty_total,
                            reduce_only=_reduce_only_for_close(pos_mode),
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
                        desk_log(logging.INFO, "SL_OK", sym, tid, sl=q_sl, planId=sl_plan_id)
                        await send_telegram(_mk_exec_msg("SL_OK", sym, tid, sl=q_sl, planId=sl_plan_id))

                    # 2) TP1
                    if not st.get("tp1_order_id") and (not st.get("tp1_inflight", False)):
                        st["tp1_inflight"] = True
                        desk_log(logging.INFO, "TP1_SEND", sym, tid, close_side=close_side, price_q=q_tp1, qty=qty_tp1, tick=tick_used, trade_side=tp_trade_side, reduceOnly=_reduce_only_for_close(pos_mode))

                        tp1_resp = await trader.place_limit(
                            symbol=sym,
                            side=close_side.lower(),
                            price=q_tp1,
                            size=qty_tp1,
                            client_oid=_oid("tp1", tid, attempts),
                            trade_side=tp_trade_side,
                            reduce_only=_reduce_only_for_close(pos_mode),
                            tick_hint=tick_used,
                            debug_tag="TP1",
                        )
                        st["tp1_inflight"] = False

                        code = str(tp1_resp.get("code", ""))

                        # MIN USDT -> adjust qty_tp1 once then retry later
                        if (not _is_ok(tp1_resp)) and code == "45110":
                            msg = str(tp1_resp.get("msg") or "")
                            mmin = _parse_min_usdt(msg) or MIN_TP_USDT_FALLBACK
                            st["tp1_min_usdt"] = float(mmin)

                            # recompute required qty and set it (next attempt will use it)
                            req_qty2 = _q_qty_ceil((float(mmin) / max(q_tp1, 1e-12)), qty_step)
                            req_qty2 = max(req_qty2, min_qty) if min_qty > 0 else req_qty2
                            req_qty2 = min(req_qty2, qty_total)

                            # if runner becomes dust -> full close
                            rem2 = _q_qty_floor(qty_total - req_qty2, qty_step)
                            if rem2 < (min_qty if min_qty > 0 else 0.0):
                                req_qty2 = qty_total
                                rem2 = 0.0

                            st["qty_tp1"] = req_qty2
                            st["qty_rem"] = rem2

                            desk_log(logging.WARNING, "TP1_MINUSDT_ADJUST", sym, tid, min_usdt=mmin, new_qty_tp1=req_qty2, new_qty_rem=rem2, price=q_tp1)
                            # cooldown + retry later
                            st["arm_attempts"] = attempts + 1
                            st["last_arm_fail_ts"] = time.time()
                            continue

                        # price band clamp
                        if (not _is_ok(tp1_resp)) and code == "22047":
                            mn, mx = _parse_band(str(tp1_resp.get("msg") or ""))
                            clamped = _clamp_and_quantize(q_tp1, tick_used, mn, mx)
                            desk_log(logging.WARNING, "TP1_22047", sym, tid, mn=mn, mx=mx, before=q_tp1, after=clamped, tick=tick_used)
                            if clamped is None:
                                st["arm_attempts"] = attempts + 1
                                st["last_arm_fail_ts"] = time.time()
                                continue
                            tp1_resp = await trader.place_limit(
                                symbol=sym,
                                side=close_side.lower(),
                                price=clamped,
                                size=qty_tp1,
                                client_oid=_oid("tp1", tid, attempts + 1),
                                trade_side=tp_trade_side,
                                reduce_only=True,
                                tick_hint=tick_used,
                                debug_tag="TP1",
                            )

                        if not _is_ok(tp1_resp):
                            # hard fail: stop looping too fast
                            st["arm_attempts"] = attempts + 1
                            st["last_arm_fail_ts"] = time.time()
                            desk_log(logging.ERROR, "TP1_FAIL", sym, tid, code=tp1_resp.get("code"), msg=tp1_resp.get("msg"), dbg=tp1_resp.get("_debug"))
                            desk_log(logging.ERROR, "META_DUMP", sym, tid, meta=await trader.debug_meta(sym))
                            await send_telegram(_mk_exec_msg("TP1_FAIL", sym, tid, code=tp1_resp.get("code"), msg=tp1_resp.get("msg")))
                            continue

                        tp1_order_id = (tp1_resp.get("data") or {}).get("orderId") or tp1_resp.get("orderId")
                        async with PENDING_LOCK:
                            if tid in PENDING:
                                PENDING[tid]["tp1_order_id"] = str(tp1_order_id) if tp1_order_id else "ok"
                        desk_log(logging.INFO, "TP1_OK", sym, tid, tp1=q_tp1, orderId=tp1_order_id, qty_tp1=qty_tp1, qty_rem=qty_rem)
                        await send_telegram(_mk_exec_msg("TP1_OK", sym, tid, tp1=q_tp1, orderId=tp1_order_id, qty_tp1=qty_tp1, qty_rem=qty_rem))

                    # ARMED
                    async with PENDING_LOCK:
                        if tid in PENDING:
                            PENDING[tid]["armed"] = True

                    desk_log(logging.INFO, "ARMED", sym, tid, qty_total=qty_total, close_side=close_side, direction=direction, pos_mode=pos_mode, runner_qty=qty_rem)
                    continue

                # ---------------------------------------------------------
                # BE after TP1 filled (and SL resized to remaining qty)
                # ---------------------------------------------------------
                if st["armed"] and (not st.get("be_done", False)):
                    tp1_order_id = st.get("tp1_order_id")
                    if not tp1_order_id:
                        continue

                    tp1_detail = await trader.get_order_detail(sym, order_id=tp1_order_id)
                    if not trader.is_filled(tp1_detail):
                        continue

                    qty_total = float(st.get("qty_total") or 0.0)
                    qty_tp1 = float(st.get("qty_tp1") or 0.0)
                    qty_step = float(st.get("qty_step") or 1.0)

                    qty_rem = _q_qty_floor(max(0.0, qty_total - qty_tp1), qty_step)
                    if qty_rem <= 0:
                        desk_log(logging.INFO, "BE_SKIP", sym, tid, reason="no_runner_remaining")
                        # position should fully close once TP1 is filled; keep watching to cleanup risk/state
                        st["be_done"] = True
                        st["watch_close"] = True
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
                                await trader.cancel_plan_orders(symbol=sym, plan_ids=[old_plan])
                            elif hasattr(trader, "cancel_plan_order"):
                                # legacy (si jamais)
                                await trader.cancel_plan_order(symbol=sym, plan_id=old_plan)
                            desk_log(logging.INFO, "SL_CANCEL_OK", sym, tid, planId=old_plan)
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

                    # keep watching until the runner is fully closed
                    st["watch_close"] = True
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
