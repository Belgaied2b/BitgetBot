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
from typing import Any, Dict, Tuple, Optional, List, Set

import pandas as pd

# Desk context (macro + options) — optional
try:
    from macro_data import MacroCache  # type: ignore
    from options_data import OptionsCache  # type: ignore
except Exception:
    MacroCache = None  # type: ignore
    OptionsCache = None  # type: ignore

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
    POSITION_MODE,
)

from bitget_client import get_client
from bitget_trader import BitgetTrader
from analyze_signal import SignalAnalyzer

from structure_utils import analyze_structure, htf_trend_ok

from duplicate_guard import DuplicateGuard, fingerprint as make_fingerprint
from risk_manager import RiskManager
from retry_utils import retry_async

logger = logging.getLogger(__name__)

# =====================================================================
# Optional: institutional_data runtime mode switch (LIGHT/FULL)
# =====================================================================

try:
    import institutional_data as _inst_mod  # type: ignore
except Exception:
    _inst_mod = None  # type: ignore


def _set_inst_mode(mode: str) -> None:
    """
    Switch institutional_data.INST_MODE at runtime.
    Works with the institutional_data.py version that stores INST_MODE as a module global.
    Safe when you do it BETWEEN passes (no concurrent analyze running).
    """
    global _inst_mod
    if _inst_mod is None:
        return
    try:
        m = str(mode or "").upper().strip()
        if m not in ("LIGHT", "NORMAL", "FULL"):
            m = "LIGHT"
        setattr(_inst_mod, "INST_MODE", m)
    except Exception:
        pass


def _inst_is_banned_now() -> bool:
    """
    Reads circuit breaker state from institutional_data if available.
    (Uses private/internal vars if present; fallback False.)
    """
    global _inst_mod
    if _inst_mod is None:
        return False
    try:
        fn = getattr(_inst_mod, "_is_banned_now", None)
        if callable(fn):
            return bool(fn())
    except Exception:
        pass
    return False


def _inst_ban_until_ms() -> int:
    global _inst_mod
    if _inst_mod is None:
        return 0
    try:
        v = getattr(_inst_mod, "_BINANCE_BAN_UNTIL_MS", 0)
        return int(v or 0)
    except Exception:
        return 0


# =====================================================================
# Runtime globals
# =====================================================================

ALERT_GUARD = DuplicateGuard(ttl_seconds=1800)   # anti spam telegram
TRADE_GUARD = DuplicateGuard(ttl_seconds=3600)   # anti double trade

RISK = RiskManager()

TF_H1 = "1H"
TF_H4 = "4H"
CANDLE_LIMIT = 200

MAX_CONCURRENT_ANALYZE = 6  # limite scan

TP1_CLOSE_PCT = 0.50
WATCH_INTERVAL_S = 3.0

# Entry watcher policy
ENTRY_TTL_MARKET_S = int(os.getenv("ENTRY_TTL_MARKET_S", "300"))
ENTRY_TTL_OTE_S = int(os.getenv("ENTRY_TTL_OTE_S", "3600"))
ENTRY_TTL_FVG_S = int(os.getenv("ENTRY_TTL_FVG_S", "2700"))
ENTRY_TTL_RAID_S = int(os.getenv("ENTRY_TTL_RAID_S", "1800"))
ENTRY_TTL_DEFAULT_S = int(os.getenv("ENTRY_TTL_DEFAULT_S", "1800"))

RUNAWAY_ATR_MULT_MARKET = float(os.getenv("RUNAWAY_ATR_MULT_MARKET", "1.0"))
RUNAWAY_ATR_MULT_PULLBACK = float(os.getenv("RUNAWAY_ATR_MULT_PULLBACK", "1.5"))

PRICE_CHECK_INTERVAL_S = float(os.getenv("PRICE_CHECK_INTERVAL_S", "6"))
HEAVY_CHECK_INTERVAL_S = float(os.getenv("HEAVY_CHECK_INTERVAL_S", "45"))

REJECT_DEBUG_SAMPLES = 25

ARM_MAX_ATTEMPTS = 40
ARM_COOLDOWN_S = 8.0

FETCH_TIMEOUT_S = 12.0
ANALYZE_TIMEOUT_S = 22.0
META_TIMEOUT_S = 10.0
ORDER_TIMEOUT_S = 15.0
DETAIL_TIMEOUT_S = 10.0
TICK_TIMEOUT_S = 8.0

MIN_TP_USDT_FALLBACK = 5.0
_MIN_USDT_RE = re.compile(r"minimum amount\s*([0-9]*\.?[0-9]+)\s*USDT", re.IGNORECASE)

_MAX_RE = re.compile(r"maximum price limit:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_MIN_RE = re.compile(r"minimum price limit:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

# Alert policy (spam control)
ALERT_MODE = str(os.getenv("ALERT_MODE", "TOP_RANK")).strip().upper()  # TOP_RANK | ALL_VALID | EXEC_ONLY | NONE
MAX_ALERTS_PER_SCAN = int(os.getenv("MAX_ALERTS_PER_SCAN", str(max(5, int(MAX_ORDERS_PER_SCAN) * 2))))

# 2-pass pipeline
PASS2_ENABLED = str(os.getenv("INST_PASS2", "1")).strip() == "1"
SHORTLIST_SIZE = int(os.getenv("SHORTLIST_SIZE", str(max(6, min(14, int(MAX_ORDERS_PER_SCAN) * 3)))))
PASS2_CONCURRENCY = int(os.getenv("PASS2_CONCURRENCY", "2"))
PASS2_MODE = str(os.getenv("PASS2_MODE", "FULL")).strip().upper()
if PASS2_MODE not in ("FULL", "NORMAL"):
    PASS2_MODE = "FULL"

# Cooldowns
GLOBAL_COOLDOWN_UNTIL = 0.0
SYMBOL_COOLDOWN: Dict[str, float] = {}
SYMBOL_COOLDOWN_REASON: Dict[str, str] = {}

# Pending persistence
PENDING_STATE_FILE = os.getenv("PENDING_STATE_FILE", "pending_state.json")
PENDING_SAVE_THROTTLE_S = 4.0
_PENDING_LAST_SAVE_TS = 0.0

PENDING: Dict[str, Dict[str, Any]] = {}
PENDING_LOCK = asyncio.Lock()
WATCHER_TASK: Optional[asyncio.Task] = None

TICK_CACHE: Dict[str, float] = {}
TICK_LOCK = asyncio.Lock()

# Macro/options caches (single fetch per scan)
MACRO_CACHE = MacroCache() if MacroCache else None
OPTIONS_CACHE = OptionsCache() if OptionsCache else None


def _cooldown_global(until_ts: float, reason: str) -> None:
    global GLOBAL_COOLDOWN_UNTIL
    try:
        u = float(until_ts)
        if u > GLOBAL_COOLDOWN_UNTIL:
            GLOBAL_COOLDOWN_UNTIL = u
            logger.warning("[COOLDOWN] GLOBAL until=%.0f reason=%s", GLOBAL_COOLDOWN_UNTIL, reason)
    except Exception:
        pass


def _cooldown_symbol(symbol: str, seconds: float, reason: str) -> None:
    sym = str(symbol or "").upper()
    if not sym:
        return
    try:
        until = time.time() + float(seconds)
        prev = float(SYMBOL_COOLDOWN.get(sym) or 0.0)
        if until > prev:
            SYMBOL_COOLDOWN[sym] = until
            SYMBOL_COOLDOWN_REASON[sym] = str(reason or "")
    except Exception:
        pass


def _is_on_cooldown(symbol: str) -> bool:
    sym = str(symbol or "").upper()
    if not sym:
        return False
    now = time.time()
    if GLOBAL_COOLDOWN_UNTIL > now:
        return True
    until = float(SYMBOL_COOLDOWN.get(sym) or 0.0)
    return until > now


def _pending_serializable(state: Dict[str, Any]) -> Dict[str, Any]:
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
        "freeze_until",
        "sl_inflight",
        "tp1_inflight",
        "tp1_min_usdt",
        "qty_step",
        "min_qty",
        "q_tp1",
        "q_sl",
        "risk_rid",
        "risk_confirmed",
        "notional",
        "setup",
        "entry_type",
        "rr",
        "inst_score",
        "runner_monitor",
        "runner_qty",
        "tier",
        "confidence",
        "sizing_mult",
        "atr",
        "pd_mid",
        "last_price_ts",
        "last_price",
        "last_heavy_ts",
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
                sym = str(st.get("symbol") or "").upper()
                if not sym:
                    continue
                PENDING[str(tid)] = st
        logger.info("[BOOT] restored pending=%d from %s", len(PENDING), PENDING_STATE_FILE)
    except Exception as e:
        logger.warning("pending_load failed: %s", e)


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


def _as_dict(x: Any) -> Dict[str, Any]:
    """
    Defensive: some APIs may return list instead of dict.
    Converts:
      - dict -> dict
      - [dict] -> first dict
      - {"data":[dict]} -> first
    else -> {}
    """
    if isinstance(x, dict):
        # unwrap one-level list containers
        for k in ("data", "list", "rows"):
            v = x.get(k)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v[0]
        return x
    if isinstance(x, list) and x and isinstance(x[0], dict):
        return x[0]
    return {}


# =====================================================================
# Telegram hardened
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
    tier: Optional[str] = None,
    confidence: Optional[float] = None,
) -> str:
    tier_s = str(tier) if tier else "-"
    conf_s = _fmt_num(float(confidence)) if confidence is not None else "-"
    return (
        f"*📌 SIGNAL*\n"
        f"*{_tg_escape(symbol)}* — {_tg_escape(_fmt_side(side))}\n"
        f"setup: `{_tg_escape(setup)}` | inst: `{inst}` | tier: `{_tg_escape(tier_s)}` | conf: `{_tg_escape(conf_s)}`\n"
        f"entry_type: `{_tg_escape(entry_type)}` | pos_mode: `{_tg_escape(pos_mode)}`\n"
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
    return "BUY" if str(direction).upper() == "LONG" else "SELL"


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
    if isinstance(inst, dict):
        iscore = inst.get("institutional_score")
        if iscore is not None:
            return "inst_score_low" if int(iscore) < 2 else f"inst_score={iscore}"
        warnings = inst.get("warnings")
        if isinstance(warnings, list) and warnings:
            return "inst_unavailable"

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


def _order_state(detail: Any) -> str:
    try:
        if not isinstance(detail, dict):
            detail = _as_dict(detail)
        if not isinstance(detail, dict) or not detail:
            return "unknown"
        data = detail.get("data") or {}
        if isinstance(data, list):
            data = _as_dict(data)
        st = str((data or {}).get("state") or (data or {}).get("status") or "").lower()
        return st or "unknown"
    except Exception:
        return "unknown"


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


def _calc_atr14(df: pd.DataFrame, period: int = 14) -> float:
    try:
        if df is None or df.empty:
            return 0.0
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        close = pd.to_numeric(df["close"], errors="coerce")
        prev_close = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        v = float(atr.iloc[-1])
        return v if math.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _pd_mid(df: pd.DataFrame, lookback: int = 80) -> float:
    try:
        if df is None or df.empty:
            return 0.0
        dd = df.tail(int(lookback)).copy()
        hi = float(pd.to_numeric(dd["high"], errors="coerce").max())
        lo = float(pd.to_numeric(dd["low"], errors="coerce").min())
        if hi <= 0 or lo <= 0 or not math.isfinite(hi) or not math.isfinite(lo):
            return 0.0
        return (hi + lo) / 2.0
    except Exception:
        return 0.0


def _entry_ttl_s(entry_type: str, setup: str) -> int:
    et = str(entry_type or "").upper()
    sp = str(setup or "").upper()
    if "MARKET" in et:
        return int(ENTRY_TTL_MARKET_S)
    if "OTE" in sp:
        return int(ENTRY_TTL_OTE_S)
    if "FVG" in sp:
        return int(ENTRY_TTL_FVG_S)
    if "RAID" in sp or "SWEEP" in sp:
        return int(ENTRY_TTL_RAID_S)
    return int(ENTRY_TTL_DEFAULT_S)


def _runaway_mult(entry_type: str) -> float:
    et = str(entry_type or "").upper()
    return float(RUNAWAY_ATR_MULT_MARKET if "MARKET" in et else RUNAWAY_ATR_MULT_PULLBACK)


# =====================================================================
# ✅ FIX: robust last price (Bitget may return dict OR list)
# =====================================================================

def _norm_sym(sym: str) -> str:
    return str(sym or "").upper().replace("-", "").replace("_", "").replace("USDTM", "USDT").replace("UMCBL", "")


def _pick_ticker_row(data: Any, symbol: str) -> Dict[str, Any]:
    symn = _norm_sym(symbol)

    if isinstance(data, dict):
        for k in ("data", "list", "tickers", "rows"):
            v = data.get(k)
            if isinstance(v, list) and v:
                data = v
                break
        if isinstance(data, dict):
            return data

    if isinstance(data, list):
        for it in data:
            if not isinstance(it, dict):
                continue
            s = it.get("symbol") or it.get("symbolName") or it.get("instId") or it.get("contractCode")
            if s and _norm_sym(str(s)) == symn:
                return it
        for it in data:
            if isinstance(it, dict):
                return it

    return {}


async def _get_last_price(trader: BitgetTrader, symbol: str) -> float:
    sym = str(symbol).upper()
    try:
        resp = await trader.client._request(
            "GET",
            "/api/v2/mix/market/ticker",
            params={"symbol": sym, "productType": trader.product_type},
            auth=False,
        )
    except Exception:
        return 0.0

    if not _is_ok(resp):
        return 0.0

    raw = resp.get("data")
    row = _pick_ticker_row(raw, sym)

    for k in ("last", "lastPr", "lastPrice", "close", "price", "markPrice"):
        v = row.get(k)
        if v is None:
            continue
        px = _safe_float(v, 0.0)
        if px > 0:
            return px

    if isinstance(raw, dict):
        for k in ("last", "lastPr", "lastPrice", "close", "price", "markPrice"):
            v = raw.get(k)
            if v is None:
                continue
            px = _safe_float(v, 0.0)
            if px > 0:
                return px

    return 0.0


async def _invalidation_check(client, symbol: str, direction: str, entry_price: float) -> Tuple[bool, str, Dict[str, Any]]:
    sym = str(symbol).upper()

    async def _do():
        df_h1 = await client.get_klines_df(sym, TF_H1, 140)
        df_h4 = await client.get_klines_df(sym, TF_H4, 140)
        return df_h1, df_h4

    try:
        df_h1, df_h4 = await asyncio.wait_for(retry_async(_do, retries=1, base_delay=0.25), timeout=FETCH_TIMEOUT_S)
    except Exception:
        return True, "", {}

    if df_h1 is None or df_h4 is None or getattr(df_h1, "empty", True) or getattr(df_h4, "empty", True):
        return True, "", {}

    try:
        if not htf_trend_ok(df_h4, direction):
            return False, "invalidate:htf_veto", {}
    except Exception:
        pass

    try:
        st1 = analyze_structure(df_h1)
        tr = str(st1.get("trend") or "").upper()
        if tr in ("LONG", "SHORT") and tr != str(direction).upper():
            return False, "invalidate:trend_flip_h1", {"trend_h1": tr}
        if bool(st1.get("choch")):
            bos_dir = str(st1.get("bos_direction") or "").upper()
            if bos_dir in ("LONG", "SHORT") and bos_dir != str(direction).upper():
                return False, "invalidate:choch_flip", {"bos_dir": bos_dir}
    except Exception:
        pass

    mid = _pd_mid(df_h1, 80)
    if mid > 0 and math.isfinite(mid):
        if str(direction).upper() == "LONG" and float(entry_price) > float(mid):
            return False, "invalidate:entry_premium", {"mid": mid}
        if str(direction).upper() == "SHORT" and float(entry_price) < float(mid):
            return False, "invalidate:entry_discount", {"mid": mid}

    return True, "", {"mid": mid}


async def _cancel_pending_entry(
    trader: BitgetTrader,
    tid: str,
    st: Dict[str, Any],
    reason: str,
    **kv: Any,
) -> None:
    sym = str(st.get("symbol") or "").upper()
    direction = str(st.get("direction") or "")
    try:
        oid = st.get("entry_order_id")
        coid = st.get("entry_client_oid")
        if oid or coid:
            try:
                resp = await asyncio.wait_for(trader.cancel_order(sym, order_id=oid, client_oid=coid), timeout=ORDER_TIMEOUT_S)
            except Exception as e:
                resp = {"code": "EXC", "msg": str(e)}
            desk_log(logging.INFO, "ENTRY_CANCEL", sym, tid, reason=reason, code=resp.get("code"), msg=resp.get("msg"))
    except Exception:
        pass

    if bool(st.get("risk_confirmed", False)):
        await _risk_close_trade(trader, sym, direction, reason=reason, pnl_override=0.0)
    else:
        try:
            await RISK.cancel_reservation(st.get("risk_rid"))
        except Exception:
            pass

    async with PENDING_LOCK:
        PENDING.pop(tid, None)

    await send_telegram(_mk_exec_msg("ENTRY_CANCEL", sym, tid, reason=reason, **kv))


# =====================================================================
# Position / PnL helpers (risk close + bootstrap)
# =====================================================================

def _hold_side(direction: str) -> str:
    return "long" if str(direction).upper() == "LONG" else "short"


async def _get_position_total(trader: BitgetTrader, symbol: str, direction: str) -> float:
    sym = str(symbol).upper()
    hold = _hold_side(direction)
    try:
        resp = await trader.client._request(
            "GET",
            "/api/v2/mix/position/single-position",
            params={
                "productType": trader.product_type,
                "symbol": sym,
                "marginCoin": trader.margin_coin,
            },
            auth=True,
        )
    except Exception:
        return -1.0

    if not _is_ok(resp):
        return -1.0

    data = resp.get("data") or []
    if isinstance(data, dict):
        data = data.get("list") or data.get("data") or []
    if not isinstance(data, list):
        return 0.0

    for row in data:
        if not isinstance(row, dict):
            continue
        hs = str(row.get("holdSide") or row.get("posSide") or row.get("positionSide") or "").lower()
        if hs and hs != hold:
            continue
        tot = _safe_float(row.get("total"), 0.0)
        if tot != 0:
            return abs(tot)

    for row in data:
        if isinstance(row, dict):
            tot = _safe_float(row.get("total"), 0.0)
            if tot != 0:
                return abs(tot)

    if len(data) == 1 and isinstance(data[0], dict):
        return _safe_float(data[0].get("total"), 0.0)

    return 0.0


async def _fetch_last_closed_pnl(trader: BitgetTrader, symbol: str, direction: str) -> float:
    sym = str(symbol).upper()
    hold = _hold_side(direction)
    try:
        resp = await trader.client._request(
            "GET",
            "/api/v2/mix/position/history-position",
            params={
                "symbol": sym,
                "productType": trader.product_type,
                "limit": "20",
            },
            auth=True,
        )
    except Exception:
        return 0.0

    if not _is_ok(resp):
        return 0.0

    data = resp.get("data") or {}
    lst = data.get("list") if isinstance(data, dict) else None
    if not isinstance(lst, list):
        return 0.0

    for row in lst:
        if not isinstance(row, dict):
            continue
        if str(row.get("holdSide") or "").lower() != hold:
            continue
        pnl = row.get("pnl")
        if pnl is None:
            pnl = row.get("netProfit")
        return _safe_float(pnl, 0.0)

    return 0.0


async def _risk_close_trade(trader: BitgetTrader, symbol: str, direction: str, reason: str, pnl_override: Optional[float] = None) -> None:
    pnl = float(pnl_override) if pnl_override is not None else await _fetch_last_closed_pnl(trader, symbol, direction)
    try:
        RISK.register_closed(symbol, direction, pnl)
    except Exception:
        pass
    desk_log(logging.INFO, "RISK_CLOSE", symbol, "-", side=direction, pnl=pnl, reason=reason)


async def _bootstrap_risk_open_positions(trader: BitgetTrader) -> None:
    try:
        resp = await trader.client._request(
            "GET",
            "/api/v2/mix/position/all-position",
            params={
                "productType": trader.product_type,
                "marginCoin": trader.margin_coin,
            },
            auth=True,
        )
    except Exception as e:
        logger.warning("[BOOT] risk bootstrap failed: %s", e)
        return

    if not _is_ok(resp):
        return

    data = resp.get("data") or []
    if not isinstance(data, list):
        return

    added = 0
    for row in data:
        if not isinstance(row, dict):
            continue
        total = _safe_float(row.get("total"), 0.0)
        if total <= 0:
            continue
        sym = str(row.get("symbol") or "").upper()
        if not sym:
            continue
        hs = str(row.get("holdSide") or "").lower()
        direction = "LONG" if hs == "long" else "SHORT"
        mark = _safe_float(row.get("markPrice"), 0.0) or _safe_float(row.get("openPriceAvg"), 0.0)
        notional = abs(total) * float(mark)
        risk_used = RISK.risk_for_this_trade()
        try:
            RISK.register_open(sym, direction, notional, risk_used)
            added += 1
        except Exception:
            pass

    if added:
        logger.info("[BOOT] risk bootstrap: registered %d open positions from exchange", added)


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
        self.alert_sent = 0
        self.duplicates_alert = 0
        self.duplicates_trade = 0
        self.risk_rejects = 0
        self.exec_selected = 0
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
# Analyze worker (Pass 1 / Pass 2)
# =====================================================================

def _extract_tier_confidence(result: Dict[str, Any]) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """
    Reads optional tier/confidence/sizing_mult if analyze_signal provides them.
    Supports multiple shapes to avoid coupling.
    """
    tier = result.get("tier")
    conf = result.get("confidence")
    sizing = result.get("sizing_mult")
    if tier is None:
        d = result.get("decision") or {}
        if isinstance(d, dict):
            tier = d.get("tier") or tier
            conf = d.get("confidence") if conf is None else conf
            sizing = d.get("sizing_mult") if sizing is None else sizing
    try:
        tier_s = str(tier) if tier is not None else None
    except Exception:
        tier_s = None
    try:
        conf_f = float(conf) if conf is not None else None
    except Exception:
        conf_f = None
    try:
        sizing_f = float(sizing) if sizing is not None else None
    except Exception:
        sizing_f = None
    return tier_s, conf_f, sizing_f


def _maybe_update_ban_and_cooldowns_from_result(sym: str, result: Dict[str, Any]) -> None:
    """
    If institutional_data circuit breaker is on / warnings contain ban-until,
    set global cooldown to ban expiry (plus small buffer).
    """
    inst = result.get("institutional")
    if not isinstance(inst, dict):
        return

    warnings = inst.get("warnings")
    if isinstance(warnings, list):
        for w in warnings:
            ws = str(w or "")
            if "binance_ip_banned_until_ms=" in ws:
                try:
                    ms = int(ws.split("binance_ip_banned_until_ms=")[-1].strip())
                    until = (ms / 1000.0) + 3.0
                    _cooldown_global(until, reason="binance_ban_circuit_breaker")
                    _cooldown_symbol(sym, max(60.0, until - time.time()), reason="binance_ban_circuit_breaker")
                except Exception:
                    pass


async def analyze_symbol(
    sym: str,
    client,
    analyze_sem: asyncio.Semaphore,
    stats: ScanStats,
    analyzer: SignalAnalyzer,
    macro_ctx: Optional[Dict[str, Any]] = None,
    *,
    reuse_tid: Optional[str] = None,
    pass_tag: str = "P1",
) -> Optional[Dict[str, Any]]:
    sym = str(sym).upper()
    if _is_on_cooldown(sym):
        await stats.inc("skips", 1)
        await stats.add_reason("cooldown")
        return None

    tid = reuse_tid or _new_tid(sym)
    await stats.inc("total", 1)

    async with analyze_sem:
        t0 = time.time()
        df_h1, df_h4 = await _fetch_dfs(client, sym)
        fetch_ms = int((time.time() - t0) * 1000)

        if df_h1 is None or df_h4 is None or getattr(df_h1, "empty", True) or getattr(df_h4, "empty", True):
            await stats.inc("skips", 1)
            await stats.add_reason("fetch_empty")
            return None
        if len(df_h1) < 80 or len(df_h4) < 80:
            await stats.inc("skips", 1)
            await stats.add_reason("fetch_short")
            return None

        t1 = time.time()
        try:
            result = await asyncio.wait_for(analyzer.analyze(sym, df_h1, df_h4, macro=(macro_ctx or {})), timeout=ANALYZE_TIMEOUT_S)
        except asyncio.TimeoutError:
            result = {"valid": False, "reject_reason": "analyze_timeout"}
        except Exception as e:
            result = {"valid": False, "reject_reason": f"analyze_exc:{type(e).__name__}"}
        analyze_ms = int((time.time() - t1) * 1000)

        if not result or not isinstance(result, dict) or not result.get("valid"):
            reason = _extract_reject_reason(result)
            if await stats.take_reject_debug_slot():
                has_fields = "Y" if (isinstance(result, dict) and _has_key_fields_for_trade(result)) else "N"
                desk_log(logging.INFO, "REJ", sym, tid, pass_=pass_tag, fetch_ms=fetch_ms, analyze_ms=analyze_ms, reason=reason, has_fields=has_fields)
            await stats.inc("rejects", 1)
            await stats.add_reason(reason)

            # soft cooldown on repeated timeouts / exceptions
            if reason in ("analyze_timeout",) or reason.startswith("analyze_exc:"):
                _cooldown_symbol(sym, 30.0, reason=reason)
            return None

        # update ban/cooldowns if circuit breaker warns
        try:
            _maybe_update_ban_and_cooldowns_from_result(sym, result)
        except Exception:
            pass

        side = str(result.get("side", "")).upper()
        entry = _safe_float(result.get("entry"), 0.0)
        sl = _safe_float(result.get("sl"), 0.0)
        tp1 = _safe_float(result.get("tp1"), 0.0)
        rr = _safe_float(result.get("rr"), 0.0)
        setup = str(result.get("setup_type") or "-")
        entry_type = str(result.get("entry_type") or "MARKET")

        inst = result.get("institutional") or {}
        inst_score_eff = int(result.get("inst_score_eff") or (inst.get("institutional_score") if isinstance(inst, dict) else 0) or 0)
        comp = result.get("composite") or {}
        comp_score = float(comp.get("score") or result.get("composite_score") or 0.0)

        tier, confidence, sizing_mult = _extract_tier_confidence(result)

        desk_log(
            logging.INFO,
            "EXITS",
            sym,
            tid,
            pass_=pass_tag,
            side=side,
            entry=entry,
            sl=sl,
            tp1=tp1,
            rr=rr,
            setup=setup,
            entry_type=entry_type,
            inst=inst_score_eff,
            comp=comp_score,
            tier=tier,
            conf=(float(confidence) if confidence is not None else None),
            size_mult=(float(sizing_mult) if sizing_mult is not None else None),
            fetch_ms=fetch_ms,
            analyze_ms=analyze_ms,
        )

        if entry <= 0 or sl <= 0 or tp1 <= 0 or rr <= 0 or side not in ("BUY", "SELL"):
            await stats.inc("skips", 1)
            await stats.add_reason("missing_exits")
            return None

        direction = _direction_from_side(side)
        close_side = _close_side_from_direction(direction)
        pos_mode = str(POSITION_MODE or "hedge")

        tick_est = _estimate_tick_from_price(entry)
        q_entry_fp = _q_entry(entry, tick_est, side)
        q_sl_fp = _q_sl(sl, tick_est, direction)
        q_tp1_fp = _q_tp_limit(tp1, tick_est, direction)

        atr14 = _calc_atr14(df_h1, 14)
        mid_pd = _pd_mid(df_h1, 80)

        fp_alert = make_fingerprint(
            sym, side, q_entry_fp, q_sl_fp, q_tp1_fp,
            extra=f"{setup}|{entry_type}|{pos_mode}",
            precision=10
        )

        return {
            "tid": tid,
            "symbol": sym,
            "side": side,
            "direction": direction,
            "close_side": close_side,
            "pos_mode": pos_mode,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "rr": rr,
            "setup": setup,
            "entry_type": entry_type,
            "inst_score": inst_score_eff,
            "comp_score": comp_score,
            "fp_alert": fp_alert,
            "fetch_ms": fetch_ms,
            "analyze_ms": analyze_ms,
            "atr": float(atr14),
            "pd_mid": float(mid_pd),
            "tier": tier,
            "confidence": confidence,
            "sizing_mult": sizing_mult,
            "pass_tag": pass_tag,
        }


# =====================================================================
# Ranking (phase B)
# =====================================================================

def _setup_priority(setup: str) -> int:
    s = (setup or "").upper()
    if "BOS" in s:
        return 2
    if "INST" in s:
        return 1
    return 0


def rank_candidates(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(c: Dict[str, Any]):
        return (
            int(c.get("inst_score") or 0),
            _setup_priority(c.get("setup") or ""),
            float(c.get("rr") or 0.0),
            float(c.get("comp_score") or 0.0),
        )
    return sorted(cands, key=key, reverse=True)


def _unique_syms_top(cands: List[Dict[str, Any]], n: int) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for c in cands:
        s = str(c.get("symbol") or "").upper()
        if not s or s in seen:
            continue
        out.append(s)
        seen.add(s)
        if len(out) >= n:
            break
    return out


# =====================================================================
# Execution (phase C)
# =====================================================================

async def execute_candidate(candidate: Dict[str, Any], client, trader: BitgetTrader, stats: ScanStats) -> None:
    sym = candidate["symbol"]
    tid = candidate["tid"]
    side = candidate["side"]
    direction = candidate["direction"]
    close_side = candidate["close_side"]
    pos_mode = candidate["pos_mode"]

    entry = float(candidate["entry"])
    sl = float(candidate["sl"])
    tp1 = float(candidate["tp1"])
    rr = float(candidate["rr"])
    setup = str(candidate["setup"])
    entry_type = str(candidate.get("entry_type") or "MARKET")
    inst_score = int(candidate.get("inst_score") or 0)

    tier = candidate.get("tier")
    confidence = candidate.get("confidence")
    sizing_mult = candidate.get("sizing_mult")

    # sizing multiplier (A/B/C tiers later will feed this)
    try:
        sm = float(sizing_mult) if sizing_mult is not None else 1.0
    except Exception:
        sm = 1.0
    sm = max(0.05, min(1.0, sm))

    notional = float(MARGIN_USDT) * float(LEVERAGE) * sm
    post_only_entry = str(entry_type).upper() != "MARKET"

    # avoid stacking same symbol+direction while a previous trade is still pending/managed
    try:
        async with PENDING_LOCK:
            for _tid, _st in PENDING.items():
                if str(_st.get("symbol") or "").upper() != sym:
                    continue
                if str(_st.get("direction") or "").upper() != direction:
                    continue
                await stats.add_reason("skip:symbol_pending")
                await send_telegram(_mk_exec_msg("EXEC_SKIPPED", sym, tid, reason="symbol_pending"))
                return
    except Exception:
        pass

    allowed, rreason, risk_rid = await RISK.reserve_trade(
        symbol=sym,
        side=direction,
        notional=notional,
        rr=rr if rr > 0 else None,
        inst_score=int(candidate.get("inst_score") or 0),
        commitment=None,
    )
    if not allowed:
        await stats.inc("risk_rejects", 1)
        await stats.add_reason(f"risk:{rreason}")
        await send_telegram(_mk_exec_msg("EXEC_SKIPPED", sym, tid, reason=f"risk:{rreason}"))
        return

    # meta/tick
    try:
        meta_dbg_raw = await asyncio.wait_for(trader.debug_meta(sym), timeout=META_TIMEOUT_S)
    except Exception:
        meta_dbg_raw = {}

    meta_dbg = _as_dict(meta_dbg_raw) if not isinstance(meta_dbg_raw, dict) else meta_dbg_raw

    tick_meta = await _get_tick_cached(trader, sym)
    tick_used = _sanitize_tick(sym, entry, tick_meta, tid)
    q_entry = _q_entry(entry, tick_used, side)

    # ---- TRADE fingerprint (real tick, prevents duplicates) ----
    q_sl_fp = _q_sl(sl, tick_used, direction)
    q_tp1_fp = _q_tp_limit(tp1, tick_used, direction)
    fp_trade = make_fingerprint(
        sym, side, q_entry, q_sl_fp, q_tp1_fp,
        extra=f"{setup}|{candidate.get('entry_type') or 'MARKET'}|{pos_mode}",
        precision=12
    )

    if TRADE_GUARD.is_duplicate(fp_trade):
        await stats.inc("duplicates_trade", 1)
        await stats.add_reason("duplicate_trade")
        await RISK.cancel_reservation(risk_rid)
        return

    desk_log(
        logging.INFO,
        "EXEC_PRE",
        sym,
        tid,
        entry_raw=entry,
        tick_meta=tick_meta,
        tick_used=tick_used,
        q_entry=q_entry,
        direction=direction,
        pos_mode=pos_mode,
        tier=tier,
        conf=(float(confidence) if confidence is not None else None),
        size_mult=sm,
        meta_pricePlace=meta_dbg.get("pricePlace"),
        meta_priceTick=meta_dbg.get("priceTick"),
        meta_qtyStep=meta_dbg.get("qtyStep"),
        meta_minQty=meta_dbg.get("minQty"),
    )

    if q_entry <= 0:
        await stats.inc("exec_failed", 1)
        await stats.add_reason("entry_q_zero")
        await RISK.cancel_reservation(risk_rid)
        return

    await stats.inc("exec_sent", 1)
    desk_log(logging.INFO, "EXEC", sym, tid, action="entry_send", entry=q_entry, notional=round(notional, 2), setup=setup, tier=tier, conf=confidence)
    await send_telegram(_mk_exec_msg("ENTRY_SEND", sym, tid, entry=q_entry, notional=round(notional, 2), setup=setup, tier=tier, conf=str(confidence) if confidence is not None else None))

    entry_client_oid = f"entry-{tid}"

    # fallback factors for 40762 etc
    factors = [1.0, 0.5, 0.25]

    entry_resp: Dict[str, Any] = {}
    for k, f in enumerate(factors):
        async def _place_entry():
            if f >= 0.999:
                return await trader.place_limit(
                    symbol=sym,
                    side=side.lower(),
                    price=q_entry,
                    size=None,
                    client_oid=entry_client_oid,
                    trade_side="open",
                    reduce_only=False,
                    post_only=post_only_entry,
                    tick_hint=tick_used,
                    debug_tag="ENTRY",
                )

            raw_qty = (notional * f) / max(q_entry, 1e-12)
            return await trader.place_limit(
                symbol=sym,
                side=side.lower(),
                price=q_entry,
                size=float(raw_qty),
                client_oid=f"{entry_client_oid}-{int(f*100)}",
                trade_side="open",
                reduce_only=False,
                tick_hint=tick_used,
                debug_tag="ENTRY",
            )

        try:
            entry_resp = await asyncio.wait_for(retry_async(_place_entry, retries=2, base_delay=0.35), timeout=ORDER_TIMEOUT_S)
        except Exception as e:
            entry_resp = {"code": "EXC", "msg": str(e)}

        if _is_ok(entry_resp):
            break

        if str(entry_resp.get("code")) == "40762":
            desk_log(logging.WARNING, "ENTRY_40762", sym, tid, attempt=k, factor=f, msg=entry_resp.get("msg"))
            continue
        else:
            break

    if not _is_ok(entry_resp):
        await stats.inc("exec_failed", 1)
        desk_log(logging.ERROR, "ENTRY_FAIL", sym, tid, code=entry_resp.get("code"), msg=entry_resp.get("msg"), dbg=(entry_resp.get("_debug") or {}))
        try:
            md = await asyncio.wait_for(trader.debug_meta(sym), timeout=META_TIMEOUT_S)
        except Exception:
            md = {}
        desk_log(logging.ERROR, "META_DUMP", sym, tid, meta=md)
        await send_telegram(_mk_exec_msg("ENTRY_FAIL", sym, tid, code=entry_resp.get("code"), msg=entry_resp.get("msg")))
        await RISK.cancel_reservation(risk_rid)
        return

    entry_order_id = (entry_resp.get("data") or {}).get("orderId") or entry_resp.get("orderId")
    qty_total = _safe_float(entry_resp.get("qty"), 0.0)

    desk_log(logging.INFO, "ENTRY_OK", sym, tid, orderId=entry_order_id, qty=qty_total)

    try:
        await RISK.confirm_open(risk_rid)
    except Exception:
        pass

    TRADE_GUARD.mark(fp_trade)

    async with PENDING_LOCK:
        PENDING[tid] = {
            "symbol": sym,
            "entry_side": side.upper(),
            "direction": direction,
            "close_side": close_side,
            "pos_mode": pos_mode,
            "setup": setup,
            "entry_type": entry_type,
            "inst_score": inst_score,
            "rr": rr,
            "tier": tier,
            "confidence": confidence,
            "sizing_mult": sm,
            "notional": float(notional),
            "risk_rid": str(risk_rid),
            "risk_confirmed": True,
            "freeze_until": 0.0,
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
            "atr": float(candidate.get("atr") or 0.0),
            "pd_mid": float(candidate.get("pd_mid") or 0.0),
            "last_price_ts": 0.0,
            "last_price": 0.0,
            "last_heavy_ts": 0.0,
        }

    await _pending_save(force=True)
    desk_log(logging.INFO, "PENDING_NEW", sym, tid, setup=setup, rr=rr, tier=tier, conf=confidence, size_mult=sm)


# =====================================================================
# WATCHER
# =====================================================================

async def _watcher_loop(client, trader: BitgetTrader) -> None:
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
                close_side = _close_side_from_direction(direction)
                pos_mode = str(st.get("pos_mode") or "one_way")

                entry = float(st.get("entry") or 0.0)
                sl = float(st.get("sl") or 0.0)
                tp1 = float(st.get("tp1") or 0.0)

                qty_step = float(st.get("qty_step") or 1.0)
                min_qty = float(st.get("min_qty") or 0.0)

                last_fail = float(st.get("last_arm_fail_ts") or 0.0)
                if last_fail > 0 and (time.time() - last_fail) < ARM_COOLDOWN_S:
                    continue

                freeze_until = float(st.get("freeze_until") or 0.0)
                if freeze_until > time.time():
                    continue
                if freeze_until > 0:
                    st["freeze_until"] = 0.0
                    dirty = True

                attempts = int(st.get("arm_attempts") or 0)
                if attempts >= ARM_MAX_ATTEMPTS:
                    freeze_s = int(os.getenv("ARM_FREEZE_SECONDS", "900"))
                    freeze_s = max(60, int(freeze_s))
                    st["freeze_until"] = time.time() + float(freeze_s)
                    st["arm_attempts"] = 0
                    st["last_arm_fail_ts"] = time.time()
                    desk_log(logging.ERROR, "ARM_FREEZE", sym, tid, attempts=attempts, freeze_s=freeze_s)
                    dirty = True
                    await send_telegram(_mk_exec_msg("ARM_FREEZE", sym, tid, attempts=attempts, freeze_s=freeze_s))
                    continue

                # If BE already placed (runner mode), just monitor position closure
                if bool(st.get("be_done", False)):
                    pos_total = await _get_position_total(trader, sym, direction)
                    if pos_total >= 0 and pos_total <= 0:
                        await _risk_close_trade(trader, sym, direction, reason="position_closed_after_be")
                        async with PENDING_LOCK:
                            PENDING.pop(tid, None)
                        dirty = True
                        await send_telegram(_mk_exec_msg("DONE", sym, tid, reason="position_closed"))
                    continue

                # ------------------------------------------------------------
                # Phase 0: pending entry not filled yet -> TTL/runaway/heavy invalidation
                # ------------------------------------------------------------
                if not bool(st.get("armed", False)):
                    # TTL cancel (unfilled entry)
                    created_ts = float(st.get("created_ts") or 0.0)
                    ttl_s = _entry_ttl_s(str(st.get("entry_type") or ""), str(st.get("setup") or ""))
                    if created_ts > 0 and ttl_s > 0 and (time.time() - created_ts) > float(ttl_s):
                        await _cancel_pending_entry(trader, tid, st, reason="ttl_expired", ttl_s=ttl_s)
                        dirty = True
                        continue

                    # Price runaway check
                    atr = float(st.get("atr") or 0.0)
                    if atr > 0:
                        last_px_ts = float(st.get("last_price_ts") or 0.0)
                        if (time.time() - last_px_ts) >= float(PRICE_CHECK_INTERVAL_S):
                            px = await _get_last_price(trader, sym)
                            if px > 0:
                                st["last_price_ts"] = time.time()
                                st["last_price"] = float(px)
                                dirty = True

                                mult = _runaway_mult(str(st.get("entry_type") or ""))
                                dist = abs(float(px) - float(entry))
                                if dist >= float(atr) * float(mult):
                                    await _cancel_pending_entry(
                                        trader, tid, st,
                                        reason="runaway",
                                        last_price=float(px),
                                        atr=float(atr),
                                        mult=float(mult),
                                        dist=float(dist),
                                    )
                                    dirty = True
                                    continue

                    # Heavy revalidation (HTF/trend/CHoCH/premium-discount)
                    last_hvy_ts = float(st.get("last_heavy_ts") or 0.0)
                    if (time.time() - last_hvy_ts) >= float(HEAVY_CHECK_INTERVAL_S):
                        st["last_heavy_ts"] = time.time()
                        dirty = True
                        ok, why, extra = await _invalidation_check(client, sym, direction, entry_price=entry)
                        if not ok:
                            await _cancel_pending_entry(trader, tid, st, reason=why, **(extra or {}))
                            dirty = True
                            continue

                    # entry order detail (arm on fill)
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

                        if bool(st.get("risk_confirmed", False)):
                            await _risk_close_trade(trader, sym, direction, reason=f"entry_state={state}", pnl_override=0.0)
                        else:
                            try:
                                await RISK.cancel_reservation(st.get("risk_rid"))
                            except Exception:
                                pass

                        async with PENDING_LOCK:
                            PENDING.pop(tid, None)
                        dirty = True
                        continue

                    # still not filled
                    try:
                        if not trader.is_filled(detail):
                            continue
                    except Exception:
                        continue

                    qty_total = float(st.get("qty_total") or 0.0)
                    if qty_total <= 0:
                        dct = _as_dict(detail)
                        data = dct.get("data") or {}
                        if isinstance(data, list):
                            data = _as_dict(data)
                        qty_total = float((data or {}).get("size") or (data or {}).get("quantity") or (data or {}).get("qty") or 0.0)

                    if qty_total <= 0:
                        st["arm_attempts"] = attempts + 1
                        st["last_arm_fail_ts"] = time.time()
                        desk_log(logging.WARNING, "ARM", sym, tid, step="no_qty_from_fill")
                        dirty = True
                        continue

                    # Wait position visible before close orders (avoid 22002)
                    pos_total = await _get_position_total(trader, sym, direction)
                    if pos_total == 0:
                        desk_log(logging.INFO, "ARM_WAIT_POS", sym, tid, step="pos_not_ready")
                        st["last_arm_fail_ts"] = time.time()
                        dirty = True
                        continue
                    if pos_total > 0:
                        qty_total = min(qty_total, float(pos_total))

                    tick_meta = await _get_tick_cached(trader, sym)
                    tick_used = _sanitize_tick(sym, entry, tick_meta, tid)

                    q_sl = _q_sl(sl, tick_used, direction)
                    q_tp1 = _q_tp_limit(tp1, tick_used, direction)

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
                    if qty_rem < (min_qty if min_qty > 0 else 0.0):
                        qty_tp1 = qty_total
                        qty_rem = 0.0

                    st["qty_total"] = qty_total
                    st["qty_tp1"] = qty_tp1
                    st["qty_rem"] = qty_rem
                    st["q_tp1"] = q_tp1
                    st["q_sl"] = q_sl

                    # 1) SL
                    if not st.get("sl_plan_id") and (not bool(st.get("sl_inflight", False))):
                        st["sl_inflight"] = True
                        dirty = True

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
                            code2 = str(sl_resp.get("code", ""))
                            if code2 == "22002":
                                st["last_arm_fail_ts"] = time.time()
                                dirty = True
                                desk_log(logging.WARNING, "SL_WAIT_POS", sym, tid, code=code2, msg=sl_resp.get("msg"))
                                continue
                            st["arm_attempts"] = attempts + 1
                            st["last_arm_fail_ts"] = time.time()
                            desk_log(logging.ERROR, "SL_FAIL", sym, tid, code=sl_resp.get("code"), msg=sl_resp.get("msg"), dbg=sl_resp.get("_debug"))
                            await send_telegram(_mk_exec_msg("SL_FAIL", sym, tid, code=sl_resp.get("code"), msg=sl_resp.get("msg")))
                            continue

                        sl_plan_id = (
                            (sl_resp.get("data") or {}).get("orderId")
                            or (sl_resp.get("data") or {}).get("planOrderId")
                            or sl_resp.get("orderId")
                        )
                        st["sl_plan_id"] = str(sl_plan_id) if sl_plan_id else "ok"
                        dirty = True
                        await send_telegram(_mk_exec_msg("SL_OK", sym, tid, sl=q_sl, planId=sl_plan_id))

                    # 2) TP1
                    if not st.get("tp1_order_id") and (not bool(st.get("tp1_inflight", False))):
                        st["tp1_inflight"] = True
                        dirty = True

                        async def _place_tp1(price_use: float, qty_use: float, attempt_k: int):
                            return await trader.place_limit(
                                symbol=sym,
                                side=close_side.lower(),
                                price=price_use,
                                size=qty_use,
                                client_oid=_oid("tp1", tid, attempt_k),
                                trade_side="close",
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

                        if (not _is_ok(tp1_resp)) and code == "22002":
                            st["last_arm_fail_ts"] = time.time()
                            dirty = True
                            desk_log(logging.WARNING, "TP1_WAIT_POS", sym, tid, code=code, msg=tp1_resp.get("msg"))
                            continue

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
                            st["arm_attempts"] = attempts + 1
                            st["last_arm_fail_ts"] = time.time()
                            dirty = True
                            continue

                        if not _is_ok(tp1_resp):
                            st["arm_attempts"] = attempts + 1
                            st["last_arm_fail_ts"] = time.time()
                            dirty = True
                            desk_log(logging.ERROR, "TP1_FAIL", sym, tid, code=tp1_resp.get("code"), msg=tp1_resp.get("msg"), dbg=tp1_resp.get("_debug"))
                            await send_telegram(_mk_exec_msg("TP1_FAIL", sym, tid, code=tp1_resp.get("code"), msg=tp1_resp.get("msg")))
                            continue

                        tp1_order_id = (tp1_resp.get("data") or {}).get("orderId") or tp1_resp.get("orderId")
                        st["tp1_order_id"] = str(tp1_order_id) if tp1_order_id else "ok"
                        dirty = True
                        await send_telegram(_mk_exec_msg("TP1_OK", sym, tid, tp1=q_tp1, orderId=tp1_order_id, qty_tp1=qty_tp1, qty_rem=qty_rem))

                    if st.get("sl_plan_id") and st.get("tp1_order_id"):
                        st["armed"] = True
                        dirty = True
                    continue

                # Phase 2: BE after TP1 filled
                if bool(st.get("armed", False)) and (not bool(st.get("be_done", False))):
                    pos_total = await _get_position_total(trader, sym, direction)
                    if pos_total >= 0 and pos_total <= 0:
                        await _risk_close_trade(trader, sym, direction, reason="position_closed_before_tp1")
                        async with PENDING_LOCK:
                            PENDING.pop(tid, None)
                        dirty = True
                        await send_telegram(_mk_exec_msg("DONE", sym, tid, reason="position_closed"))
                        continue

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

                    try:
                        if not trader.is_filled(tp1_detail):
                            continue
                    except Exception:
                        continue

                    qty_total = float(st.get("qty_total") or 0.0)
                    qty_tp1 = float(st.get("qty_tp1") or 0.0)
                    qty_step = float(st.get("qty_step") or 1.0)

                    qty_rem = _q_qty_floor(max(0.0, qty_total - qty_tp1), qty_step)
                    if qty_rem <= 0:
                        await _risk_close_trade(trader, sym, direction, reason="tp1_full_close")
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

                    old_plan = st.get("sl_plan_id")
                    if old_plan and old_plan not in ("ok", ""):
                        try:
                            await asyncio.wait_for(trader.cancel_plan_orders(sym, [str(old_plan)]), timeout=ORDER_TIMEOUT_S)
                        except Exception:
                            pass

                    async def _place_be():
                        return await trader.place_stop_market_sl(
                            symbol=sym,
                            close_side=close_side.lower(),
                            trigger_price=be_q,
                            qty=qty_rem,
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
                        st["last_arm_fail_ts"] = time.time()
                        dirty = True
                        await send_telegram(_mk_exec_msg("BE_FAIL", sym, tid, code=sl_be_resp.get("code"), msg=sl_be_resp.get("msg")))
                        continue

                    new_plan = (
                        (sl_be_resp.get("data") or {}).get("orderId")
                        or (sl_be_resp.get("data") or {}).get("planOrderId")
                        or sl_be_resp.get("orderId")
                    )

                    st["sl_plan_id"] = str(new_plan) if new_plan else st.get("sl_plan_id")
                    st["be_done"] = True
                    dirty = True

                    await send_telegram(_mk_exec_msg("BE_OK", sym, tid, be=be_q, planId=new_plan, runner_qty=qty_rem))

                    st["runner_monitor"] = True
                    st["runner_qty"] = qty_rem
                    dirty = True
                    continue

            except Exception as e:
                logger.exception("[WATCHER] tid=%s error=%s", tid, e)

        if dirty:
            await _pending_save(force=False)


def _ensure_watcher(client, trader: BitgetTrader) -> None:
    global WATCHER_TASK
    if WATCHER_TASK is None or WATCHER_TASK.done():
        WATCHER_TASK = asyncio.create_task(_watcher_loop(client, trader))


# =====================================================================
# Scan loop
# =====================================================================

async def scan_once(client, trader: BitgetTrader) -> None:
    stats = ScanStats()
    t_scan0 = time.time()

    symbols = await client.get_contracts_list()
    if not symbols:
        logger.warning("⚠️ get_contracts_list() vide")
        return

    symbols = sorted(set(map(str.upper, symbols)))[: int(TOP_N_SYMBOLS)]
    logger.info("📊 Scan %d symboles (TOP_N_SYMBOLS=%s) | pass2=%s shortlist=%s", len(symbols), TOP_N_SYMBOLS, PASS2_ENABLED, SHORTLIST_SIZE)

    # ---- Fetch desk context once per scan (non-blocking) ----
    macro_ctx: Dict[str, Any] = {}
    if MACRO_CACHE and OPTIONS_CACHE:
        try:
            msnap = await asyncio.wait_for(MACRO_CACHE.get(force=False), timeout=8)
            osnap = await asyncio.wait_for(OPTIONS_CACHE.get(force=False), timeout=8)
            macro_ctx = {
                "macro": msnap.__dict__ if hasattr(msnap, "__dict__") else msnap,
                "options": osnap.__dict__ if hasattr(osnap, "__dict__") else osnap,
            }
        except Exception as e:
            logger.debug("desk ctx fetch failed: %s", e)
            macro_ctx = {}

    # ------------------------------------------------------------
    # PASS 1: LIGHT (cheap) across all symbols
    # ------------------------------------------------------------
    if _inst_mod is not None:
        _set_inst_mode("LIGHT")

    analyzer_p1 = SignalAnalyzer()
    analyze_sem_p1 = asyncio.Semaphore(int(MAX_CONCURRENT_ANALYZE))

    async def _worker_p1(sym: str):
        return await analyze_symbol(sym, client, analyze_sem_p1, stats, analyzer_p1, macro_ctx, pass_tag="P1")

    results = await asyncio.gather(*[_worker_p1(sym) for sym in symbols], return_exceptions=True)

    candidates: List[Dict[str, Any]] = []
    for r in results:
        if isinstance(r, dict):
            candidates.append(r)
        elif isinstance(r, Exception):
            logger.debug("worker exception: %s", r)

    ranked = rank_candidates(candidates)

    # ------------------------------------------------------------
    # PASS 2: FULL only on shortlist (if possible)
    # ------------------------------------------------------------
    ranked_final = ranked

    pass2_ok = PASS2_ENABLED and (_inst_mod is not None)
    if pass2_ok:
        # if circuit breaker already on, skip
        if _inst_is_banned_now():
            ban_ms = _inst_ban_until_ms()
            if ban_ms > 0:
                _cooldown_global((ban_ms / 1000.0) + 3.0, reason="binance_ban_pre_pass2")
            pass2_ok = False

    if pass2_ok and ranked:
        shortlist_syms = _unique_syms_top(ranked, int(SHORTLIST_SIZE))
        if shortlist_syms:
            logger.info("🔁 PASS2 %s on shortlist=%d", PASS2_MODE, len(shortlist_syms))
            _set_inst_mode(PASS2_MODE)

            analyzer_p2 = SignalAnalyzer()
            analyze_sem_p2 = asyncio.Semaphore(max(1, int(PASS2_CONCURRENCY)))

            # Re-analyze only symbols; if pass2 fails, keep pass1 candidate
            by_sym: Dict[str, Dict[str, Any]] = {str(c["symbol"]).upper(): c for c in ranked}

            async def _worker_p2(sym: str):
                base = by_sym.get(str(sym).upper())
                reuse_tid = base.get("tid") if isinstance(base, dict) else None
                return await analyze_symbol(sym, client, analyze_sem_p2, stats, analyzer_p2, macro_ctx, reuse_tid=reuse_tid, pass_tag="P2")

            res2 = await asyncio.gather(*[_worker_p2(s) for s in shortlist_syms], return_exceptions=True)

            for r in res2:
                if isinstance(r, dict):
                    by_sym[str(r["symbol"]).upper()] = r
                elif isinstance(r, Exception):
                    logger.debug("pass2 exception: %s", r)

            ranked_final = rank_candidates(list(by_sym.values()))

            # restore LIGHT for next scan
            _set_inst_mode("LIGHT")

    # ------------------------------------------------------------
    # Select top N for execution from final ranking
    # ------------------------------------------------------------
    N = int(MAX_ORDERS_PER_SCAN)
    selected = ranked_final[: max(0, N)]
    await stats.inc("exec_selected", len(selected))

    for c in ranked_final[N:]:
        await stats.add_reason("budget:ranked_out")

    # ------------------------------------------------------------
    # ALERT POLICY (anti spam)
    # ------------------------------------------------------------
    alert_list: List[Dict[str, Any]] = []
    if ALERT_MODE == "ALL_VALID":
        alert_list = ranked_final
    elif ALERT_MODE == "TOP_RANK":
        alert_list = ranked_final[: max(0, int(MAX_ALERTS_PER_SCAN))]
    elif ALERT_MODE == "EXEC_ONLY":
        alert_list = selected
    elif ALERT_MODE == "NONE":
        alert_list = []
    else:
        alert_list = ranked_final[: max(0, int(MAX_ALERTS_PER_SCAN))]

    for c in alert_list:
        fp = str(c.get("fp_alert") or "")
        if not fp:
            continue

        if ALERT_GUARD.seen(fp):
            await stats.inc("duplicates_alert", 1)
            continue

        await send_telegram(
            _mk_signal_msg(
                c["symbol"], c["tid"], c["side"], c["setup"],
                float(c["entry"]), float(c["sl"]), float(c["tp1"]),
                float(c["rr"]), int(c.get("inst_score") or 0),
                str(c.get("entry_type") or "MARKET"),
                str(c.get("pos_mode") or POSITION_MODE),
                tier=c.get("tier"),
                confidence=c.get("confidence"),
            )
        )
        await stats.inc("alert_sent", 1)

    await stats.inc("valids", len(ranked_final))

    if DRY_RUN or N <= 0:
        dt = time.time() - t_scan0
        reasons = stats.reasons.most_common(12)
        reasons_str = ", ".join([f"{k}:{v}" for k, v in reasons]) if reasons else "-"
        logger.info(
            "🧾 Scan summary: total=%s valids=%s rejects=%s skips=%s alert_sent=%s dup_alert=%s exec_selected=%s exec_sent=%s exec_failed=%s time=%.1fs | top_reasons=%s",
            stats.total, stats.valids, stats.rejects, stats.skips, stats.alert_sent, stats.duplicates_alert,
            stats.exec_selected, stats.exec_sent, stats.exec_failed, dt, reasons_str
        )
        return

    # Execute selected sequentially (desk control)
    for c in selected:
        try:
            await execute_candidate(c, client, trader, stats)
        except Exception as e:
            await stats.inc("exec_failed", 1)
            desk_log(logging.ERROR, "EXEC_ERR", c.get("symbol", "?"), c.get("tid", "?"), err=str(e))

    dt = time.time() - t_scan0
    reasons = stats.reasons.most_common(12)
    reasons_str = ", ".join([f"{k}:{v}" for k, v in reasons]) if reasons else "-"

    logger.info(
        "🧾 Scan summary: total=%s valids=%s rejects=%s skips=%s alert_sent=%s dup_alert=%s dup_trade=%s risk_rejects=%s exec_selected=%s exec_sent=%s exec_failed=%s time=%.1fs | top_reasons=%s",
        stats.total,
        stats.valids,
        stats.rejects,
        stats.skips,
        stats.alert_sent,
        stats.duplicates_alert,
        stats.duplicates_trade,
        stats.risk_rejects,
        stats.exec_selected,
        stats.exec_sent,
        stats.exec_failed,
        dt,
        reasons_str,
    )


async def start_scanner() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")

    await _pending_load()

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        await send_telegram(
            f"✅ *Bot démarré* \\(ALERT_MODE={_tg_escape(ALERT_MODE)} MAX_ALERTS_PER_SCAN={MAX_ALERTS_PER_SCAN} MAX_ORDERS_PER_SCAN={MAX_ORDERS_PER_SCAN} PASS2={_tg_escape(str(PASS2_ENABLED))}\\)"
        )
    else:
        logger.warning("Telegram disabled: missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")

    client = await get_client(API_KEY, API_SECRET, API_PASSPHRASE)

    trader = BitgetTrader(
        client,
        margin_usdt=float(MARGIN_USDT),
        leverage=float(LEVERAGE),
        margin_mode="isolated",
    )

    await _bootstrap_risk_open_positions(trader)

    _ensure_watcher(client, trader)

    logger.info(
        "🚀 Scanner started | interval=%s min | dry_run=%s | max_orders_per_scan=%s | alert_mode=%s | max_alerts_per_scan=%s | pass2=%s shortlist=%s",
        SCAN_INTERVAL_MIN, DRY_RUN, MAX_ORDERS_PER_SCAN, ALERT_MODE, MAX_ALERTS_PER_SCAN, PASS2_ENABLED, SHORTLIST_SIZE
    )

    while True:
        t0 = time.time()
        try:
            await scan_once(client, trader)
        except Exception:
            logger.exception("SCAN ERROR")

        dt = time.time() - t0
        sleep_s = max(1, int(float(SCAN_INTERVAL_MIN) * 60 - dt))
        await asyncio.sleep(sleep_s)


if __name__ == "__main__":
    asyncio.run(start_scanner())
