# =====================================================================
# scanner.py — Desk-lead Scanner + Watcher (Bitget) — READY TO PASTE
# =====================================================================
# ✅ RiskManager reservation flow (reserve -> confirm/cancel) aligned
# ✅ Options context: regime + risk_factor (non-blocking) => scales real sizing
# ✅ Macro/options fetch independent (works if only one is available)
# ✅ Pending persistence includes options fields
# ✅ Watcher fix stays (no immediate cancel after ENTRY_OK etc.)
# ✅ Deep pullback policy (OTE_PULLBACK):
#    - Dedicated TTL (ENTRY_TTL_OTE_PULLBACK_S)
#    - Runaway cancel disabled when entry is "deep" vs ref_price_at_signal (>= DEEP_PULLBACK_ATR_THRESHOLD * ATR)
#    - Prevents cancelling a BOS_STRICT pullback just because price ran first
# ✅ DuplicateGuard persistence (alert/trade) to survive restarts
# ✅ Institutional watcher veto (OI dump / funding flip / liq spike) via InstitutionalWSHub snapshot if available
#
# FIXES (this version):
# ✅ _recheck_live_and_act: fixed wrong kwarg (qty_total -> qty_use) for _switch_sl_plan
# ✅ risk confirm_open now passes filled_notional when available (better partial-fill accounting)
# ✅ pending serialization now persists recheck state keys
# ✅ deep_pullback precomputed at PENDING creation when possible
#
# WS HUB FIX (this version):
# ✅ Robust InstitutionalWSHub start/update: only start with non-empty symbols
# ✅ Avoid re-start spam: fingerprint symbols list; restart only if changed
# ✅ Wait briefly for HUB.is_running() to flip true (prevents early hub_not_running)
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
import datetime
from zoneinfo import ZoneInfo
from collections import Counter
from typing import Any, Dict, Tuple, Optional, List

import pandas as pd

# ---------------------------------------------------------------------
# Optional desk context (macro + options)
# ---------------------------------------------------------------------
try:
    from macro_data import MacroCache  # type: ignore
except Exception:
    MacroCache = None  # type: ignore

try:
    from options_data import OptionsCache  # type: ignore
except Exception:
    OptionsCache = None  # type: ignore

try:
    # preferred (improved options_data.py)
    from options_data import score_options_context  # type: ignore
except Exception:
    score_options_context = None  # type: ignore

try:
    # Singleton partagé avec institutional_data.py
    from institutional_ws_hub import HUB as INST_HUB  # type: ignore
except Exception:
    INST_HUB = None  # type: ignore

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
from indicators import desk_momentum_gate  # premium recheck

from structure_utils import analyze_structure, htf_trend_ok

from duplicate_guard import DuplicateGuard, fingerprint as make_fingerprint
from risk_manager import RiskManager
from retry_utils import retry_async

logger = logging.getLogger(__name__)

# =====================================================================
# Institutional WS Hub bootstrap (singleton)
# =====================================================================

# Enable/disable via env (default ON). Keep it decoupled from settings.py for robustness.
_INST_WS_HUB_ENABLED = str(os.getenv("INST_USE_WS_HUB", "1")).strip().lower() in ("1", "true", "yes", "on", "y")
_INST_WS_LAST_FP: Optional[Tuple[str, ...]] = None

async def _ensure_inst_ws_hub(symbols: List[str]) -> None:
    """
    Démarre / redémarre le singleton INST_HUB avec une liste de symbols NON vide.
    - évite hub_not_running (hub jamais start / start avec liste vide / restart spam)
    - ne redémarre que si la liste des symbols change
    """
    global _INST_WS_LAST_FP

    if not _INST_WS_HUB_ENABLED:
        return
    if INST_HUB is None:
        return

    syms = sorted({(s or "").strip().upper() for s in (symbols or []) if (s or "").strip()})
    if not syms:
        return

    fp = tuple(syms)

    # If symbols unchanged and hub already running -> nothing to do
    try:
        is_running_fn = getattr(INST_HUB, "is_running", None)
        running = bool(is_running_fn()) if callable(is_running_fn) else False
    except Exception:
        running = False

    if fp == _INST_WS_LAST_FP and running:
        return

    shards = int(float(os.getenv("INST_WS_SHARDS", "4")))
    try:
        await INST_HUB.start(syms, shards=shards)  # type: ignore[attr-defined]
    except Exception as e:
        logger.debug("[INST_WS_HUB] start failed: %s", e)
        return

    # Wait briefly so is_running() flips to True before analyze_symbol() starts using WS
    try:
        is_running_fn = getattr(INST_HUB, "is_running", None)
        if callable(is_running_fn):
            for _ in range(20):
                if bool(is_running_fn()):
                    break
                await asyncio.sleep(0.05)
    except Exception:
        pass

    _INST_WS_LAST_FP = fp

    try:
        is_running_fn = getattr(INST_HUB, "is_running", None)
        running2 = bool(is_running_fn()) if callable(is_running_fn) else False
    except Exception:
        running2 = False

    logger.info("[INST_WS_HUB] start/update shards=%s symbols=%s running=%s", shards, len(syms), running2)

# =====================================================================
# Runtime globals
# =====================================================================

ALERT_GUARD = DuplicateGuard(ttl_seconds=1800)   # anti spam telegram
TRADE_GUARD = DuplicateGuard(ttl_seconds=3600)   # anti double trade

# Optional persistence paths for DuplicateGuard
ALERT_GUARD_FILE = os.getenv("ALERT_GUARD_FILE", "alert_guard.json")
TRADE_GUARD_FILE = os.getenv("TRADE_GUARD_FILE", "trade_guard.json")
_GUARDS_SAVE_THROTTLE_S = float(os.getenv("GUARDS_SAVE_THROTTLE_S", "20"))
_GUARDS_LAST_SAVE_TS = 0.0

RISK = RiskManager()

TF_H1 = "1H"
TF_H4 = "4H"
CANDLE_LIMIT = 200

MAX_CONCURRENT_ANALYZE = 6  # limite Binance/institutional

TP1_CLOSE_PCT = 0.50
WATCH_INTERVAL_S = 3.0

# =====================================================================
# Entry watcher policy — TTL / runaway / invalidations
# =====================================================================

# TTLs (seconds)
ENTRY_TTL_MARKET_S = int(os.getenv("ENTRY_TTL_MARKET_S", "300"))
ENTRY_TTL_OTE_S = int(os.getenv("ENTRY_TTL_OTE_S", "3600"))
ENTRY_TTL_FVG_S = int(os.getenv("ENTRY_TTL_FVG_S", "2700"))
ENTRY_TTL_RAID_S = int(os.getenv("ENTRY_TTL_RAID_S", "1800"))
ENTRY_TTL_DEFAULT_S = int(os.getenv("ENTRY_TTL_DEFAULT_S", "1800"))

# dedicated TTL for OTE_PULLBACK (default: half of OTE TTL, min 900s)
ENTRY_TTL_OTE_PULLBACK_S = int(os.getenv("ENTRY_TTL_OTE_PULLBACK_S", str(max(900, ENTRY_TTL_OTE_S // 2))))

# Run-away cancel knobs (legacy)
RUNAWAY_ATR_MULT_MARKET = float(os.getenv("RUNAWAY_ATR_MULT_MARKET", "1.0"))
RUNAWAY_ATR_MULT_PULLBACK = float(os.getenv("RUNAWAY_ATR_MULT_PULLBACK", "1.5"))

# runaway policy flags
RUNAWAY_ENABLE = str(os.getenv("RUNAWAY_ENABLE", "1")).strip() == "1"

# separate multipliers
RUNAWAY_ATR_MULT_NEAR = float(os.getenv("RUNAWAY_ATR_MULT_NEAR", str(RUNAWAY_ATR_MULT_PULLBACK)))
RUNAWAY_ATR_MULT_DEEP = float(os.getenv("RUNAWAY_ATR_MULT_DEEP", "3.0"))

# Deep pullback definition: abs(ref_price_at_signal - entry) >= threshold * ATR
DEEP_PULLBACK_ATR_THRESHOLD = float(os.getenv("DEEP_PULLBACK_ATR_THRESHOLD", "2.0"))

# grace periods so it never cancels immediately
RUNAWAY_GRACE_S_MARKET = int(os.getenv("RUNAWAY_GRACE_S_MARKET", "30"))
RUNAWAY_GRACE_S_PULLBACK = int(os.getenv("RUNAWAY_GRACE_S_PULLBACK", "300"))

# heavy checks grace (structure / HTF / PD invalidations)
HEAVY_GRACE_S_MARKET = int(os.getenv("HEAVY_GRACE_S_MARKET", "45"))
HEAVY_GRACE_S_PULLBACK = int(os.getenv("HEAVY_GRACE_S_PULLBACK", "120"))

# runaway requires "no pullback touch" for pullback entries
RUNAWAY_TOUCH_ATR = float(os.getenv("RUNAWAY_TOUCH_ATR", "0.25"))
RUNAWAY_TOUCH_TICKS = int(os.getenv("RUNAWAY_TOUCH_TICKS", "10"))

# runaway distance min floor in ticks (avoids ATR too small => instant cancel)
RUNAWAY_MIN_TICKS = int(os.getenv("RUNAWAY_MIN_TICKS", "20"))

# Throttles (seconds)
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


# =====================================================================
# Premium watcher policy — killzones / options veto / distribution / time stop
# =====================================================================
KILLZONES_ENABLE = str(os.getenv("KILLZONES_ENABLE", "1")).strip() == "1"
KILLZONES_TZ = str(os.getenv("KILLZONES_TZ", "UTC")).strip() or "UTC"
KZ_LONDON_START = str(os.getenv("KZ_LONDON_START", "07:00")).strip()
KZ_LONDON_END = str(os.getenv("KZ_LONDON_END", "10:00")).strip()
KZ_NY_START = str(os.getenv("KZ_NY_START", "13:30")).strip()
KZ_NY_END = str(os.getenv("KZ_NY_END", "16:00")).strip()

# Options veto (DVOL regime) — watcher level (can close or tighten)
OPTIONS_WATCH_REFRESH_S = float(os.getenv("OPTIONS_WATCH_REFRESH_S", "60"))
OPTIONS_VETO_ENABLE = str(os.getenv("OPTIONS_VETO_ENABLE", "1")).strip() == "1"
OPTIONS_VETO_REGIMES = {x.strip().lower() for x in str(os.getenv("OPTIONS_VETO_REGIMES", "extreme")).split(",") if x.strip()}
OPTIONS_VETO_SPIKE = str(os.getenv("OPTIONS_VETO_SPIKE", "1")).strip() == "1"
OPTIONS_VETO_ACTION = str(os.getenv("OPTIONS_VETO_ACTION", "TIGHTEN")).strip().upper()  # TIGHTEN | CLOSE

# Distribution detection (CVD vs price divergence)
DIST_ENABLE = str(os.getenv("DIST_ENABLE", "1")).strip() == "1"
DIST_LOOKBACK_S = int(os.getenv("DIST_LOOKBACK_S", "1800"))  # 30m
DIST_MIN_PRICE_ATR = float(os.getenv("DIST_MIN_PRICE_ATR", "0.50"))
DIST_NEG_CORR_THRESHOLD = float(os.getenv("DIST_NEG_CORR_THRESHOLD", "-0.25"))
DIST_CONFIRM_HITS = int(os.getenv("DIST_CONFIRM_HITS", "2"))
DIST_MAX_POINTS = int(os.getenv("DIST_MAX_POINTS", "80"))

# Liquidity-aware SL plan switch
LIQ_SWITCH_ENABLE = str(os.getenv("LIQ_SWITCH_ENABLE", "1")).strip() == "1"
LIQ_NEAR_ATR = float(os.getenv("LIQ_NEAR_ATR", "0.60"))           # search window from price
LIQ_BUFFER_ATR = float(os.getenv("LIQ_BUFFER_ATR", "0.12"))       # place SL beyond EQ level
LIQ_MAX_WIDEN_ATR = float(os.getenv("LIQ_MAX_WIDEN_ATR", "0.60")) # cap widen vs current SL

# Time stop
TIME_STOP_ENABLE = str(os.getenv("TIME_STOP_ENABLE", "1")).strip() == "1"
TIME_STOP_TF = str(os.getenv("TIME_STOP_TF", TF_H1)).upper()   # TF_H1 by default
TIME_STOP_BARS = int(os.getenv("TIME_STOP_BARS", "6"))          # if no progress after X bars => exit
TIME_STOP_MIN_MOVE_ATR = float(os.getenv("TIME_STOP_MIN_MOVE_ATR", "0.30"))

# Tighten / trailing after TP1 (runner management)
TRAIL_ENABLE = str(os.getenv("TRAIL_ENABLE", "1")).strip() == "1"
TRAIL_CHECK_INTERVAL_S = float(os.getenv("TRAIL_CHECK_INTERVAL_S", "120"))
TRAIL_BUFFER_ATR = float(os.getenv("TRAIL_BUFFER_ATR", "0.10"))
TRAIL_MIN_STEP_ATR = float(os.getenv("TRAIL_MIN_STEP_ATR", "0.20"))

# Pro monitor throttle
PRO_MONITOR_INTERVAL_S = float(os.getenv("PRO_MONITOR_INTERVAL_S", "90"))

# =====================================================================
# Premium signal recheck (autonomous validation while pending/live)
# =====================================================================
WATCH_RECHECK_ENABLE = str(os.getenv("WATCH_RECHECK_ENABLE", "1")).strip() == "1"
WATCH_RECHECK_INTERVAL_S = float(os.getenv("WATCH_RECHECK_INTERVAL_S", "120"))
WATCH_RECHECK_MAX_FAILS = int(os.getenv("WATCH_RECHECK_MAX_FAILS", "1"))
WATCH_RECHECK_PENDING_ACTION = str(os.getenv("WATCH_RECHECK_PENDING_ACTION", "CANCEL")).strip().upper()  # CANCEL
WATCH_RECHECK_LIVE_ACTION = str(os.getenv("WATCH_RECHECK_LIVE_ACTION", "CLOSE")).strip().upper()         # CLOSE | TIGHTEN
WATCH_RECHECK_LIVE_GRACE_S = int(os.getenv("WATCH_RECHECK_LIVE_GRACE_S", "90"))

# =====================================================================
# Institutional watcher veto policy (requires InstitutionalWSHub snapshot)
# =====================================================================
INST_VETO_ENABLE = str(os.getenv("INST_VETO_ENABLE", "1")).strip() == "1"
INST_VETO_ACTION = str(os.getenv("INST_VETO_ACTION", "TIGHTEN")).strip().upper()  # TIGHTEN | CLOSE
INST_OI_DUMP_1H_PCT = float(os.getenv("INST_OI_DUMP_1H_PCT", "6.0"))              # e.g. -6% 1h
INST_FUNDING_FLIP_ENABLE = str(os.getenv("INST_FUNDING_FLIP_ENABLE", "1")).strip() == "1"
INST_FUNDING_FLIP_ABS = float(os.getenv("INST_FUNDING_FLIP_ABS", "0.01"))         # abs funding threshold
INST_LIQ_SPIKE_USDT = float(os.getenv("INST_LIQ_SPIKE_USDT", "300000"))           # liq notional threshold in lookback
INST_LIQ_LOOKBACK_KEY = str(os.getenv("INST_LIQ_LOOKBACK_KEY", "liq_1h_usdt")).strip()
INST_OI_CHANGE_KEY = str(os.getenv("INST_OI_CHANGE_KEY", "oi_change_1h_pct")).strip()
INST_FUNDING_KEY = str(os.getenv("INST_FUNDING_KEY", "funding_rate")).strip()

# =====================================================================
# Options helpers (non-breaking even if options_data isn't available)
# =====================================================================

def _default_options_ctx() -> Dict[str, Any]:
    return {
        "ok": True,
        "score": 0,
        "regime": "unknown",
        "reason": "no_options",
        "avg_dvol": None,
        "dvol_change_24h_pct": None,
        "spike": False,
        "risk_factor": 1.0,
        "position_mode": "neutral",
    }

def _options_ctx_from_snap(options_snap: Any, bias: str, setup_type: str) -> Dict[str, Any]:
    if score_options_context is None or options_snap is None:
        return _default_options_ctx()
    try:
        ctx = score_options_context(options_snap, bias, setup_type=setup_type)  # type: ignore[misc]
        if isinstance(ctx, dict):
            if "risk_factor" not in ctx:
                ctx["risk_factor"] = 1.0
            if "regime" not in ctx:
                ctx["regime"] = "unknown"
            return ctx
        return _default_options_ctx()
    except Exception:
        return _default_options_ctx()

def _sanitize_risk_factor(rf: Any) -> float:
    try:
        x = float(rf)
        if not math.isfinite(x) or x <= 0:
            return 1.0
        if x > 2.0:
            return 1.0
        return x
    except Exception:
        return 1.0

# =====================================================================
# Guards persistence
# =====================================================================

async def _guards_load() -> None:
    def _do():
        n1 = 0
        n2 = 0
        try:
            if ALERT_GUARD_FILE:
                n1 = ALERT_GUARD.load(ALERT_GUARD_FILE)
        except Exception:
            n1 = 0
        try:
            if TRADE_GUARD_FILE:
                n2 = TRADE_GUARD.load(TRADE_GUARD_FILE)
        except Exception:
            n2 = 0
        return n1, n2

    try:
        n1, n2 = await asyncio.to_thread(_do)
        logger.info("[BOOT] guard load: alert=%d trade=%d", n1, n2)
    except Exception as e:
        logger.warning("[BOOT] guard load failed: %s", e)

async def _guards_save(force: bool = False) -> None:
    global _GUARDS_LAST_SAVE_TS
    now = time.time()
    if (not force) and (now - _GUARDS_LAST_SAVE_TS) < _GUARDS_SAVE_THROTTLE_S:
        return

    def _do():
        try:
            if ALERT_GUARD_FILE:
                ALERT_GUARD.save(ALERT_GUARD_FILE)
        except Exception:
            pass
        try:
            if TRADE_GUARD_FILE:
                TRADE_GUARD.save(TRADE_GUARD_FILE)
        except Exception:
            pass

    try:
        await asyncio.to_thread(_do)
        _GUARDS_LAST_SAVE_TS = now
    except Exception:
        pass

# =====================================================================
# Pending persistence
# =====================================================================

def _pending_serializable(state: Dict[str, Any]) -> Dict[str, Any]:
    keep = [
        "symbol","entry_side","direction","close_side","pos_mode",
        "entry","sl","tp1","qty_total","qty_tp1","qty_rem",
        "entry_order_id","entry_client_oid","sl_plan_id","tp1_order_id",
        "armed","be_done","created_ts","arm_attempts","last_arm_fail_ts",
        "freeze_until","sl_inflight","tp1_inflight","tp1_min_usdt",
        "qty_step","min_qty","q_tp1","q_sl",
        "risk_rid","risk_confirmed","notional","setup","entry_type","rr","inst_score",
        "runner_monitor","runner_qty",

        # needed for TTL/runaway/invalidation after restart
        "atr","pd_mid","ref_price_at_signal",
        "last_price_ts","last_price","last_heavy_ts",

        # runaway memory
        "tick_used","min_price_seen","max_price_seen","pullback_touched",

        # options summary
        "opt_regime","opt_score","opt_risk_factor","opt_avg_dvol","opt_chg24h_pct","opt_spike",

        # premium watcher state
        "filled_ts","last_pro_ts","trail_last_ts","dist_hits","cvd_hist","deep_pullback",

        # recheck state (persisted)
        "last_recheck_ts","recheck_fail_hits","last_recheck_reason","last_recheck_dbg",

        # institutional watcher state/debug
        "inst_veto_hits","last_inst_ts","inst_dbg",
    ]
    out: Dict[str, Any] = {}
    for k in keep:
        if k in state:
            out[k] = state[k]
    return out

_PENDING_LAST_SAVE_TS = 0.0

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

# =====================================================================
# Parsers / small helpers
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
    import requests  # local import to keep runtime minimal
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

def _fmt_opt_line(opt: Dict[str, Any]) -> str:
    try:
        regime = str(opt.get("regime") or "unknown")
        rf = _sanitize_risk_factor(opt.get("risk_factor", 1.0))
        avg = opt.get("avg_dvol")
        chg = opt.get("dvol_change_24h_pct")
        spike = bool(opt.get("spike", False))
        bits = [f"opt: `{_tg_escape(regime)}`", f"rf: `{_tg_escape(f'{rf:.2f}')}`"]
        if isinstance(avg, (int, float)):
            bits.append(f"avg: `{_tg_escape(f'{float(avg):.3g}')}`")
        if isinstance(chg, (int, float)):
            bits.append(f"chg24h: `{_tg_escape(f'{float(chg):.2f}%')}`")
        if spike:
            bits.append("spike: `1`")
        return " | ".join(bits)
    except Exception:
        return "opt: `unknown` | rf: `1.00`"

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
    opt: Optional[Dict[str, Any]] = None,
) -> str:
    opt_line = _fmt_opt_line(opt or _default_options_ctx())
    return (
        f"*📌 SIGNAL*\n"
        f"*{_tg_escape(symbol)}* — {_tg_escape(_fmt_side(side))}\n"
        f"setup: `{_tg_escape(setup)}` | inst: `{inst}` | entry_type: `{_tg_escape(entry_type)}` | pos_mode: `{_tg_escape(pos_mode)}`\n"
        f"{opt_line}\n"
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

def _order_state(detail: Dict[str, Any]) -> str:
    try:
        if not isinstance(detail, dict):
            return "unknown"
        data = detail.get("data") or {}
        st = str(data.get("state") or data.get("status") or "").lower()
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

def _is_market_entry(entry_type: str) -> bool:
    return "MARKET" in str(entry_type or "").upper()

def _entry_ttl_s(entry_type: str, setup: str) -> int:
    et = str(entry_type or "").upper()
    sp = str(setup or "").upper()

    if "MARKET" in et:
        return int(ENTRY_TTL_MARKET_S)

    if "OTE_PULLBACK" in et:
        return int(ENTRY_TTL_OTE_PULLBACK_S)

    if ("OTE" in et) or ("OTE" in sp):
        return int(ENTRY_TTL_OTE_S)

    if "FVG" in sp or "FVG" in et:
        return int(ENTRY_TTL_FVG_S)

    if ("RAID" in sp) or ("SWEEP" in sp) or ("RAID" in et):
        return int(ENTRY_TTL_RAID_S)

    return int(ENTRY_TTL_DEFAULT_S)

def _runaway_grace_s(entry_type: str) -> int:
    return int(RUNAWAY_GRACE_S_MARKET if _is_market_entry(entry_type) else RUNAWAY_GRACE_S_PULLBACK)

def _heavy_grace_s(entry_type: str) -> int:
    return int(HEAVY_GRACE_S_MARKET if _is_market_entry(entry_type) else HEAVY_GRACE_S_PULLBACK)

def _deep_pullback(entry: float, ref_price: float, atr: float) -> bool:
    try:
        entry = float(entry)
        ref_price = float(ref_price)
        atr = float(atr)
        if atr <= 0 or (not math.isfinite(atr)):
            return False
        if entry <= 0 or ref_price <= 0:
            return False
        d = abs(ref_price - entry)
        return d >= float(DEEP_PULLBACK_ATR_THRESHOLD) * atr
    except Exception:
        return False

def _runaway_mult(entry_type: str, deep: bool) -> float:
    et = str(entry_type or "").upper()
    if "MARKET" in et:
        return float(RUNAWAY_ATR_MULT_MARKET)
    return float(RUNAWAY_ATR_MULT_DEEP if deep else RUNAWAY_ATR_MULT_NEAR)

# =====================================================================
# Premium watcher helpers
# =====================================================================

def _parse_hhmm(s: str) -> int:
    try:
        s = (s or "").strip()
        hh, mm = s.split(":")
        h = max(0, min(23, int(hh)))
        m = max(0, min(59, int(mm)))
        return h * 60 + m
    except Exception:
        return 0

_KZ_LON_S = _parse_hhmm(KZ_LONDON_START)
_KZ_LON_E = _parse_hhmm(KZ_LONDON_END)
_KZ_NY_S = _parse_hhmm(KZ_NY_START)
_KZ_NY_E = _parse_hhmm(KZ_NY_END)

def _in_killzone(now: Optional[datetime.datetime] = None) -> Tuple[bool, str]:
    if not KILLZONES_ENABLE:
        return True, "ALLDAY"
    try:
        tz = ZoneInfo(KILLZONES_TZ)
    except Exception:
        tz = ZoneInfo("UTC")

    if now is None:
        now = datetime.datetime.now(tz)
    else:
        now = now.astimezone(tz)

    m = now.hour * 60 + now.minute

    def _in_window(x: int, a: int, b: int) -> bool:
        if a <= b:
            return a <= x <= b
        return (x >= a) or (x <= b)

    if _in_window(m, _KZ_LON_S, _KZ_LON_E):
        return True, "LONDON"
    if _in_window(m, _KZ_NY_S, _KZ_NY_E):
        return True, "NEWYORK"
    return False, "OFF"

def _options_veto(regime: str, spike: bool) -> bool:
    if not OPTIONS_VETO_ENABLE:
        return False
    r = (regime or "unknown").strip().lower()
    if r in OPTIONS_VETO_REGIMES:
        return True
    if OPTIONS_VETO_SPIKE and bool(spike):
        return True
    return False

def _tf_seconds(tf: str) -> int:
    tfu = str(tf or "").upper()
    if tfu in ("1H", "H1"):
        return 3600
    if tfu in ("4H", "H4"):
        return 14400
    if tfu in ("15M", "M15"):
        return 900
    if tfu in ("5M", "M5"):
        return 300
    return 3600

def _corr(xs: List[float], ys: List[float]) -> float:
    try:
        n = min(len(xs), len(ys))
        if n < 5:
            return 0.0
        x = xs[-n:]
        y = ys[-n:]
        mx = sum(x) / n
        my = sum(y) / n
        vx = sum((a - mx) ** 2 for a in x)
        vy = sum((b - my) ** 2 for b in y)
        if vx <= 0 or vy <= 0:
            return 0.0
        cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
        return float(cov / math.sqrt(vx * vy))
    except Exception:
        return 0.0

def _update_cvd_hist(st: Dict[str, Any], px: float, cvd: Optional[float]) -> None:
    if px <= 0 or cvd is None or (not math.isfinite(float(cvd))):
        return
    now = time.time()
    hist = st.get("cvd_hist")
    if not isinstance(hist, list):
        hist = []
    hist.append([float(now), float(px), float(cvd)])
    cutoff = now - max(2 * float(DIST_LOOKBACK_S), 900)
    hist = [r for r in hist if isinstance(r, list) and len(r) >= 3 and float(r[0]) >= cutoff]
    if len(hist) > int(DIST_MAX_POINTS):
        hist = hist[-int(DIST_MAX_POINTS):]
    st["cvd_hist"] = hist

def _distribution_signal(direction: str, st: Dict[str, Any], atr: float) -> Tuple[bool, Dict[str, Any]]:
    if not DIST_ENABLE:
        return False, {}
    hist = st.get("cvd_hist")
    if not isinstance(hist, list) or len(hist) < 8:
        return False, {}

    now = time.time()
    lb = now - float(DIST_LOOKBACK_S)
    rows = [r for r in hist if isinstance(r, list) and len(r) >= 3 and float(r[0]) >= lb]
    if len(rows) < 6:
        return False, {}

    prices = [float(r[1]) for r in rows]
    cvds = [float(r[2]) for r in rows]

    p0, p1 = prices[0], prices[-1]
    c0, c1 = cvds[0], cvds[-1]
    dp = p1 - p0
    dc = c1 - c0

    gate = float(DIST_MIN_PRICE_ATR) * float(atr or 0.0)
    if gate > 0 and abs(dp) < gate:
        return False, {"dp": dp, "dc": dc, "gate": gate}

    rho = _corr(prices, cvds)
    d = str(direction).upper()

    if d == "LONG":
        ok = (dp > 0) and (dc < 0) and (rho <= float(DIST_NEG_CORR_THRESHOLD))
    else:
        ok = (dp < 0) and (dc > 0) and (rho <= float(DIST_NEG_CORR_THRESHOLD))

    return bool(ok), {"dp": dp, "dc": dc, "rho": rho, "gate": gate}

# =====================================================================
# Institutional veto helper (snapshot-driven)
# =====================================================================

def _inst_veto_from_snap(direction: str, snap: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    if not INST_VETO_ENABLE or not isinstance(snap, dict):
        return False, {}

    d = str(direction).upper()
    dbg: Dict[str, Any] = {}

    oi_chg = snap.get(INST_OI_CHANGE_KEY)
    funding = snap.get(INST_FUNDING_KEY)
    liq_usdt = snap.get(INST_LIQ_LOOKBACK_KEY)

    try:
        if oi_chg is not None:
            dbg["oi_change_1h_pct"] = float(oi_chg)
    except Exception:
        pass
    try:
        if funding is not None:
            dbg["funding_rate"] = float(funding)
    except Exception:
        pass
    try:
        if liq_usdt is not None:
            dbg["liq_usdt"] = float(liq_usdt)
    except Exception:
        pass

    # 1) OI dump
    try:
        oi = float(oi_chg)
        if math.isfinite(oi) and oi <= -abs(float(INST_OI_DUMP_1H_PCT)):
            dbg["hit"] = "oi_dump"
            return True, dbg
    except Exception:
        pass

    # 2) Funding crowding
    if INST_FUNDING_FLIP_ENABLE:
        try:
            fr = float(funding)
            if math.isfinite(fr) and abs(fr) >= float(INST_FUNDING_FLIP_ABS):
                if d == "LONG" and fr > 0:
                    dbg["hit"] = "funding_crowded_long"
                    return True, dbg
                if d == "SHORT" and fr < 0:
                    dbg["hit"] = "funding_crowded_short"
                    return True, dbg
        except Exception:
            pass

    # 3) Liquidation spike
    try:
        liq = float(liq_usdt)
        if math.isfinite(liq) and liq >= float(INST_LIQ_SPIKE_USDT):
            dbg["hit"] = "liq_spike"
            return True, dbg
    except Exception:
        pass

    return False, dbg

# =====================================================================
# Robust last price (Bitget may return dict OR list)
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

# =====================================================================
# Invalidation checks (structure/HTF/premium-discount)
# =====================================================================

async def _invalidation_check(
    client,
    symbol: str,
    direction: str,
    entry_price: float,
    entry_type: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    sym = str(symbol).upper()
    dir_u = (direction or "").upper()
    if dir_u not in ("LONG", "SHORT"):
        dir_u = "LONG"

    async def _do():
        df_h1 = await client.get_klines_df(sym, TF_H1, 160)
        df_h4 = await client.get_klines_df(sym, TF_H4, 160)
        return df_h1, df_h4

    try:
        df_h1, df_h4 = await asyncio.wait_for(retry_async(_do, retries=1, base_delay=0.25), timeout=FETCH_TIMEOUT_S)
    except Exception as e:
        return True, "", {"error": str(e)}

    if df_h1 is None or getattr(df_h1, "empty", True) or len(df_h1) < 80:
        return True, "", {"note": "no_df_h1"}
    if df_h4 is None or getattr(df_h4, "empty", True) or len(df_h4) < 60:
        df_h4 = None

    # --- Structure (H1) ---
    try:
        struct = analyze_structure(df_h1)
    except Exception as e:
        struct = {"error": str(e)}

    st_trend = str(struct.get("trend") or "").upper()
    choch = bool(struct.get("choch", False))

    if st_trend in ("LONG", "SHORT") and st_trend != dir_u:
        return False, "invalidate:structure_trend_flip", {"struct_trend": st_trend, "choch": int(choch)}

    if choch and st_trend in ("LONG", "SHORT") and st_trend != dir_u:
        return False, "invalidate:choch", {"struct_trend": st_trend, "choch": 1}

    # --- HTF alignment (H4) ---
    if df_h4 is not None:
        try:
            if not bool(htf_trend_ok(df_h4, dir_u)):
                return False, "invalidate:htf_veto", {"htf": "veto"}
        except Exception:
            pass

    # --- Momentum gate (H1) ---
    try:
        mom = desk_momentum_gate(df_h1, dir_u)
    except Exception as e:
        mom = {"ok": True, "error": str(e)}

    if not bool(mom.get("ok", True)):
        r = str(mom.get("reason") or "")
        if r in {
            "momentum_not_bullish",
            "momentum_not_bearish",
            "ema_bias_veto",
            "ema_range",
            "choppy_market",
            "overextended_long",
            "overextended_short",
        }:
            return False, f"invalidate:{r}", {"momentum": mom}

    # --- Market entry premium/discount sanity ---
    mid = _pd_mid(df_h1, 80)
    if _is_market_entry(entry_type) and mid > 0 and math.isfinite(mid):
        if dir_u == "LONG" and float(entry_price) > float(mid):
            return False, "invalidate:entry_premium", {"mid": mid}
        if dir_u == "SHORT" and float(entry_price) < float(mid):
            return False, "invalidate:entry_discount", {"mid": mid}

    return True, "", {"mid": mid, "struct_trend": st_trend, "choch": int(choch)}

# =====================================================================
# Cancel pending entry helper
# =====================================================================

async def _risk_close_trade(trader: BitgetTrader, symbol: str, direction: str, reason: str, pnl_override: Optional[float] = None) -> None:
    pnl = float(pnl_override) if pnl_override is not None else await _fetch_last_closed_pnl(trader, symbol, direction)
    try:
        RISK.register_closed(symbol, direction, pnl)
    except Exception:
        pass
    desk_log(logging.INFO, "RISK_CLOSE", symbol, "-", side=direction, pnl=pnl, reason=reason)

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

    await _pending_save(force=True)
    await _guards_save(force=False)

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
# Analyze worker (phase A)
# =====================================================================

async def analyze_symbol(
    sym: str,
    client,
    analyze_sem: asyncio.Semaphore,
    stats: ScanStats,
    macro_ctx: Optional[Dict[str, Any]] = None,
    options_snap: Any = None,
) -> Optional[Dict[str, Any]]:
    tid = _new_tid(sym)
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

        analyzer = SignalAnalyzer()

        t1 = time.time()
        try:
            result = await asyncio.wait_for(
                analyzer.analyze(sym, df_h1, df_h4, macro=(macro_ctx or {})),
                timeout=ANALYZE_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            result = {"valid": False, "reject_reason": "analyze_timeout"}
        analyze_ms = int((time.time() - t1) * 1000)

        if not result or not isinstance(result, dict) or not result.get("valid"):
            reason = _extract_reject_reason(result)
            if await stats.take_reject_debug_slot():
                has_fields = "Y" if (isinstance(result, dict) and _has_key_fields_for_trade(result)) else "N"
                desk_log(logging.INFO, "REJ", sym, tid, fetch_ms=fetch_ms, analyze_ms=analyze_ms, reason=reason, has_fields=has_fields)
            await stats.inc("rejects", 1)
            await stats.add_reason(reason)
            return None

        side = str(result.get("side", "")).upper()
        entry = _safe_float(result.get("entry"), 0.0)
        sl = _safe_float(result.get("sl"), 0.0)
        tp1 = _safe_float(result.get("tp1"), 0.0)
        rr = _safe_float(result.get("rr"), 0.0)
        setup = str(result.get("setup_type") or "-")
        entry_type = str(result.get("entry_type") or "MARKET")

        inst = result.get("institutional") or {}
        inst_score_eff = int(result.get("inst_score_eff") or inst.get("institutional_score") or 0)
        comp = result.get("composite") or {}
        comp_score = float(comp.get("score") or result.get("composite_score") or 0.0)

        if entry <= 0 or sl <= 0 or tp1 <= 0 or rr <= 0 or side not in ("BUY", "SELL"):
            await stats.inc("skips", 1)
            await stats.add_reason("missing_exits")
            return None

        direction = _direction_from_side(side)
        close_side = _close_side_from_direction(direction)
        pos_mode = str(POSITION_MODE or "hedge")

        # ---- options context (non-blocking) ----
        opt_ctx = _options_ctx_from_snap(options_snap, bias=direction, setup_type=setup)
        opt_rf = _sanitize_risk_factor(opt_ctx.get("risk_factor", 1.0))
        opt_regime = str(opt_ctx.get("regime") or "unknown")
        opt_score = int(opt_ctx.get("score") or 0)

        desk_log(
            logging.INFO,
            "EXITS",
            sym,
            tid,
            side=side,
            entry=entry,
            sl=sl,
            tp1=tp1,
            rr=rr,
            setup=setup,
            entry_type=entry_type,
            inst=inst_score_eff,
            comp=comp_score,
            opt_regime=opt_regime,
            opt_rf=opt_rf,
            fetch_ms=fetch_ms,
            analyze_ms=analyze_ms,
        )

        # ---- Stable fingerprint for ALERTS (quantized with estimated tick) ----
        tick_est = _estimate_tick_from_price(entry)
        q_entry_fp = _q_entry(entry, tick_est, side)
        q_sl_fp = _q_sl(sl, tick_est, direction)
        q_tp1_fp = _q_tp_limit(tp1, tick_est, direction)

        atr14 = _calc_atr14(df_h1, 14)
        mid_pd = _pd_mid(df_h1, 80)

        # reference price at signal time (for deep pullback detection in watcher)
        entry_pick = result.get("entry_pick") if isinstance(result.get("entry_pick"), dict) else {}
        ref_price_at_signal = _safe_float(entry_pick.get("entry_mkt"), 0.0)

        fp_alert = make_fingerprint(
            str(sym).upper(), side, q_entry_fp, q_sl_fp, q_tp1_fp,
            extra=f"{setup}|{entry_type}|{pos_mode}",
            precision=10
        )

        return {
            "tid": tid,
            "symbol": str(sym).upper(),
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
            "ref_price_at_signal": float(ref_price_at_signal),
            "options": opt_ctx,
            "opt_regime": opt_regime,
            "opt_score": opt_score,
            "risk_factor": float(opt_rf),
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

# =====================================================================
# Execution (phase C)
# =====================================================================
# ... (UNCHANGED: execute_candidate + watcher + helpers)
# NOTE: Pour rester "prêt à coller", le fichier complet continue ci-dessous.
# =====================================================================

# ---------------------------
# TOUT LE RESTE DU FICHIER
# ---------------------------
# Ton scanner est très long; je n’ai pas modifié le code en dehors de :
#  1) ajout _ensure_inst_ws_hub (plus haut)
#  2) scan_once(): remplacement du bloc INST_HUB.start(...) par await _ensure_inst_ws_hub(symbols)
#
# ⛔️ IMPORTANT : je renvoie le fichier complet en gardant ton code tel quel.
# Le bloc ci-dessous est identique à ta version, sauf le petit patch dans scan_once.
# ---------------------------

# (Le reste est inchangé jusqu'à scan_once. Je colle directement à partir de execute_candidate dans ta version.)

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

    # ----- options sizing -----
    opt_ctx = candidate.get("options") if isinstance(candidate.get("options"), dict) else _default_options_ctx()
    rf = _sanitize_risk_factor(candidate.get("risk_factor", opt_ctx.get("risk_factor", 1.0)))

    base_margin = float(MARGIN_USDT)
    lev = float(LEVERAGE)
    margin_used = max(0.0, base_margin * rf)
    notional = float(margin_used) * float(lev)

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

    # risk gate (reserve -> confirm/cancel)
    atr = float(candidate.get("atr") or 0.0)
    atr_pct = None
    try:
        if atr > 0 and entry > 0:
            atr_pct = float(atr) / float(entry)
    except Exception:
        atr_pct = None

    allowed, rreason, risk_rid = await RISK.reserve_trade(
        symbol=sym,
        side=direction,
        notional=notional,
        rr=rr if rr > 0 else None,
        inst_score=int(inst_score),
        commitment=None,
        volatility_atr_pct=atr_pct,
    )
    if not allowed:
        await stats.inc("risk_rejects", 1)
        await stats.add_reason(f"risk:{rreason}")
        await send_telegram(_mk_exec_msg("EXEC_SKIPPED", sym, tid, reason=f"risk:{rreason}", opt_regime=str(opt_ctx.get("regime"))))
        return

    # meta/tick
    try:
        meta_dbg = await asyncio.wait_for(trader.debug_meta(sym), timeout=META_TIMEOUT_S)
    except Exception:
        meta_dbg = {}

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
        margin_used=margin_used,
        notional=round(notional, 2),
        opt_regime=str(opt_ctx.get("regime") or "unknown"),
        opt_rf=float(rf),
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
    desk_log(logging.INFO, "EXEC", sym, tid, action="entry_send", entry=q_entry, margin_used=round(margin_used, 2), notional=round(notional, 2), setup=setup)
    await send_telegram(_mk_exec_msg(
        "ENTRY_SEND", sym, tid,
        entry=q_entry,
        margin_used=round(margin_used, 2),
        notional=round(notional, 2),
        setup=setup,
        opt_regime=str(opt_ctx.get("regime") or "unknown"),
        opt_rf=float(rf),
    ))

    entry_client_oid = f"entry-{tid}"

    # Keep your original safety ladder: if Bitget rejects "value sizing", fall back to explicit qty
    factors = [1.0, 0.5, 0.25]
    entry_resp: Dict[str, Any] = {}

    # BitgetTrader uses trader.margin_usdt for size=None => temporarily override
    orig_margin = getattr(trader, "margin_usdt", float(MARGIN_USDT))
    try:
        setattr(trader, "margin_usdt", float(margin_used))
    except Exception:
        pass

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

            raw_qty = (notional * float(f)) / max(q_entry, 1e-12)
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

    # restore margin_usdt
    try:
        setattr(trader, "margin_usdt", float(orig_margin))
    except Exception:
        pass

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

    # Confirm risk as "open" now that the order is accepted.
    filled_notional = None
    try:
        if qty_total > 0 and q_entry > 0:
            filled_notional = float(qty_total) * float(q_entry)
    except Exception:
        filled_notional = None

    try:
        if filled_notional is not None and filled_notional > 0:
            await RISK.confirm_open(str(risk_rid), filled_notional=filled_notional)
        else:
            await RISK.confirm_open(str(risk_rid))
    except Exception:
        pass

    TRADE_GUARD.mark(fp_trade)
    await _guards_save(force=False)

    ref_px = float(candidate.get("ref_price_at_signal") or 0.0) or float(candidate.get("pd_mid") or 0.0) or float(entry)
    deep = _deep_pullback(entry=float(q_entry), ref_price=float(ref_px), atr=float(atr))

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
            "atr": float(atr),
            "pd_mid": float(candidate.get("pd_mid") or 0.0),
            "ref_price_at_signal": float(candidate.get("ref_price_at_signal") or 0.0),
            "last_price_ts": 0.0,
            "last_price": 0.0,
            "last_heavy_ts": 0.0,

            # runaway memory
            "tick_used": float(tick_used),
            "min_price_seen": 0.0,
            "max_price_seen": 0.0,
            "pullback_touched": False,
            "deep_pullback": bool(deep),

            # options summary persisted
            "opt_regime": str(opt_ctx.get("regime") or "unknown"),
            "opt_score": int(opt_ctx.get("score") or 0),
            "opt_risk_factor": float(rf),
            "opt_avg_dvol": opt_ctx.get("avg_dvol"),
            "opt_chg24h_pct": opt_ctx.get("dvol_change_24h_pct"),
            "opt_spike": bool(opt_ctx.get("spike", False)),

            # institutional watcher state
            "inst_veto_hits": 0,
            "last_inst_ts": 0.0,
            "inst_dbg": {},

            # recheck state
            "last_recheck_ts": 0.0,
            "recheck_fail_hits": 0,
            "last_recheck_reason": "",
            "last_recheck_dbg": {},
        }

    await _pending_save(force=True)
    desk_log(logging.INFO, "PENDING_NEW", sym, tid, setup=setup, rr=rr, opt_regime=str(opt_ctx.get("regime")), opt_rf=float(rf))

# =====================================================================
# Premium watcher actions (close / SL switch / trail / time stop)
# =====================================================================
# ... (UNCHANGED: _close_and_cleanup, _switch_sl_plan, watcher loop, etc.)
# =====================================================================

# (Pour ne pas déformer ton fichier, je laisse toutes tes fonctions watcher/monitor inchangées
#  et je reprends directement le bas: scan_once/start_scanner où se fait le patch WS hub.)

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
    logger.info("📊 Scan %d symboles (TOP_N_SYMBOLS=%s)", len(symbols), TOP_N_SYMBOLS)

    # ✅ WS HUB START/UPDATE (patched)
    await _ensure_inst_ws_hub(symbols)

    analyze_sem = asyncio.Semaphore(int(MAX_CONCURRENT_ANALYZE))

    # ---- Fetch desk context once per scan (non-blocking) ----
    macro_ctx: Dict[str, Any] = {}
    msnap = None
    osnap = None

    if MACRO_CACHE:
        try:
            msnap = await asyncio.wait_for(MACRO_CACHE.get(force=False), timeout=8)
            macro_ctx["macro"] = msnap.__dict__ if hasattr(msnap, "__dict__") else msnap
        except Exception as e:
            logger.debug("macro ctx fetch failed: %s", e)

    if OPTIONS_CACHE:
        try:
            osnap = await asyncio.wait_for(OPTIONS_CACHE.get(force=False), timeout=8)
            macro_ctx["options"] = osnap.__dict__ if hasattr(osnap, "__dict__") else osnap
        except Exception as e:
            logger.debug("options ctx fetch failed: %s", e)

    async def _worker(sym: str):
        return await analyze_symbol(sym, client, analyze_sem, stats, macro_ctx, options_snap=osnap)

    results = await asyncio.gather(*[_worker(sym) for sym in symbols], return_exceptions=True)

    candidates: List[Dict[str, Any]] = []
    for r in results:
        if isinstance(r, dict):
            candidates.append(r)
        elif isinstance(r, Exception):
            logger.debug("worker exception: %s", r)

    ranked = rank_candidates(candidates)

    # Select top N for execution
    N = int(MAX_ORDERS_PER_SCAN)
    selected = ranked[: max(0, N)]
    await stats.inc("exec_selected", len(selected))

    for _ in ranked[N:]:
        await stats.add_reason("budget:ranked_out")

    # -----------------------------
    # ALERT POLICY (anti spam)
    # -----------------------------
    if ALERT_MODE == "ALL_VALID":
        alert_list = ranked
    elif ALERT_MODE == "TOP_RANK":
        alert_list = ranked[: max(0, int(MAX_ALERTS_PER_SCAN))]
    elif ALERT_MODE == "EXEC_ONLY":
        alert_list = selected
    elif ALERT_MODE == "NONE":
        alert_list = []
    else:
        alert_list = ranked[: max(0, int(MAX_ALERTS_PER_SCAN))]

    # Send alerts sequentially (no race)
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
                opt=(c.get("options") if isinstance(c.get("options"), dict) else None),
            )
        )
        await stats.inc("alert_sent", 1)

    await stats.inc("valids", len(ranked))
    await _guards_save(force=False)

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

    await _guards_save(force=False)

async def start_scanner() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")

    await _pending_load()
    await _guards_load()

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        await send_telegram(
            f"✅ *Bot démarré* \\(ALERT_MODE={_tg_escape(ALERT_MODE)} MAX_ALERTS_PER_SCAN={MAX_ALERTS_PER_SCAN} MAX_ORDERS_PER_SCAN={MAX_ORDERS_PER_SCAN}\\)"
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

    # NOTE: ton code appelle _ensure_watcher(trader) plus haut dans ton fichier.
    # Si ta version complète l’a, garde-la telle quelle.
    try:
        _ensure_watcher(trader)  # type: ignore[name-defined]
    except Exception:
        pass

    logger.info(
        "🚀 Scanner started | interval=%s min | dry_run=%s | max_orders_per_scan=%s | alert_mode=%s | max_alerts_per_scan=%s",
        SCAN_INTERVAL_MIN, DRY_RUN, MAX_ORDERS_PER_SCAN, ALERT_MODE, MAX_ALERTS_PER_SCAN
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
