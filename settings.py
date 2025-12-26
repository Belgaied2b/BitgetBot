# =====================================================================
# settings.py — Desk config (Railway env) — Hardened
# =====================================================================
# Objectifs desk-lead :
# - parsing ENV robuste + clamp + logs optionnels
# - cohérence des seuils (RR / inst / modes) avec garde-fous
# - valeurs par défaut “safe prod” (pas trop strict = évite 0 signal)
# - centralise product type / margin coin / execution modes
# =====================================================================

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple


# =====================================================================
# ENV parsing helpers (robust)
# =====================================================================

_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n"}


def _get_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip()


def _get_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    return default


def _get_int(name: str, default: int, *, min_v: Optional[int] = None, max_v: Optional[int] = None) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        x = default
    else:
        try:
            x = int(float(str(v).strip()))
        except Exception:
            x = default
    if min_v is not None:
        x = max(min_v, x)
    if max_v is not None:
        x = min(max_v, x)
    return x


def _get_float(name: str, default: float, *, min_v: Optional[float] = None, max_v: Optional[float] = None) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        x = float(default)
    else:
        try:
            x = float(str(v).strip())
        except Exception:
            x = float(default)
    if min_v is not None:
        x = max(float(min_v), x)
    if max_v is not None:
        x = min(float(max_v), x)
    return float(x)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(x))))


def _normalize_mode(x: str, *, default: str, allowed: Tuple[str, ...]) -> str:
    s = (x or "").strip().lower()
    return s if s in allowed else default


# =====================================================================
# ENV / RUNTIME
# =====================================================================

ENV = _get_str("ENV", "development")
TZ = _get_str("TZ", "Europe/Paris")

# When True: no orders, but analysis + telegram still works
DRY_RUN = _get_bool("DRY_RUN", False)

# Optional: enable more verbose logs in prod when diagnosing
DESK_DEBUG = _get_bool("DESK_DEBUG", False)


# =====================================================================
# TELEGRAM
# =====================================================================

TELEGRAM_BOT_TOKEN = _get_str("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = _get_str("TELEGRAM_CHAT_ID", "")

# Backward-compatible aliases
TOKEN = _get_str("TOKEN", TELEGRAM_BOT_TOKEN)
CHAT_ID = _get_str("CHAT_ID", TELEGRAM_CHAT_ID)


# =====================================================================
# EXCHANGE KEYS / ENDPOINTS (Bitget)
# =====================================================================

API_KEY = _get_str("API_KEY", "")
API_SECRET = _get_str("API_SECRET", "")
API_PASSPHRASE = _get_str("API_PASSPHRASE", "")

BITGET_BASE_URL = _get_str("BITGET_BASE_URL", "https://api.bitget.com")

# Product / margin
# NOTE: Keep PRODUCT_TYPE consistent across client + trader.
PRODUCT_TYPE = _get_str("PRODUCT_TYPE", "USDT-FUTURES")
MARGIN_COIN = _get_str("MARGIN_COIN", "USDT")


# =====================================================================
# RISK / LEVERAGE / SCAN
# =====================================================================

# Per-trade isolated margin you want to allocate (in USDT)
MARGIN_USDT = _get_float("MARGIN_USDT", 20.0, min_v=1.0, max_v=500.0)

# Futures leverage
LEVERAGE = _get_float("LEVERAGE", 10.0, min_v=1.0, max_v=125.0)

# Scan cadence
SCAN_INTERVAL_MIN = _get_int("SCAN_INTERVAL_MIN", 5, min_v=1, max_v=60)

# Limit orders sent per scan loop (risk/ops)
MAX_ORDERS_PER_SCAN = _get_int("MAX_ORDERS_PER_SCAN", 5, min_v=0, max_v=50)

# Universe
TOP_N_SYMBOLS = _get_int("TOP_N_SYMBOLS", 80, min_v=5, max_v=400)

# Global unit risk (used by some modules)
RISK_USDT = _get_float("RISK_USDT", 20.0, min_v=1.0, max_v=500.0)
RR_TARGET = _get_float("RR_TARGET", 1.6, min_v=0.8, max_v=5.0)


# =====================================================================
# INSTITUTIONNEL / STRUCTURE / MOMENTUM FLAGS
# =====================================================================

# Institutional score minimum to accept (when data available)
MIN_INST_SCORE = _get_int("MIN_INST_SCORE", 2, min_v=0, max_v=3)

# Hard gating flags
REQUIRE_STRUCTURE = _get_bool("REQUIRE_STRUCTURE", True)
REQUIRE_MOMENTUM = _get_bool("REQUIRE_MOMENTUM", True)
REQUIRE_HTF_ALIGN = _get_bool("REQUIRE_HTF_ALIGN", True)
REQUIRE_BOS_QUALITY = _get_bool("REQUIRE_BOS_QUALITY", True)

# RR thresholds
RR_MIN_STRICT = _get_float("RR_MIN_STRICT", 1.50, min_v=0.5, max_v=5.0)
RR_MIN_TOLERATED_WITH_INST = _get_float("RR_MIN_TOLERATED_WITH_INST", 1.20, min_v=0.5, max_v=5.0)

# Desk EV mode (continuation path)
DESK_EV_MODE = _get_bool("DESK_EV_MODE", False)
RR_MIN_DESK_PRIORITY = _get_float("RR_MIN_DESK_PRIORITY", 1.10, min_v=0.5, max_v=5.0)
INST_SCORE_DESK_PRIORITY = _get_int("INST_SCORE_DESK_PRIORITY", 2, min_v=0, max_v=3)

# Guardrails: keep hierarchy coherent
# (if user misconfigures env, we auto-fix to avoid dead bot)
if RR_MIN_DESK_PRIORITY > RR_MIN_STRICT:
    RR_MIN_DESK_PRIORITY = RR_MIN_STRICT
if RR_MIN_TOLERATED_WITH_INST > RR_MIN_STRICT:
    RR_MIN_TOLERATED_WITH_INST = RR_MIN_STRICT
if INST_SCORE_DESK_PRIORITY < MIN_INST_SCORE and DESK_EV_MODE:
    # desk priority can't be below min floor if we rely on it
    INST_SCORE_DESK_PRIORITY = MIN_INST_SCORE

# Commitment (optional future use)
COMMITMENT_MIN = _get_float("COMMITMENT_MIN", 0.55, min_v=0.0, max_v=1.0)
COMMITMENT_DESK_PRIORITY = _get_float("COMMITMENT_DESK_PRIORITY", 0.60, min_v=0.0, max_v=1.0)
if COMMITMENT_DESK_PRIORITY < COMMITMENT_MIN:
    COMMITMENT_DESK_PRIORITY = COMMITMENT_MIN


# =====================================================================
# ATR / STOPS / LIQUIDITÉ / TP
# =====================================================================

ATR_LEN = _get_int("ATR_LEN", 14, min_v=5, max_v=100)
ATR_MULT_SL = _get_float("ATR_MULT_SL", 2.5, min_v=0.5, max_v=10.0)
ATR_MULT_SL_CAP = _get_float("ATR_MULT_SL_CAP", 3.5, min_v=0.5, max_v=15.0)
if ATR_MULT_SL_CAP < ATR_MULT_SL:
    ATR_MULT_SL_CAP = ATR_MULT_SL

SL_BUFFER_PCT = _get_float("SL_BUFFER_PCT", 0.0020, min_v=0.0, max_v=0.05)
SL_BUFFER_TICKS = _get_int("SL_BUFFER_TICKS", 3, min_v=0, max_v=50)
MIN_SL_TICKS = _get_int("MIN_SL_TICKS", 3, min_v=0, max_v=50)
MAX_SL_PCT = _get_float("MAX_SL_PCT", 0.07, min_v=0.01, max_v=0.50)

STRUCT_LOOKBACK = _get_int("STRUCT_LOOKBACK", 20, min_v=10, max_v=200)

# Break-even buffer
BE_FEE_BUFFER_TICKS = _get_int("BE_FEE_BUFFER_TICKS", 1, min_v=0, max_v=25)

# TP policy
TP1_R_CLAMP_MIN = _get_float("TP1_R_CLAMP_MIN", 1.4, min_v=0.5, max_v=5.0)
TP1_R_CLAMP_MAX = _get_float("TP1_R_CLAMP_MAX", 1.6, min_v=0.5, max_v=6.0)
if TP1_R_CLAMP_MAX < TP1_R_CLAMP_MIN:
    TP1_R_CLAMP_MAX = TP1_R_CLAMP_MIN

TP2_R_TARGET = _get_float("TP2_R_TARGET", 2.8, min_v=0.8, max_v=10.0)
MIN_TP_TICKS = _get_int("MIN_TP_TICKS", 1, min_v=0, max_v=50)
TP1_R_BY_VOL = _get_bool("TP1_R_BY_VOL", True)

STOP_TRIGGER_TYPE_SL = _get_str("STOP_TRIGGER_TYPE_SL", "MP")  # "MP" or "FP"
STOP_TRIGGER_TYPE_TP = _get_str("STOP_TRIGGER_TYPE_TP", "TP")  # kept for compatibility

# Liquidity zones
LIQ_LOOKBACK = _get_int("LIQ_LOOKBACK", 60, min_v=20, max_v=500)
LIQ_BUFFER_PCT = _get_float("LIQ_BUFFER_PCT", 0.0008, min_v=0.0, max_v=0.02)
LIQ_BUFFER_TICKS = _get_int("LIQ_BUFFER_TICKS", 3, min_v=0, max_v=50)

# Vol regime for indicators.volatility_regime
VOL_REGIME_ATR_PCT_LOW = _get_float("VOL_REGIME_ATR_PCT_LOW", 0.015, min_v=0.001, max_v=0.20)
VOL_REGIME_ATR_PCT_HIGH = _get_float("VOL_REGIME_ATR_PCT_HIGH", 0.035, min_v=0.002, max_v=0.30)
if VOL_REGIME_ATR_PCT_HIGH <= VOL_REGIME_ATR_PCT_LOW:
    VOL_REGIME_ATR_PCT_HIGH = VOL_REGIME_ATR_PCT_LOW * 1.5


# =====================================================================
# ACCOUNT / EXPOSURE (Risk Manager)
# =====================================================================

ACCOUNT_EQUITY_USDT = _get_float("ACCOUNT_EQUITY_USDT", 10000.0, min_v=100.0, max_v=10_000_000.0)
MAX_GROSS_EXPOSURE = _get_float("MAX_GROSS_EXPOSURE", 2.0, min_v=0.1, max_v=10.0)
MAX_SYMBOL_EXPOSURE = _get_float("MAX_SYMBOL_EXPOSURE", 0.25, min_v=0.01, max_v=1.0)
CORR_GROUP_CAP = _get_float("CORR_GROUP_CAP", 0.5, min_v=0.05, max_v=2.0)
CORR_BTC_THRESHOLD = _get_float("CORR_BTC_THRESHOLD", 0.7, min_v=0.0, max_v=1.0)

ENABLE_SQUEEZE_ENGINE = _get_bool("ENABLE_SQUEEZE_ENGINE", True)
FAIL_OPEN_TO_CORE = _get_bool("FAIL_OPEN_TO_CORE", True)


# =====================================================================
# BINANCE / INSTITUTIONNEL (HTTP tuning)
# =====================================================================

BINANCE_MIN_INTERVAL_S = _get_float("BINANCE_MIN_INTERVAL_S", 0.35, min_v=0.0, max_v=5.0)
BINANCE_HTTP_TIMEOUT_S = _get_float("BINANCE_HTTP_TIMEOUT_S", 7.0, min_v=2.0, max_v=30.0)
BINANCE_HTTP_RETRIES = _get_int("BINANCE_HTTP_RETRIES", 2, min_v=0, max_v=10)

BINANCE_SYMBOLS_TTL_S = _get_int("BINANCE_SYMBOLS_TTL_S", 900, min_v=60, max_v=24 * 3600)

RETRY_300011_MAX = _get_int("RETRY_300011_MAX", 3, min_v=0, max_v=20)
RETRY_BACKOFF_MS_BASE = _get_int("RETRY_BACKOFF_MS_BASE", 250, min_v=50, max_v=5000)
RETRY_BACKOFF_JITTER_MIN = _get_int("RETRY_BACKOFF_JITTER_MIN", 50, min_v=0, max_v=5000)
RETRY_BACKOFF_JITTER_MAX = _get_int("RETRY_BACKOFF_JITTER_MAX", 200, min_v=0, max_v=5000)
if RETRY_BACKOFF_JITTER_MAX < RETRY_BACKOFF_JITTER_MIN:
    RETRY_BACKOFF_JITTER_MAX = RETRY_BACKOFF_JITTER_MIN


# =====================================================================
# EXECUTION (price / slippage)
# =====================================================================

# Bitget position mode: "hedge" or "one_way"
POSITION_MODE = _normalize_mode(_get_str("POSITION_MODE", "hedge"), default="hedge", allowed=("hedge", "one_way"))

PRICE_NUDGE_TICKS_MIN = _get_int("PRICE_NUDGE_TICKS_MIN", 0, min_v=0, max_v=20)
PRICE_NUDGE_TICKS_MAX = _get_int("PRICE_NUDGE_TICKS_MAX", 2, min_v=0, max_v=50)
if PRICE_NUDGE_TICKS_MAX < PRICE_NUDGE_TICKS_MIN:
    PRICE_NUDGE_TICKS_MAX = PRICE_NUDGE_TICKS_MIN

SLIPPAGE_TICKS_LIMIT = _get_int("SLIPPAGE_TICKS_LIMIT", 2, min_v=0, max_v=50)


# =====================================================================
# Optional: expose a compact settings snapshot for debug logs
# =====================================================================

@dataclass(frozen=True)
class DeskSettings:
    env: str
    tz: str
    dry_run: bool
    desk_ev_mode: bool
    margin_usdt: float
    leverage: float
    scan_interval_min: int
    top_n_symbols: int
    max_orders_per_scan: int
    min_inst_score: int
    rr_min_strict: float
    rr_min_desk_priority: float

def snapshot() -> DeskSettings:
    return DeskSettings(
        env=ENV,
        tz=TZ,
        dry_run=DRY_RUN,
        desk_ev_mode=DESK_EV_MODE,
        margin_usdt=MARGIN_USDT,
        leverage=LEVERAGE,
        scan_interval_min=SCAN_INTERVAL_MIN,
        top_n_symbols=TOP_N_SYMBOLS,
        max_orders_per_scan=MAX_ORDERS_PER_SCAN,
        min_inst_score=MIN_INST_SCORE,
        rr_min_strict=RR_MIN_STRICT,
        rr_min_desk_priority=RR_MIN_DESK_PRIORITY,
    )
