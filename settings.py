# =====================================================================
# settings.py — Desk config (Railway env) — Hardened (Bitget-aligned)
# =====================================================================
# Objectifs desk-lead :
# - parsing ENV robuste + clamp
# - cohérence des seuils (RR / inst / modes) avec garde-fous
# - valeurs par défaut “safe prod” (pas trop strict = évite 0 signal)
# - centralise product type / margin coin / execution modes
# - grading A→E + règles d’exécution associées (auto-order vs alert-only)
# - TTL / runaway / WS hub flags centralisés
#
# IMPORTANT (migration Binance -> Bitget):
# - On conserve des noms d'ENV historiques "BINANCE_*" en alias pour compat,
#   mais ce fichier expose désormais des variables "BITGET_*" + "INST_*".
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
# Mix WS endpoint used in your logs: wss://ws.bitget.com/mix/v1/stream
BITGET_MIX_WS_URL = _get_str("BITGET_MIX_WS_URL", "wss://ws.bitget.com/mix/v1/stream")

# Product / margin (used by private endpoints / trader)
# NOTE: keep PRODUCT_TYPE consistent across bitget_client + bitget_trader
PRODUCT_TYPE = _get_str("PRODUCT_TYPE", "USDT-FUTURES").strip()
MARGIN_COIN = _get_str("MARGIN_COIN", "USDT").strip()


def _product_type_to_inst_product_type(product_type: str) -> str:
    """
    institutional_data.py (Bitget v2 Mix market endpoints) expects lower-case
    productType like "usdt-futures".
    """
    s = (product_type or "").strip().lower()
    if s in ("usdt-futures", "coin-futures", "usdc-futures"):
        return s
    # Common internal formats:
    if s in ("usdt-futures", "usdtfutures", "usdt_futures"):
        return "usdt-futures"
    if s in ("coin-futures", "coinfutures", "coin_futures"):
        return "coin-futures"
    if s in ("usdc-futures", "usdcfutures", "usdc_futures"):
        return "usdc-futures"
    # Uppercase formats:
    if s == "usdt-futures" or s == "usdt_futures":
        return "usdt-futures"
    if s == "coin-futures" or s == "coin_futures":
        return "coin-futures"
    if s == "usdc-futures" or s == "usdc_futures":
        return "usdc-futures"
    # Bitget default (USDT-M)
    return "usdt-futures"


# Institutional (public endpoints) productType
# If user sets INST_BITGET_PRODUCT_TYPE explicitly, it wins.
INST_BITGET_PRODUCT_TYPE = _get_str(
    "INST_BITGET_PRODUCT_TYPE",
    _product_type_to_inst_product_type(PRODUCT_TYPE),
).strip()


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
MAX_ORDERS_PER_SCAN = _get_int("MAX_ORDERS_PER_SCAN", 10, min_v=0, max_v=50)

# Universe
TOP_N_SYMBOLS = _get_int("TOP_N_SYMBOLS", 150, min_v=5, max_v=400)

# Execution logs (TCA / audit)
EXEC_LOG_ENABLE = _get_bool("EXEC_LOG_ENABLE", False)
EXEC_LOG_PATH = _get_str("EXEC_LOG_PATH", "exec_log.jsonl")

# Global unit risk (used by some modules)
RISK_USDT = _get_float("RISK_USDT", 20.0, min_v=1.0, max_v=500.0)
RR_TARGET = _get_float("RR_TARGET", 1.6, min_v=0.8, max_v=5.0)

# Desk risk limits (used by RiskManager)
MAX_DAILY_LOSS = _get_float("MAX_DAILY_LOSS", 60.0, min_v=0.0, max_v=100000.0)
MAX_TRADES_PER_DAY = _get_int("MAX_TRADES_PER_DAY", 500, min_v=0, max_v=10000)
MAX_OPEN_POSITIONS = _get_int("MAX_OPEN_POSITIONS", 20, min_v=0, max_v=200)
MAX_LONG_POSITIONS = _get_int("MAX_LONG_POSITIONS", 15, min_v=0, max_v=200)
MAX_SHORT_POSITIONS = _get_int("MAX_SHORT_POSITIONS", 15, min_v=0, max_v=200)
MAX_CONSECUTIVE_LOSSES = _get_int("MAX_CONSECUTIVE_LOSSES", 5, min_v=0, max_v=200)
TILT_COOLDOWN_SECONDS = _get_int("TILT_COOLDOWN_SECONDS", 3600, min_v=0, max_v=86400)
DRAWDOWN_RISK_FACTOR = _get_float("DRAWDOWN_RISK_FACTOR", 0.5, min_v=0.1, max_v=1.0)

# Per-symbol risk caps (0 disables)
SYMBOL_MAX_DAILY_LOSS = _get_float("SYMBOL_MAX_DAILY_LOSS", 0.0, min_v=0.0, max_v=100000.0)
SYMBOL_MAX_TRADES_PER_DAY = _get_int("SYMBOL_MAX_TRADES_PER_DAY", 0, min_v=0, max_v=10000)


# =====================================================================
# INSTITUTIONNEL / STRUCTURE / MOMENTUM FLAGS
# =====================================================================

# Align with your current scoring scale seen in logs (inst_score up to 4)
INST_SCORE_MAX = _get_int("INST_SCORE_MAX", 4, min_v=1, max_v=10)

# Institutional score minimum to accept (when data available)
MIN_INST_SCORE = _get_int("MIN_INST_SCORE", 2, min_v=0, max_v=INST_SCORE_MAX)

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
INST_SCORE_DESK_PRIORITY = _get_int("INST_SCORE_DESK_PRIORITY", 2, min_v=0, max_v=INST_SCORE_MAX)

# Guardrails: keep hierarchy coherent
if RR_MIN_DESK_PRIORITY > RR_MIN_STRICT:
    RR_MIN_DESK_PRIORITY = RR_MIN_STRICT
if RR_MIN_TOLERATED_WITH_INST > RR_MIN_STRICT:
    RR_MIN_TOLERATED_WITH_INST = RR_MIN_STRICT
if DESK_EV_MODE and INST_SCORE_DESK_PRIORITY < MIN_INST_SCORE:
    INST_SCORE_DESK_PRIORITY = MIN_INST_SCORE
if MIN_INST_SCORE > INST_SCORE_MAX:
    MIN_INST_SCORE = INST_SCORE_MAX

# Commitment (optional future use)
COMMITMENT_MIN = _get_float("COMMITMENT_MIN", 0.55, min_v=0.0, max_v=1.0)
COMMITMENT_DESK_PRIORITY = _get_float("COMMITMENT_DESK_PRIORITY", 0.60, min_v=0.0, max_v=1.0)
if COMMITMENT_DESK_PRIORITY < COMMITMENT_MIN:
    COMMITMENT_DESK_PRIORITY = COMMITMENT_MIN


# =====================================================================
# INSTITUTIONAL DATA (public endpoints) — align with institutional_data.py
# =====================================================================

# Mode used by institutional_data/analyze_signal: LIGHT / NORMAL / FULL
INST_MODE = _get_str("INST_MODE", "LIGHT").strip().upper()
if INST_MODE not in ("LIGHT", "NORMAL", "FULL"):
    INST_MODE = "LIGHT"

# Optional best-effort features (whether institutional_data should attempt them)
INST_ENABLE_OPEN_INTEREST = _get_bool("INST_ENABLE_OPEN_INTEREST", True)
INST_ENABLE_CURRENT_FUNDING = _get_bool("INST_ENABLE_CURRENT_FUNDING", True)
INST_ENABLE_RECENT_FILLS = _get_bool("INST_ENABLE_RECENT_FILLS", False)
INST_ENABLE_CANDLES = _get_bool("INST_ENABLE_CANDLES", False)

# Normalisation (rolling z-scores)
INST_NORM_ENABLED = _get_bool("INST_NORM_ENABLED", True)
INST_NORM_MIN_POINTS = _get_int("INST_NORM_MIN_POINTS", 20, min_v=5, max_v=500)
INST_NORM_WINDOW = _get_int("INST_NORM_WINDOW", 120, min_v=20, max_v=5000)

# WS hub controls (institutional_ws_hub)
# - INST_USE_WS_HUB is the switch used by institutional_data.py in your version
# - INST_WS_HUB_ENABLE is kept as alias for older code paths
INST_USE_WS_HUB = _get_bool("INST_USE_WS_HUB", True)
INST_WS_HUB_ENABLE = _get_bool("INST_WS_HUB_ENABLE", _get_bool("ENABLE_INST_WS_HUB", INST_USE_WS_HUB))
WS_STALE_SEC = _get_float("INST_WS_STALE_SEC", 15.0, min_v=1.0, max_v=300.0)


# =====================================================================
# BITGET (HTTP tuning) — replaces BINANCE_* but keeps aliases
# =====================================================================

# These names match your institutional_data.py defaults:
BITGET_HTTP_CONCURRENCY = _get_int("BITGET_HTTP_CONCURRENCY", 4, min_v=1, max_v=50)

# Prefer BITGET_MIN_INTERVAL_SEC; allow legacy BINANCE_MIN_INTERVAL_S too.
BITGET_MIN_INTERVAL_SEC = _get_float(
    "BITGET_MIN_INTERVAL_SEC",
    _get_float("BITGET_MIN_INTERVAL_S", _get_float("BINANCE_MIN_INTERVAL_S", 0.08, min_v=0.0, max_v=5.0), min_v=0.0, max_v=5.0),
    min_v=0.0,
    max_v=5.0,
)

BITGET_HTTP_TIMEOUT_S = _get_float(
    "BITGET_HTTP_TIMEOUT_S",
    _get_float("BINANCE_HTTP_TIMEOUT_S", 10.0, min_v=2.0, max_v=30.0),
    min_v=2.0,
    max_v=60.0,
)

BITGET_HTTP_RETRIES = _get_int(
    "BITGET_HTTP_RETRIES",
    _get_int("BINANCE_HTTP_RETRIES", 2, min_v=0, max_v=10),
    min_v=0,
    max_v=20,
)

# Contracts / symbols cache TTL (Bitget)
BITGET_SYMBOLS_TTL_S = _get_int(
    "BITGET_SYMBOLS_TTL_S",
    _get_int("BINANCE_SYMBOLS_TTL_S", 900, min_v=60, max_v=24 * 3600),
    min_v=60,
    max_v=24 * 3600,
)

# ---- Deprecated aliases (do not remove yet; other modules may import them) ----
BINANCE_MIN_INTERVAL_S = BITGET_MIN_INTERVAL_SEC
BINANCE_HTTP_TIMEOUT_S = BITGET_HTTP_TIMEOUT_S
BINANCE_HTTP_RETRIES = BITGET_HTTP_RETRIES
BINANCE_SYMBOLS_TTL_S = BITGET_SYMBOLS_TTL_S

# Retry controls (kept generic, used by retry_utils/rate-limit errors)
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
# PRIORITY (A→E) + EXECUTION POLICY
# =====================================================================

PRIORITY_LEVELS: Tuple[str, ...] = ("A", "B", "C", "D", "E")
_PRIORITY_RANK = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}


def _get_priority(name: str, default: str = "C") -> str:
    """
    Parse une priorité A/B/C/D/E depuis ENV. Si invalide → default.
    """
    v = _get_str(name, default).strip().upper()
    return v if v in PRIORITY_LEVELS else str(default).strip().upper()


def priority_rank(p: str) -> int:
    """
    A=5 ... E=1, invalide=0
    """
    if not p:
        return 0
    return int(_PRIORITY_RANK.get(str(p).strip().upper(), 0))


def priority_at_least(p: str, min_p: str) -> bool:
    """
    True si p >= min_p (en ranking A>B>...>E)
    """
    return priority_rank(p) >= priority_rank(min_p)


# Auto-order seulement si priority >= AUTO_ORDER_MIN_PRIORITY
AUTO_ORDER_MIN_PRIORITY = _get_priority("AUTO_ORDER_MIN_PRIORITY", "C")

# Pass2 (= datafeeds/inst lourd) seulement si pre_priority >= PASS2_ONLY_FOR_PRIORITY
PASS2_ONLY_FOR_PRIORITY = _get_priority("PASS2_ONLY_FOR_PRIORITY", "C")


# =====================================================================
# SIGNAL TTL / RUNAWAY / PENDING POLICY
# =====================================================================

ENTRY_TTL_MINUTES = _get_int("ENTRY_TTL_MINUTES", 45, min_v=1, max_v=24 * 60)

# distance relative entre prix actuel et entry (ex: 0.006 = 0.6%)
RUNAWAY_PCT = _get_float("RUNAWAY_PCT", 0.006, min_v=0.0, max_v=0.20)

PENDING_RECHECK_INTERVAL_S = _get_float("PENDING_RECHECK_INTERVAL_S", 20.0, min_v=0.0, max_v=600.0)


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

BE_FEE_BUFFER_TICKS = _get_int("BE_FEE_BUFFER_TICKS", 1, min_v=0, max_v=25)

LIQ_LOOKBACK = _get_int("LIQ_LOOKBACK", 60, min_v=20, max_v=500)
LIQ_BUFFER_PCT = _get_float("LIQ_BUFFER_PCT", 0.0008, min_v=0.0, max_v=0.02)
LIQ_BUFFER_TICKS = _get_int("LIQ_BUFFER_TICKS", 3, min_v=0, max_v=50)

LIQ_BUFFER_ATR_MULT = _get_float("LIQ_BUFFER_ATR_MULT", 0.12, min_v=0.0, max_v=2.0)
SL_BUFFER_ATR_MULT = _get_float("SL_BUFFER_ATR_MULT", 0.00, min_v=0.0, max_v=2.0)

SL_POLICY_DEFAULT = _get_str("SL_POLICY_DEFAULT", "TIGHT").strip().upper()
SL_POLICY_BOS = _get_str("SL_POLICY_BOS", "SWING,LIQ,ATR").strip().upper()
SL_POLICY_OTE = _get_str("SL_POLICY_OTE", "LIQ,SWING,ATR").strip().upper()
SL_POLICY_INST = _get_str("SL_POLICY_INST", "LIQ,SWING,ATR").strip().upper()

TP1_R_CLAMP_MIN = _get_float("TP1_R_CLAMP_MIN", 1.4, min_v=0.5, max_v=5.0)
TP1_R_CLAMP_MAX = _get_float("TP1_R_CLAMP_MAX", 1.6, min_v=0.5, max_v=6.0)
if TP1_R_CLAMP_MAX < TP1_R_CLAMP_MIN:
    TP1_R_CLAMP_MAX = TP1_R_CLAMP_MIN

TP2_R_TARGET = _get_float("TP2_R_TARGET", 2.8, min_v=0.8, max_v=10.0)
MIN_TP_TICKS = _get_int("MIN_TP_TICKS", 1, min_v=0, max_v=50)
TP1_R_BY_VOL = _get_bool("TP1_R_BY_VOL", True)

STOP_TRIGGER_TYPE_SL = _get_str("STOP_TRIGGER_TYPE_SL", "MP")  # "MP" or "FP"
STOP_TRIGGER_TYPE_TP = _get_str("STOP_TRIGGER_TYPE_TP", "TP")  # kept for compatibility

TP1_USE_EQUAL_LEVELS = _get_bool("TP1_USE_EQUAL_LEVELS", True)
TP1_USE_SWINGS = _get_bool("TP1_USE_SWINGS", True)

TP1_LIQ_MAX_DIST_FACTOR = _get_float("TP1_LIQ_MAX_DIST_FACTOR", 1.00, min_v=0.2, max_v=3.0)
TP1_LIQ_MIN_RR_EPS = _get_float("TP1_LIQ_MIN_RR_EPS", 0.08, min_v=0.0, max_v=0.50)

TP_LIQ_BUFFER_PCT = _get_float("TP_LIQ_BUFFER_PCT", 0.0000, min_v=0.0, max_v=0.01)
TP_LIQ_BUFFER_TICKS = _get_int("TP_LIQ_BUFFER_TICKS", 1, min_v=0, max_v=20)
TP_LIQ_BUFFER_ATR_MULT = _get_float("TP_LIQ_BUFFER_ATR_MULT", 0.06, min_v=0.0, max_v=1.0)

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

    # institutional gating
    inst_mode: str
    min_inst_score: int
    inst_score_max: int

    rr_min_strict: float
    rr_min_desk_priority: float

    # bitget inst http
    bitget_http_concurrency: int
    bitget_min_interval_sec: float
    bitget_http_timeout_s: float
    bitget_http_retries: int

    # ws hub
    inst_use_ws_hub: bool
    ws_stale_sec: float
    inst_product_type: str


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

        inst_mode=INST_MODE,
        min_inst_score=MIN_INST_SCORE,
        inst_score_max=INST_SCORE_MAX,

        rr_min_strict=RR_MIN_STRICT,
        rr_min_desk_priority=RR_MIN_DESK_PRIORITY,

        bitget_http_concurrency=BITGET_HTTP_CONCURRENCY,
        bitget_min_interval_sec=BITGET_MIN_INTERVAL_SEC,
        bitget_http_timeout_s=BITGET_HTTP_TIMEOUT_S,
        bitget_http_retries=BITGET_HTTP_RETRIES,

        inst_use_ws_hub=INST_USE_WS_HUB,
        ws_stale_sec=WS_STALE_SEC,
        inst_product_type=INST_BITGET_PRODUCT_TYPE,
    )
