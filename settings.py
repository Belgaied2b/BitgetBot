# =====================================================================
# settings.py — Desk config, basé sur variables d'environnement Railway
# =====================================================================

from __future__ import annotations

import os
from typing import Optional


def _get_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _get_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


def _get_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


# =====================================================================
# ENV / RUNTIME
# =====================================================================

ENV = os.getenv("ENV", "development")
TZ = os.getenv("TZ", "Europe/Paris")

DRY_RUN = _get_bool("DRY_RUN", False)

# =====================================================================
# TELEGRAM / BOT
# =====================================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TOKEN = os.getenv("TOKEN", TELEGRAM_BOT_TOKEN)
CHAT_ID = os.getenv("CHAT_ID", TELEGRAM_CHAT_ID)

# =====================================================================
# EXCHANGE KEYS / ENDPOINTS (Bitget)
# =====================================================================

API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")
API_PASSPHRASE = os.getenv("API_PASSPHRASE", "")

BITGET_BASE_URL = os.getenv("BITGET_BASE_URL", "https://api.bitget.com")

# Produit / margin par défaut
PRODUCT_TYPE = "USDT-FUTURES"
MARGIN_COIN = "USDT"

# =====================================================================
# RISK / LEVIER / MARGE / SCAN
# =====================================================================

MARGIN_USDT = _get_float("MARGIN_USDT", 20.0)
LEVERAGE = _get_float("LEVERAGE", 10.0)

SCAN_INTERVAL_MIN = _get_int("SCAN_INTERVAL_MIN", 5)
MAX_ORDERS_PER_SCAN = _get_int("MAX_ORDERS_PER_SCAN", 5)

# Risk unit globale
RISK_USDT = _get_float("RISK_USDT", 20.0)
RR_TARGET = _get_float("RR_TARGET", 1.6)

TOP_N_SYMBOLS = _get_int("TOP_N_SYMBOLS", 80)

# =====================================================================
# INSTITUTIONNEL / STRUCTURE / MOMENTUM FLAGS
# =====================================================================

MIN_INST_SCORE = _get_int("MIN_INST_SCORE", 2)

REQUIRE_STRUCTURE = _get_bool("REQUIRE_STRUCTURE", True)
REQUIRE_MOMENTUM = _get_bool("REQUIRE_MOMENTUM", True)
REQUIRE_HTF_ALIGN = _get_bool("REQUIRE_HTF_ALIGN", True)
REQUIRE_BOS_QUALITY = _get_bool("REQUIRE_BOS_QUALITY", True)

# RR thresholds (stricte / tolérée / desk EV)
RR_MIN_STRICT = _get_float("RR_MIN_STRICT", 1.5)
RR_MIN_TOLERATED_WITH_INST = _get_float("RR_MIN_TOLERATED_WITH_INST", 1.20)

DESK_EV_MODE = _get_bool("DESK_EV_MODE", False)
RR_MIN_DESK_PRIORITY = _get_float("RR_MIN_DESK_PRIORITY", 1.10)
INST_SCORE_DESK_PRIORITY = _get_int("INST_SCORE_DESK_PRIORITY", 2)

# Commitment (si utilisé plus tard)
COMMITMENT_MIN = _get_float("COMMITMENT_MIN", 0.55)
COMMITMENT_DESK_PRIORITY = _get_float("COMMITMENT_DESK_PRIORITY", 0.60)

# =====================================================================
# ATR / STOPS / LIQUIDITÉ / TP
# =====================================================================

ATR_LEN = _get_int("ATR_LEN", 14)
ATR_MULT_SL = _get_float("ATR_MULT_SL", 2.5)
ATR_MULT_SL_CAP = _get_float("ATR_MULT_SL_CAP", 3.5)

SL_BUFFER_PCT = _get_float("SL_BUFFER_PCT", 0.0020)
SL_BUFFER_TICKS = _get_int("SL_BUFFER_TICKS", 3)
MIN_SL_TICKS = _get_int("MIN_SL_TICKS", 3)
MAX_SL_PCT = _get_float("MAX_SL_PCT", 0.07)

STRUCT_LOOKBACK = _get_int("STRUCT_LOOKBACK", 20)

BE_FEE_BUFFER_TICKS = _get_int("BE_FEE_BUFFER_TICKS", 1)

TP1_R_CLAMP_MIN = _get_float("TP1_R_CLAMP_MIN", 1.4)
TP1_R_CLAMP_MAX = _get_float("TP1_R_CLAMP_MAX", 1.6)
TP2_R_TARGET = _get_float("TP2_R_TARGET", 2.8)
MIN_TP_TICKS = _get_int("MIN_TP_TICKS", 1)
TP1_R_BY_VOL = _get_bool("TP1_R_BY_VOL", True)

STOP_TRIGGER_TYPE_SL = os.getenv("STOP_TRIGGER_TYPE_SL", "MP")
STOP_TRIGGER_TYPE_TP = os.getenv("STOP_TRIGGER_TYPE_TP", "TP")

LIQ_LOOKBACK = _get_int("LIQ_LOOKBACK", 60)
LIQ_BUFFER_PCT = _get_float("LIQ_BUFFER_PCT", 0.0008)
LIQ_BUFFER_TICKS = _get_int("LIQ_BUFFER_TICKS", 3)

# Vol regime pour indicators.volatility_regime
VOL_REGIME_ATR_PCT_LOW = _get_float("VOL_REGIME_ATR_PCT_LOW", 0.015)
VOL_REGIME_ATR_PCT_HIGH = _get_float("VOL_REGIME_ATR_PCT_HIGH", 0.035)

# =====================================================================
# COMPTE / EXPOSURE (Risk Manager)
# =====================================================================

ACCOUNT_EQUITY_USDT = _get_float("ACCOUNT_EQUITY_USDT", 10000.0)
MAX_GROSS_EXPOSURE = _get_float("MAX_GROSS_EXPOSURE", 2.0)
MAX_SYMBOL_EXPOSURE = _get_float("MAX_SYMBOL_EXPOSURE", 0.25)
CORR_GROUP_CAP = _get_float("CORR_GROUP_CAP", 0.5)
CORR_BTC_THRESHOLD = _get_float("CORR_BTC_THRESHOLD", 0.7)

ENABLE_SQUEEZE_ENGINE = _get_bool("ENABLE_SQUEEZE_ENGINE", True)
FAIL_OPEN_TO_CORE = _get_bool("FAIL_OPEN_TO_CORE", True)

# =====================================================================
# BINANCE / INSTITUTIONNEL (HTTP tuning)
# =====================================================================

BINANCE_MIN_INTERVAL_S = _get_float("BINANCE_MIN_INTERVAL_S", 0.35)
BINANCE_HTTP_TIMEOUT_S = _get_float("BINANCE_HTTP_TIMEOUT_S", 7.0)
BINANCE_HTTP_RETRIES = _get_int("BINANCE_HTTP_RETRIES", 2)

BINANCE_SYMBOLS_TTL_S = _get_int("BINANCE_SYMBOLS_TTL_S", 900)

RETRY_300011_MAX = _get_int("RETRY_300011_MAX", 3)
RETRY_BACKOFF_MS_BASE = _get_int("RETRY_BACKOFF_MS_BASE", 250)
RETRY_BACKOFF_JITTER_MIN = _get_int("RETRY_BACKOFF_JITTER_MIN", 50)
RETRY_BACKOFF_JITTER_MAX = _get_int("RETRY_BACKOFF_JITTER_MAX", 200)

# =====================================================================
# EXECUTION (prix / slippage)
# =====================================================================
# Bitget position mode: "hedge" (recommandé vu tes erreurs) ou "oneway"

POSITION_MODE = "hedge"

PRICE_NUDGE_TICKS_MIN = _get_int("PRICE_NUDGE_TICKS_MIN", 0)
PRICE_NUDGE_TICKS_MAX = _get_int("PRICE_NUDGE_TICKS_MAX", 2)
SLIPPAGE_TICKS_LIMIT = _get_int("SLIPPAGE_TICKS_LIMIT", 2)
