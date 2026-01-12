# =====================================================================
# analyze_signal.py — Desk institutional analyzer (INSTITUTIONAL MAX PACK)
# - 2-pass institutional policy (LIGHT then NORMAL/FULL)
# - Liquidations PASS2-only by default (anti-ban)
# - TTL-compatible setup_type suffixing for scanner.py
# - Optional: SMT divergence (cross-symbol) + normalized inst score (0..100)
# - Options/IV regime context (Deribit DVOL snapshot) + IV macro hint
# - TrendGuard: HTF bias mismatch + ADX/squeeze + overextension MARKET veto
# - Institutional MAX: Killzones, Liquidity/Tradability filter, Microstructure & Inst alignment filters,
#   confidence score and stricter continuation policy.
# =====================================================================

from __future__ import annotations

import logging
import os
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from structure_utils import (
    analyze_structure,
    htf_trend_ok,
    bos_quality_details,
    detect_equal_levels,
    liquidity_sweep_details,
)

from indicators import (
    institutional_momentum,
    compute_ote,
    volatility_regime,
    extension_signal,
    composite_momentum,
    true_atr,
)

from stops import protective_stop_long, protective_stop_short
from tp_clamp import compute_tp1
from institutional_data import compute_full_institutional_analysis

from settings import (
    MIN_INST_SCORE,
    RR_MIN_STRICT,
    RR_MIN_DESK_PRIORITY,
    INST_SCORE_DESK_PRIORITY,
    DESK_EV_MODE,
    REQUIRE_STRUCTURE,
    REQUIRE_MOMENTUM,
    REQUIRE_HTF_ALIGN,
    REQUIRE_BOS_QUALITY,
    RR_MIN_TOLERATED_WITH_INST,
    # grading / pass2 policy
    PASS2_ONLY_FOR_PRIORITY,
    priority_at_least,
)

LOGGER = logging.getLogger(__name__)

REQUIRED_COLS = ("open", "high", "low", "close", "volume")

# Optional: allow technical fallback if Binance is banned/down (default False)
ALLOW_TECH_FALLBACK_WHEN_INST_DOWN = str(os.getenv("ALLOW_TECH_FALLBACK_WHEN_INST_DOWN", "0")).strip() == "1"

# =====================================================================
# ✅ Optional SMT divergence (cross-symbol) + options regime helper
# =====================================================================
try:
    from smt_utils import compute_smt_divergence  # type: ignore
except Exception:
    compute_smt_divergence = None  # type: ignore

try:
    from options_data import OptionsSnapshot, score_options_context  # type: ignore
except Exception:
    OptionsSnapshot = None  # type: ignore
    score_options_context = None  # type: ignore


# =====================================================================
# ✅ ENV helpers
# =====================================================================
def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip() == "1"


def _env_str(name: str, default: str) -> str:
    return str(os.getenv(name, default)).strip()


def _env_int(name: str, default: str) -> int:
    try:
        return int(_env_str(name, default))
    except Exception:
        return int(default)


def _env_float(name: str, default: str) -> float:
    try:
        return float(_env_str(name, default))
    except Exception:
        return float(default)


# =====================================================================
# ✅ INSTITUTIONAL MAX: Session / Killzones (UTC windows by default)
# =====================================================================
# If enabled: outside killzones -> reject in non-desk, soft veto in desk.
SESSION_FILTER_ENABLE = _env_flag("SESSION_FILTER_ENABLE", "1")
SESSION_FILTER_STRICT = _env_flag("SESSION_FILTER_STRICT", "1")  # if 0 and desk -> always soft veto only
SESSION_TZ = _env_str("SESSION_TZ", "UTC")  # "UTC" recommended for stability
# London killzone default (UTC)
LONDON_START = _env_str("LONDON_START", "07:00")
LONDON_END = _env_str("LONDON_END", "10:00")
# NY killzone default (UTC)
NY_START = _env_str("NY_START", "13:00")
NY_END = _env_str("NY_END", "16:00")


def _parse_hhmm(s: str) -> time:
    try:
        hh, mm = str(s).strip().split(":")
        return time(hour=int(hh), minute=int(mm))
    except Exception:
        return time(0, 0)


def _now_in_tz(tz_name: str) -> datetime:
    tz_name = (tz_name or "UTC").strip()
    try:
        tz = timezone.utc if tz_name.upper() == "UTC" else ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc
    return datetime.now(tz=tz)


def _in_time_window(now_t: time, start: time, end: time) -> bool:
    # Handles standard window same-day only (start < end)
    if start <= end:
        return (now_t >= start) and (now_t <= end)
    # If someone configures wrap-around (rare), handle it
    return (now_t >= start) or (now_t <= end)


def _session_ok() -> Tuple[bool, str, Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    if not SESSION_FILTER_ENABLE:
        return True, "OK", {"enabled": False}

    now_dt = _now_in_tz(SESSION_TZ)
    now_t = now_dt.time()

    l_start = _parse_hhmm(LONDON_START)
    l_end = _parse_hhmm(LONDON_END)
    ny_start = _parse_hhmm(NY_START)
    ny_end = _parse_hhmm(NY_END)

    in_london = _in_time_window(now_t, l_start, l_end)
    in_ny = _in_time_window(now_t, ny_start, ny_end)
    ok = bool(in_london or in_ny)

    meta.update(
        {
            "enabled": True,
            "tz": SESSION_TZ,
            "now": now_dt.isoformat(),
            "in_london": bool(in_london),
            "in_ny": bool(in_ny),
            "london": f"{LONDON_START}-{LONDON_END}",
            "ny": f"{NY_START}-{NY_END}",
        }
    )
    return ok, ("OK" if ok else "outside_killzones"), meta


# =====================================================================
# ✅ TrendGuard (anti-contre-tendance / anti-chop MARKET)
# =====================================================================
TG_STRICT_HTF_BIAS = _env_flag("TG_STRICT_HTF_BIAS", "1")
TG_STRICT_REGIME_MARKET = _env_flag("TG_STRICT_REGIME_MARKET", "1")
TG_ADX_PERIOD = _env_int("TG_ADX_PERIOD", "14")
TG_ADX_MIN = _env_float("TG_ADX_MIN", "20")
TG_BB_PERIOD = _env_int("TG_BB_PERIOD", "20")
TG_BB_K = _env_float("TG_BB_K", "2")
TG_BB_SQUEEZE_BW = _env_float("TG_BB_SQUEEZE_BW", "0.04")
TG_OVEREXT_ATR = _env_float("TG_OVEREXT_ATR", "1.2")


def _ema(series: pd.Series, span: int) -> pd.Series:
    return pd.Series(series).astype(float).ewm(span=int(span), adjust=False).mean()


def _ensure_ohlcv(df: pd.DataFrame) -> bool:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False
    for c in REQUIRED_COLS:
        if c not in df.columns:
            return False
    return True


def _last_close(df: pd.DataFrame) -> float:
    try:
        return float(df["close"].astype(float).iloc[-1])
    except Exception:
        return 0.0


def _atr(df: pd.DataFrame, n: int = 14) -> float:
    try:
        if not _ensure_ohlcv(df) or len(df) < n + 2:
            return 0.0
        s = true_atr(df, length=n)
        v = float(s.iloc[-1]) if s is not None and len(s) else 0.0
        return float(max(0.0, v)) if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _adx_wilder(df: pd.DataFrame, period: int = 14) -> float:
    try:
        if df is None or df.empty or len(df) < period * 3:
            return 0.0
        for c in ("high", "low", "close"):
            if c not in df.columns:
                return 0.0

        h = pd.to_numeric(df["high"], errors="coerce").astype(float)
        l = pd.to_numeric(df["low"], errors="coerce").astype(float)
        c = pd.to_numeric(df["close"], errors="coerce").astype(float)

        up = h.diff()
        dn = -l.diff()

        plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
        minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)

        prev_close = c.shift(1)
        tr = pd.concat([(h - l).abs(), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)

        alpha = 1.0 / float(max(1, int(period)))
        atr = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean()

        pdi = 100.0 * (pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan))
        mdi = 100.0 * (pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan))

        dx = 100.0 * (abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan))
        adx_s = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean()
        v = float(adx_s.iloc[-1])
        return v if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _bb_bandwidth(df: pd.DataFrame, period: int = 20, k: float = 2.0) -> float:
    try:
        if df is None or df.empty or len(df) < period + 5:
            return 0.0
        if "close" not in df.columns:
            return 0.0
        c = pd.to_numeric(df["close"], errors="coerce").astype(float)
        mid = c.rolling(int(period)).mean()
        sd = c.rolling(int(period)).std(ddof=0)
        upper = mid + float(k) * sd
        lower = mid - float(k) * sd
        denom = mid.replace(0, np.nan)
        bw = (upper - lower) / denom
        v = float(bw.iloc[-1])
        return v if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _is_squeeze(df: pd.DataFrame) -> bool:
    bw = _bb_bandwidth(df, period=TG_BB_PERIOD, k=TG_BB_K)
    return bool(bw > 0 and bw < TG_BB_SQUEEZE_BW)


def _is_market_entry(entry_type: str) -> bool:
    return "MARKET" in str(entry_type or "").upper()


def _htf_bias_from_df(df_h4: pd.DataFrame) -> str:
    try:
        st = analyze_structure(df_h4)
        t = str(st.get("trend") or "").upper()
        if t in ("LONG", "SHORT"):
            return t
    except Exception:
        pass

    try:
        if df_h4 is None or df_h4.empty or len(df_h4) < 60 or "close" not in df_h4.columns:
            return "RANGE"
        c = pd.to_numeric(df_h4["close"], errors="coerce").astype(float)
        e20 = _ema(c, 20)
        e50 = _ema(c, 50)
        if float(e20.iloc[-1]) > float(e50.iloc[-1]):
            return "LONG"
        if float(e20.iloc[-1]) < float(e50.iloc[-1]):
            return "SHORT"
        return "RANGE"
    except Exception:
        return "RANGE"


def _trend_guard(
    *,
    df_h1: pd.DataFrame,
    df_h4: pd.DataFrame,
    bias: str,
    entry: Optional[float],
    entry_type: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    b = str(bias or "").upper()
    et = str(entry_type or "MARKET")

    htf_bias = _htf_bias_from_df(df_h4)
    meta["htf_bias"] = htf_bias
    meta["bias"] = b
    if TG_STRICT_HTF_BIAS and htf_bias in ("LONG", "SHORT") and b in ("LONG", "SHORT") and htf_bias != b:
        return False, "reject:htf_bias_mismatch", meta

    adx_v = _adx_wilder(df_h4, period=TG_ADX_PERIOD)
    meta["adx_h4"] = float(adx_v)
    sq = _is_squeeze(df_h1)
    meta["squeeze_h1"] = bool(sq)

    if _is_market_entry(et) and TG_STRICT_REGIME_MARKET:
        if adx_v > 0 and adx_v < TG_ADX_MIN:
            return False, "reject:weak_trend_no_market", meta
        if sq:
            return False, "reject:squeeze_no_market", meta

        try:
            if entry is not None and "close" in df_h1.columns:
                a = _atr(df_h1, 14)
                meta["atr_h1"] = float(a)
                if a > 0:
                    c = pd.to_numeric(df_h1["close"], errors="coerce").astype(float)
                    ema20 = float(_ema(c, 20).iloc[-1])
                    meta["ema20_h1"] = float(ema20)
                    if abs(float(entry) - ema20) > float(TG_OVEREXT_ATR) * a:
                        return False, "reject:overextended_market", meta
        except Exception:
            pass

    return True, "OK", meta


# =====================================================================
# Macro unwrap helpers
# =====================================================================
def _unwrap_macro(macro: Any) -> Dict[str, Any]:
    if isinstance(macro, dict):
        return macro
    return {}


def _get_macro_ref_df(macro: Any) -> Optional[pd.DataFrame]:
    try:
        m = _unwrap_macro(macro)
        for k in ("btc_h1", "btc_df_h1", "btc_h1_df", "BTC_H1", "BTC", "btc"):
            v = m.get(k)
            if isinstance(v, pd.DataFrame) and (not v.empty):
                return v
        mm = m.get("macro")
        if isinstance(mm, dict):
            for k in ("btc_h1", "btc_df_h1", "btc_h1_df", "BTC_H1", "BTC", "btc"):
                v = mm.get(k)
                if isinstance(v, pd.DataFrame) and (not v.empty):
                    return v
        return None
    except Exception:
        return None


def _get_macro_implied_vol(macro: Any, symbol: str) -> Optional[float]:
    try:
        m = _unwrap_macro(macro)

        def _probe(d: Dict[str, Any]) -> Optional[float]:
            iv = d.get("implied_vol", None)
            if iv is None:
                iv = d.get("iv", None)
            if iv is None:
                return None

            if isinstance(iv, (int, float, np.floating)):
                v = float(iv)
                return v if np.isfinite(v) and v > 0 else None

            if isinstance(iv, dict):
                if symbol in iv:
                    v = float(iv[symbol])
                    return v if np.isfinite(v) and v > 0 else None
                sym2 = str(symbol).replace("-USDT", "").replace("USDTM", "").replace("USDT", "")
                if sym2 in iv:
                    v = float(iv[sym2])
                    return v if np.isfinite(v) and v > 0 else None
            return None

        v1 = _probe(m)
        if v1 is not None:
            return v1
        mm = m.get("macro")
        if isinstance(mm, dict):
            return _probe(mm)
        return None
    except Exception:
        return None


def _get_options_obj(macro: Any) -> Any:
    try:
        m = _unwrap_macro(macro)
        if "options" in m:
            return m.get("options")
        if "options_snapshot" in m:
            return m.get("options_snapshot")
        mm = m.get("macro")
        if isinstance(mm, dict):
            if "options" in mm:
                return mm.get("options")
            if "options_snapshot" in mm:
                return mm.get("options_snapshot")
        return None
    except Exception:
        return None


def _score_options_context_safe(opt_obj: Any, *, bias: str, setup_type: Optional[str] = None) -> Dict[str, Any]:
    if score_options_context is None:
        return {"ok": True, "score": 0, "regime": "unavailable", "reason": "no_options_module"}
    try:
        if setup_type is not None:
            return score_options_context(opt_obj, bias=bias, setup_type=setup_type)  # type: ignore
        return score_options_context(opt_obj, bias=bias)  # type: ignore
    except TypeError:
        try:
            if setup_type is not None:
                return score_options_context(opt_obj, bias, setup_type=setup_type)  # type: ignore
            return score_options_context(opt_obj, bias)  # type: ignore
        except Exception:
            return {"ok": True, "score": 0, "regime": "unknown", "reason": "options_error"}
    except Exception:
        return {"ok": True, "score": 0, "regime": "unknown", "reason": "options_error"}


# =====================================================================
# ✅ scanner.py compatibility helper (TTL policy uses setup string)
# =====================================================================
def _setup_ttl_compatible(setup_type: str, entry_type: str) -> str:
    base = str(setup_type or "").strip()
    if not base:
        base = "OTHER"
    s = base.upper()
    et = str(entry_type or "").upper()

    # Don't append _FVG on RAID/SWEEP setups (TTL should follow setup, not generic zone)
    if ("RAID" in s) or ("SWEEP" in s):
        return base

    if "RAID" in et and ("RAID" not in s and "SWEEP" not in s):
        return f"{base}_RAID"
    if "OTE" in et and "OTE" not in s:
        return f"{base}_OTE"
    if "FVG" in et and "FVG" not in s:
        return f"{base}_FVG"
    return base


# =====================================================================
# ✅ Liquidations + 2-pass institutional policy
# =====================================================================
INST_ENABLE_LIQUIDATIONS = _env_flag("INST_ENABLE_LIQUIDATIONS", "1")
INST_LIQ_PASS2_ONLY = _env_flag("INST_LIQ_PASS2_ONLY", "1")
INST_PASS1_MODE = _env_str("INST_PASS1_MODE", "LIGHT").upper()
INST_PASS2_MODE = _env_str("INST_PASS2_MODE", "NORMAL").upper()
INST_PASS2_ENABLED = _env_flag("INST_PASS2_ENABLED", "1")
INST_PASS2_MIN_GATE = _env_int("INST_PASS2_MIN_GATE", "1")

if INST_PASS1_MODE not in ("LIGHT", "NORMAL", "FULL"):
    INST_PASS1_MODE = "LIGHT"
if INST_PASS2_MODE not in ("LIGHT", "NORMAL", "FULL"):
    INST_PASS2_MODE = "NORMAL"

# =====================================================================
# ✅ INST_CONTINUATION ULTRA-STRICT policy (env-tunable)
# =====================================================================
INST_CONT_REQUIRE_TRIGGER = _env_flag("INST_CONT_REQUIRE_TRIGGER", "1")
INST_CONT_REQUIRE_PASS2 = _env_flag("INST_CONT_REQUIRE_PASS2", "1")
INST_CONT_DISALLOW_MARKET = _env_flag("INST_CONT_DISALLOW_MARKET", "1")
INST_CONT_REQUIRE_NO_SOFT_VETO = _env_flag("INST_CONT_REQUIRE_NO_SOFT_VETO", "1")
INST_CONT_REQUIRE_STRONG_MOM = _env_flag("INST_CONT_REQUIRE_STRONG_MOM", "1")
INST_CONT_REQUIRE_CLEAR_BIAS = _env_flag("INST_CONT_REQUIRE_CLEAR_BIAS", "1")
INST_CONT_MIN_COMPOSITE_THR = _env_float("INST_CONT_MIN_COMPOSITE_THR", "70")
INST_CONT_MIN_GATE = _env_int("INST_CONT_MIN_GATE", str(int(INST_SCORE_DESK_PRIORITY)))
INST_CONT_RR_MIN = _env_float("INST_CONT_RR_MIN", str(float(max(RR_MIN_STRICT, RR_MIN_DESK_PRIORITY))))

# =====================================================================
# ✅ INSTITUTIONAL MAX: Tradability / Liquidity filters (OHLCV only)
# =====================================================================
LIQ_FILTER_ENABLE = _env_flag("LIQ_FILTER_ENABLE", "1")
LIQ_FILTER_STRICT = _env_flag("LIQ_FILTER_STRICT", "1")  # if 0 and desk -> soft veto only

MIN_DOLLAR_VOL_20 = _env_float("MIN_DOLLAR_VOL_20", "250000")  # avg(volume*close) over last 20
MAX_SPREAD_PROXY_20 = _env_float("MAX_SPREAD_PROXY_20", "0.020")  # median((high-low)/close) over last 20
MAX_WICKINESS_20 = _env_float("MAX_WICKINESS_20", "0.70")  # avg wick ratio over last 20
MAX_RANGE_ATR_MULT = _env_float("MAX_RANGE_ATR_MULT", "3.5")  # last candle range / ATR(14) too big => noisy/manip
MIN_BARS_FOR_LIQ = _env_int("MIN_BARS_FOR_LIQ", "120")


def _tradability_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        if not _ensure_ohlcv(df) or len(df) < MIN_BARS_FOR_LIQ:
            return {"ok": True, "reason": "insufficient_bars", "enabled": LIQ_FILTER_ENABLE}

        w = df.tail(20).copy()
        o = pd.to_numeric(w["open"], errors="coerce").astype(float)
        h = pd.to_numeric(w["high"], errors="coerce").astype(float)
        l = pd.to_numeric(w["low"], errors="coerce").astype(float)
        c = pd.to_numeric(w["close"], errors="coerce").astype(float)
        v = pd.to_numeric(w["volume"], errors="coerce").astype(float)

        dollar_vol = float(np.nanmean(v * c))
        spread_proxy = float(np.nanmedian((h - l).abs() / c.replace(0, np.nan)))

        rng = (h - l).abs()
        upper_wick = (h - np.maximum(o, c)).clip(lower=0)
        lower_wick = (np.minimum(o, c) - l).clip(lower=0)
        wick = (upper_wick + lower_wick)
        wickiness = float(np.nanmean((wick / rng.replace(0, np.nan)).clip(0, 1)))

        a = _atr(df, 14)
        last_range = float(abs(float(df["high"].astype(float).iloc[-1]) - float(df["low"].astype(float).iloc[-1])))
        range_atr = float(last_range / a) if a > 0 else 0.0

        out.update(
            {
                "enabled": LIQ_FILTER_ENABLE,
                "dollar_vol_20": dollar_vol,
                "spread_proxy_20": spread_proxy,
                "wickiness_20": wickiness,
                "atr14": float(a),
                "last_range": float(last_range),
                "range_atr_mult": float(range_atr),
            }
        )

        if not LIQ_FILTER_ENABLE:
            out["ok"] = True
            out["reason"] = "disabled"
            return out

        if np.isfinite(dollar_vol) and dollar_vol < MIN_DOLLAR_VOL_20:
            out["ok"] = False
            out["reason"] = "low_dollar_volume"
            return out

        if np.isfinite(spread_proxy) and spread_proxy > MAX_SPREAD_PROXY_20:
            out["ok"] = False
            out["reason"] = "wide_spread_proxy"
            return out

        if np.isfinite(wickiness) and wickiness > MAX_WICKINESS_20:
            out["ok"] = False
            out["reason"] = "wicky_noisy_market"
            return out

        if a > 0 and np.isfinite(range_atr) and range_atr > MAX_RANGE_ATR_MULT:
            out["ok"] = False
            out["reason"] = "range_spike_vs_atr"
            return out

        out["ok"] = True
        out["reason"] = "OK"
        return out
    except Exception:
        return {"ok": True, "reason": "metrics_error", "enabled": LIQ_FILTER_ENABLE}


# =====================================================================
# ✅ Institutional Micro-Filters (robust key extraction)
# =====================================================================
INST_MICRO_FILTER_ENABLE = _env_flag("INST_MICRO_FILTER_ENABLE", "1")
INST_MICRO_FILTER_STRICT = _env_flag("INST_MICRO_FILTER_STRICT", "1")  # if 0 and desk -> soft only

# Funding crowding extremes
FUNDING_EXTREME_ABS = _env_float("FUNDING_EXTREME_ABS", "0.0015")  # absolute funding rate threshold (per interval)
# OI delta thresholds
OI_DELTA_MIN_ABS = _env_float("OI_DELTA_MIN_ABS", "0.03")  # interpret as pct if institutional_data provides pct-like
# Orderbook imbalance
OB_IMB_MIN_ABS = _env_float("OB_IMB_MIN_ABS", "0.15")  # imbalance in [-1..1], require sign align if above abs threshold
# Alignment requirements
INST_ALIGN_MIN_CONT = _env_int("INST_ALIGN_MIN_CONT", "2")  # BOS / continuation
INST_ALIGN_MIN_REV = _env_int("INST_ALIGN_MIN_REV", "1")  # RAID / SWEEP


def _pick_first(inst: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in inst and inst.get(k) is not None:
            return inst.get(k)
    return None


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _sign_align(bias: str, v: float) -> bool:
    b = (bias or "").upper()
    if b == "LONG":
        return v > 0
    if b == "SHORT":
        return v < 0
    return False


def _inst_micro_filters(inst: Dict[str, Any], bias: str, setup_intent: str) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Returns:
      hard_ok (bool),
      vetoes (list[str])  -> can be treated as soft vetoes in desk
      meta (dict)
    """
    meta: Dict[str, Any] = {"enabled": INST_MICRO_FILTER_ENABLE, "setup_intent": setup_intent}
    vetoes: List[str] = []
    if not INST_MICRO_FILTER_ENABLE:
        return True, vetoes, meta

    b = (bias or "").upper()

    funding = _as_float(_pick_first(inst, ["funding_rate", "fundingRate", "funding", "funding_rate_last"]))
    oi_delta = _as_float(_pick_first(inst, ["oi_delta", "open_interest_delta", "oi_change", "oi_change_pct", "oi_pct_change"]))
    ob_imb = _as_float(_pick_first(inst, ["orderbook_imbalance", "orderbook_imb", "ob_imbalance", "ob_imb"]))
    cvd_slope = _as_float(_pick_first(inst, ["cvd_slope", "cvdSlope", "cvd_trend"]))
    flow = str(_pick_first(inst, ["flow_regime", "flowRegime", "flow"]) or "").lower()
    crowd = str(_pick_first(inst, ["crowding_regime", "crowdingRegime", "crowding"]) or "").lower()

    meta.update(
        {
            "funding": funding,
            "oi_delta": oi_delta,
            "orderbook_imb": ob_imb,
            "cvd_slope": cvd_slope,
            "flow_regime": flow,
            "crowding_regime": crowd,
        }
    )

    # Crowding veto (hard-ish)
    if any(x in crowd for x in ["risky", "crowded", "overcrowded"]):
        vetoes.append("crowding_risky")

    # Funding extreme: if funding is extreme and aligns with bias -> often crowded
    if funding is not None and abs(funding) >= FUNDING_EXTREME_ABS:
        # If extreme funding same direction as bias -> crowded risk
        if _sign_align(b, funding):
            vetoes.append("funding_extreme_crowded")
        else:
            # opposite funding can be "contrarian tailwind" -> no veto
            pass

    # Directional alignment counts
    align_count = 0
    align_tags: List[str] = []

    # CVD alignment (continuation wants align)
    if cvd_slope is not None and abs(cvd_slope) > 0:
        if _sign_align(b, cvd_slope):
            align_count += 1
            align_tags.append("cvd_align")
        else:
            vetoes.append("cvd_misalign")

    # OI alignment (only if meaningful magnitude)
    if oi_delta is not None and abs(oi_delta) >= OI_DELTA_MIN_ABS:
        if _sign_align(b, oi_delta):
            align_count += 1
            align_tags.append("oi_align")
        else:
            vetoes.append("oi_misalign")

    # Orderbook imbalance alignment (only if meaningful)
    if ob_imb is not None and abs(ob_imb) >= OB_IMB_MIN_ABS:
        if _sign_align(b, ob_imb):
            align_count += 1
            align_tags.append("ob_align")
        else:
            vetoes.append("orderbook_misalign")

    # Flow regime alignment
    if flow:
        if b == "LONG" and "buy" in flow:
            align_count += 1
            align_tags.append("flow_buy")
        if b == "SHORT" and "sell" in flow:
            align_count += 1
            align_tags.append("flow_sell")
        # strong opposite flow -> veto
        if b == "LONG" and "sell" in flow and "strong" in flow:
            vetoes.append("flow_strong_sell_vs_long")
        if b == "SHORT" and "buy" in flow and "strong" in flow:
            vetoes.append("flow_strong_buy_vs_short")

    meta["align_count"] = int(align_count)
    meta["align_tags"] = align_tags

    # Apply intent-specific minimum alignment
    intent = (setup_intent or "OTHER").upper()
    min_align = INST_ALIGN_MIN_CONT if intent == "CONTINUATION" else INST_ALIGN_MIN_REV
    meta["min_align_required"] = int(min_align)

    if align_count < min_align:
        vetoes.append(f"inst_align_low({align_count}<{min_align})")

    # Hard decision:
    # - continuation should be very strict
    # - reversal can accept some misalign but not "crowding_risky" + "funding_extreme_crowded" together
    hard_ok = True
    if intent == "CONTINUATION":
        if any(v in vetoes for v in ["crowding_risky", "funding_extreme_crowded", "inst_align_low(0<", "flow_strong_sell_vs_long", "flow_strong_buy_vs_short"]):
            hard_ok = False
        if any(v.startswith("inst_align_low") for v in vetoes):
            hard_ok = False
    else:
        # reversal/raid/sweep: crowding risky still matters
        if "crowding_risky" in vetoes and "funding_extreme_crowded" in vetoes:
            hard_ok = False

    return bool(hard_ok), vetoes, meta


# =====================================================================
# Premium/Discount helper
# =====================================================================
def compute_premium_discount(df: pd.DataFrame, lookback: int = 80) -> Tuple[bool, bool]:
    if not _ensure_ohlcv(df) or len(df) < lookback:
        return False, False

    w = df.tail(lookback)
    hi = float(w["high"].astype(float).max())
    lo = float(w["low"].astype(float).min())
    last = float(w["close"].astype(float).iloc[-1])

    if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
        return False, False

    mid = (hi + lo) / 2.0
    premium = bool(last > mid)
    discount = bool(last < mid)
    return premium, discount


# =====================================================================
# Tick helpers
# =====================================================================
def _safe_tick(tick: float) -> float:
    try:
        t = float(tick)
        if not np.isfinite(t) or t <= 0:
            return 0.0
        return t
    except Exception:
        return 0.0


def _floor_to_tick(price: float, tick: float) -> float:
    t = _safe_tick(tick)
    if t <= 0:
        return float(price)
    p = float(price)
    return float(np.floor(p / t) * t)


def _ceil_to_tick(price: float, tick: float) -> float:
    t = _safe_tick(tick)
    if t <= 0:
        return float(price)
    p = float(price)
    return float(np.ceil(p / t) * t)


def _round_sl_for_side(sl: float, entry: float, bias: str, tick: float) -> float:
    b = (bias or "").upper()
    t = _safe_tick(tick)
    sl_f = float(sl)
    entry_f = float(entry)

    if b == "LONG":
        sl_f = _floor_to_tick(sl_f, t)
        if sl_f >= entry_f:
            gap = max((2.0 * t if t > 0 else 0.0), abs(entry_f) * 0.001)
            sl_f = _floor_to_tick(entry_f - gap, t)
        return float(sl_f)

    if b == "SHORT":
        sl_f = _ceil_to_tick(sl_f, t)
        if sl_f <= entry_f:
            gap = max((2.0 * t if t > 0 else 0.0), abs(entry_f) * 0.001)
            sl_f = _ceil_to_tick(entry_f + gap, t)
        return float(sl_f)

    return float(sl_f)


# =====================================================================
# Composite helpers
# =====================================================================
def _composite_bias_ok(bias: str, score: float, label: str, thr: float = 65.0) -> bool:
    try:
        b = (bias or "").upper()
        s = float(score)
        lab = str(label or "").upper()

        if b == "LONG":
            return (s >= thr) or ("BULL" in lab and s >= (thr - 5.0))
        if b == "SHORT":
            return (s <= (100.0 - thr)) or ("BEAR" in lab and s <= (100.0 - thr + 5.0))
        return False
    except Exception:
        return False


def _composite_bias_fallback(score: float, label: str, mom: str) -> Optional[str]:
    try:
        s = float(score)
        lab = str(label or "").upper()
        m = str(mom or "").upper()

        if m in ("STRONG_BULLISH", "BULLISH"):
            return "LONG"
        if m in ("STRONG_BEARISH", "BEARISH"):
            return "SHORT"

        if (s >= 58.0) or ("BULL" in lab and s >= 55.0):
            return "LONG"
        if (s <= 42.0) or ("BEAR" in lab and s <= 45.0):
            return "SHORT"
        return None
    except Exception:
        return None


def _safe_rr(entry: float, sl: float, tp1: float, bias: str) -> Optional[float]:
    try:
        entry = float(entry)
        sl = float(sl)
        tp1 = float(tp1)
        if not (np.isfinite(entry) and np.isfinite(sl) and np.isfinite(tp1)):
            return None

        b = (bias or "").upper()
        if b == "LONG":
            risk = entry - sl
            reward = tp1 - entry
        elif b == "SHORT":
            risk = sl - entry
            reward = entry - tp1
        else:
            return None

        if risk <= 0:
            return None
        return float(reward / risk)
    except Exception:
        return None


def estimate_tick_from_price(price: float) -> float:
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


def _compute_exits(
    df: pd.DataFrame,
    entry: float,
    bias: str,
    tick: float,
    *,
    setup: Optional[str] = None,
    entry_type: Optional[str] = None,
    htf_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    if bias == "LONG":
        sl, meta = protective_stop_long(
            df,
            entry,
            tick,
            return_meta=True,
            setup=setup,
            entry_type=entry_type,
            htf_df=htf_df,
        )
    else:
        sl, meta = protective_stop_short(
            df,
            entry,
            tick,
            return_meta=True,
            setup=setup,
            entry_type=entry_type,
            htf_df=htf_df,
        )

    tp1, rr_used = compute_tp1(entry, sl, bias, df=df, tick=tick)
    return {"sl": float(sl), "tp1": float(tp1), "rr_used": float(rr_used), "sl_meta": meta}


def _entry_pullback_ok(entry: float, entry_mkt: float, bias: str, atr: float) -> bool:
    try:
        if atr <= 0:
            return True
        entry = float(entry)
        entry_mkt = float(entry_mkt)
        b = (bias or "").upper()

        if b == "LONG":
            return (entry_mkt - entry) >= 0.25 * atr
        if b == "SHORT":
            return (entry - entry_mkt) >= 0.25 * atr
        return True
    except Exception:
        return True


# =====================================================================
# Zone parsing / FVG entry
# =====================================================================
def _parse_zone_bounds(z: Dict[str, Any]) -> Optional[Tuple[float, float, str]]:
    try:
        a = z.get("low")
        b = z.get("high")
        if a is None or b is None:
            a = z.get("bottom")
            b = z.get("top")
        if a is None or b is None:
            a = z.get("start")
            b = z.get("end")
        if a is None or b is None:
            return None

        zl = float(min(a, b))
        zh = float(max(a, b))
        if not (np.isfinite(zl) and np.isfinite(zh)) or zh <= zl:
            return None

        zdir = (z.get("direction") or z.get("dir") or z.get("type") or "").lower()
        return zl, zh, zdir
    except Exception:
        return None


def _pick_fvg_entry(
    struct: Dict[str, Any],
    entry_mkt: float,
    bias: str,
    atr: float,
    max_dist_atr: float = 4.0,
) -> Tuple[Optional[float], str]:
    zones = struct.get("fvg_zones") or []
    if not zones:
        return None, "no_fvg"

    best_mid = None
    best_dist = 1e18
    b = (bias or "").upper()

    for z in zones:
        parsed = _parse_zone_bounds(z)
        if not parsed:
            continue
        zl, zh, zdir = parsed

        if b == "LONG" and zdir and "bear" in zdir:
            continue
        if b == "SHORT" and zdir and "bull" in zdir:
            continue

        mid = (zl + zh) / 2.0
        dist = abs(float(entry_mkt) - mid)
        if zl <= float(entry_mkt) <= zh:
            dist = 0.0

        if dist < best_dist:
            best_dist = dist
            best_mid = mid

    if best_mid is None:
        return None, "no_fvg_parse"

    try:
        if atr > 0 and best_dist > 0:
            if b == "LONG" and float(best_mid) > float(entry_mkt) + 0.1 * atr:
                return None, "fvg_not_better_side_long"
            if b == "SHORT" and float(best_mid) < float(entry_mkt) - 0.1 * atr:
                return None, "fvg_not_better_side_short"
    except Exception:
        pass

    if atr > 0 and best_dist > float(max_dist_atr) * atr:
        return None, f"fvg_too_far dist={best_dist:.6g} atr={atr:.6g} max_atr={max_dist_atr}"

    return float(best_mid), f"fvg_ok dist={best_dist:.6g} atr={atr:.6g} max_atr={max_dist_atr}"


def _pick_entry(df_h1: pd.DataFrame, struct: Dict[str, Any], bias: str) -> Dict[str, Any]:
    entry_mkt = _last_close(df_h1)
    atr = _atr(df_h1, 14)

    raid = struct.get("raid_displacement") or {}
    try:
        if isinstance(raid, dict) and raid.get("ok") and raid.get("entry") is not None:
            raid_entry = float(raid["entry"])
            dist = abs(entry_mkt - raid_entry)
            if atr <= 0 or dist <= 2.8 * atr:
                return {
                    "entry_used": raid_entry,
                    "entry_type": "RAID_FVG",
                    "order_type": "LIMIT",
                    "in_zone": True,
                    "note": f"raid_ok note={raid.get('note')} dist={dist:.6g} atr={atr:.6g}",
                    "entry_mkt": entry_mkt,
                    "atr": atr,
                    "raid": raid,
                }
    except Exception:
        pass

    # OTE
    ote_entry = None
    ote_in_zone = False
    ote_note = "ote_unavailable"
    try:
        ote = compute_ote(df_h1, bias)

        if isinstance(ote, dict) and ("in_ote" in ote or "ote_low" in ote or "ote_high" in ote):
            in_ote = bool(ote.get("in_ote", False))
            lo = ote.get("ote_low")
            hi = ote.get("ote_high")
            if lo is not None and hi is not None:
                lo = float(lo)
                hi = float(hi)
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    ote_entry = (lo + hi) / 2.0
                    ote_in_zone = in_ote
                    dist = abs(entry_mkt - ote_entry)
                    ote_note = f"ote(in_ote={ote_in_zone}) low={lo} high={hi} dist={dist:.6g} atr={atr:.6g}"
                else:
                    ote_note = f"ote_bad_bounds low={lo} high={hi} atr={atr:.6g}"
            else:
                ote_note = f"ote_missing_bounds atr={atr:.6g}"

        elif isinstance(ote, dict):
            ote_entry = ote.get("entry") or ote.get("entry_price") or ote.get("price")
            ote_in_zone = bool(ote.get("in_zone") or ote.get("ok") or False)
            dist = ote.get("dist") or ote.get("distance")
            ote_note = f"ote(oldfmt in_zone={ote_in_zone}) dist={dist} atr={atr:.6g}"

        elif isinstance(ote, (tuple, list)) and len(ote) >= 2:
            ote_entry = float(ote[0])
            ote_in_zone = bool(ote[1])
            ote_note = f"ote tuple in_zone={ote_in_zone} atr={atr:.6g}"

    except Exception as e:
        ote_note = f"ote_error {e}"

    if ote_entry is not None and ote_in_zone:
        return {
            "entry_used": float(ote_entry),
            "entry_type": "OTE",
            "order_type": "LIMIT",
            "in_zone": True,
            "note": ote_note,
            "entry_mkt": entry_mkt,
            "atr": atr,
        }

    # OTE pullback (même si pas in-zone)
    if ote_entry is not None and (not ote_in_zone):
        try:
            dist = abs(float(entry_mkt) - float(ote_entry))
            if bias == "LONG" and float(ote_entry) <= float(entry_mkt) and (atr <= 0 or dist <= 4.0 * atr):
                return {
                    "entry_used": float(ote_entry),
                    "entry_type": "OTE_PULLBACK",
                    "order_type": "LIMIT",
                    "in_zone": False,
                    "note": f"{ote_note} (pullback dist={dist:.6g} atr={atr:.6g})",
                    "entry_mkt": entry_mkt,
                    "atr": atr,
                }
            if bias == "SHORT" and float(ote_entry) >= float(entry_mkt) and (atr <= 0 or dist <= 4.0 * atr):
                return {
                    "entry_used": float(ote_entry),
                    "entry_type": "OTE_PULLBACK",
                    "order_type": "LIMIT",
                    "in_zone": False,
                    "note": f"{ote_note} (pullback dist={dist:.6g} atr={atr:.6g})",
                    "entry_mkt": entry_mkt,
                    "atr": atr,
                }
        except Exception:
            pass

    # FVG
    fvg_entry, fvg_note = _pick_fvg_entry(struct, entry_mkt, bias, atr, max_dist_atr=4.0)
    if fvg_entry is not None:
        return {
            "entry_used": float(fvg_entry),
            "entry_type": "FVG",
            "order_type": "LIMIT",
            "in_zone": False,
            "note": fvg_note,
            "entry_mkt": entry_mkt,
            "atr": atr,
        }

    # fallback
    return {
        "entry_used": float(entry_mkt),
        "entry_type": "MARKET",
        "order_type": "LIMIT",
        "in_zone": False,
        "note": "no_zone_entry",
        "entry_mkt": entry_mkt,
        "atr": atr,
    }


# =====================================================================
# Institutional gate helpers (score integration)
# =====================================================================
def _inst_ok_count(inst: Dict[str, Any]) -> int:
    comp = inst.get("score_components") or {}
    try:
        return int(sum(1 for v in comp.values() if int(v) > 0))
    except Exception:
        return 0


def _inst_gate_value(inst: Dict[str, Any]) -> Tuple[int, int]:
    try:
        inst_score = int(inst.get("institutional_score") or 0)
        meta = inst.get("score_meta") or {}
        raw_sum = meta.get("raw_components_sum")
        gate = int(raw_sum) if raw_sum is not None else inst_score
        ok_count = int(meta.get("ok_count") or _inst_ok_count(inst))
        return gate, ok_count
    except Exception:
        return 0, _inst_ok_count(inst)


def _inst_score_normalized_0_100(inst: Dict[str, Any], gate: int) -> int:
    try:
        meta = inst.get("score_meta") or {}
        comps = inst.get("score_components") or {}

        max_sum = meta.get("max_components_sum")
        if max_sum is None:
            max_sum = len(comps) if isinstance(comps, dict) and len(comps) > 0 else 1

        max_sum_i = int(max(1, int(max_sum)))
        g = int(gate)
        s = int(round(100.0 * float(g) / float(max_sum_i)))
        return int(max(0, min(100, s)))
    except Exception:
        return 0


def _desk_inst_gate_override(inst: Dict[str, Any], bias: str, gate: int) -> Tuple[int, Optional[str]]:
    try:
        if gate >= int(MIN_INST_SCORE):
            return gate, None

        b = (bias or "").upper()
        flow = str(inst.get("flow_regime") or "").lower()
        crowd = str(inst.get("crowding_regime") or "").lower()
        cvd = _as_float(inst.get("cvd_slope"))
        cvd = cvd if cvd is not None else 0.0

        if "risky" in crowd or "crowded" in crowd:
            return gate, None

        if b == "LONG" and flow == "strong_buy" and cvd >= 0.2:
            return int(MIN_INST_SCORE), "override_flow_strong_buy"
        if b == "SHORT" and flow == "strong_sell" and cvd <= -0.2:
            return int(MIN_INST_SCORE), "override_flow_strong_sell"
        return gate, None
    except Exception:
        return gate, None


def _reject(reason: str, **extra: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"valid": False, "reject_reason": reason}
    out.update(extra)
    return out


# =====================================================================
# Priority helpers (A→E)
# =====================================================================
_PRIORITY_ORDER = ("A", "B", "C", "D", "E")
_PRIORITY_IDX = {p: i for i, p in enumerate(_PRIORITY_ORDER)}


def _prio_norm(p: str, default: str = "C") -> str:
    s = (p or "").strip().upper()
    return s if s in _PRIORITY_IDX else default


def _downgrade(p: str, steps: int = 1) -> str:
    p = _prio_norm(p)
    i = _PRIORITY_IDX[p]
    i2 = min(len(_PRIORITY_ORDER) - 1, i + max(0, int(steps)))
    return _PRIORITY_ORDER[i2]


def _upgrade(p: str, steps: int = 1) -> str:
    p = _prio_norm(p)
    i = _PRIORITY_IDX[p]
    i2 = max(0, i - max(0, int(steps)))
    return _PRIORITY_ORDER[i2]


def _pre_grade_candidate(
    *,
    bos_flag: bool,
    raid_ok: bool,
    sweep_ok: bool,
    mom: str,
    comp_score: float,
    used_bias_fallback: bool,
    ext_sig: str,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    m = (mom or "").upper()
    ext = (ext_sig or "").upper()

    if bos_flag:
        p = "B"
        reasons.append("pre:bos_trigger")
    elif raid_ok or sweep_ok:
        p = "C"
        reasons.append("pre:raid_or_sweep")
    else:
        p = "D"
        reasons.append("pre:no_trigger")

    if m in ("STRONG_BULLISH", "STRONG_BEARISH"):
        p = _upgrade(p, 1)
        reasons.append("pre:strong_momentum")
    if float(comp_score) >= 67.0:
        p = _upgrade(p, 1)
        reasons.append("pre:high_composite")
    elif float(comp_score) <= 45.0:
        p = _downgrade(p, 1)
        reasons.append("pre:weak_composite")

    if used_bias_fallback:
        p = _downgrade(p, 1)
        reasons.append("pre:bias_fallback")

    if ext.startswith("OVEREXTENDED"):
        p = _downgrade(p, 1)
        reasons.append("pre:overextended")

    return _prio_norm(p), reasons


def _final_grade(
    *,
    setup_type: str,
    setup_variant: str,
    rr: float,
    gate: int,
    ok_count: int,
    inst_available: bool,
    bos_quality_ok: bool,
    used_bias_fallback: bool,
    entry_type: str,
    ext_sig: str,
    unfavorable_market: bool,
    mom: str,
    soft_vetoes: Optional[List[str]] = None,
    inst_micro_vetoes: Optional[List[str]] = None,
    confidence: Optional[int] = None,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    st = (setup_type or "").upper()
    sv = (setup_variant or "").upper()
    et = (entry_type or "").upper()
    ext = (ext_sig or "").upper()
    m = (mom or "").upper()
    svs = soft_vetoes or []
    imv = inst_micro_vetoes or []

    if st == "BOS_STRICT":
        if "RR_STRICT" in sv and int(gate) >= int(MIN_INST_SCORE) and bos_quality_ok:
            p = "A"
            reasons.append("setup:BOS_STRICT+RR_STRICT")
        else:
            p = "B"
            reasons.append("setup:BOS_STRICT")
    elif st in ("RAID_DISPLACEMENT", "LIQ_SWEEP"):
        if int(gate) >= int(INST_SCORE_DESK_PRIORITY):
            p = "B"
            reasons.append(f"setup:{st}+inst_ok")
        else:
            p = "C"
            reasons.append(f"setup:{st}")
    elif st == "INST_CONTINUATION":
        p = "C"
        reasons.append("setup:INST_CONTINUATION")
    else:
        p = "D"
        reasons.append("setup:OTHER")

    reasons.append(f"rr:{float(rr):.3f}")
    reasons.append(f"inst_gate:{int(gate)} ok_count:{int(ok_count)}")
    if confidence is not None:
        reasons.append(f"confidence:{int(confidence)}")

    if bos_quality_ok:
        reasons.append("bos_quality:ok")

    if not inst_available:
        p = _downgrade(p, 1)
        reasons.append("downgrade:inst_unavailable")

    if used_bias_fallback:
        p = _downgrade(p, 1)
        reasons.append("downgrade:bias_fallback")

    if et == "MARKET":
        p = _downgrade(p, 1)
        reasons.append("downgrade:market_entry")

    if unfavorable_market:
        p = _downgrade(p, 1)
        reasons.append("downgrade:premium_discount_veto_path")

    if ext.startswith("OVEREXTENDED"):
        p = _downgrade(p, 1)
        reasons.append("downgrade:overextended")

    if imv:
        p = _downgrade(p, 1)
        reasons.append(f"downgrade:inst_micro({','.join(imv)})")

    if svs:
        p = _downgrade(p, 1)
        reasons.append(f"downgrade:soft_vetoes({','.join(svs)})")

    if m in ("STRONG_BULLISH", "STRONG_BEARISH"):
        p = _upgrade(p, 1)
        reasons.append("upgrade:strong_momentum")

    return _prio_norm(p), reasons


# =====================================================================
# Confidence score (0..100) — "institutional cleanliness"
# =====================================================================
SOFT_VETO_PENALTY = _env_int("SOFT_VETO_PENALTY", "6")
MICRO_VETO_PENALTY = _env_int("MICRO_VETO_PENALTY", "8")
OPTIONS_PENALTY = _env_int("OPTIONS_PENALTY", "6")
SESSION_PENALTY = _env_int("SESSION_PENALTY", "8")
LIQ_PENALTY = _env_int("LIQ_PENALTY", "10")
IV_HIGH_THRESHOLD = _env_float("IV_HIGH_THRESHOLD", "1.0")  # if macro iv > this => risk flag


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _confidence_score(
    *,
    inst_score_norm: int,
    bos_q: Dict[str, Any],
    comp_score: float,
    bias: str,
    used_bias_fallback: bool,
    soft_vetoes: List[str],
    micro_vetoes: List[str],
    options_filter: Optional[Dict[str, Any]],
    session_ok: bool,
    tradability_ok: bool,
    iv_hint: Optional[float],
) -> int:
    # base pillars
    inst_p = _clamp01(float(inst_score_norm) / 100.0)
    bos_s = _as_float(bos_q.get("score")) or 50.0
    bos_p = _clamp01(float(bos_s) / 100.0)

    # composite alignment to bias
    try:
        if (bias or "").upper() == "LONG":
            comp_al = _clamp01(float(comp_score) / 100.0)
        else:
            comp_al = _clamp01((100.0 - float(comp_score)) / 100.0)
    except Exception:
        comp_al = 0.5

    base = 100.0 * (0.50 * inst_p + 0.25 * bos_p + 0.25 * comp_al)

    # penalties
    pen = 0.0
    if used_bias_fallback:
        pen += 10.0
    pen += float(len(soft_vetoes)) * float(SOFT_VETO_PENALTY)
    pen += float(len(micro_vetoes)) * float(MICRO_VETO_PENALTY)

    if options_filter is not None:
        reg = str(options_filter.get("regime") or "").lower()
        if reg in ("high_vol", "crisis", "event"):
            pen += float(OPTIONS_PENALTY)

    if not session_ok and SESSION_FILTER_ENABLE:
        pen += float(SESSION_PENALTY)

    if not tradability_ok and LIQ_FILTER_ENABLE:
        pen += float(LIQ_PENALTY)

    if iv_hint is not None and np.isfinite(iv_hint) and float(iv_hint) >= IV_HIGH_THRESHOLD:
        pen += 6.0

    score = int(round(max(0.0, min(100.0, base - pen))))
    return score


# =====================================================================
# Analyzer
# =====================================================================
class SignalAnalyzer:
    def __init__(self, *args, **kwargs):
        pass

    async def analyze(
        self,
        symbol: str,
        df_h1: pd.DataFrame,
        df_h4: pd.DataFrame,
        macro: Any = None,
    ) -> Dict[str, Any]:

        LOGGER.info("[EVAL] ▶ START %s", symbol)

        if not _ensure_ohlcv(df_h1) or len(df_h1) < 80:
            LOGGER.info("[EVAL_REJECT] %s bad_df_h1", symbol)
            return _reject("bad_df_h1")

        if not _ensure_ohlcv(df_h4) or len(df_h4) < 80:
            LOGGER.info("[EVAL_REJECT] %s bad_df_h4", symbol)
            return _reject("bad_df_h4")

        # =================================================================
        # Session filter (institutional killzones)
        # =================================================================
        sess_ok, sess_why, sess_meta = _session_ok()
        if not sess_ok:
            if (not DESK_EV_MODE) and SESSION_FILTER_STRICT:
                LOGGER.info("[EVAL_REJECT] %s outside_killzones meta=%s", symbol, sess_meta)
                return _reject("outside_killzones", session=sess_meta)
        # desk soft veto
        soft_vetoes: List[str] = []
        if (not sess_ok) and DESK_EV_MODE:
            soft_vetoes.append("outside_killzones")

        # =================================================================
        # Tradability filter (liquidity/market quality)
        # =================================================================
        trad = _tradability_metrics(df_h1)
        trad_ok = bool(trad.get("ok", True))
        if LIQ_FILTER_ENABLE and (not trad_ok):
            if (not DESK_EV_MODE) and LIQ_FILTER_STRICT:
                LOGGER.info("[EVAL_REJECT] %s tradability_veto reason=%s metrics=%s", symbol, trad.get("reason"), trad)
                return _reject("tradability_veto", tradability=trad)
            if DESK_EV_MODE:
                soft_vetoes.append(f"tradability:{trad.get('reason')}")

        # ---- cheap direction hints first ----
        mom = institutional_momentum(df_h1)
        comp = composite_momentum(df_h1)
        vol_reg = volatility_regime(df_h1)
        ext_sig = extension_signal(df_h1)

        comp_score = float(comp.get("score", 50.0)) if isinstance(comp, dict) else 50.0
        comp_label = str(comp.get("label", "NEUTRAL")) if isinstance(comp, dict) else "NEUTRAL"

        # ---- structure ----
        struct = analyze_structure(df_h1)
        struct_trend = str(struct.get("trend", "")).upper()
        bias = str(struct_trend).upper()

        # Fallback bias if range
        used_bias_fallback = False
        if bias not in ("LONG", "SHORT"):
            fb = _composite_bias_fallback(comp_score, comp_label, mom)
            if fb is not None:
                bias = fb
                used_bias_fallback = True
                LOGGER.info("[EVAL_PRE] %s bias_fallback_for_inst=%s", symbol, bias)
            else:
                LOGGER.info("[EVAL_REJECT] %s no_bias_fallback_for_inst", symbol)
                return _reject("no_bias_fallback_for_inst", structure=struct, momentum=mom, composite=comp)

        LOGGER.info(
            "[EVAL_PRE] %s STRUCT trend=%s bos=%s choch=%s cos=%s",
            symbol, struct_trend, struct.get("bos"), struct.get("choch"), struct.get("cos")
        )
        LOGGER.info("[EVAL_PRE] %s MOMENTUM=%s", symbol, mom)
        LOGGER.info("[EVAL_PRE] %s MOMENTUM_COMPOSITE score=%.2f label=%s", symbol, comp_score, comp_label)
        LOGGER.info("[EVAL_PRE] %s VOL_REGIME=%s EXTENSION=%s", symbol, vol_reg, ext_sig)

        # ---------------------------------------------------------------
        # SMT divergence (optional)
        # ---------------------------------------------------------------
        smt: Dict[str, Any] = {"available": False}
        smt_veto = False
        ref_df = _get_macro_ref_df(macro)

        if compute_smt_divergence is not None and ref_df is not None:
            try:
                smt = compute_smt_divergence(df_h1, ref_df, lookback=160)
            except Exception:
                smt = {"available": False}

            if bool(smt.get("available")):
                if bias == "LONG" and bool(smt.get("bearish_smt")):
                    smt_veto = True
                if bias == "SHORT" and bool(smt.get("bullish_smt")):
                    smt_veto = True

        if smt_veto and (not DESK_EV_MODE):
            LOGGER.info("[EVAL_REJECT] %s smt_divergence_veto bias=%s smt=%s", symbol, bias, smt)
            return _reject("smt_divergence_veto", smt=smt, smt_veto=True, structure=struct, momentum=mom, composite=comp)

        if smt_veto and DESK_EV_MODE:
            soft_vetoes.append("smt_veto")

        # ---------------------------------------------------------------
        # options_data regime + IV hint (soft context)
        # ---------------------------------------------------------------
        options_filter: Optional[Dict[str, Any]] = None
        opt_obj = _get_options_obj(macro)
        iv_hint = _get_macro_implied_vol(macro, symbol)

        # setup guess (cheap)
        bos_flag = bool(struct.get("bos", False))
        raid_ok = bool(isinstance(struct.get("raid_displacement"), dict) and struct["raid_displacement"].get("ok"))
        liq_sweep_pre = liquidity_sweep_details(df_h1, bias, lookback=180)
        sweep_ok = bool(isinstance(liq_sweep_pre, dict) and liq_sweep_pre.get("ok"))

        if score_options_context is not None:
            if bos_flag:
                setup_guess = "BOS_STRICT"
            elif raid_ok:
                setup_guess = "RAID_DISPLACEMENT"
            elif sweep_ok:
                setup_guess = "LIQ_SWEEP"
            else:
                setup_guess = "OTHER"
            options_filter = _score_options_context_safe(opt_obj, bias=bias, setup_type=setup_guess)

            try:
                reg = str((options_filter or {}).get("regime") or "").lower()
                if DESK_EV_MODE and reg in ("high_vol", "crisis", "event"):
                    soft_vetoes.append(f"options_regime:{reg}")
            except Exception:
                pass

        if iv_hint is not None and np.isfinite(iv_hint) and float(iv_hint) >= IV_HIGH_THRESHOLD:
            if DESK_EV_MODE:
                soft_vetoes.append("iv_high")
            else:
                # non-desk: keep it as veto only if strict session filter already enabled
                # (institutional max still allows, because RR/SL already adapts)
                pass

        # ---------------------------------------------------------------
        # STRICT HTF mismatch (anti-contre-tendance)
        # ---------------------------------------------------------------
        htf_bias_now = _htf_bias_from_df(df_h4)
        if htf_bias_now in ("LONG", "SHORT") and bias in ("LONG", "SHORT") and (htf_bias_now != bias):
            if TG_STRICT_HTF_BIAS or (not DESK_EV_MODE):
                LOGGER.info("[EVAL_REJECT] %s htf_bias_mismatch bias=%s htf_bias=%s", symbol, bias, htf_bias_now)
                return _reject(
                    "htf_bias_mismatch",
                    structure=struct,
                    momentum=mom,
                    composite=comp,
                    htf_bias=htf_bias_now,
                    smt=smt,
                    smt_veto=bool(smt_veto),
                    options_filter=options_filter,
                    session=sess_meta,
                    tradability=trad,
                )
            soft_vetoes.append("htf_bias_mismatch")

        if REQUIRE_STRUCTURE and used_bias_fallback and (not DESK_EV_MODE):
            LOGGER.info("[EVAL_REJECT] %s no_clear_trend_range", symbol)
            return _reject("no_clear_trend_range", structure=struct, momentum=mom, composite=comp, smt=smt, smt_veto=bool(smt_veto))

        if REQUIRE_HTF_ALIGN and bias in ("LONG", "SHORT"):
            if not htf_trend_ok(df_h4, bias):
                if DESK_EV_MODE:
                    soft_vetoes.append("htf_veto")
                    LOGGER.info("[EVAL_PRE] %s HTF_VETO (soft, desk_mode)", symbol)
                else:
                    LOGGER.info("[EVAL_REJECT] %s htf_veto", symbol)
                    return _reject("htf_veto", structure=struct, smt=smt, smt_veto=bool(smt_veto), options_filter=options_filter)

        if (not DESK_EV_MODE) and (not (bos_flag or raid_ok or sweep_ok)):
            LOGGER.info("[EVAL_REJECT] %s no_structure_trigger", symbol)
            return _reject("no_structure_trigger", structure=struct, sweep=liq_sweep_pre, smt=smt, smt_veto=bool(smt_veto), options_filter=options_filter)

        if REQUIRE_MOMENTUM:
            if bias == "LONG" and mom not in ("BULLISH", "STRONG_BULLISH"):
                if DESK_EV_MODE:
                    soft_vetoes.append("momentum_not_bullish")
                else:
                    return _reject("momentum_not_bullish", structure=struct, smt=smt, smt_veto=bool(smt_veto), options_filter=options_filter)
            if bias == "SHORT" and mom not in ("BEARISH", "STRONG_BEARISH"):
                if DESK_EV_MODE:
                    soft_vetoes.append("momentum_not_bearish")
                else:
                    return _reject("momentum_not_bearish", structure=struct, smt=smt, smt_veto=bool(smt_veto), options_filter=options_filter)

        # ===============================================================
        # PRE-PRIORITY (cheap) → controls PASS2
        # ===============================================================
        pre_priority, pre_priority_reasons = _pre_grade_candidate(
            bos_flag=bos_flag,
            raid_ok=raid_ok,
            sweep_ok=sweep_ok,
            mom=mom,
            comp_score=comp_score,
            used_bias_fallback=used_bias_fallback,
            ext_sig=ext_sig,
        )
        pass2_allowed_by_priority = priority_at_least(pre_priority, PASS2_ONLY_FOR_PRIORITY)

        LOGGER.info(
            "[EVAL_PRE] %s PRE_PRIORITY=%s pass2_allowed=%s reasons=%s soft_vetoes=%s smt_veto=%s options_filter=%s session=%s trad=%s iv=%s",
            symbol,
            pre_priority,
            pass2_allowed_by_priority,
            pre_priority_reasons,
            soft_vetoes,
            smt_veto,
            options_filter,
            sess_meta,
            trad,
            iv_hint,
        )

        # ===============================================================
        # 1) INSTITUTIONAL — 2-PASS (anti-ban) + Liquidations
        # ===============================================================
        inst: Dict[str, Any]

        pass1_liq = bool(INST_ENABLE_LIQUIDATIONS and (not INST_LIQ_PASS2_ONLY))
        try:
            inst = await compute_full_institutional_analysis(
                symbol,
                bias,
                mode=INST_PASS1_MODE,
                include_liquidations=pass1_liq,
            )
        except Exception as e:
            inst = {
                "institutional_score": 0,
                "bitget_symbol": None,
                "available": False,
                "warnings": [f"inst_exception:{e}"],
                "score_components": {"flow": 0, "oi": 0, "crowding": 0, "orderbook": 0},
                "score_meta": {"raw_components_sum": 0, "ok_count": 0, "mode": INST_PASS1_MODE},
            }

        available = bool(inst.get("available", False))
        bitget_symbol = (inst.get("bitget_symbol") or inst.get("symbol") or symbol)

        gate, ok_count = _inst_gate_value(inst)
        gate, override = _desk_inst_gate_override(inst, bias, int(gate))
        inst_score_norm = _inst_score_normalized_0_100(inst, int(gate))

        LOGGER.info(
            "[INST_RAW] %s pass=1 mode=%s liq_req=%s inst_score=%s ok_count=%s gate=%s inst_score_norm=%s override=%s available=%s bitget_symbol=%s bias=%s comps=%s",
            symbol,
            (inst.get("score_meta") or {}).get("mode") or INST_PASS1_MODE,
            pass1_liq,
            inst.get("institutional_score"),
            ok_count,
            gate,
            inst_score_norm,
            override,
            available,
            bitget_symbol,
            bias,
            inst.get("available_components") or [],
        )

        if (not available) or (bitget_symbol is None):
            if not ALLOW_TECH_FALLBACK_WHEN_INST_DOWN:
                return _reject(
                    "inst_unavailable",
                    institutional=inst,
                    structure=struct,
                    momentum=mom,
                    composite=comp,
                    pre_priority=pre_priority,
                    pre_priority_reasons=pre_priority_reasons,
                    soft_vetoes=soft_vetoes,
                    smt=smt,
                    smt_veto=bool(smt_veto),
                    options_filter=options_filter,
                    session=sess_meta,
                    tradability=trad,
                    iv=iv_hint,
                    inst_gate=int(gate),
                    inst_ok_count=int(ok_count),
                    inst_score_norm=int(inst_score_norm),
                )
            LOGGER.warning("[INST_DOWN] %s tech_fallback_enabled", symbol)

        do_pass2 = bool(
            INST_PASS2_ENABLED
            and pass2_allowed_by_priority
            and available
            and (bitget_symbol is not None)
            and (int(gate) >= int(INST_PASS2_MIN_GATE) or bool(DESK_EV_MODE))
        )

        if do_pass2:
            pass2_liq = bool(INST_ENABLE_LIQUIDATIONS)
            try:
                inst2 = await compute_full_institutional_analysis(
                    symbol,
                    bias,
                    mode=INST_PASS2_MODE,
                    include_liquidations=pass2_liq,
                )
            except Exception as e:
                inst2 = {
                    "institutional_score": 0,
                    "bitget_symbol": bitget_symbol,
                    "available": False,
                    "warnings": [f"inst_pass2_exception:{e}"],
                    "score_components": {"flow": 0, "oi": 0, "crowding": 0, "orderbook": 0},
                    "score_meta": {"raw_components_sum": 0, "ok_count": 0, "mode": INST_PASS2_MODE},
                }

            if isinstance(inst2, dict) and bool(inst2.get("available")) :
                inst = inst2
                available = bool(inst.get("available", False))
                bitget_symbol = (inst.get("bitget_symbol") or inst.get("symbol") or symbol)
                gate, ok_count = _inst_gate_value(inst)
                gate, override = _desk_inst_gate_override(inst, bias, int(gate))
                inst_score_norm = _inst_score_normalized_0_100(inst, int(gate))

                LOGGER.info(
                    "[INST_RAW] %s pass=2 mode=%s liq_req=%s inst_score=%s ok_count=%s gate=%s inst_score_norm=%s override=%s available=%s bitget_symbol=%s bias=%s comps=%s",
                    symbol,
                    (inst.get("score_meta") or {}).get("mode") or INST_PASS2_MODE,
                    pass2_liq,
                    inst.get("institutional_score"),
                    ok_count,
                    gate,
                    inst_score_norm,
                    override,
                    available,
                    bitget_symbol,
                    bias,
                    inst.get("available_components") or [],
                )

        if (not ALLOW_TECH_FALLBACK_WHEN_INST_DOWN) and int(gate) < int(MIN_INST_SCORE):
            return _reject(
                "inst_gate_low",
                institutional=inst,
                structure=struct,
                inst_gate=int(gate),
                ok_count=int(ok_count),
                inst_score_norm=int(inst_score_norm),
                pre_priority=pre_priority,
                pre_priority_reasons=pre_priority_reasons,
                soft_vetoes=soft_vetoes,
                smt=smt,
                smt_veto=bool(smt_veto),
                options_filter=options_filter,
                session=sess_meta,
                tradability=trad,
                iv=iv_hint,
            )

        # ===============================================================
        # 2) STRUCTURE / QUALITY / ENTRY / EXITS
        # ===============================================================
        entry_mkt = _last_close(df_h1)

        bos_dir = struct.get("bos_direction", None)
        bos_q = bos_quality_details(
            df_h1,
            oi_series=struct.get("oi_series", None),
            df_liq=df_h1,
            price=entry_mkt,
            direction=bos_dir,
        )
        bos_quality_ok = bool(bos_q.get("ok", True))

        if REQUIRE_BOS_QUALITY and not bos_quality_ok:
            return _reject(
                "bos_quality_low",
                institutional=inst,
                structure=struct,
                bos_quality=bos_q,
                pre_priority=pre_priority,
                pre_priority_reasons=pre_priority_reasons,
                soft_vetoes=soft_vetoes,
                smt=smt,
                smt_veto=bool(smt_veto),
                options_filter=options_filter,
                session=sess_meta,
                tradability=trad,
                iv=iv_hint,
                inst_gate=int(gate),
                inst_ok_count=int(ok_count),
                inst_score_norm=int(inst_score_norm),
            )

        premium, discount = compute_premium_discount(df_h1)
        liq_sweep = liq_sweep_pre

        entry_pick = _pick_entry(df_h1, struct, bias)
        entry = float(entry_pick["entry_used"])
        entry_type = str(entry_pick["entry_type"])
        atr14 = float(entry_pick.get("atr") or _atr(df_h1, 14))

        # ---------------------------------------------------------------
        # TrendGuard
        # ---------------------------------------------------------------
        tg_ok, tg_why, tg_meta = _trend_guard(
            df_h1=df_h1,
            df_h4=df_h4,
            bias=bias,
            entry=float(entry) if entry > 0 else None,
            entry_type=entry_type,
        )
        if not tg_ok:
            if DESK_EV_MODE and (not TG_STRICT_REGIME_MARKET) and (tg_why != "reject:htf_bias_mismatch"):
                soft_vetoes.append(tg_why.replace("reject:", "tg:"))
            else:
                return _reject(
                    tg_why.replace("reject:", ""),
                    institutional=inst,
                    structure=struct,
                    entry_pick=entry_pick,
                    trend_guard=tg_meta,
                    pre_priority=pre_priority,
                    pre_priority_reasons=pre_priority_reasons,
                    soft_vetoes=soft_vetoes,
                    smt=smt,
                    smt_veto=bool(smt_veto),
                    options_filter=options_filter,
                    session=sess_meta,
                    tradability=trad,
                    iv=iv_hint,
                    inst_gate=int(gate),
                    inst_ok_count=int(ok_count),
                    inst_score_norm=int(inst_score_norm),
                )

        # Extension hygiene
        if ext_sig == "OVEREXTENDED_LONG" and bias == "LONG":
            if entry_type == "MARKET":
                return _reject("overextended_long_market", institutional=inst, structure=struct, entry_pick=entry_pick)
            if not _entry_pullback_ok(entry, entry_mkt, bias, atr14):
                return _reject("overextended_long_no_pullback", institutional=inst, structure=struct, entry_pick=entry_pick)

        if ext_sig == "OVEREXTENDED_SHORT" and bias == "SHORT":
            if entry_type == "MARKET":
                return _reject("overextended_short_market", institutional=inst, structure=struct, entry_pick=entry_pick)
            if not _entry_pullback_ok(entry, entry_mkt, bias, atr14):
                return _reject("overextended_short_no_pullback", institutional=inst, structure=struct, entry_pick=entry_pick)

        unfavorable_market = bool(entry_type == "MARKET" and ((bias == "LONG" and premium) or (bias == "SHORT" and discount)))
        if unfavorable_market and (not DESK_EV_MODE):
            if bias == "LONG" and premium:
                return _reject("long_in_premium_market", institutional=inst, structure=struct, entry_pick=entry_pick)
            if bias == "SHORT" and discount:
                return _reject("short_in_discount_market", institutional=inst, structure=struct, entry_pick=entry_pick)

        # --------------------------
        # Exits (policy-aware SL v3)
        # --------------------------
        setup_hint: Optional[str] = None
        if bos_flag:
            setup_hint = "BOS_STRICT"
        elif DESK_EV_MODE and raid_ok:
            setup_hint = "RAID_DISPLACEMENT"
        elif DESK_EV_MODE and sweep_ok:
            setup_hint = "LIQ_SWEEP"
        elif DESK_EV_MODE:
            setup_hint = "INST_CONTINUATION"

        tick = estimate_tick_from_price(entry)
        exits = _compute_exits(
            df_h1,
            entry,
            bias,
            tick=tick,
            setup=setup_hint,
            entry_type=entry_type,
            htf_df=df_h4,
        )

        sl = float(exits["sl"])
        tp1 = float(exits["tp1"])
        rr = _safe_rr(entry, sl, tp1, bias)

        # SL hygiene beyond sweep / eq levels
        atr_last = _atr(df_h1, 14)
        buf = max((atr_last * 0.08) if atr_last > 0 else 0.0, float(tick) * 2.0, entry * 0.0004)

        sl_pre = float(sl)
        sl_adj: Dict[str, Any] = {
            "setup_hint": setup_hint,
            "entry_type": entry_type,
            "sl_pre": sl_pre,
            "buf": float(buf),
            "did_sweep": False,
            "did_eq": False,
            "sweep_extreme": None,
            "eq_level": None,
            "sl_post_round": None,
        }

        try:
            if isinstance(liq_sweep, dict) and liq_sweep.get("ok"):
                ext = float(liq_sweep.get("sweep_extreme") or 0.0)
                if ext > 0 and buf > 0:
                    if bias == "LONG":
                        new_sl = min(sl, ext - buf)
                    else:
                        new_sl = max(sl, ext + buf)
                    if np.isfinite(new_sl) and float(new_sl) != float(sl):
                        sl = float(new_sl)
                        sl_adj["did_sweep"] = True
                        sl_adj["sweep_extreme"] = float(ext)
        except Exception:
            pass

        try:
            lv = detect_equal_levels(df_h1.tail(200), max_window=200, tol_mult_atr=0.10)
            eq_highs = lv.get("eq_highs", []) or []
            eq_lows = lv.get("eq_lows", []) or []

            if bias == "LONG" and eq_lows:
                lvl = min(eq_lows, key=lambda x: abs(entry - float(x)))
                lvl = float(lvl)
                if lvl > 0 and lvl < entry and sl > (lvl - buf):
                    new_sl = min(sl, lvl - buf)
                    if np.isfinite(new_sl) and float(new_sl) != float(sl):
                        sl = float(new_sl)
                        sl_adj["did_eq"] = True
                        sl_adj["eq_level"] = float(lvl)

            if bias == "SHORT" and eq_highs:
                lvl = min(eq_highs, key=lambda x: abs(entry - float(x)))
                lvl = float(lvl)
                if lvl > 0 and lvl > entry and sl < (lvl + buf):
                    new_sl = max(sl, lvl + buf)
                    if np.isfinite(new_sl) and float(new_sl) != float(sl):
                        sl = float(new_sl)
                        sl_adj["did_eq"] = True
                        sl_adj["eq_level"] = float(lvl)
        except Exception:
            pass

        sl = _round_sl_for_side(sl, entry, bias, tick)
        sl_adj["sl_post_round"] = float(sl)

        if sl <= 0:
            return _reject("sl_invalid_after_liq_adj", institutional=inst, structure=struct, sl_meta=sl_adj)

        # recalc TP1 after SL adjustments
        try:
            tp1, _rr_used2 = compute_tp1(entry, sl, bias, df=df_h1, tick=tick)
            tp1 = float(tp1)
        except Exception:
            tp1 = float(tp1)

        rr = _safe_rr(entry, sl, tp1, bias)
        if rr is None or rr <= 0:
            return _reject("rr_invalid", institutional=inst, structure=struct, entry_pick=entry_pick)

        base_sl_meta = exits.get("sl_meta") if isinstance(exits.get("sl_meta"), dict) else {}
        sl_meta_out = dict(base_sl_meta)
        sl_meta_out["post_adjust"] = sl_adj

        # ===============================================================
        # Institutional micro filters (intent-aware)
        # ===============================================================
        setup_intent = "CONTINUATION" if bos_flag else ("REVERSAL" if (raid_ok or sweep_ok) else "OTHER")
        hard_ok_micro, micro_vetoes, micro_meta = _inst_micro_filters(inst, bias, setup_intent=setup_intent)

        if not hard_ok_micro:
            if (not DESK_EV_MODE) and INST_MICRO_FILTER_STRICT:
                return _reject(
                    "inst_micro_veto",
                    institutional=inst,
                    inst_micro=micro_meta,
                    inst_micro_vetoes=micro_vetoes,
                    structure=struct,
                    entry_pick=entry_pick,
                    rr=float(rr),
                    soft_vetoes=soft_vetoes,
                    session=sess_meta,
                    tradability=trad,
                    iv=iv_hint,
                )
            if DESK_EV_MODE:
                soft_vetoes.extend([f"inst_micro:{v}" for v in micro_vetoes])

        # ===============================================================
        # Confidence score
        # ===============================================================
        confidence = _confidence_score(
            inst_score_norm=int(inst_score_norm),
            bos_q=bos_q,
            comp_score=float(comp_score),
            bias=bias,
            used_bias_fallback=bool(used_bias_fallback),
            soft_vetoes=list(soft_vetoes),
            micro_vetoes=list(micro_vetoes),
            options_filter=options_filter,
            session_ok=bool(sess_ok),
            tradability_ok=bool(trad_ok),
            iv_hint=iv_hint,
        )

        # ===============================================================
        # SETUPS
        # ===============================================================
        can_bos = bool(bos_flag and ((not REQUIRE_BOS_QUALITY) or bos_quality_ok))
        can_raid = bool(isinstance(struct.get("raid_displacement"), dict) and struct["raid_displacement"].get("ok"))
        can_sweep = bool(isinstance(liq_sweep, dict) and liq_sweep.get("ok"))

        # BOS_STRICT — now requires micro alignment if enabled (true institutional continuation)
        bos_ok = False
        bos_variant = "NO"
        if can_bos:
            # enforce continuation cleanliness (confidence and micro alignment)
            if INST_MICRO_FILTER_ENABLE and (not hard_ok_micro) and (not DESK_EV_MODE):
                bos_ok = False
            else:
                if rr >= RR_MIN_STRICT:
                    bos_ok = True
                    bos_variant = "RR_STRICT"
                elif rr >= RR_MIN_TOLERATED_WITH_INST and confidence >= 70:
                    bos_ok = True
                    bos_variant = "RR_RELAX_WITH_INST"

        if bos_ok:
            setup_core = "BOS_STRICT"
            setup_ttl = _setup_ttl_compatible(setup_core, entry_type)
            priority, priority_reasons = _final_grade(
                setup_type=setup_core,
                setup_variant=bos_variant,
                rr=float(rr),
                gate=int(gate),
                ok_count=int(ok_count),
                inst_available=bool(available),
                bos_quality_ok=bool(bos_quality_ok),
                used_bias_fallback=bool(used_bias_fallback),
                entry_type=str(entry_type),
                ext_sig=str(ext_sig),
                unfavorable_market=bool(unfavorable_market),
                mom=str(mom),
                soft_vetoes=soft_vetoes,
                inst_micro_vetoes=micro_vetoes,
                confidence=int(confidence),
            )

            # recompute options context with final setup type (if module exists)
            if score_options_context is not None:
                options_filter = _score_options_context_safe(opt_obj, bias=bias, setup_type=setup_core)

            return {
                "valid": True,
                "symbol": symbol,
                "side": "BUY" if bias == "LONG" else "SELL",
                "bias": bias,
                "entry": entry,
                "entry_type": entry_type,
                "sl": float(sl),
                "tp1": float(tp1),
                "tp2": None,
                "rr": float(rr),
                "qty": 1,
                "setup_type": setup_ttl,
                "setup_type_core": setup_core,
                "setup_variant": bos_variant,
                "priority": priority,
                "priority_reasons": priority_reasons,
                "pre_priority": pre_priority,
                "pre_priority_reasons": pre_priority_reasons,
                "pass2_done": bool(do_pass2),
                "soft_vetoes": soft_vetoes,
                "session": sess_meta,
                "tradability": trad,
                "trend_guard": tg_meta,
                "structure": struct,
                "bos_quality": bos_q,
                "institutional": inst,
                "inst_micro": micro_meta,
                "inst_micro_vetoes": micro_vetoes,
                "inst_gate": int(gate),
                "inst_ok_count": int(ok_count),
                "inst_score_eff": int(gate),
                "inst_score_norm": int(inst_score_norm),
                "confidence": int(confidence),
                "momentum": mom,
                "composite": comp,
                "composite_score": comp_score,
                "composite_label": comp_label,
                "premium": premium,
                "discount": discount,
                "entry_pick": entry_pick,
                "sl_meta": sl_meta_out,
                "smt": smt,
                "smt_veto": bool(smt_veto),
                "options_filter": options_filter,
                "iv": iv_hint,
            }

        # RAID_DISPLACEMENT (desk)
        if DESK_EV_MODE and can_raid and float(rr) >= float(RR_MIN_DESK_PRIORITY):
            setup_core = "RAID_DISPLACEMENT"
            setup_ttl = _setup_ttl_compatible(setup_core, entry_type)
            variant = str((struct.get("raid_displacement") or {}).get("note") or "raid_ok")
            priority, priority_reasons = _final_grade(
                setup_type=setup_core,
                setup_variant=variant,
                rr=float(rr),
                gate=int(gate),
                ok_count=int(ok_count),
                inst_available=bool(available),
                bos_quality_ok=bool(bos_quality_ok),
                used_bias_fallback=bool(used_bias_fallback),
                entry_type=str(entry_type),
                ext_sig=str(ext_sig),
                unfavorable_market=bool(unfavorable_market),
                mom=str(mom),
                soft_vetoes=soft_vetoes,
                inst_micro_vetoes=micro_vetoes,
                confidence=int(confidence),
            )

            if score_options_context is not None:
                options_filter = _score_options_context_safe(opt_obj, bias=bias, setup_type=setup_core)

            return {
                "valid": True,
                "symbol": symbol,
                "side": "BUY" if bias == "LONG" else "SELL",
                "bias": bias,
                "entry": entry,
                "entry_type": entry_type,
                "sl": float(sl),
                "tp1": float(tp1),
                "tp2": None,
                "rr": float(rr),
                "qty": 1,
                "setup_type": setup_ttl,
                "setup_type_core": setup_core,
                "setup_variant": variant,
                "priority": priority,
                "priority_reasons": priority_reasons,
                "pre_priority": pre_priority,
                "pre_priority_reasons": pre_priority_reasons,
                "pass2_done": bool(do_pass2),
                "soft_vetoes": soft_vetoes,
                "session": sess_meta,
                "tradability": trad,
                "trend_guard": tg_meta,
                "structure": struct,
                "bos_quality": bos_q,
                "institutional": inst,
                "inst_micro": micro_meta,
                "inst_micro_vetoes": micro_vetoes,
                "inst_gate": int(gate),
                "inst_ok_count": int(ok_count),
                "inst_score_eff": int(gate),
                "inst_score_norm": int(inst_score_norm),
                "confidence": int(confidence),
                "momentum": mom,
                "composite": comp,
                "composite_score": comp_score,
                "composite_label": comp_label,
                "premium": premium,
                "discount": discount,
                "entry_pick": entry_pick,
                "liquidity_sweep": liq_sweep,
                "sl_meta": sl_meta_out,
                "smt": smt,
                "smt_veto": bool(smt_veto),
                "options_filter": options_filter,
                "iv": iv_hint,
            }

        # LIQ_SWEEP (desk)
        if DESK_EV_MODE and can_sweep and float(rr) >= float(RR_MIN_DESK_PRIORITY):
            setup_core = "LIQ_SWEEP"
            setup_ttl = _setup_ttl_compatible(setup_core, entry_type)
            variant = str(liq_sweep.get("kind") if isinstance(liq_sweep, dict) else "liq_ok")
            priority, priority_reasons = _final_grade(
                setup_type=setup_core,
                setup_variant=variant,
                rr=float(rr),
                gate=int(gate),
                ok_count=int(ok_count),
                inst_available=bool(available),
                bos_quality_ok=bool(bos_quality_ok),
                used_bias_fallback=bool(used_bias_fallback),
                entry_type=str(entry_type),
                ext_sig=str(ext_sig),
                unfavorable_market=bool(unfavorable_market),
                mom=str(mom),
                soft_vetoes=soft_vetoes,
                inst_micro_vetoes=micro_vetoes,
                confidence=int(confidence),
            )

            if score_options_context is not None:
                options_filter = _score_options_context_safe(opt_obj, bias=bias, setup_type=setup_core)

            return {
                "valid": True,
                "symbol": symbol,
                "side": "BUY" if bias == "LONG" else "SELL",
                "bias": bias,
                "entry": entry,
                "entry_type": entry_type,
                "sl": float(sl),
                "tp1": float(tp1),
                "tp2": None,
                "rr": float(rr),
                "qty": 1,
                "setup_type": setup_ttl,
                "setup_type_core": setup_core,
                "setup_variant": variant,
                "priority": priority,
                "priority_reasons": priority_reasons,
                "pre_priority": pre_priority,
                "pre_priority_reasons": pre_priority_reasons,
                "pass2_done": bool(do_pass2),
                "soft_vetoes": soft_vetoes,
                "session": sess_meta,
                "tradability": trad,
                "trend_guard": tg_meta,
                "structure": struct,
                "bos_quality": bos_q,
                "institutional": inst,
                "inst_micro": micro_meta,
                "inst_micro_vetoes": micro_vetoes,
                "inst_gate": int(gate),
                "inst_ok_count": int(ok_count),
                "inst_score_eff": int(gate),
                "inst_score_norm": int(inst_score_norm),
                "confidence": int(confidence),
                "momentum": mom,
                "composite": comp,
                "composite_score": comp_score,
                "composite_label": comp_label,
                "premium": premium,
                "discount": discount,
                "entry_pick": entry_pick,
                "liquidity_sweep": liq_sweep,
                "sl_meta": sl_meta_out,
                "smt": smt,
                "smt_veto": bool(smt_veto),
                "options_filter": options_filter,
                "iv": iv_hint,
            }

        # INST_CONTINUATION (desk) — ULTRA STRICT + micro + confidence
        if DESK_EV_MODE and bias in ("LONG", "SHORT"):
            inst_cont_trigger_ok = (can_bos or can_raid or can_sweep)
            inst_cont_pass2_ok = bool(do_pass2)
            inst_cont_entry_ok = (not _is_market_entry(entry_type))
            inst_cont_gate_ok = int(gate) >= int(max(MIN_INST_SCORE, INST_CONT_MIN_GATE))
            inst_cont_rr_ok = float(rr) >= float(INST_CONT_RR_MIN)
            inst_cont_comp_ok = _composite_bias_ok(bias, comp_score, comp_label, thr=float(INST_CONT_MIN_COMPOSITE_THR))
            inst_cont_soft_ok = (len(soft_vetoes) == 0)
            inst_cont_bias_ok = (not used_bias_fallback)

            # Institutional MAX: require micro alignment success + confidence high
            inst_cont_micro_ok = bool(hard_ok_micro) and (len([v for v in micro_vetoes if v.startswith("inst_align_low")]) == 0)
            inst_cont_conf_ok = bool(int(confidence) >= 80)

            m = str(mom or "").upper()
            if INST_CONT_REQUIRE_STRONG_MOM:
                inst_cont_mom_ok = (m == "STRONG_BULLISH") if bias == "LONG" else (m == "STRONG_BEARISH")
            else:
                inst_cont_mom_ok = True

            if not INST_CONT_REQUIRE_TRIGGER:
                inst_cont_trigger_ok = True
            if not INST_CONT_REQUIRE_PASS2:
                inst_cont_pass2_ok = True
            if not INST_CONT_DISALLOW_MARKET:
                inst_cont_entry_ok = True
            if not INST_CONT_REQUIRE_NO_SOFT_VETO:
                inst_cont_soft_ok = True
            if not INST_CONT_REQUIRE_CLEAR_BIAS:
                inst_cont_bias_ok = True

            inst_cont_ok = bool(
                inst_cont_gate_ok
                and inst_cont_rr_ok
                and inst_cont_comp_ok
                and inst_cont_mom_ok
                and inst_cont_trigger_ok
                and inst_cont_pass2_ok
                and inst_cont_entry_ok
                and inst_cont_soft_ok
                and inst_cont_bias_ok
                and inst_cont_micro_ok
                and inst_cont_conf_ok
            )

            LOGGER.info(
                "[EVAL_PRE] %s INST_CONT_MAX ok=%s gate_ok=%s rr_ok=%s comp_ok=%s mom_ok=%s trigger_ok=%s pass2_ok=%s entry_ok=%s soft_ok=%s bias_ok=%s micro_ok=%s conf_ok=%s "
                "(gate=%s rr=%.4f conf=%s micro=%s soft_vetoes=%s)",
                symbol,
                inst_cont_ok,
                inst_cont_gate_ok,
                inst_cont_rr_ok,
                inst_cont_comp_ok,
                inst_cont_mom_ok,
                inst_cont_trigger_ok,
                inst_cont_pass2_ok,
                inst_cont_entry_ok,
                inst_cont_soft_ok,
                inst_cont_bias_ok,
                inst_cont_micro_ok,
                inst_cont_conf_ok,
                gate,
                float(rr),
                confidence,
                micro_vetoes,
                soft_vetoes,
            )

            if inst_cont_ok:
                setup_core = "INST_CONTINUATION"
                setup_ttl = _setup_ttl_compatible(setup_core, entry_type)
                priority, priority_reasons = _final_grade(
                    setup_type=setup_core,
                    setup_variant="INST_CONTINUATION_MAX",
                    rr=float(rr),
                    gate=int(gate),
                    ok_count=int(ok_count),
                    inst_available=bool(available),
                    bos_quality_ok=bool(bos_quality_ok),
                    used_bias_fallback=bool(used_bias_fallback),
                    entry_type=str(entry_type),
                    ext_sig=str(ext_sig),
                    unfavorable_market=bool(unfavorable_market),
                    mom=str(mom),
                    soft_vetoes=soft_vetoes,
                    inst_micro_vetoes=micro_vetoes,
                    confidence=int(confidence),
                )

                if score_options_context is not None:
                    options_filter = _score_options_context_safe(opt_obj, bias=bias, setup_type=setup_core)

                return {
                    "valid": True,
                    "symbol": symbol,
                    "side": "BUY" if bias == "LONG" else "SELL",
                    "bias": bias,
                    "entry": entry,
                    "entry_type": entry_type,
                    "sl": float(sl),
                    "tp1": float(tp1),
                    "tp2": None,
                    "rr": float(rr),
                    "qty": 1,
                    "setup_type": setup_ttl,
                    "setup_type_core": setup_core,
                    "setup_variant": "INST_CONTINUATION_MAX",
                    "priority": priority,
                    "priority_reasons": priority_reasons,
                    "pre_priority": pre_priority,
                    "pre_priority_reasons": pre_priority_reasons,
                    "pass2_done": bool(do_pass2),
                    "soft_vetoes": soft_vetoes,
                    "session": sess_meta,
                    "tradability": trad,
                    "trend_guard": tg_meta,
                    "structure": struct,
                    "bos_quality": bos_q,
                    "institutional": inst,
                    "inst_micro": micro_meta,
                    "inst_micro_vetoes": micro_vetoes,
                    "inst_gate": int(gate),
                    "inst_ok_count": int(ok_count),
                    "inst_score_eff": int(gate),
                    "inst_score_norm": int(inst_score_norm),
                    "confidence": int(confidence),
                    "momentum": mom,
                    "composite": comp,
                    "composite_score": comp_score,
                    "composite_label": comp_label,
                    "premium": premium,
                    "discount": discount,
                    "entry_pick": entry_pick,
                    "sl_meta": sl_meta_out,
                    "smt": smt,
                    "smt_veto": bool(smt_veto),
                    "options_filter": options_filter,
                    "iv": iv_hint,
                }

        LOGGER.info("[EVAL_REJECT] %s no_setup_validated (DESK_EV_MODE=%s)", symbol, DESK_EV_MODE)
        return {
            "valid": False,
            "reject_reason": "no_setup_validated",
            "structure": struct,
            "institutional": inst,
            "inst_micro": micro_meta,
            "inst_micro_vetoes": micro_vetoes,
            "bos_quality": bos_q,
            "entry_pick": entry_pick,
            "rr": float(rr) if rr is not None else None,
            "inst_gate": int(gate),
            "inst_ok_count": int(ok_count),
            "inst_score_eff": int(gate),
            "inst_score_norm": int(inst_score_norm),
            "confidence": int(confidence),
            "composite_score": comp_score,
            "composite_label": comp_label,
            "sl_meta": sl_meta_out,
            "pre_priority": pre_priority,
            "pre_priority_reasons": pre_priority_reasons,
            "soft_vetoes": soft_vetoes,
            "session": sess_meta,
            "tradability": trad,
            "trend_guard": tg_meta,
            "smt": smt,
            "smt_veto": bool(smt_veto),
            "options_filter": options_filter,
            "iv": iv_hint,
        }
