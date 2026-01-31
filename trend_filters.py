# trend_filters.py
from __future__ import annotations

import os
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd


def _has_cols(df: pd.DataFrame, cols: Tuple[str, ...]) -> bool:
    return isinstance(df, pd.DataFrame) and (not df.empty) and all(c in df.columns for c in cols)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return pd.Series(series).astype(float).ewm(span=int(span), adjust=False).mean()


def ema_trend(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> str:
    """Return LONG/SHORT/RANGE."""
    if not _has_cols(df, ("close",)) or len(df) < slow + 8:
        return "RANGE"
    c = df["close"].astype(float)
    ef = _ema(c, fast)
    es = _ema(c, slow)
    slope = float(ef.iloc[-1] - ef.iloc[-8])
    spread = float(ef.iloc[-1] - es.iloc[-1])
    base = float(abs(es.iloc[-1])) if abs(float(es.iloc[-1])) > 1e-12 else 1.0
    spread_pct = spread / base

    min_spread = float(os.getenv("TG_MIN_EMA_SPREAD_PCT", "0.0008"))  # 0.08% default
    if ef.iloc[-1] > es.iloc[-1] and slope > 0 and spread_pct > min_spread:
        return "LONG"
    if ef.iloc[-1] < es.iloc[-1] and slope < 0 and spread_pct < -min_spread:
        return "SHORT"
    return "RANGE"


def atr14(df: pd.DataFrame, period: int = 14) -> float:
    if not _has_cols(df, ("high", "low", "close")) or len(df) < period + 2:
        return 0.0
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    v = float(tr.rolling(int(period)).mean().iloc[-1])
    return v if np.isfinite(v) else 0.0


def adx(df: pd.DataFrame, period: int = 14) -> float:
    """
    Wilder ADX (lightweight). Returns last ADX value.
    """
    if not _has_cols(df, ("high", "low", "close")) or len(df) < period * 3:
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

    # Wilder smoothing via EMA(alpha=1/period)
    alpha = 1.0 / float(period)
    atr = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean()

    pdi = 100.0 * (pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan))
    mdi = 100.0 * (pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan))

    dx = 100.0 * (abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan))
    adx_s = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean()
    v = float(adx_s.iloc[-1])
    return v if np.isfinite(v) else 0.0


def bollinger_bandwidth(df: pd.DataFrame, period: int = 20, k: float = 2.0) -> float:
    """
    Bandwidth = (upper - lower) / middle. Returns last value.
    """
    if not _has_cols(df, ("close",)) or len(df) < period + 5:
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


def is_squeeze(df: pd.DataFrame) -> bool:
    """
    Simple squeeze heuristic with env-config threshold.
    """
    thr = float(os.getenv("TG_BB_SQUEEZE_BW", "0.04"))  # 4% default
    bw = bollinger_bandwidth(df, period=int(os.getenv("TG_BB_PERIOD", "20")), k=float(os.getenv("TG_BB_K", "2")))
    return bool(bw > 0 and bw < thr)


def supertrend_dir(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> str:
    """
    Minimal Supertrend direction (LONG/SHORT), last value only.
    """
    if not _has_cols(df, ("high", "low", "close")) or len(df) < period * 3:
        return "RANGE"

    h = pd.to_numeric(df["high"], errors="coerce").astype(float)
    l = pd.to_numeric(df["low"], errors="coerce").astype(float)
    c = pd.to_numeric(df["close"], errors="coerce").astype(float)

    hl2 = (h + l) / 2.0
    a = atr14(df, period=period)
    if a <= 0:
        return "RANGE"

    upper = hl2 + float(multiplier) * a
    lower = hl2 - float(multiplier) * a

    # simplified band flip
    st = np.zeros(len(df), dtype=float)
    dirn = np.zeros(len(df), dtype=int)

    st[0] = upper.iloc[0]
    dirn[0] = -1

    for i in range(1, len(df)):
        prev_st = st[i - 1]
        prev_dir = dirn[i - 1]
        close_i = float(c.iloc[i])

        up_i = float(upper.iloc[i])
        lo_i = float(lower.iloc[i])

        if prev_dir == 1:
            st[i] = max(lo_i, prev_st)
            if close_i < st[i]:
                dirn[i] = -1
                st[i] = up_i
            else:
                dirn[i] = 1
        else:
            st[i] = min(up_i, prev_st)
            if close_i > st[i]:
                dirn[i] = 1
                st[i] = lo_i
            else:
                dirn[i] = -1

    last_dir = int(dirn[-1])
    return "LONG" if last_dir == 1 else "SHORT"


def trend_guard(
    df_h1: pd.DataFrame,
    df_h4: pd.DataFrame,
    direction: str,
    *,
    entry: Optional[float] = None,
    entry_type: str = "MARKET",
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Returns (ok, reason, meta). This is meant to be called inside analyze_signal.
    """
    d = str(direction).upper()
    meta: Dict[str, Any] = {"direction": d}

    # 1) HTF bias (Supertrend first, EMA fallback)
    st_p = int(os.getenv("TG_ST_PERIOD", "10"))
    st_m = float(os.getenv("TG_ST_MULT", "3"))
    try:
        bias_h4 = supertrend_dir(df_h4, period=st_p, multiplier=st_m)
    except Exception:
        bias_h4 = "RANGE"

    if bias_h4 not in ("LONG", "SHORT"):
        bias_h4 = ema_trend(df_h4)

    meta["bias_h4"] = bias_h4

    if bias_h4 in ("LONG", "SHORT") and bias_h4 != d:
        return False, "reject:htf_bias_mismatch", meta

    # 2) Regime (ADX + squeeze)
    adx_min = float(os.getenv("TG_ADX_MIN", "20"))
    adx_v = adx(df_h4, period=int(os.getenv("TG_ADX_PERIOD", "14")))
    meta["adx_h4"] = float(adx_v)

    squeeze = is_squeeze(df_h1)
    meta["squeeze_h1"] = bool(squeeze)

    is_market = "MARKET" in str(entry_type or "").upper()

    # If weak trend OR squeeze, block MARKET entries (chase)
    if is_market and (adx_v > 0 and adx_v < adx_min):
        return False, "reject:weak_trend_no_market", meta
    if is_market and squeeze:
        return False, "reject:squeeze_no_market", meta

    # 3) Overextension guard (market only)
    if is_market and entry is not None and _has_cols(df_h1, ("close",)):
        a = atr14(df_h1, 14)
        meta["atr_h1"] = float(a)
        if a > 0:
            ema20 = float(_ema(df_h1["close"].astype(float), 20).iloc[-1])
            meta["ema20_h1"] = float(ema20)
            over_mult = float(os.getenv("TG_OVEREXT_ATR", "1.2"))
            if abs(float(entry) - ema20) > (over_mult * a):
                return False, "reject:overextended_market", meta

    return True, "OK", meta
