# =====================================================================
# indicators.py — Core + Institutional Indicators (Desk-lead, Hardened)
# =====================================================================
# ✅ Robust OHLCV checks (fail-safe, no crashes)
# ✅ ATR / true_atr stable (used by stops/tp) + SERIES SHAPE SAFE
# ✅ OTE upgraded for analyze_signal.py compatibility (entry + in_zone + dist)
# ✅ Vol regime improved (ATR% + smoothing)
# ✅ Extension signal dynamic (threshold adapts to ATR%)
# ✅ Momentum upgraded (EMA slope + MACD + RSI + RVOL + ADX + OBV + BB chop)
# ✅ Composite momentum upgraded (0-100 + label + rich components)
#
# NEW (Desk filters to reduce over-signals):
# ✅ ema_trend_bias(): EMA20/50 bias with slope+spread threshold (anti-range)
# ✅ is_choppy_market(): BB width + ADX chop detector
# ✅ desk_momentum_gate(): boolean gate per direction + reason
#
# Backward compatibility:
# - compute_ote keeps keys: in_ote, ote_low, ote_high
# - atr/compute_atr/true_atr kept
# - institutional_momentum returns same labels
# - composite_momentum returns {score,label,components}
# =====================================================================

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Optional settings overrides
try:
    from settings import VOL_REGIME_ATR_PCT_LOW, VOL_REGIME_ATR_PCT_HIGH
except Exception:
    VOL_REGIME_ATR_PCT_LOW = 0.015
    VOL_REGIME_ATR_PCT_HIGH = 0.035


# =====================================================================
# Internal helpers
# =====================================================================

def _has_cols(df: pd.DataFrame, cols: Tuple[str, ...]) -> bool:
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return False
        return all(c in df.columns for c in cols)
    except Exception:
        return False


def _shape_series(df: Optional[pd.DataFrame], fill: float, *, name: Optional[str] = None) -> pd.Series:
    """
    Returns a Series aligned on df.index (shape-safe).
    If df is None/empty -> returns length-1 series.
    """
    if isinstance(df, pd.DataFrame) and not df.empty:
        s = pd.Series(np.full(len(df), float(fill), dtype=float), index=df.index)
        if name:
            s.name = name
        return s
    s = pd.Series([float(fill)], dtype=float)
    if name:
        s.name = name
    return s


def _to_close_series(x: Any) -> pd.Series:
    """Accepts DataFrame(OHLCV) with 'close' or a Series."""
    if isinstance(x, pd.Series):
        return x.astype(float)
    if isinstance(x, pd.DataFrame):
        if "close" not in x.columns:
            raise TypeError("Expected DataFrame with 'close'")
        return x["close"].astype(float)
    raise TypeError("Expected DataFrame with 'close' or Series for price input")


def _safe_last(series: pd.Series, default: float = np.nan) -> float:
    try:
        if series is None or len(series) == 0:
            return float(default)
        v = float(series.iloc[-1])
        return v
    except Exception:
        return float(default)


def _nan_to(x: float, default: float) -> float:
    try:
        xf = float(x)
        if not np.isfinite(xf):
            return float(default)
        return xf
    except Exception:
        return float(default)


def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        x = float(x)
        lo = float(lo)
        hi = float(hi)
        if lo > hi:
            lo, hi = hi, lo
        return max(lo, min(hi, x))
    except Exception:
        return float(lo)


def _tanh_score(x: float) -> float:
    try:
        return float(np.tanh(float(x)))
    except Exception:
        return 0.0


def _linreg_slope(y: pd.Series, n: int = 12) -> float:
    """
    Linear regression slope over last n points (normalized by price level).
    Returns small number: positive uptrend, negative downtrend.
    """
    try:
        if y is None or len(y) < max(5, n):
            return 0.0
        w = y.tail(n).astype(float).values
        if w.size < 5:
            return 0.0
        x = np.arange(len(w), dtype=float)
        x = x - x.mean()
        w = w - w.mean()
        denom = float(np.sum(x * x))
        if denom <= 0:
            return 0.0
        slope = float(np.sum(x * w) / denom)
        level = float(abs(y.iloc[-1]))
        if level <= 1e-12:
            level = 1.0
        return float(slope / level)
    except Exception:
        return 0.0


# =====================================================================
# Institutional MAX++ hardening (optional, backward-compatible)
# =====================================================================
# These features are OFF by default to preserve legacy behavior.
# Enable via env vars when you want “desk/institutional” strictness.
import os
import time
from datetime import datetime, timezone

def _env_flag(name: str, default: str = "0") -> bool:
    try:
        return str(os.getenv(name, default)).strip() == "1"
    except Exception:
        return default == "1"

def _env_float(name: str, default: str) -> float:
    try:
        return float(str(os.getenv(name, default)).strip())
    except Exception:
        return float(default)

# If enabled, we drop the last candle when it looks like an “incomplete/live” bar.
# This reduces false ADX/squeeze/ATR% signals on streaming data.
IND_DROP_LIVE_BAR = _env_flag("IND_DROP_LIVE_BAR", "0")
IND_LIVE_BAR_GRACE = _env_float("IND_LIVE_BAR_GRACE", "0.55")  # fraction of bar duration

def _to_utc_ts_seconds(ts) -> Optional[float]:
    try:
        if ts is None:
            return None
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts.timestamp()
        return None
    except Exception:
        return None

def _infer_bar_seconds(idx: pd.Index) -> Optional[float]:
    try:
        if idx is None or len(idx) < 10:
            return None
        if not isinstance(idx, (pd.DatetimeIndex,)):
            return None
        # Use median of last deltas for robustness
        deltas = (idx[-10:].to_series().diff().dropna()).dt.total_seconds().values
        if deltas is None or len(deltas) < 3:
            return None
        med = float(np.nanmedian(deltas))
        return med if np.isfinite(med) and med > 0 else None
    except Exception:
        return None

def _maybe_drop_live_bar_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    try:
        if not IND_DROP_LIVE_BAR:
            return df
        if df is None or not isinstance(df, pd.DataFrame) or df.empty or len(df) < 5:
            return df
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            return df
        bar_s = _infer_bar_seconds(idx)
        if bar_s is None or bar_s <= 0:
            return df
        last_ts = _to_utc_ts_seconds(idx[-1])
        if last_ts is None:
            return df
        now_s = time.time()
        age_s = max(0.0, now_s - float(last_ts))
        # If last bar timestamp is “too recent” relative to bar duration, treat it as live.
        if age_s < float(IND_LIVE_BAR_GRACE) * float(bar_s):
            return df.iloc[:-1].copy()
        return df
    except Exception:
        return df

def _efficiency_ratio(close: pd.Series, n: int = 20) -> float:
    """Kaufman Efficiency Ratio: trendiness vs chop. 0..1."""
    try:
        c = pd.to_numeric(close, errors="coerce").astype(float)
        if c is None or len(c) < n + 2:
            return 0.0
        win = c.tail(n + 1)
        net = abs(float(win.iloc[-1] - win.iloc[0]))
        den = float(win.diff().abs().sum())
        if den <= 0:
            return 0.0
        v = net / den
        return float(v) if np.isfinite(v) else 0.0
    except Exception:
        return 0.0

def _realized_vol_ewma(close: pd.Series, span: int = 32) -> float:
    """EWMA of log returns std, expressed as a fraction (e.g., 0.02 = 2%)."""
    try:
        c = pd.to_numeric(close, errors="coerce").astype(float)
        if c is None or len(c) < span + 5:
            return 0.0
        r = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
        if r.empty:
            return 0.0
        var = r.ewm(span=int(span), adjust=False).var(bias=False)
        v = float(np.sqrt(max(0.0, float(var.iloc[-1]))))
        return float(v) if np.isfinite(v) else 0.0
    except Exception:
        return 0.0

# =====================================================================
# MA / EMA
# =====================================================================

def ema(series_or_df: Any, length: int = 20) -> pd.Series:
    c = _to_close_series(series_or_df)
    return c.ewm(span=int(length), adjust=False).mean()


def sma(series_or_df: Any, length: int = 20) -> pd.Series:
    c = _to_close_series(series_or_df)
    return c.rolling(window=int(length), min_periods=1).mean()


# =====================================================================
# EMA trend (Desk anti-range)
# =====================================================================

def ema_trend_bias(
    df: pd.DataFrame,
    fast: int = 20,
    slow: int = 50,
    *,
    spread_min_pct: float = 0.0008,  # 0.08%
    slope_min: float = 0.0006,
) -> str:
    """
    Returns: LONG / SHORT / RANGE
    Desk-style: requires EMA stack + slope + min spread percent.
    """
    try:
        df = _maybe_drop_live_bar_df(df)
        if df is None or df.empty or "close" not in df.columns or len(df) < slow + 8:
            return "RANGE"
        c = df["close"].astype(float)
        ef = c.ewm(span=int(fast), adjust=False).mean()
        es = c.ewm(span=int(slow), adjust=False).mean()

        slope = _linreg_slope(ef, n=12)
        spread = float(ef.iloc[-1] - es.iloc[-1])
        level = float(abs(es.iloc[-1])) if abs(float(es.iloc[-1])) > 1e-12 else 1.0
        spread_pct = spread / level

        if (ef.iloc[-1] > es.iloc[-1]) and (slope > slope_min) and (spread_pct > spread_min_pct):
            return "LONG"
        if (ef.iloc[-1] < es.iloc[-1]) and (slope < -slope_min) and (spread_pct < -spread_min_pct):
            return "SHORT"
        return "RANGE"
    except Exception:
        return "RANGE"


# =====================================================================
# RSI
# =====================================================================

def rsi(series_or_df: Any, length: int = 14) -> pd.Series:
    c = _to_close_series(series_or_df)
    length = int(max(2, length))
    delta = c.diff()

    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain_series = pd.Series(gain, index=c.index)
    loss_series = pd.Series(loss, index=c.index)

    avg_gain = gain_series.ewm(alpha=1.0 / length, adjust=False).mean()
    avg_loss = loss_series.ewm(alpha=1.0 / length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.replace([np.inf, -np.inf], np.nan).fillna(50.0)


# =====================================================================
# MACD
# =====================================================================

def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    if df is None or df.empty or "close" not in df.columns:
        return pd.DataFrame({"macd": [], "signal": [], "hist": []})

    close = df["close"].astype(float)
    ema_fast = close.ewm(span=int(fast), adjust=False).mean()
    ema_slow = close.ewm(span=int(slow), adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=int(signal), adjust=False).mean()
    hist = macd_line - signal_line

    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist}, index=df.index)


# =====================================================================
# ATR / true_atr (stable, shape-safe)
# =====================================================================

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    df = _maybe_drop_live_bar_df(df)
    if df is None or df.empty:
        return _shape_series(df, 0.0, name="atr")

    if not _has_cols(df, ("high", "low", "close")):
        return _shape_series(df, 0.0, name="atr")

    length = int(max(2, length))
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    out = tr.rolling(window=length, min_periods=1).mean()
    out = out.bfill().fillna(0.0)
    out.name = "atr"
    return out


def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    df = _maybe_drop_live_bar_df(df)
    return atr(df, length=length)


def true_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    df = _maybe_drop_live_bar_df(df)
    return atr(df, length=length)


def atr_pct(df: pd.DataFrame, length: int = 14) -> pd.Series:
    df = _maybe_drop_live_bar_df(df)
    if df is None or df.empty:
        return _shape_series(df, np.nan, name="atr_pct")
    if not _has_cols(df, ("close", "high", "low")):
        return _shape_series(df, np.nan, name="atr_pct")
    a = atr(df, length=int(length))
    c = df["close"].astype(float).replace(0.0, np.nan)
    out = (a / c).replace([np.inf, -np.inf], np.nan)
    out.name = "atr_pct"
    return out


# =====================================================================
# ADX (trend strength) — shape-safe
# =====================================================================

def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    df = _maybe_drop_live_bar_df(df)
    """
    ADX (Average Directional Index) — trend strength [0..100]
    Uses Wilder smoothing (EMA alpha=1/length).
    """
    if df is None or df.empty or (not _has_cols(df, ("high", "low", "close"))) or len(df) < max(20, length + 5):
        return _shape_series(df, np.nan, name="adx")

    length = int(max(2, length))

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    up = high.diff()
    down = -low.diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_w = tr.ewm(alpha=1.0 / length, adjust=False).mean().replace(0.0, np.nan)

    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / length, adjust=False).mean() / atr_w
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / length, adjust=False).mean() / atr_w

    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    out = dx.ewm(alpha=1.0 / length, adjust=False).mean()
    out = out.clip(lower=0.0, upper=100.0)
    out.name = "adx"
    return out


# =====================================================================
# Volume tools (RVOL / OBV) — shape-safe
# =====================================================================

def rel_volume(df: pd.DataFrame, lookback: int = 40) -> float:
    df = _maybe_drop_live_bar_df(df)
    if not _has_cols(df, ("volume",)):
        return 1.0
    v = df["volume"].astype(float)
    w = v.tail(int(max(10, lookback)))
    avg = float(w.mean()) if len(w) else 0.0
    last = float(v.iloc[-1]) if len(v) else 0.0
    if avg <= 0:
        return 1.0
    return float(last / avg)


def obv(df: pd.DataFrame) -> pd.Series:
    df = _maybe_drop_live_bar_df(df)
    if df is None or df.empty:
        return _shape_series(df, 0.0, name="obv")
    if not _has_cols(df, ("close", "volume")):
        return _shape_series(df, 0.0, name="obv")

    close = df["close"].astype(float)
    vol = df["volume"].astype(float)
    direction = np.sign(close.diff().fillna(0.0))
    out = (direction * vol).fillna(0.0).cumsum()
    out.name = "obv"
    return out


# =====================================================================
# Bollinger tools (optional but useful for chop/squeeze)
# =====================================================================

def bollinger(df: pd.DataFrame, length: int = 20, n_std: float = 2.0) -> Dict[str, pd.Series]:
    df = _maybe_drop_live_bar_df(df)
    if df is None or df.empty or "close" not in df.columns:
        z = _shape_series(df, np.nan)
        return {"mid": z, "upper": z, "lower": z, "width": z}

    c = _to_close_series(df)
    length = int(max(5, length))
    ma = c.rolling(length, min_periods=1).mean()
    sd = c.rolling(length, min_periods=1).std(ddof=0).fillna(0.0)
    upper = ma + float(n_std) * sd
    lower = ma - float(n_std) * sd
    width = (upper - lower) / ma.replace(0.0, np.nan)
    return {"mid": ma, "upper": upper, "lower": lower, "width": width.replace([np.inf, -np.inf], np.nan)}


def is_choppy_market(df: pd.DataFrame, *, bb_len: int = 20, adx_len: int = 14) -> bool:
    df = _maybe_drop_live_bar_df(df)
    """Desk chop filter (Institutional MAX++).

    Signals “chop/range” when:
    - BB bandwidth is tight AND ADX is weak, OR
    - Efficiency Ratio (Kaufman) indicates low directional efficiency.
    """
    try:
        if df is None or df.empty or len(df) < 80 or not _has_cols(df, ("close", "high", "low")):
            return True

        bb = bollinger(df, bb_len, 2.0)
        bb_width = _nan_to(_safe_last(bb.get("width"), np.nan), 0.06)
        a = _nan_to(_safe_last(adx(df, adx_len), np.nan), 18.0)

        er = _efficiency_ratio(df["close"].astype(float), n=20)

        # Institutional thresholds (conservative):
        tight = (bb_width < 0.028)
        very_tight = (bb_width < 0.020)
        weak_trend = (a < 16.0)

        # ER: <0.22 tends to be “mean-reverting / noisy” on many intraday series
        er_chop = (er > 0 and er < 0.22)

        if very_tight:
            return True
        if tight and weak_trend:
            return True
        if er_chop and (weak_trend or tight):
            return True

        return False
    except Exception:
        return False

def compute_ote(df: pd.DataFrame, bias: str, lookback: int = 60) -> Dict[str, Any]:
    df = _maybe_drop_live_bar_df(df)
    """
    Desk-grade OTE approximation for LIMIT entries.

    Output keys (compat with analyze_signal.py):
      - entry (zone mid)
      - in_zone (bool)
      - ok (alias)
      - dist (0 if in zone else distance to nearest bound)
      - ote_low / ote_high
      - in_ote (legacy alias)
    """
    if df is None or df.empty or len(df) < 20 or not _has_cols(df, ("high", "low", "close")):
        return {"in_zone": False, "ok": False, "in_ote": False, "ote_low": None, "ote_high": None, "entry": None}

    b = (bias or "").upper()
    if b not in ("LONG", "SHORT"):
        b = "LONG"

    sub = df.tail(int(max(30, lookback)))
    high = float(sub["high"].max())
    low = float(sub["low"].min())
    last = float(sub["close"].iloc[-1])

    if not np.isfinite(high) or not np.isfinite(low) or high <= low:
        return {"in_zone": False, "ok": False, "in_ote": False, "ote_low": None, "ote_high": None, "entry": None}

    diff = high - low

    if b == "LONG":
        fib_62 = high - 0.62 * diff
        fib_705 = high - 0.705 * diff
    else:
        fib_62 = low + 0.62 * diff
        fib_705 = low + 0.705 * diff

    ote_low = float(min(fib_62, fib_705))
    ote_high = float(max(fib_62, fib_705))
    entry = float((ote_low + ote_high) / 2.0)

    in_zone = bool(ote_low <= last <= ote_high)

    dist = 0.0 if in_zone else (float(ote_low - last) if last < ote_low else float(last - ote_high))

    return {
        "entry": entry,
        "entry_price": entry,
        "in_zone": in_zone,
        "ok": in_zone,
        "dist": dist,
        "in_ote": in_zone,
        "ote_low": ote_low,
        "ote_high": ote_high,
        "swing_high": high,
        "swing_low": low,
        "last": last,
        "bias": b,
    }


# =====================================================================
# Volatility regime (ATR% with smoothing)
# =====================================================================

def volatility_regime(df: pd.DataFrame, atr_length: int = 14) -> str:
    df = _maybe_drop_live_bar_df(df)
    """Volatility regime classifier (LOW/MEDIUM/HIGH).

    Institutional MAX++:
    - Uses ATR% (robust median) as primary signal
    - Adds EWMA realized vol (log-returns) as a cross-check
    - Blends static thresholds (settings) with adaptive quantiles for symbol-specific behavior
    """
    try:
        if df is None or df.empty or len(df) < int(atr_length) + 40 or not _has_cols(df, ("high", "low", "close")):
            return "UNKNOWN"

        ap = atr_pct(df, length=int(atr_length))
        if ap is None or len(ap) < 20:
            return "UNKNOWN"

        ap_tail = pd.to_numeric(ap.tail(10), errors="coerce").astype(float)
        last_atr = float(np.nanmedian(ap_tail.values)) if len(ap_tail) else np.nan
        if not np.isfinite(last_atr) or last_atr <= 0:
            return "UNKNOWN"

        close = pd.to_numeric(df["close"], errors="coerce").astype(float)
        rv = float(_realized_vol_ewma(close, span=32))
        # rv is log-return std (fraction). Convert to a comparable “pct move” proxy.
        rv_proxy = float(max(0.0, rv))

        # Adaptive thresholds (quantile-based), fallback to settings
        q_low = q_high = None
        try:
            if len(ap) >= 220:
                base = pd.to_numeric(ap.tail(220), errors="coerce").astype(float).dropna()
                if len(base) >= 120:
                    q_low = float(base.quantile(0.25))
                    q_high = float(base.quantile(0.75))
        except Exception:
            q_low = q_high = None

        low_thr = float(VOL_REGIME_ATR_PCT_LOW)
        high_thr = float(VOL_REGIME_ATR_PCT_HIGH)

        if q_low is not None and np.isfinite(q_low) and q_low > 0:
            low_thr = max(low_thr, 0.85 * float(q_low))
        if q_high is not None and np.isfinite(q_high) and q_high > 0:
            high_thr = max(high_thr, 1.10 * float(q_high))

        # Combine: ATR% dominates, RV nudges classification upward if consistently elevated
        combo = max(float(last_atr), float(rv_proxy) * 1.25)

        if combo < low_thr:
            return "LOW"
        if combo > high_thr:
            return "HIGH"
        return "MEDIUM"
    except Exception:
        return "UNKNOWN"

def extension_signal(df: pd.DataFrame, ema_fast_len: int = 20, ema_slow_len: int = 50) -> Optional[str]:
    df = _maybe_drop_live_bar_df(df)
    """
    Overextension detection:
      - distance vs EMA slow (dynamic threshold)
      - RSI extreme
    """
    if df is None or df.empty or len(df) < max(int(ema_slow_len), 40) or not _has_cols(df, ("close", "high", "low")):
        return None

    close = df["close"].astype(float)
    ema_slow = ema(close, int(ema_slow_len))
    r = rsi(close, 14)

    last_close = float(close.iloc[-1])
    last_ema = float(ema_slow.iloc[-1])
    last_rsi = float(r.iloc[-1])

    if not np.isfinite(last_close) or not np.isfinite(last_ema) or last_ema <= 0:
        return None

    dist_pct = float((last_close - last_ema) / last_ema)

    ap = _nan_to(_safe_last(atr_pct(df, 14), np.nan), 0.02)
    thr = _clamp(2.5 * ap, 0.045, 0.12)  # desk: ~4.5%..12%

    if dist_pct > thr and last_rsi > 70:
        return "OVEREXTENDED_LONG"
    if dist_pct < -thr and last_rsi < 30:
        return "OVEREXTENDED_SHORT"
    return None


# =====================================================================
# Institutional momentum (desk-grade)
# =====================================================================

def institutional_momentum(df: pd.DataFrame) -> str:
    df = _maybe_drop_live_bar_df(df)
    """
    Returns:
      STRONG_BULLISH / BULLISH / NEUTRAL / BEARISH / STRONG_BEARISH
    """
    if df is None or df.empty or len(df) < 80 or not _has_cols(df, ("close", "volume", "high", "low")):
        return "NEUTRAL"

    close = df["close"].astype(float)

    e20 = ema(close, 20)
    e50 = ema(close, 50)
    spread_last = float((e20 - e50).iloc[-1])

    e20_slope = _linreg_slope(e20, n=12)

    m = macd(df)
    macd_last = float(m["macd"].iloc[-1]) if "macd" in m and len(m) else 0.0
    hist_last = float(m["hist"].iloc[-1]) if "hist" in m and len(m) else 0.0
    hist_slope = _linreg_slope(m["hist"], n=10) if "hist" in m and len(m) >= 20 else 0.0

    r = rsi(close, 14)
    r_last = float(r.iloc[-1])

    rv = rel_volume(df, 40)
    adx_last = _nan_to(_safe_last(adx(df, 14), np.nan), 18.0)
    obv_slope = _linreg_slope(obv(df), n=20)

    score = 0.0

    score += 1.0 if spread_last > 0 else (-1.0 if spread_last < 0 else 0.0)
    score += 0.5 if e20_slope > 0.0006 else (-0.5 if e20_slope < -0.0006 else 0.0)

    if macd_last > 0 and hist_last > 0:
        score += 1.0
    elif macd_last < 0 and hist_last < 0:
        score -= 1.0

    score += 0.25 if hist_slope > 0.0005 else (-0.25 if hist_slope < -0.0005 else 0.0)

    score += 0.5 if r_last > 60 else (-0.5 if r_last < 40 else 0.0)
    score += 0.5 if rv > 1.6 else (-0.35 if rv < 0.7 else 0.0)
    score += 0.35 if adx_last >= 28 else (-0.35 if adx_last <= 14 else 0.0)
    score += 0.25 if obv_slope > 0.0005 else (-0.25 if obv_slope < -0.0005 else 0.0)

    if score >= 2.2:
        return "STRONG_BULLISH"
    if score >= 0.8:
        return "BULLISH"
    if score <= -2.2:
        return "STRONG_BEARISH"
    if score <= -0.8:
        return "BEARISH"
    return "NEUTRAL"


# =====================================================================
# Composite momentum (0..100 + label + components)
# =====================================================================

def composite_momentum(df: pd.DataFrame) -> Dict[str, Any]:
    df = _maybe_drop_live_bar_df(df)
    """
    Desk composite score in [0,100] with richer components.

    Labels include STRONG_* to match analyze_signal.py filters.
    """
    if df is None or df.empty or len(df) < 80 or not _has_cols(df, ("close", "volume", "high", "low")):
        return {"score": 50.0, "label": "NEUTRAL", "components": {}}

    close = df["close"].astype(float)

    e20 = ema(close, 20)
    e50 = ema(close, 50)
    spread_last = float((e20 - e50).iloc[-1])

    trend_dir = 1.0 if spread_last > 0 else (-1.0 if spread_last < 0 else 0.0)
    trend_slope = _linreg_slope(e20, n=12)

    m = macd(df)
    macd_last = float(m["macd"].iloc[-1])
    hist_last = float(m["hist"].iloc[-1])
    macd_score = _tanh_score(macd_last + 0.6 * hist_last)

    r = rsi(close, 14)
    r_last = float(r.iloc[-1])
    rsi_score = float(np.clip((r_last - 50.0) / 20.0, -2.0, 2.0))

    rv = rel_volume(df, 40)
    rvol_score = float(np.clip((rv - 1.0) / 0.8, -1.0, 1.0))

    adx_last = _nan_to(_safe_last(adx(df, 14), np.nan), 18.0)
    adx_score = float(np.clip((adx_last - 20.0) / 15.0, -1.0, 1.0))

    obv_s = obv(df)
    obv_slope = _linreg_slope(obv_s, n=20)
    obv_flow = float(np.clip(obv_slope / 0.0015, -1.0, 1.0))

    bb = bollinger(df, 20, 2.0)
    bb_width_last = _nan_to(_safe_last(bb["width"], np.nan), 0.05)

    chop_penalty = 0.0
    if bb_width_last < 0.03 and adx_last < 16:
        chop_penalty = -0.8
    elif bb_width_last < 0.02:
        chop_penalty = -0.6

    ext = extension_signal(df)
    ext_penalty = -0.7 if ext in ("OVEREXTENDED_LONG", "OVEREXTENDED_SHORT") else 0.0

    raw = (
        1.00 * trend_dir
        + 1.10 * (trend_slope / 0.002)
        + 0.95 * macd_score
        + 0.85 * (rsi_score / 2.0)
        + 0.60 * rvol_score
        + 0.70 * adx_score
        + 0.55 * obv_flow
        + 1.00 * chop_penalty
        + 1.00 * ext_penalty
    )

    score = 50.0 + 45.0 * _tanh_score(raw)
    score = float(np.clip(score, 0.0, 100.0))

    if score >= 86:
        label = "STRONG_BULLISH"
    elif score >= 72:
        label = "BULLISH"
    elif score >= 56:
        label = "SLIGHT_BULLISH"
    elif score <= 14:
        label = "STRONG_BEARISH"
    elif score <= 28:
        label = "BEARISH"
    elif score <= 44:
        label = "SLIGHT_BEARISH"
    else:
        label = "NEUTRAL"

    return {
        "score": score,
        "label": label,
        "components": {
            "trend_dir": trend_dir,
            "trend_slope": float(trend_slope),
            "macd_score": float(macd_score),
            "rsi": float(r_last),
            "rsi_score": float(rsi_score),
            "rvol": float(rv),
            "rvol_score": float(rvol_score),
            "adx": float(adx_last),
            "adx_score": float(adx_score),
            "obv_flow": float(obv_flow),
            "bb_width": float(bb_width_last),
            "chop_penalty": float(chop_penalty),
            "ext": ext,
            "ext_penalty": float(ext_penalty),
            "raw": float(raw),
        },
    }


# =====================================================================
# Desk gate to reduce signals (simple + effective)
# =====================================================================

def desk_momentum_gate(df: pd.DataFrame, direction: str) -> Dict[str, Any]:
    df = _maybe_drop_live_bar_df(df)
    """
    Returns:
      {"ok": bool, "reason": str, "label": str, "score": float, "chop": bool, "ext": str|None, "ema_bias": str}

    Use this in analyze_signal to harden entries and reduce spam.
    """
    d = (direction or "").upper()
    if d not in ("LONG", "SHORT"):
        d = "LONG"

    ema_bias = ema_trend_bias(df, 20, 50)
    chop = is_choppy_market(df)
    ext = extension_signal(df)

    cm = composite_momentum(df)
    label = str(cm.get("label") or "NEUTRAL")
    score = float(cm.get("score") or 50.0)

    # Hard rules (desk)
    if chop:
        return {"ok": False, "reason": "choppy_market", "label": label, "score": score, "chop": chop, "ext": ext, "ema_bias": ema_bias}

    if d == "LONG" and ext == "OVEREXTENDED_LONG":
        return {"ok": False, "reason": "overextended_long", "label": label, "score": score, "chop": chop, "ext": ext, "ema_bias": ema_bias}
    if d == "SHORT" and ext == "OVEREXTENDED_SHORT":
        return {"ok": False, "reason": "overextended_short", "label": label, "score": score, "chop": chop, "ext": ext, "ema_bias": ema_bias}

    # Trend alignment (optional hardening)
    if ema_bias == "RANGE":
        return {"ok": False, "reason": "ema_range", "label": label, "score": score, "chop": chop, "ext": ext, "ema_bias": ema_bias}
    if ema_bias == "LONG" and d == "SHORT":
        return {"ok": False, "reason": "ema_bias_veto", "label": label, "score": score, "chop": chop, "ext": ext, "ema_bias": ema_bias}
    if ema_bias == "SHORT" and d == "LONG":
        return {"ok": False, "reason": "ema_bias_veto", "label": label, "score": score, "chop": chop, "ext": ext, "ema_bias": ema_bias}

    # Momentum threshold
    if d == "LONG" and label not in ("BULLISH", "STRONG_BULLISH"):
        return {"ok": False, "reason": "momentum_not_bullish", "label": label, "score": score, "chop": chop, "ext": ext, "ema_bias": ema_bias}
    if d == "SHORT" and label not in ("BEARISH", "STRONG_BEARISH"):
        return {"ok": False, "reason": "momentum_not_bearish", "label": label, "score": score, "chop": chop, "ext": ext, "ema_bias": ema_bias}

    return {"ok": True, "reason": "ok", "label": label, "score": score, "chop": chop, "ext": ext, "ema_bias": ema_bias}
