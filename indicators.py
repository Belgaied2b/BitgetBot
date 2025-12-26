# =====================================================================
# indicators.py — Core + Institutional Indicators (Desk-lead, Hardened)
# =====================================================================
# ✅ Robust OHLCV checks (fail-safe, no crashes)
# ✅ ATR / true_atr stable (used by stops/tp)
# ✅ OTE upgraded for analyze_signal.py compatibility (entry + in_zone + dist)
# ✅ Vol regime improved (ATR% + smoothing)
# ✅ Extension signal dynamic (threshold adapts to ATR%)
# ✅ Momentum upgraded (EMA slope + MACD + RSI + RVOL + ADX + OBV)
# ✅ Composite momentum upgraded (0-100 + label + rich components)
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


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, float(x))))


def _nan_to(x: float, default: float) -> float:
    try:
        if not np.isfinite(float(x)):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _tanh_score(x: float) -> float:
    # stable tanh mapping
    try:
        return float(np.tanh(float(x)))
    except Exception:
        return 0.0


def _linreg_slope(y: pd.Series, n: int = 12) -> float:
    """
    Linear regression slope over last n points (normalized by price level).
    Returns ~ small number, positive uptrend, negative downtrend.
    """
    try:
        if y is None or len(y) < max(5, n):
            return 0.0
        w = y.tail(n).astype(float).values
        x = np.arange(len(w), dtype=float)
        x = x - x.mean()
        w = w - w.mean()
        denom = float(np.sum(x * x))
        if denom <= 0:
            return 0.0
        slope = float(np.sum(x * w) / denom)
        level = float(abs(y.iloc[-1])) if float(abs(y.iloc[-1])) > 1e-12 else 1.0
        return float(slope / level)
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

    rs = avg_gain / loss_series.replace(0.0, np.nan).ewm(alpha=1.0 / length, adjust=False).mean()
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
# ATR / true_atr (stable)
# =====================================================================

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    if not _has_cols(df, ("high", "low", "close")):
        return pd.Series([0.0])

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
    return out.bfill().fillna(0.0)


def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return atr(df, length=length)


def true_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return atr(df, length=length)


def atr_pct(df: pd.DataFrame, length: int = 14) -> pd.Series:
    if not _has_cols(df, ("close", "high", "low")):
        return pd.Series([np.nan])
    a = atr(df, length=length)
    c = df["close"].astype(float).replace(0.0, np.nan)
    return (a / c).replace([np.inf, -np.inf], np.nan)


# =====================================================================
# ADX (trend strength)
# =====================================================================

def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    ADX (Average Directional Index) — trend strength [0..100]
    Uses Wilder smoothing (EMA alpha=1/length).
    """
    if not _has_cols(df, ("high", "low", "close")) or len(df) < max(20, length + 5):
        return pd.Series([np.nan])

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
    return out.clip(lower=0.0, upper=100.0)


# =====================================================================
# Volume tools (RVOL / OBV)
# =====================================================================

def rel_volume(df: pd.DataFrame, lookback: int = 40) -> float:
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
    if not _has_cols(df, ("close", "volume")):
        return pd.Series([0.0])
    close = df["close"].astype(float)
    vol = df["volume"].astype(float)
    direction = np.sign(close.diff().fillna(0.0))
    out = (direction * vol).fillna(0.0).cumsum()
    return out


# =====================================================================
# Bollinger tools (optional but useful for chop/squeeze)
# =====================================================================

def bollinger(df: pd.DataFrame, length: int = 20, n_std: float = 2.0) -> Dict[str, pd.Series]:
    c = _to_close_series(df)
    length = int(max(5, length))
    ma = c.rolling(length, min_periods=1).mean()
    sd = c.rolling(length, min_periods=1).std(ddof=0).fillna(0.0)
    upper = ma + float(n_std) * sd
    lower = ma - float(n_std) * sd
    width = (upper - lower) / ma.replace(0.0, np.nan)
    return {"mid": ma, "upper": upper, "lower": lower, "width": width.replace([np.inf, -np.inf], np.nan)}


# =====================================================================
# OTE (Optimal Trade Entry) — Desk-grade
# =====================================================================

def compute_ote(df: pd.DataFrame, bias: str, lookback: int = 60) -> Dict[str, Any]:
    """
    Desk-grade OTE approximation for LIMIT entries.

    We use swing range over lookback:
      range = [low, high]

    LONG:
      - assume impulse from low -> high
      - OTE zone = retracement from high: 0.62..0.705 of the range
      - buy inside zone (discount)

    SHORT:
      - assume impulse from high -> low
      - OTE zone = retracement from low: 0.62..0.705 of the range
      - sell inside zone (premium)

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
        # retracement from high
        fib_62 = high - 0.62 * diff
        fib_705 = high - 0.705 * diff
    else:
        # retracement from low (after dump)
        fib_62 = low + 0.62 * diff
        fib_705 = low + 0.705 * diff

    ote_low = float(min(fib_62, fib_705))
    ote_high = float(max(fib_62, fib_705))
    entry = float((ote_low + ote_high) / 2.0)

    in_zone = bool(ote_low <= last <= ote_high)

    if in_zone:
        dist = 0.0
    else:
        # distance to nearest bound
        if last < ote_low:
            dist = float(ote_low - last)
        else:
            dist = float(last - ote_high)

    return {
        # analyzer-friendly
        "entry": entry,
        "entry_price": entry,
        "in_zone": in_zone,
        "ok": in_zone,
        "dist": dist,
        # legacy keys
        "in_ote": in_zone,
        "ote_low": ote_low,
        "ote_high": ote_high,
        # useful debug
        "swing_high": high,
        "swing_low": low,
        "last": last,
        "bias": b,
    }


# =====================================================================
# Volatility regime (ATR% with smoothing)
# =====================================================================

def volatility_regime(df: pd.DataFrame, atr_length: int = 14) -> str:
    """
    Uses smoothed ATR% over last bars to reduce noise.
    """
    if df is None or df.empty or len(df) < int(atr_length) + 10:
        return "UNKNOWN"

    ap = atr_pct(df, length=int(atr_length))
    if ap is None or len(ap) < 5:
        return "UNKNOWN"

    last = float(ap.tail(5).mean())  # smooth
    if not np.isfinite(last):
        return "UNKNOWN"

    if last < float(VOL_REGIME_ATR_PCT_LOW):
        return "LOW"
    if last > float(VOL_REGIME_ATR_PCT_HIGH):
        return "HIGH"
    return "MEDIUM"


# =====================================================================
# Extension signal (dynamic thresholds vs ATR%)
# =====================================================================

def extension_signal(df: pd.DataFrame, ema_fast_len: int = 20, ema_slow_len: int = 50) -> Optional[str]:
    """
    Overextension detection:
      - distance vs EMA slow (dynamic threshold)
      - RSI extreme
    """
    if df is None or df.empty or len(df) < max(int(ema_slow_len), 40) or not _has_cols(df, ("close",)):
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

    # dynamic dist threshold: bigger in high vol, smaller in low vol
    ap = _safe_last(atr_pct(df, 14), np.nan)
    ap = _nan_to(ap, 0.02)
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
    """
    Desk-grade momentum label:
      - EMA20/EMA50 direction + EMA20 slope
      - MACD direction + histogram
      - RSI location
      - RVOL (relative volume)
      - ADX trend strength (filters chop)
      - OBV slope (volume flow)

    Returns:
      STRONG_BULLISH / BULLISH / NEUTRAL / BEARISH / STRONG_BEARISH
    """
    if df is None or df.empty or len(df) < 80 or not _has_cols(df, ("close", "volume", "high", "low")):
        return "NEUTRAL"

    close = df["close"].astype(float)

    e20 = ema(close, 20)
    e50 = ema(close, 50)
    spread = e20 - e50
    spread_last = float(spread.iloc[-1])

    # slope of EMA20 (trend persistence)
    e20_slope = _linreg_slope(e20, n=12)

    m = macd(df)
    macd_last = float(m["macd"].iloc[-1]) if "macd" in m else 0.0
    hist_last = float(m["hist"].iloc[-1]) if "hist" in m else 0.0
    hist_slope = _linreg_slope(m["hist"], n=10) if "hist" in m and len(m) >= 20 else 0.0

    r = rsi(close, 14)
    r_last = float(r.iloc[-1])

    rv = rel_volume(df, 40)

    a = adx(df, 14)
    adx_last = _safe_last(a, np.nan)
    adx_last = _nan_to(adx_last, 18.0)

    obv_s = obv(df)
    obv_slope = _linreg_slope(obv_s, n=20)

    score = 0.0

    # EMA direction + slope
    if spread_last > 0:
        score += 1.0
    elif spread_last < 0:
        score -= 1.0

    if e20_slope > 0.0006:
        score += 0.5
    elif e20_slope < -0.0006:
        score -= 0.5

    # MACD + hist + hist slope
    if macd_last > 0 and hist_last > 0:
        score += 1.0
    elif macd_last < 0 and hist_last < 0:
        score -= 1.0

    if hist_slope > 0.0005:
        score += 0.25
    elif hist_slope < -0.0005:
        score -= 0.25

    # RSI
    if r_last > 60:
        score += 0.5
    elif r_last < 40:
        score -= 0.5

    # RVOL
    if rv > 1.6:
        score += 0.5
    elif rv < 0.7:
        score -= 0.35

    # ADX (avoid giving strong labels in chop)
    if adx_last >= 28:
        score += 0.35
    elif adx_last <= 14:
        score -= 0.35

    # OBV slope (volume flow confirmation)
    if obv_slope > 0.0005:
        score += 0.25
    elif obv_slope < -0.0005:
        score -= 0.25

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
    """
    Desk composite score in [0,100] with richer components.

    Components:
      - trend_dir (EMA spread)
      - trend_slope (EMA20 slope)
      - macd (tanh of macd+hist)
      - rsi (normalized)
      - rvol (relative volume)
      - adx (trend strength)
      - obv_flow (obv slope)
      - chop_penalty (bb width low + adx low)
      - ext_penalty (overextension)
    """
    if df is None or df.empty or len(df) < 80 or not _has_cols(df, ("close", "volume", "high", "low")):
        return {"score": 50.0, "label": "NEUTRAL", "components": {}}

    close = df["close"].astype(float)

    e20 = ema(close, 20)
    e50 = ema(close, 50)
    spread_last = float((e20 - e50).iloc[-1])

    trend_dir = 1.0 if spread_last > 0 else (-1.0 if spread_last < 0 else 0.0)
    trend_slope = _linreg_slope(e20, n=12)  # small

    m = macd(df)
    macd_last = float(m["macd"].iloc[-1])
    hist_last = float(m["hist"].iloc[-1])
    macd_score = _tanh_score(macd_last + 0.6 * hist_last)

    r = rsi(close, 14)
    r_last = float(r.iloc[-1])
    rsi_score = float(np.clip((r_last - 50.0) / 20.0, -2.0, 2.0))  # stronger scaling than before

    rv = rel_volume(df, 40)
    # map rvol around 1.0 into [-1..+1] with saturation
    rvol_score = float(np.clip((rv - 1.0) / 0.8, -1.0, 1.0))

    a = adx(df, 14)
    adx_last = _nan_to(_safe_last(a, np.nan), 18.0)
    # ADX score: <15 bearish for trend quality, >30 bullish (trend healthy)
    adx_score = float(np.clip((adx_last - 20.0) / 15.0, -1.0, 1.0))

    obv_s = obv(df)
    obv_slope = _linreg_slope(obv_s, n=20)
    obv_flow = float(np.clip(obv_slope / 0.0015, -1.0, 1.0))

    # Chop/squeeze detection
    bb = bollinger(df, 20, 2.0)
    bb_width_last = _safe_last(bb["width"], np.nan)
    bb_width_last = _nan_to(bb_width_last, 0.05)
    # low width + low adx => chop penalty
    chop_penalty = 0.0
    if bb_width_last < 0.03 and adx_last < 16:
        chop_penalty = -0.8
    elif bb_width_last < 0.02:
        chop_penalty = -0.6

    # Extension penalty
    ext = extension_signal(df)
    ext_penalty = -0.7 if ext in ("OVEREXTENDED_LONG", "OVEREXTENDED_SHORT") else 0.0

    # Combine (desk weights)
    raw = (
        1.00 * trend_dir
        + 1.10 * (trend_slope / 0.002)          # normalize slope into ~[-1..+1] region
        + 0.95 * macd_score
        + 0.85 * (rsi_score / 2.0)
        + 0.60 * rvol_score
        + 0.70 * adx_score
        + 0.55 * obv_flow
        + 1.00 * chop_penalty
        + 1.00 * ext_penalty
    )

    # score 0..100 via tanh
    score = 50.0 + 45.0 * _tanh_score(raw)   # wider spread (closer to desk intuition)
    score = float(np.clip(score, 0.0, 100.0))

    if score >= 72:
        label = "BULLISH"
    elif score >= 56:
        label = "SLIGHT_BULLISH"
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
