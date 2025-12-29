# =====================================================================
# structure_utils.py — Institutional Structure Engine ++ (Desk v3) [FIXED]
# BOS / CHOCH / COS / Internal vs External / Liquidity / OB / FVG / HTF
# =====================================================================
# ✅ Fix FVG zones format (low/high + direction) -> compatible avec analyze_signal._pick_fvg_entry()
# ✅ BOS détection plus "desk" (buffer ATR, internal/external propre, broken_level)
# ✅ Trend EMA20/50 + slope + spread threshold (moins de faux signaux RANGE)
# ✅ Liquidity pools (equal highs/lows) ATR-aware + clustering + recency
# ✅ BOS quality score renforcé (volume, body, impulse vs ATR, close location, sweep, OI slope)
# ✅ Fail-safe partout (pas de crash si df incomplet)
#
# API stable utilisée par analyze_signal.py :
#   - analyze_structure(df) -> dict avec keys: trend, bos, bos_direction, bos_type, fvg_zones, oi_series...
#   - htf_trend_ok(df_htf, bias) -> bool
#   - bos_quality_details(...)
#   - has_liquidity_zone(df, direction)
# =====================================================================

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd


# =====================================================================
# Helpers
# =====================================================================

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _has_cols(df: pd.DataFrame, cols: Tuple[str, ...]) -> bool:
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return False
        return all(c in df.columns for c in cols)
    except Exception:
        return False


# Try to reuse true ATR from indicators if available (preferred)
_compute_atr_ext = None
try:
    from indicators import true_atr as _compute_atr_ext  # type: ignore
except Exception:
    try:
        from indicators import compute_atr as _compute_atr_ext  # type: ignore
    except Exception:
        _compute_atr_ext = None


def _atr(df: pd.DataFrame, length: int = 14) -> float:
    """
    Returns last ATR value (float). Uses indicators.true_atr/compute_atr if available,
    otherwise uses a local ATR implementation.
    """
    try:
        if df is None or df.empty or len(df) < int(length) + 3:
            return 0.0

        if _compute_atr_ext is not None:
            # Support both signatures: (df, length=..) or (df, period=..)
            try:
                s = _compute_atr_ext(df, length=int(length))  # type: ignore
            except Exception:
                s = _compute_atr_ext(df, period=int(length))  # type: ignore
            try:
                v = float(s.iloc[-1]) if s is not None and len(s) else 0.0
                return max(0.0, v)
            except Exception:
                return 0.0

        # local ATR
        if not _has_cols(df, ("high", "low", "close")):
            return 0.0

        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)

        tr = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)

        a = tr.rolling(window=int(max(2, length)), min_periods=1).mean()
        v = float(a.iloc[-1])
        return max(0.0, v)
    except Exception:
        return 0.0


def _median_range(df: pd.DataFrame, n: int = 60) -> float:
    try:
        if not _has_cols(df, ("high", "low")) or len(df) < 10:
            return 0.0
        w = df.tail(int(max(10, n)))
        r = (w["high"].astype(float) - w["low"].astype(float)).abs()
        v = float(np.nanmedian(r.values)) if len(r) else 0.0
        return max(0.0, v)
    except Exception:
        return 0.0


# =====================================================================
# SWINGS (pivot highs / lows)
# =====================================================================

def find_swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> Dict[str, List[Tuple[int, float]]]:
    """
    Pivot swings (fractal) detection.
    Returns:
      {"highs": [(idx, price), ...], "lows": [(idx, price), ...]}
    """
    highs: List[Tuple[int, float]] = []
    lows: List[Tuple[int, float]] = []

    if df is None or len(df) < left + right + 5 or not _has_cols(df, ("high", "low")):
        return {"highs": highs, "lows": lows}

    h = df["high"].astype(float).to_numpy()
    l = df["low"].astype(float).to_numpy()

    for i in range(left, len(df) - right):
        win_h = h[i - left : i + right + 1]
        win_l = l[i - left : i + right + 1]

        hi = float(h[i])
        lo = float(l[i])

        if not np.isfinite(hi) or not np.isfinite(lo):
            continue

        # pivot high
        if hi >= float(np.max(win_h)):
            # keep last occurrence to avoid duplicates
            if i == (i - left + int(np.argmax(win_h))):
                highs.append((i, hi))

        # pivot low
        if lo <= float(np.min(win_l)):
            if i == (i - left + int(np.argmin(win_l))):
                lows.append((i, lo))

    return {"highs": highs, "lows": lows}


# =====================================================================
# LIQUIDITY (equal highs / equal lows)
# =====================================================================

def _cluster_levels(levels: List[float], tolerance: float) -> List[float]:
    """
    Groups nearby price levels into clusters (liquidity pools).
    Returns cluster representative (mean) only if >=2 touches.
    """
    if not levels:
        return []
    tolerance = abs(float(tolerance))
    if tolerance <= 0:
        return []

    lv = sorted([float(x) for x in levels if np.isfinite(float(x))])
    if not lv:
        return []

    clusters: List[List[float]] = [[lv[0]]]
    for p in lv[1:]:
        if abs(p - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(p)
        else:
            clusters.append([p])

    out: List[float] = []
    for c in clusters:
        if len(c) >= 2:
            out.append(float(np.mean(c)))
    return out


def detect_equal_levels(
    df: pd.DataFrame,
    left: int = 3,
    right: int = 3,
    max_window: int = 200,
    *,
    tol_mult_atr: float = 0.12,
    tol_mult_range: float = 0.15,
    tol_fallback_pct: float = 0.001,
) -> Dict[str, List[float]]:
    """
    Detects equal highs/lows pools from swing points.
    Tolerance is ATR-aware (preferred) with robust fallback.

    Returns:
      {"eq_highs": [float,...], "eq_lows": [float,...]}
    """
    if df is None or len(df) < left + right + 10 or not _has_cols(df, ("high", "low", "close")):
        return {"eq_highs": [], "eq_lows": []}

    sub = df.tail(int(max_window)).reset_index(drop=True)
    swings = find_swings(sub, left=left, right=right)

    high_prices = [p for _, p in swings.get("highs", [])]
    low_prices = [p for _, p in swings.get("lows", [])]

    last_price = float(sub["close"].astype(float).iloc[-1])

    a = _atr(sub, 14)
    base_range = _median_range(sub, 60)

    tol = 0.0
    if a > 0:
        tol = a * float(tol_mult_atr)
    if tol <= 0 and base_range > 0:
        tol = base_range * float(tol_mult_range)
    if tol <= 0:
        tol = max(1e-12, last_price * float(tol_fallback_pct))

    eq_highs = sorted(_cluster_levels(high_prices, tol))
    eq_lows = sorted(_cluster_levels(low_prices, tol))

    return {"eq_highs": eq_highs, "eq_lows": eq_lows}


def has_liquidity_zone(df: pd.DataFrame, direction: str, lookback: int = 200) -> bool:
    """
    Quick helper (pour SL beyond liquidity):
      - LONG  => cherche EQ_LOWS proches (liquidité sous le prix)
      - SHORT => cherche EQ_HIGHS proches (liquidité au-dessus)
    """
    try:
        if df is None or df.empty or len(df) < 40 or not _has_cols(df, ("close",)):
            return False

        d = (direction or "").upper()
        lv = detect_equal_levels(df.tail(int(lookback)))
        close = float(df["close"].astype(float).iloc[-1])
        a = _atr(df, 14)
        buf = max(a * 0.15, close * 0.0008)

        if d == "LONG":
            for x in lv.get("eq_lows", []) or []:
                if abs(float(x) - close) <= buf:
                    return True

        if d == "SHORT":
            for x in lv.get("eq_highs", []) or []:
                if abs(float(x) - close) <= buf:
                    return True

        return False
    except Exception:
        return False


def liquidity_sweep_details(
    df: pd.DataFrame,
    direction: str,
    lookback: int = 160,
    tol_mult_atr: float = 0.12,
    wick_ratio_min: float = 0.55,
) -> Dict[str, Any]:
    """
    Stop hunt / liquidity sweep (desk-style).

    - LONG  : wick sous un EQ_LOW puis close au-dessus
    - SHORT : wick au-dessus d'un EQ_HIGH puis close en-dessous

    Returns dict:
      {
        "ok": bool,
        "direction": "LONG"/"SHORT",
        "level": float|None,
        "sweep_extreme": float|None,
        "reclaim_close": float|None,
        "wick_ratio": float,
        "body_ratio": float,
        "kind": "EQ_LOW_SWEEP"|"EQ_HIGH_SWEEP"|None
      }
    """
    out: Dict[str, Any] = {
        "ok": False,
        "direction": str(direction).upper(),
        "level": None,
        "sweep_extreme": None,
        "reclaim_close": None,
        "wick_ratio": 0.0,
        "body_ratio": 0.0,
        "kind": None,
    }

    if df is None or getattr(df, "empty", True) or len(df) < 60 or not _has_cols(df, ("open", "high", "low", "close")):
        return out

    d = str(direction).upper()
    if d not in {"LONG", "SHORT"}:
        return out

    dfw = df.tail(int(max(80, lookback))).copy()

    atr_last = _atr(dfw, 14)
    tol = float(tol_mult_atr) * max(atr_last, 1e-12)

    eq = detect_equal_levels(dfw.tail(200), tol_mult_atr=float(tol_mult_atr))
    eq_highs = eq.get("eq_highs", []) or []
    eq_lows = eq.get("eq_lows", []) or []

    last = dfw.iloc[-1]
    o = float(last["open"])
    h = float(last["high"])
    l = float(last["low"])
    c = float(last["close"])
    rng = max(h - l, 1e-12)
    body = abs(c - o)

    out["reclaim_close"] = c
    out["body_ratio"] = float(body / rng)

    lower_wick = max(min(o, c) - l, 0.0)
    upper_wick = max(h - max(o, c), 0.0)

    if d == "LONG":
        if not eq_lows:
            return out
        # pick closest eq_low to current close
        lvl = float(min(eq_lows, key=lambda x: abs(float(x) - c)))
        out["level"] = lvl
        out["sweep_extreme"] = l
        out["wick_ratio"] = float(lower_wick / rng)

        if (l < (lvl - tol)) and (c > (lvl + tol * 0.25)) and (out["wick_ratio"] >= float(wick_ratio_min)):
            out["ok"] = True
            out["kind"] = "EQ_LOW_SWEEP"
        return out

    # SHORT
    if not eq_highs:
        return out
    lvl = float(min(eq_highs, key=lambda x: abs(float(x) - c)))
    out["level"] = lvl
    out["sweep_extreme"] = h
    out["wick_ratio"] = float(upper_wick / rng)

    if (h > (lvl + tol)) and (c < (lvl - tol * 0.25)) and (out["wick_ratio"] >= float(wick_ratio_min)):
        out["ok"] = True
        out["kind"] = "EQ_HIGH_SWEEP"
    return out


# =====================================================================
# TREND via EMA 20/50 (LTF & HTF)
# =====================================================================

def _trend_from_ema(close: pd.Series, fast: int = 20, slow: int = 50) -> str:
    """
    Desk bias from EMA20/EMA50 with slope + spread threshold.

    LONG  if ema_fast > ema_slow AND slope_fast > 0 AND spread% > threshold
    SHORT if ema_fast < ema_slow AND slope_fast < 0 AND spread% < -threshold
    else RANGE
    """
    try:
        if close is None or len(close) < slow + 8:
            return "RANGE"

        c = pd.Series(close).astype(float)
        ef = c.ewm(span=int(fast), adjust=False).mean()
        es = c.ewm(span=int(slow), adjust=False).mean()

        ef_tail = ef.tail(8)
        es_tail = es.tail(8)
        if len(ef_tail) < 8 or len(es_tail) < 8:
            return "RANGE"

        slope = float(ef_tail.iloc[-1] - ef_tail.iloc[0])
        spread = float(ef.iloc[-1] - es.iloc[-1])
        level = float(abs(es.iloc[-1])) if abs(float(es.iloc[-1])) > 1e-12 else 1.0
        spread_pct = spread / level

        min_spread = 0.0008  # 0.08%
        if ef.iloc[-1] > es.iloc[-1] and slope > 0 and spread_pct > min_spread:
            return "LONG"
        if ef.iloc[-1] < es.iloc[-1] and slope < 0 and spread_pct < -min_spread:
            return "SHORT"
        return "RANGE"
    except Exception:
        return "RANGE"


# =====================================================================
# BOS / CHOCH / COS (buffered, internal vs external)
# =====================================================================

def _classify_bos(
    swings: Dict[str, List[Tuple[int, float]]],
    last_close: float,
    *,
    break_buffer: float = 0.0,
) -> Dict[str, Any]:
    """
    BOS classification using recent swing highs/lows and optional buffer.

    Returns dict with:
      bos, direction ("UP"/"DOWN"), bos_type ("INTERNAL"/"EXTERNAL"), broken_level
    """
    res = {"bos": False, "direction": None, "bos_type": None, "broken_level": None}

    highs = swings.get("highs") or []
    lows = swings.get("lows") or []
    if len(highs) < 1 and len(lows) < 1:
        return res

    last_hi = float(highs[-1][1]) if len(highs) >= 1 else None
    last_lo = float(lows[-1][1]) if len(lows) >= 1 else None

    ext_hi = float(max([p for _, p in highs[-3:]])) if len(highs) >= 2 else None
    ext_lo = float(min([p for _, p in lows[-3:]])) if len(lows) >= 2 else None

    bos_up = False
    bos_dn = False
    bos_type_up = None
    bos_type_dn = None
    broken_up = None
    broken_dn = None

    if last_hi is not None and last_close > (last_hi + break_buffer):
        bos_up = True
        broken_up = last_hi
        bos_type_up = "EXTERNAL" if (ext_hi is not None and last_close > (ext_hi + break_buffer)) else "INTERNAL"

    if last_lo is not None and last_close < (last_lo - break_buffer):
        bos_dn = True
        broken_dn = last_lo
        bos_type_dn = "EXTERNAL" if (ext_lo is not None and last_close < (ext_lo - break_buffer)) else "INTERNAL"

    # If both triggered (rare), keep the stronger displacement vs broken level
    if bos_up and bos_dn:
        du = abs(last_close - float(broken_up or last_close))
        dd = abs(last_close - float(broken_dn or last_close))
        if du >= dd:
            bos_dn = False
        else:
            bos_up = False

    if bos_up:
        res.update({"bos": True, "direction": "UP", "bos_type": bos_type_up, "broken_level": broken_up})
    elif bos_dn:
        res.update({"bos": True, "direction": "DOWN", "bos_type": bos_type_dn, "broken_level": broken_dn})

    return res


def _detect_bos_choch_cos(df: pd.DataFrame) -> Dict[str, Any]:
    """
    BOS/CHOCH/COS:
      - BOS : buffered break of last swing high/low
      - CHOCH : BOS opposite to EMA trend
      - COS : BOS aligned with EMA trend
    """
    if df is None or len(df) < 60 or not _has_cols(df, ("open", "high", "low", "close")):
        return {"bos": False, "choch": False, "cos": False, "direction": None, "bos_type": None, "broken_level": None}

    close = df["close"].astype(float)
    last_close = float(close.iloc[-1])

    swings = find_swings(df, left=3, right=3)

    a = _atr(df, 14)
    buf = max(a * 0.12, last_close * 0.0004)

    bos_info = _classify_bos(swings, last_close, break_buffer=buf)
    if not bos_info.get("bos"):
        bos_info.update({"choch": False, "cos": False})
        return bos_info

    trend = _trend_from_ema(close)
    direction = bos_info.get("direction")

    choch = False
    cos = False

    if direction == "UP":
        if trend == "SHORT":
            choch = True
        elif trend == "LONG":
            cos = True
    elif direction == "DOWN":
        if trend == "LONG":
            choch = True
        elif trend == "SHORT":
            cos = True

    bos_info.update({"choch": bool(choch), "cos": bool(cos)})
    return bos_info


# =====================================================================
# HTF trend alignment (H4 vs H1) — SOFT VETO (évite 0 signal)
# =====================================================================

def htf_trend_ok(df_htf: pd.DataFrame, bias: str) -> bool:
    """
    SOFT HTF veto:
      - HTF RANGE => allow
      - HTF opposite => veto seulement si HTF est "fort" (spread EMA20/50 au-dessus d'un seuil)
    """
    try:
        if df_htf is None or len(df_htf) < 60 or not _has_cols(df_htf, ("close",)):
            return True

        b = (bias or "").upper()
        if b not in {"LONG", "SHORT"}:
            return True

        close = df_htf["close"].astype(float)
        htf = _trend_from_ema(close)

        if htf not in {"LONG", "SHORT"}:
            return True
        if htf == b:
            return True

        # strength proxy: EMA spread%
        ef = close.ewm(span=20, adjust=False).mean()
        es = close.ewm(span=50, adjust=False).mean()
        efv = float(ef.iloc[-1])
        esv = float(es.iloc[-1])
        spread_pct = abs(efv - esv) / max(abs(esv), 1e-12)

        HTF_VETO_MIN_SPREAD = 0.0012  # 0.12%
        # If HTF trend is weak => allow
        if spread_pct < HTF_VETO_MIN_SPREAD:
            return True

        # HTF strong and opposite => veto
        return False
    except Exception:
        return True


# =====================================================================
# BOS QUALITY (Desk v3)
# =====================================================================

def bos_quality_details(
    df: pd.DataFrame,
    oi_series: Optional[pd.Series] = None,
    vol_lookback: int = 60,
    vol_pct: float = 0.8,           # kept for compatibility (not used as hard gate)
    oi_min_trend: float = 0.003,    # kept for compatibility
    oi_min_squeeze: float = -0.005, # kept for compatibility
    df_liq: Optional[pd.DataFrame] = None,
    price: Optional[float] = None,
    tick: float = 0.1,
    direction: Optional[str] = None,  # "UP"/"DOWN"
) -> Dict[str, Any]:
    """
    BOS quality scoring (non-binary, desk-friendly).

    ok = score >= 0
    """
    if df is None or len(df) < max(int(vol_lookback), 40) or not _has_cols(df, ("open", "high", "low", "close", "volume")):
        return {"ok": True, "score": 0.0, "reason": "not_enough_data", "reasons": []}

    closes = df["close"].astype(float)
    opens = df["open"].astype(float)
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)
    vols = df["volume"].astype(float)

    last_close = float(closes.iloc[-1])
    last_open = float(opens.iloc[-1])
    last_high = float(highs.iloc[-1])
    last_low = float(lows.iloc[-1])
    last_vol = float(vols.iloc[-1])

    rng = float(last_high - last_low)
    body = float(abs(last_close - last_open))
    body_ratio = (body / rng) if rng > 0 else 0.0

    a = _atr(df, 14)
    impulse = (rng / a) if a > 0 else 0.0

    w = df.tail(int(vol_lookback))
    avg_vol = float(w["volume"].astype(float).mean()) if len(w) else 0.0
    volume_factor = (last_vol / avg_vol) if avg_vol > 0 else 1.0

    range_pos = ((last_close - last_low) / rng) if rng > 0 else 0.5

    upper_wick = float(last_high - max(last_open, last_close))
    lower_wick = float(min(last_open, last_close) - last_low)
    wick_ratio = float((upper_wick + lower_wick) / rng) if rng > 0 else 0.0

    oi_slope = 0.0
    oi_used = False
    if oi_series is not None:
        try:
            s = pd.Series(oi_series).astype(float)
            if len(s) >= 12 and np.isfinite(float(s.iloc[-12])) and np.isfinite(float(s.iloc[-1])):
                base = float(s.iloc[-12])
                last = float(s.iloc[-1])
                if abs(base) > 1e-12:
                    oi_slope = (last - base) / abs(base)
                    oi_used = True
        except Exception:
            oi_slope = 0.0
            oi_used = False

    # Liquidity sweep bonus
    liquidity_sweep = False
    if df_liq is None:
        df_liq = df

    try:
        lv = detect_equal_levels(df_liq.tail(200))
        eq_highs = lv.get("eq_highs", []) or []
        eq_lows = lv.get("eq_lows", []) or []

        buf = max((a * 0.08) if a > 0 else 0.0, abs(float(tick)) * 5.0, last_close * 0.0004)
        d = (direction or "").upper()

        if d == "UP":
            for lvl in eq_highs:
                lvl = float(lvl)
                if last_high > (lvl + buf) and last_close < (lvl + 0.25 * buf):
                    liquidity_sweep = True
                    break
        elif d == "DOWN":
            for lvl in eq_lows:
                lvl = float(lvl)
                if last_low < (lvl - buf) and last_close > (lvl - 0.25 * buf):
                    liquidity_sweep = True
                    break
    except Exception:
        liquidity_sweep = False

    # ------------------------------------------------------------
    # SCORE
    # ------------------------------------------------------------
    score = 0.0
    reasons: List[str] = []

    if rng <= 0 or (a > 0 and rng < 0.25 * a):
        score -= 1.0
        reasons.append("tiny_range")
    elif a > 0 and impulse >= 1.4:
        score += 1.0
    elif a > 0 and impulse >= 0.9:
        score += 0.5

    if volume_factor >= 1.8:
        score += 1.5
    elif volume_factor >= 1.2:
        score += 1.0
    elif volume_factor >= 0.7:
        score += 0.0
    else:
        score -= 0.8
        reasons.append("low_volume")

    if body_ratio >= 0.62:
        score += 1.5
    elif body_ratio >= 0.45:
        score += 1.0
    elif body_ratio >= 0.28:
        score += 0.2
    else:
        score -= 0.8
        reasons.append("small_body")

    if wick_ratio >= 0.65 and body_ratio < 0.35:
        score -= 0.8
        reasons.append("wicky_candle")

    d = (direction or "").upper()
    if d == "UP":
        if range_pos >= 0.62:
            score += 0.7
        elif range_pos <= 0.35:
            score -= 0.7
            reasons.append("close_not_up")
    elif d == "DOWN":
        if range_pos <= 0.38:
            score += 0.7
        elif range_pos >= 0.65:
            score -= 0.7
            reasons.append("close_not_down")

    if oi_used:
        abs_oi = abs(float(oi_slope))
        if abs_oi >= 0.012:
            score += 1.2
        elif abs_oi >= 0.005:
            score += 0.6
        elif abs_oi < 0.0012:
            score -= 0.5
            reasons.append("weak_oi")

    if liquidity_sweep:
        score += 0.6

    ok = bool(score >= 0.0)

    if score >= 3.0:
        grade = "A+"
    elif score >= 2.0:
        grade = "A"
    elif score >= 1.0:
        grade = "B"
    elif score >= 0.0:
        grade = "C"
    else:
        grade = "D"

    return {
        "ok": ok,
        "score": float(score),
        "grade": grade,
        "volume_factor": float(volume_factor),
        "body_ratio": float(body_ratio),
        "impulse_atr": float(impulse),
        "wick_ratio": float(wick_ratio),
        "range_pos": float(range_pos),
        "oi_slope": float(oi_slope),
        "oi_used": bool(oi_used),
        "liquidity_sweep": bool(liquidity_sweep),
        "reasons": reasons,
    }


# =====================================================================
# ORDER BLOCKS (improved, still lightweight)
# =====================================================================

def _detect_order_blocks(df: pd.DataFrame, lookback: int = 120) -> Dict[str, Any]:
    """
    Lightweight OB detection:
      - Find an impulsive move (range/ATR) in last N candles.
      - Bullish OB: last bearish candle before impulse up.
      - Bearish OB: last bullish candle before impulse down.
    """
    if df is None or len(df) < 60 or not _has_cols(df, ("open", "high", "low", "close")):
        return {"bullish": None, "bearish": None}

    sub = df.tail(int(lookback)).reset_index(drop=True)
    idx_offset = len(df) - len(sub)

    a = _atr(sub, 14)
    if a <= 0:
        return {"bullish": None, "bearish": None}

    o = sub["open"].astype(float).to_numpy()
    c = sub["close"].astype(float).to_numpy()
    h = sub["high"].astype(float).to_numpy()
    l = sub["low"].astype(float).to_numpy()

    bullish_ob = None
    bearish_ob = None

    start = max(10, len(sub) - 50)
    for i in range(start, len(sub) - 1):
        rng = float(h[i] - l[i])
        body = float(abs(c[i] - o[i]))
        impulse = rng / a if a > 0 else 0.0

        if impulse >= 1.6 and (body / max(rng, 1e-12)) >= 0.45:
            up = c[i] > o[i]
            down = c[i] < o[i]

            if up:
                for j in range(max(0, i - 8), i):
                    if c[j] < o[j]:
                        bullish_ob = {
                            "index": int(idx_offset + j),
                            "low": float(min(o[j], c[j])),
                            "high": float(max(o[j], c[j])),
                            "direction": "bullish",
                        }
                        break
            elif down:
                for j in range(max(0, i - 8), i):
                    if c[j] > o[j]:
                        bearish_ob = {
                            "index": int(idx_offset + j),
                            "low": float(min(o[j], c[j])),
                            "high": float(max(o[j], c[j])),
                            "direction": "bearish",
                        }
                        break

    return {"bullish": bullish_ob, "bearish": bearish_ob}


# =====================================================================
# FVG (Fair Value Gaps) — FIXED FORMAT + unmitigated filter
# =====================================================================

def _detect_fvg(df: pd.DataFrame, lookback: int = 140, keep_last: int = 8) -> List[Dict[str, Any]]:
    """
    3-candle FVG detection (classic):
      - Bullish FVG at i: low[i] > high[i-2]  => gap [high[i-2], low[i]]
      - Bearish FVG at i: high[i] < low[i-2]  => gap [high[i], low[i-2]]

    Zones format:
      {"low":..., "high":..., "direction":"bullish"/"bearish", "index":..., "mitigated":bool}
    """
    zones: List[Dict[str, Any]] = []
    if df is None or len(df) < 10 or not _has_cols(df, ("high", "low")):
        return zones

    sub = df.tail(int(lookback)).reset_index(drop=True)
    idx_offset = len(df) - len(sub)

    H = sub["high"].astype(float).to_numpy()
    L = sub["low"].astype(float).to_numpy()

    for i in range(2, len(sub)):
        # Bullish FVG
        if L[i] > H[i - 2]:
            z_low = float(H[i - 2])
            z_high = float(L[i])
            if z_high > z_low:
                zones.append(
                    {
                        "low": z_low,
                        "high": z_high,
                        "bottom": z_low,
                        "top": z_high,
                        "direction": "bullish",
                        "type": "bullish",
                        "index": int(idx_offset + i),
                        "mitigated": False,
                    }
                )

        # Bearish FVG
        if H[i] < L[i - 2]:
            z_low = float(H[i])
            z_high = float(L[i - 2])
            if z_high > z_low:
                zones.append(
                    {
                        "low": z_low,
                        "high": z_high,
                        "bottom": z_low,
                        "top": z_high,
                        "direction": "bearish",
                        "type": "bearish",
                        "index": int(idx_offset + i),
                        "mitigated": False,
                    }
                )

    if not zones:
        return []

    # Mitigation check
    try:
        for z in zones:
            zi = int(z.get("index", 0)) - idx_offset
            if zi < 0 or zi >= len(sub):
                continue
            low_z = float(z["low"])
            high_z = float(z["high"])

            later = sub.iloc[zi + 1 :]
            if later.empty:
                continue
            hit = ((later["low"].astype(float) <= high_z) & (later["high"].astype(float) >= low_z)).any()
            if bool(hit):
                z["mitigated"] = True
    except Exception:
        pass

    # Prefer unmitigated, else last ones
    unmit = [z for z in zones if not z.get("mitigated")]
    use = unmit if unmit else zones
    use = sorted(use, key=lambda x: int(x.get("index", 0)), reverse=True)[: int(keep_last)]
    return list(reversed(use))


# =====================================================================
# STRUCTURE ENGINE (H1)
# =====================================================================

def analyze_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Main structure analysis entrypoint for H1.

    Output keys relied on by analyze_signal.py:
      - trend: "LONG"/"SHORT"/"RANGE"
      - bos: bool
      - bos_direction: "UP"/"DOWN"|None
      - bos_type: "INTERNAL"/"EXTERNAL"|None
      - fvg_zones: list[dict] with keys low/high + direction
      - oi_series: Series|None (if df has "oi")
    """
    if df is None or len(df) < 40 or not _has_cols(df, ("open", "high", "low", "close")):
        return {
            "trend": "RANGE",
            "swings": {"highs": [], "lows": []},
            "liquidity": {"eq_highs": [], "eq_lows": []},
            "bos": False,
            "choch": False,
            "cos": False,
            "bos_type": None,
            "bos_direction": None,
            "bos_dir": None,        # alias safety
            "broken_level": None,
            "order_blocks": {"bullish": None, "bearish": None},
            "fvg_zones": [],
            "oi_series": None,
        }

    trend = _trend_from_ema(df["close"].astype(float))
    swings = find_swings(df, left=3, right=3)
    levels = detect_equal_levels(df, left=3, right=3, max_window=200)

    bos_block = _detect_bos_choch_cos(df)
    ob = _detect_order_blocks(df, lookback=120)
    fvg_zones = _detect_fvg(df, lookback=140, keep_last=8)

    oi_series = df["oi"] if "oi" in df.columns else None

    bos_direction = bos_block.get("direction")
    return {
        "trend": trend,
        "swings": swings,
        "liquidity": levels,
        "bos": bool(bos_block.get("bos")),
        "choch": bool(bos_block.get("choch")),
        "cos": bool(bos_block.get("cos")),
        "bos_type": bos_block.get("bos_type"),
        "bos_direction": bos_direction,
        "bos_dir": bos_direction,   # alias safety
        "broken_level": bos_block.get("broken_level"),
        "order_blocks": ob,
        "fvg_zones": fvg_zones,
        "oi_series": oi_series,
    }


# =====================================================================
# COMMITMENT SCORE (OI + CVD) — kept
# =====================================================================

def commitment_score(
    df: pd.DataFrame,
    oi_series: Optional[pd.Series] = None,
    cvd_series: Optional[pd.Series] = None,
) -> float:
    """
    Lightweight commitment proxy combining OI & CVD.
    Returns score in [-1, +1].
    """
    try:
        if oi_series is None or cvd_series is None:
            return 0.0

        oi = pd.Series(oi_series).astype(float)
        cvd = pd.Series(cvd_series).astype(float)

        if len(oi) < 10 or len(cvd) < 10:
            return 0.0

        d_oi = float(oi.iloc[-1] - oi.iloc[-10])
        d_cvd = float(cvd.iloc[-1] - cvd.iloc[-10])

        score_oi = float(np.tanh(d_oi * 10.0))
        score_cvd = float(np.tanh(d_cvd * 10.0))

        score = 0.6 * score_oi + 0.4 * score_cvd
        return float(max(-1.0, min(1.0, score)))
    except Exception:
        return 0.0
