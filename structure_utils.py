
# =====================================================================
# structure_utils.py — Institutional Structure Engine (Desk v4)
# BOS / CHOCH / COS / Internal vs External / Liquidity / OB / FVG / HTF
# + RAID → DISPLACEMENT → FVG entry (desk model)
# + Volume Profile (POC/HVN/LVN) lightweight
# =====================================================================
# API stable used by analyze_signal.py:
#   - analyze_structure(df) -> dict with keys: trend, bos, bos_direction, bos_type, fvg_zones, oi_series...
#   - htf_trend_ok(df_htf, bias) -> bool
#   - bos_quality_details(...)
#   - detect_equal_levels(...)
#   - liquidity_sweep_details(...)
# =====================================================================

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd


# =====================================================================
# Base utils
# =====================================================================

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _has_cols(df: pd.DataFrame, cols: Tuple[str, ...]) -> bool:
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return False
        return all(c in df.columns for c in cols)
    except Exception:
        return False


# Optional: reuse ATR from indicators if present
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
            # be tolerant to signature differences (length / period / n)
            try:
                s = _compute_atr_ext(df, length=int(length))
            except TypeError:
                try:
                    s = _compute_atr_ext(df, period=int(length))
                except TypeError:
                    s = _compute_atr_ext(df, n=int(length))
            v = float(s.iloc[-1]) if s is not None and len(s) else 0.0
            return max(0.0, v) if np.isfinite(v) else 0.0

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
        return max(0.0, v) if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _median_range(df: pd.DataFrame, n: int = 60) -> float:
    try:
        if not _has_cols(df, ("high", "low")) or len(df) < 10:
            return 0.0
        w = df.tail(int(max(10, n)))
        r = (w["high"].astype(float) - w["low"].astype(float)).abs()
        v = float(np.nanmedian(r.values)) if len(r) else 0.0
        return max(0.0, v) if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


# =====================================================================
# Swings (pivot highs / lows)
# =====================================================================

def find_swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> Dict[str, List[Tuple[int, float]]]:
    """
    Pivot swings detection (fractal).

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

        if hi >= float(np.max(win_h)):
            # keep last occurrence if duplicates
            if np.sum(np.isclose(win_h, hi, atol=0.0, rtol=0.0)) == 1:
                highs.append((i, hi))
            else:
                highs.append((i, hi))

        if lo <= float(np.min(win_l)):
            if np.sum(np.isclose(win_l, lo, atol=0.0, rtol=0.0)) == 1:
                lows.append((i, lo))
            else:
                lows.append((i, lo))

    return {"highs": highs, "lows": lows}


# =====================================================================
# Liquidity (equal highs / equal lows)
# =====================================================================

def _cluster_levels(levels: List[float], tolerance: float) -> List[float]:
    """
    Groups nearby levels into clusters. Returns cluster mean only if >= 2 touches.
    """
    if not levels:
        return []
    tol = abs(float(tolerance))
    if tol <= 0:
        return []

    lv = sorted([float(x) for x in levels if np.isfinite(float(x))])
    if not lv:
        return []

    clusters: List[List[float]] = [[lv[0]]]
    for p in lv[1:]:
        if abs(p - clusters[-1][-1]) <= tol:
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
    tol_atr: Optional[float] = None,  # alias
) -> Dict[str, List[float]]:
    """
    Detects EQH/EQL pools from swing points.

    Returns:
      {"eq_highs": [level...], "eq_lows": [level...]}
    """
    if df is None or len(df) < left + right + 10 or not _has_cols(df, ("high", "low", "close")):
        return {"eq_highs": [], "eq_lows": []}

    if tol_atr is not None:
        tol_mult_atr = float(tol_atr)

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
        tol = max(1e-12, abs(last_price) * float(tol_fallback_pct))

    eq_highs = sorted(_cluster_levels(high_prices, tol))
    eq_lows = sorted(_cluster_levels(low_prices, tol))
    return {"eq_highs": eq_highs, "eq_lows": eq_lows}


def has_liquidity_zone(df: pd.DataFrame, direction: str, lookback: int = 200) -> bool:
    """
    Quick helper: True if EQH (for LONG) / EQL (for SHORT) exists near price.
    """
    try:
        if df is None or df.empty or len(df) < 40:
            return False
        d = (direction or "").upper()
        lv = detect_equal_levels(df.tail(int(lookback)))
        close = float(df["close"].astype(float).iloc[-1])
        a = _atr(df, 14)
        buf = max(a * 0.15, abs(close) * 0.0008)

        if d == "LONG":
            return any(abs(float(x) - close) <= buf for x in (lv.get("eq_highs") or []))
        if d == "SHORT":
            return any(abs(float(x) - close) <= buf for x in (lv.get("eq_lows") or []))
        return False
    except Exception:
        return False


def liquidity_sweep_details(
    df: pd.DataFrame,
    direction: str,
    lookback: int = 160,
    tol_atr: float = 0.12,
    wick_ratio_min: float = 0.55,
) -> Dict[str, Any]:
    """
    Stop-hunt / sweep detector (desk-style).

    LONG:
      - wick below an EQL (below lvl - tol) then close back above lvl (+ small tol)
    SHORT:
      - wick above an EQH (above lvl + tol) then close back below lvl (- small tol)

    Checks last ~3 candles and returns the first valid sweep.
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
        "index": None,
    }
    if df is None or getattr(df, "empty", True) or len(df) < 60:
        return out

    d = str(direction).upper()
    if d not in {"LONG", "SHORT"}:
        return out

    dfw = df.tail(int(max(80, lookback))).copy().reset_index(drop=True)
    a = _atr(dfw, 14)
    tol = max(a * float(tol_atr), 1e-12)

    lv = detect_equal_levels(dfw, max_window=int(max(80, lookback)), tol_atr=float(tol_atr))
    eq_highs = lv.get("eq_highs", []) or []
    eq_lows = lv.get("eq_lows", []) or []

    # pick nearest level to current price (more "desk" than oldest cluster)
    close_last = float(dfw["close"].astype(float).iloc[-1])
    if d == "LONG" and eq_lows:
        lvl = float(min(eq_lows, key=lambda x: abs(close_last - float(x))))
    elif d == "SHORT" and eq_highs:
        lvl = float(min(eq_highs, key=lambda x: abs(close_last - float(x))))
    else:
        return out

    # scan last 3 candles for a sweep
    tail_n = min(3, len(dfw))
    for i in range(len(dfw) - tail_n, len(dfw)):
        row = dfw.iloc[i]
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        rng = max(h - l, 1e-12)
        body = abs(c - o)
        body_ratio = float(body / rng)

        lower_wick = max(min(o, c) - l, 0.0)
        upper_wick = max(h - max(o, c), 0.0)

        if d == "LONG":
            wick_ratio = float(lower_wick / rng)
            if (l < (lvl - tol)) and (c > (lvl + 0.25 * tol)) and (wick_ratio >= float(wick_ratio_min)):
                out.update(
                    {
                        "ok": True,
                        "level": lvl,
                        "sweep_extreme": l,
                        "reclaim_close": c,
                        "wick_ratio": wick_ratio,
                        "body_ratio": body_ratio,
                        "kind": "EQ_LOW_SWEEP",
                        "index": int(i),
                    }
                )
                return out

        else:  # SHORT
            wick_ratio = float(upper_wick / rng)
            if (h > (lvl + tol)) and (c < (lvl - 0.25 * tol)) and (wick_ratio >= float(wick_ratio_min)):
                out.update(
                    {
                        "ok": True,
                        "level": lvl,
                        "sweep_extreme": h,
                        "reclaim_close": c,
                        "wick_ratio": wick_ratio,
                        "body_ratio": body_ratio,
                        "kind": "EQ_HIGH_SWEEP",
                        "index": int(i),
                    }
                )
                return out

    return out


# =====================================================================
# Trend via EMA20/EMA50 (slope + spread)
# =====================================================================

def _trend_from_ema(close: pd.Series, fast: int = 20, slow: int = 50) -> str:
    """
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
        if len(ef_tail) < 8:
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
    BOS classification using last swing and a buffer.
    """
    res = {"bos": False, "direction": None, "bos_type": None, "broken_level": None}

    highs = swings.get("highs") or []
    lows = swings.get("lows") or []
    if not highs and not lows:
        return res

    last_hi = float(highs[-1][1]) if highs else None
    last_lo = float(lows[-1][1]) if lows else None

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
    if df is None or len(df) < 60 or not _has_cols(df, ("open", "high", "low", "close")):
        return {"bos": False, "choch": False, "cos": False, "direction": None, "bos_type": None, "broken_level": None}

    close = df["close"].astype(float)
    last_close = float(close.iloc[-1])

    swings = find_swings(df, left=3, right=3)
    a = _atr(df, 14)
    buf = max(a * 0.12, abs(last_close) * 0.0004)

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
# HTF alignment
# =====================================================================

def htf_trend_ok(df_htf: pd.DataFrame, bias: str) -> bool:
    """
    HTF alignment veto:
      - HTF LONG  veto SHORT bias
      - HTF SHORT veto LONG bias
      - HTF RANGE => allow both
    """
    try:
        if df_htf is None or len(df_htf) < 60 or not _has_cols(df_htf, ("close",)):
            return True

        b = (bias or "").upper()
        htf = _trend_from_ema(df_htf["close"].astype(float))

        if htf == "LONG" and b == "SHORT":
            return False
        if htf == "SHORT" and b == "LONG":
            return False
        return True
    except Exception:
        return True


# =====================================================================
# BOS QUALITY (score, non-binary)
# =====================================================================

def bos_quality_details(
    df: pd.DataFrame,
    oi_series: Optional[pd.Series] = None,
    vol_lookback: int = 60,
    vol_pct: float = 0.8,  # kept for compatibility
    oi_min_trend: float = 0.003,
    oi_min_squeeze: float = -0.005,
    df_liq: Optional[pd.DataFrame] = None,
    price: Optional[float] = None,
    tick: float = 0.1,
    direction: Optional[str] = None,  # "UP"/"DOWN"
) -> Dict[str, Any]:
    """
    BOS quality scoring.

    ok = score >= 0
    """
    if df is None or len(df) < max(int(vol_lookback), 40) or not _has_cols(df, ("open", "high", "low", "close", "volume")):
        return {"ok": True, "score": 0.0, "grade": "C", "reasons": ["not_enough_data"]}

    opens = df["open"].astype(float)
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)
    closes = df["close"].astype(float)
    vols = df["volume"].astype(float)

    last_open = float(opens.iloc[-1])
    last_high = float(highs.iloc[-1])
    last_low = float(lows.iloc[-1])
    last_close = float(closes.iloc[-1])
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

    # OI slope (optional)
    oi_slope = 0.0
    oi_used = False
    if oi_series is not None:
        try:
            s = pd.Series(oi_series).astype(float)
            if len(s) >= 12:
                base = float(s.iloc[-12])
                last = float(s.iloc[-1])
                if abs(base) > 1e-12:
                    oi_slope = (last - base) / abs(base)
                    oi_used = True
        except Exception:
            oi_slope = 0.0
            oi_used = False

    # Sweep bonus: wick beyond EQ then close back (opposite of BOS direction)
    liquidity_sweep = False
    try:
        df_liq = df_liq if df_liq is not None else df
        lv = detect_equal_levels(df_liq.tail(200))
        eq_highs = lv.get("eq_highs", []) or []
        eq_lows = lv.get("eq_lows", []) or []
        buf = max((a * 0.08) if a > 0 else 0.0, abs(float(tick)) * 5.0, abs(last_close) * 0.0004)
        d = (direction or "").upper()

        if d == "UP" and eq_highs:
            for lvl in eq_highs:
                lvl = float(lvl)
                if last_high > (lvl + buf) and last_close < (lvl + 0.25 * buf):
                    liquidity_sweep = True
                    break
        if d == "DOWN" and eq_lows:
            for lvl in eq_lows:
                lvl = float(lvl)
                if last_low < (lvl - buf) and last_close > (lvl - 0.25 * buf):
                    liquidity_sweep = True
                    break
    except Exception:
        liquidity_sweep = False

    # Score
    score = 0.0
    reasons: List[str] = []

    # Range/impulse
    if rng <= 0 or (a > 0 and rng < 0.25 * a):
        score -= 1.0
        reasons.append("tiny_range")
    elif a > 0 and impulse >= 1.4:
        score += 1.0
    elif a > 0 and impulse >= 0.9:
        score += 0.5

    # Volume
    if volume_factor >= 1.8:
        score += 1.5
    elif volume_factor >= 1.2:
        score += 1.0
    elif volume_factor >= 0.7:
        score += 0.0
    else:
        score -= 0.8
        reasons.append("low_volume")

    # Body
    if body_ratio >= 0.62:
        score += 1.5
    elif body_ratio >= 0.45:
        score += 1.0
    elif body_ratio >= 0.28:
        score += 0.2
    else:
        score -= 0.8
        reasons.append("small_body")

    # Wickiness
    if wick_ratio >= 0.65 and body_ratio < 0.35:
        score -= 0.8
        reasons.append("wicky_candle")

    # Close location
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

    # OI
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
# Order Blocks (lightweight)
# =====================================================================

def _detect_order_blocks(df: pd.DataFrame, lookback: int = 120) -> Dict[str, Any]:
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

    start = max(10, len(sub) - 60)
    for i in range(start, len(sub) - 1):
        rng = float(h[i] - l[i])
        body = float(abs(c[i] - o[i]))
        impulse = rng / a if a > 0 else 0.0

        if impulse >= 1.6 and body / max(rng, 1e-12) >= 0.45:
            if c[i] > o[i]:
                for j in range(max(0, i - 8), i):
                    if c[j] < o[j]:
                        bullish_ob = {
                            "index": int(idx_offset + j),
                            "low": float(min(o[j], c[j])),
                            "high": float(max(o[j], c[j])),
                            "direction": "bullish",
                        }
                        break
            elif c[i] < o[i]:
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
# FVG — fixed format + unmitigated filter
# =====================================================================

def _detect_fvg(df: pd.DataFrame, lookback: int = 140, keep_last: int = 8) -> List[Dict[str, Any]]:
    """
    3-candle FVG detection:
      - Bullish FVG at i: low[i] > high[i-2] => gap [high[i-2], low[i]]
      - Bearish FVG at i: high[i] < low[i-2] => gap [high[i], low[i-2]]

    Return zones with keys: low, high, direction, index, mitigated
    """
    zones: List[Dict[str, Any]] = []
    if df is None or len(df) < 10 or not _has_cols(df, ("high", "low")):
        return zones

    sub = df.tail(int(lookback)).reset_index(drop=True)
    idx_offset = len(df) - len(sub)

    H = sub["high"].astype(float).to_numpy()
    L = sub["low"].astype(float).to_numpy()

    for i in range(2, len(sub)):
        # bullish
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
        # bearish
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

    # mitigated check
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

    unmit = [z for z in zones if not z.get("mitigated")]
    if unmit:
        unmit = sorted(unmit, key=lambda x: int(x.get("index", 0)), reverse=True)[: int(keep_last)]
        return list(reversed(unmit))

    zones = sorted(zones, key=lambda x: int(x.get("index", 0)), reverse=True)[: int(keep_last)]
    return list(reversed(zones))


# =====================================================================
# Volume Profile (lightweight)
# =====================================================================

def _volume_profile(df: pd.DataFrame, lookback: int = 140, bins: int = 48) -> Dict[str, Any]:
    """
    Lightweight volume profile (no tick granularity needed).
    Returns:
      {
        "poc": float|None,
        "hvn": float|None,
        "lvn": float|None,
        "range": (low, high),
      }
    """
    out = {"poc": None, "hvn": None, "lvn": None, "range": None}
    try:
        if df is None or len(df) < 30 or not _has_cols(df, ("high", "low", "close", "volume")):
            return out

        w = df.tail(int(lookback)).copy()
        hi = float(w["high"].astype(float).max())
        lo = float(w["low"].astype(float).min())
        if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
            return out

        rng = hi - lo
        nb = int(max(12, min(200, bins)))
        edges = np.linspace(lo, hi, nb + 1)
        vol_bins = np.zeros(nb, dtype=float)

        tp = (w["high"].astype(float) + w["low"].astype(float) + w["close"].astype(float)) / 3.0
        vv = w["volume"].astype(float).to_numpy()

        idx = np.clip(np.digitize(tp.to_numpy(), edges) - 1, 0, nb - 1)
        for i, b in enumerate(idx):
            if np.isfinite(vv[i]):
                vol_bins[int(b)] += float(vv[i])

        if float(np.nansum(vol_bins)) <= 0:
            return out

        poc_i = int(np.nanargmax(vol_bins))
        poc = float((edges[poc_i] + edges[poc_i + 1]) / 2.0)

        # HVN: top 3 bins average
        top_idx = np.argsort(vol_bins)[-3:]
        hvn = float(np.mean([(edges[i] + edges[i + 1]) / 2.0 for i in top_idx]))

        # LVN: among non-zero bins, choose min
        nz = np.where(vol_bins > 0)[0]
        lvn_i = int(nz[np.argmin(vol_bins[nz])]) if len(nz) else poc_i
        lvn = float((edges[lvn_i] + edges[lvn_i + 1]) / 2.0)

        out.update({"poc": poc, "hvn": hvn, "lvn": lvn, "range": (lo, hi)})
        return out
    except Exception:
        return out


# =====================================================================
# RAID → DISPLACEMENT → FVG (desk)
# =====================================================================

def _raid_displacement_fvg(df: pd.DataFrame, trend: str, lookback: int = 180) -> Dict[str, Any]:
    """
    Detects a desk-style pattern:
      1) Sweep (raid) of EQ level (EQL for longs / EQH for shorts)
      2) Displacement candle in trend direction (impulse)
      3) Fresh FVG formed by displacement -> propose entry at FVG mid

    Returns dict:
      {"ok": bool, "bias": "LONG"/"SHORT", "entry": float|None, "zone": {low/high/...}|None, "note": str, ...}
    """
    out = {"ok": False, "bias": None, "entry": None, "zone": None, "note": "no_pattern"}

    try:
        if df is None or len(df) < 80 or not _has_cols(df, ("open", "high", "low", "close", "volume")):
            return out

        bias = str(trend).upper()
        if bias not in {"LONG", "SHORT"}:
            return out
        out["bias"] = bias

        w = df.tail(int(max(120, lookback))).copy().reset_index(drop=True)
        a = _atr(w, 14)
        if a <= 0:
            return out

        lv = detect_equal_levels(w, max_window=int(max(120, lookback)), tol_mult_atr=0.10)
        eq_highs = lv.get("eq_highs", []) or []
        eq_lows = lv.get("eq_lows", []) or []
        if bias == "LONG" and not eq_lows:
            return out
        if bias == "SHORT" and not eq_highs:
            return out

        close_last = float(w["close"].astype(float).iloc[-1])
        tol = max(a * 0.10, abs(close_last) * 0.0003)

        # choose nearest level (desk)
        lvl = float(min(eq_lows if bias == "LONG" else eq_highs, key=lambda x: abs(close_last - float(x))))

        # search sweep candle in last 6 candles, then displacement in next 1-3
        tail_start = max(2, len(w) - 12)
        sweep_i = None
        for i in range(tail_start, len(w) - 1):
            o = float(w["open"].iloc[i])
            h = float(w["high"].iloc[i])
            l = float(w["low"].iloc[i])
            c = float(w["close"].iloc[i])
            rng = max(h - l, 1e-12)
            body = abs(c - o)

            lower_wick = max(min(o, c) - l, 0.0)
            upper_wick = max(h - max(o, c), 0.0)
            wick_ratio = float((lower_wick if bias == "LONG" else upper_wick) / rng)

            if bias == "LONG":
                if (l < (lvl - tol)) and (c > (lvl + 0.25 * tol)) and (wick_ratio >= 0.50):
                    sweep_i = i
                    break
            else:
                if (h > (lvl + tol)) and (c < (lvl - 0.25 * tol)) and (wick_ratio >= 0.50):
                    sweep_i = i
                    break

        if sweep_i is None:
            out["note"] = "no_sweep"
            return out

        # find displacement candle after sweep
        disp_i = None
        for j in range(sweep_i + 1, min(len(w), sweep_i + 4)):
            o = float(w["open"].iloc[j])
            h = float(w["high"].iloc[j])
            l = float(w["low"].iloc[j])
            c = float(w["close"].iloc[j])

            rng = max(h - l, 1e-12)
            body = abs(c - o)
            body_ratio = body / rng
            impulse = rng / a

            if impulse < 1.4 or body_ratio < 0.45:
                continue

            if bias == "LONG" and c > o and (c - o) > 0:
                disp_i = j
                break
            if bias == "SHORT" and c < o and (o - c) > 0:
                disp_i = j
                break

        if disp_i is None:
            out["note"] = "sweep_no_displacement"
            return out

        # find a fresh FVG created around displacement candle
        zones = _detect_fvg(w, lookback=len(w), keep_last=12)
        # keep zones created at/after displacement (the "fresh" displacement gap)
        zones = [z for z in zones if int(z.get("index", 0)) >= (len(df) - len(w) + disp_i)]
        if not zones:
            out["note"] = "no_fvg_after_displacement"
            return out

        # pick zone closest to current price and matching bias
        best = None
        best_dist = 1e18
        for z in zones:
            zdir = str(z.get("direction") or "").lower()
            if bias == "LONG" and "bear" in zdir:
                continue
            if bias == "SHORT" and "bull" in zdir:
                continue
            zl = float(z["low"])
            zh = float(z["high"])
            mid = (zl + zh) / 2.0
            dist = abs(close_last - mid)
            if dist < best_dist:
                best_dist = dist
                best = z

        if best is None:
            out["note"] = "no_fvg_bias_match"
            return out

        entry = float((float(best["low"]) + float(best["high"])) / 2.0)
        if best_dist > 2.0 * a:
            out["note"] = f"fvg_too_far dist={best_dist:.6g} atr={a:.6g}"
            return out

        out.update(
            {
                "ok": True,
                "entry": entry,
                "zone": best,
                "note": f"raid(lvl={lvl:.6g}) sweep_i={sweep_i} disp_i={disp_i} fvg_i={best.get('index')} dist={best_dist:.6g} atr={a:.6g}",
                "sweep_level": lvl,
                "sweep_index": sweep_i,
                "displacement_index": disp_i,
            }
        )
        return out
    except Exception as e:
        out["note"] = f"error:{e}"
        return out


# =====================================================================
# Structure Engine (H1)
# =====================================================================

def analyze_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Main structure analysis entrypoint (H1).

    Output keys relied on by analyze_signal.py:
      - trend: "LONG"/"SHORT"/"RANGE"
      - bos: bool
      - bos_direction: "UP"/"DOWN"|None
      - bos_type: "INTERNAL"/"EXTERNAL"|None
      - fvg_zones: list[dict] with keys low/high + direction
      - oi_series: Series|None (if df has "oi")
      - raid_displacement: dict {ok, entry, note...}
      - volume_profile: dict {poc/hvn/lvn}
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
            "broken_level": None,
            "order_blocks": {"bullish": None, "bearish": None},
            "fvg_zones": [],
            "oi_series": None,
            "raid_displacement": {"ok": False, "note": "no_data"},
            "volume_profile": {"poc": None, "hvn": None, "lvn": None, "range": None},
        }

    trend = _trend_from_ema(df["close"].astype(float))
    swings = find_swings(df, left=3, right=3)
    levels = detect_equal_levels(df, left=3, right=3, max_window=200)
    bos_block = _detect_bos_choch_cos(df)
    ob = _detect_order_blocks(df, lookback=120)
    fvg_zones = _detect_fvg(df, lookback=140, keep_last=8)

    oi_series = df["oi"] if "oi" in df.columns else None

    raid = _raid_displacement_fvg(df, trend, lookback=180)
    vp = _volume_profile(df, lookback=160, bins=48)

    return {
        "trend": trend,
        "swings": swings,
        "liquidity": levels,
        "bos": bool(bos_block.get("bos")),
        "choch": bool(bos_block.get("choch")),
        "cos": bool(bos_block.get("cos")),
        "bos_type": bos_block.get("bos_type"),
        "bos_direction": bos_block.get("direction"),
        "broken_level": bos_block.get("broken_level"),
        "order_blocks": ob,
        "fvg_zones": fvg_zones,
        "oi_series": oi_series,
        "raid_displacement": raid,
        "volume_profile": vp,
    }


# =====================================================================
# Commitment score (kept)
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
