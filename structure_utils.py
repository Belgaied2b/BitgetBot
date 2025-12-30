# =====================================================================
# structure_utils.py — Institutional Structure Engine ++ (Desk v3.2)
# BOS / CHOCH / COS / Internal vs External / Liquidity / OB / FVG / HTF
# =====================================================================
# ✅ Compatible analyze_signal desk (OTE/FVG entry + sweep EQH/EQL)
# ✅ detect_equal_levels() "unpackable" (eq_highs, eq_lows = ...)
# ✅ FVG zones: low/high + direction (+ bottom/top for backward compat)
# ✅ BOS buffered ATR, internal/external, broken_level
# ✅ Trend EMA20/50 with slope+spread threshold
# ✅ Liquidity pools clustered ATR-aware (level/touches/recency)
# ✅ Liquidity sweep detector (raid + reclaim) desk-style
# ✅ BOS quality score non-binaire
# =====================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterator

import numpy as np
import pandas as pd


# =====================================================================
# Safe helpers
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


# =====================================================================
# ATR (prefer indicators.true_atr / compute_atr)
# =====================================================================

_atr_func = None
try:
    from indicators import true_atr as _atr_func  # type: ignore
except Exception:
    try:
        from indicators import compute_atr as _atr_func  # type: ignore
    except Exception:
        _atr_func = None


def _atr_last(df: pd.DataFrame, length: int = 14) -> float:
    try:
        if df is None or df.empty or len(df) < length + 3:
            return 0.0

        if _atr_func is not None:
            s = _atr_func(df, length=length)  # series
            if s is None or len(s) == 0:
                return 0.0
            return float(max(0.0, float(s.iloc[-1])))

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
        return float(max(0.0, float(a.iloc[-1])))
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
# Trend EMA20/EMA50 (desk)
# =====================================================================

def _trend_from_ema(close: pd.Series, fast: int = 20, slow: int = 50) -> str:
    """
    LONG if ema_fast > ema_slow AND slope_fast > 0 AND spread% > threshold
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

        # 0.08% default threshold
        min_spread = 0.0008

        if ef.iloc[-1] > es.iloc[-1] and slope > 0 and spread_pct > min_spread:
            return "LONG"
        if ef.iloc[-1] < es.iloc[-1] and slope < 0 and spread_pct < -min_spread:
            return "SHORT"
        return "RANGE"
    except Exception:
        return "RANGE"


# =====================================================================
# Swings (pivot highs/lows)
# =====================================================================

def find_swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> Dict[str, List[Tuple[int, float]]]:
    highs: List[Tuple[int, float]] = []
    lows: List[Tuple[int, float]] = []

    if df is None or len(df) < left + right + 8 or not _has_cols(df, ("high", "low")):
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
            highs.append((i, hi))

        if lo <= float(np.min(win_l)):
            lows.append((i, lo))

    return {"highs": highs, "lows": lows}


# =====================================================================
# Liquidity pools EQH/EQL (clustered) + "unpackable" return
# =====================================================================

@dataclass
class LiquidityLevel:
    level: float
    touches: int
    last_index: int


class EqualLevels(dict):
    """
    Dict-like + can be unpacked:
        eq_highs, eq_lows = detect_equal_levels(...)
    Also usable as:
        lv = detect_equal_levels(...)
        lv["eq_highs"], lv["eq_lows"]
    """
    def __iter__(self) -> Iterator[List[Dict[str, Any]]]:
        yield self.get("eq_highs", [])
        yield self.get("eq_lows", [])


def _cluster_levels(levels: List[Tuple[int, float]], tolerance: float) -> List[LiquidityLevel]:
    """
    Cluster levels with tolerance.
    Input: [(idx, price), ...]
    Output: list of LiquidityLevel(level, touches, last_index)
    """
    tolerance = abs(float(tolerance))
    if tolerance <= 0 or not levels:
        return []

    # sort by price
    lv = sorted([(int(i), float(p)) for i, p in levels if np.isfinite(float(p))], key=lambda x: x[1])
    if not lv:
        return []

    clusters: List[List[Tuple[int, float]]] = [[lv[0]]]
    for it in lv[1:]:
        if abs(it[1] - clusters[-1][-1][1]) <= tolerance:
            clusters[-1].append(it)
        else:
            clusters.append([it])

    out: List[LiquidityLevel] = []
    for c in clusters:
        if len(c) < 2:
            continue
        prices = [p for _, p in c]
        last_index = max(i for i, _ in c)
        out.append(LiquidityLevel(level=float(np.mean(prices)), touches=len(c), last_index=int(last_index)))

    # prefer more touches, then recency
    out.sort(key=lambda x: (x.touches, x.last_index), reverse=True)
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
) -> EqualLevels:
    """
    Returns EqualLevels with:
      - eq_highs: [{"level":..., "touches":..., "last_index":...}, ...]
      - eq_lows:  [{"level":..., "touches":..., "last_index":...}, ...]
    Unpackable:
      eq_highs, eq_lows = detect_equal_levels(...)
    """
    res = EqualLevels(eq_highs=[], eq_lows=[])

    if df is None or len(df) < left + right + 12 or not _has_cols(df, ("high", "low", "close")):
        return res

    sub = df.tail(int(max_window)).reset_index(drop=True)
    swings = find_swings(sub, left=left, right=right)
    close = float(sub["close"].astype(float).iloc[-1])

    a = _atr_last(sub, 14)
    base_range = _median_range(sub, 60)

    tol = 0.0
    if a > 0:
        tol = a * float(tol_mult_atr)
    if tol <= 0 and base_range > 0:
        tol = base_range * float(tol_mult_range)
    if tol <= 0:
        tol = max(1e-12, close * float(tol_fallback_pct))

    highs = swings.get("highs", []) or []
    lows = swings.get("lows", []) or []

    eqh = _cluster_levels(highs, tol)
    eql = _cluster_levels(lows, tol)

    # convert to dicts
    res["eq_highs"] = [{"level": x.level, "touches": x.touches, "last_index": x.last_index} for x in eqh]
    res["eq_lows"] = [{"level": x.level, "touches": x.touches, "last_index": x.last_index} for x in eql]
    res["tol_used"] = float(tol)
    return res


def has_liquidity_zone(df: pd.DataFrame, direction: str, lookback: int = 200) -> bool:
    try:
        if df is None or df.empty or len(df) < 60:
            return False
        d = (direction or "").upper()
        lv = detect_equal_levels(df.tail(int(lookback)))
        close = float(df["close"].astype(float).iloc[-1])
        a = _atr_last(df, 14)
        buf = max(a * 0.15, close * 0.0008)

        if d == "LONG":
            for x in lv.get("eq_highs", []) or []:
                lvl = float(x.get("level") or 0.0)
                if lvl > 0 and abs(lvl - close) <= buf:
                    return True

        if d == "SHORT":
            for x in lv.get("eq_lows", []) or []:
                lvl = float(x.get("level") or 0.0)
                if lvl > 0 and abs(lvl - close) <= buf:
                    return True

        return False
    except Exception:
        return False


# =====================================================================
# Liquidity sweep (raid + reclaim) — desk style
# =====================================================================

def liquidity_sweep_details(
    df: pd.DataFrame,
    bias: str,
    lookback: int = 180,
    tol_atr: float = 0.12,
    wick_ratio_min: float = 0.55,
) -> Dict[str, Any]:
    """
    LONG:
      - raid below EQL then reclaim (close back above level)
    SHORT:
      - raid above EQH then reject (close back below level)
    """
    out: Dict[str, Any] = {
        "ok": False,
        "bias": (bias or "").upper(),
        "level": None,
        "sweep_extreme": None,
        "reclaim_close": None,
        "wick_ratio": 0.0,
        "body_ratio": 0.0,
        "kind": None,
    }

    if df is None or getattr(df, "empty", True) or len(df) < 80:
        return out

    b = (bias or "").upper()
    if b not in {"LONG", "SHORT"}:
        return out

    if not _has_cols(df, ("open", "high", "low", "close")):
        return out

    d = df.tail(int(max(80, lookback))).copy()

    atr = _atr_last(d, 14)
    tol = float(tol_atr) * max(atr, 1e-12)

    eq_highs, eq_lows = detect_equal_levels(d, max_window=int(max(120, lookback)), tol_mult_atr=float(tol_atr))

    last = d.iloc[-1]
    o = float(last["open"])
    h = float(last["high"])
    l = float(last["low"])
    c = float(last["close"])
    rng = max(h - l, 1e-12)
    body = abs(c - o)

    out["reclaim_close"] = c
    out["body_ratio"] = float(body / rng)

    if b == "LONG":
        if not eq_lows:
            return out
        lvl = float(eq_lows[0].get("level") or 0.0)
        if lvl <= 0:
            return out

        lower_wick = max(min(o, c) - l, 0.0)
        out["wick_ratio"] = float(lower_wick / rng)
        out["level"] = lvl
        out["sweep_extreme"] = l

        if (l < (lvl - tol)) and (c > (lvl + tol * 0.25)) and (out["wick_ratio"] >= float(wick_ratio_min)):
            out["ok"] = True
            out["kind"] = "EQ_LOW_SWEEP"
        return out

    # SHORT
    if not eq_highs:
        return out
    lvl = float(eq_highs[0].get("level") or 0.0)
    if lvl <= 0:
        return out

    upper_wick = max(h - max(o, c), 0.0)
    out["wick_ratio"] = float(upper_wick / rng)
    out["level"] = lvl
    out["sweep_extreme"] = h

    if (h > (lvl + tol)) and (c < (lvl - tol * 0.25)) and (out["wick_ratio"] >= float(wick_ratio_min)):
        out["ok"] = True
        out["kind"] = "EQ_HIGH_SWEEP"
    return out


# =====================================================================
# BOS / CHOCH / COS (buffered ATR, internal/external)
# =====================================================================

def _classify_bos(
    swings: Dict[str, List[Tuple[int, float]]],
    last_close: float,
    *,
    break_buffer: float = 0.0,
) -> Dict[str, Any]:
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

    # resolve conflict
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

    a = _atr_last(df, 14)
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
# HTF trend alignment
# =====================================================================

def htf_trend_ok(df_htf: pd.DataFrame, bias: str) -> bool:
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
# BOS quality score (desk)
# =====================================================================

def bos_quality_details(
    df: pd.DataFrame,
    oi_series: Optional[pd.Series] = None,
    vol_lookback: int = 60,
    df_liq: Optional[pd.DataFrame] = None,
    price: Optional[float] = None,
    tick: float = 0.1,
    direction: Optional[str] = None,  # "UP"/"DOWN"
) -> Dict[str, Any]:
    """
    ok = score >= 0
    Score components:
      + RVOL
      + Body ratio
      + Impulse vs ATR
      + Close location aligned
      + Liquidity raid bonus
      + OI slope (if available)
    """
    if df is None or len(df) < max(int(vol_lookback), 50) or not _has_cols(df, ("open", "high", "low", "close", "volume")):
        return {"ok": True, "score": 0.0, "grade": "C", "reasons": ["not_enough_data"]}

    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    v = df["volume"].astype(float)

    last_open = float(o.iloc[-1])
    last_high = float(h.iloc[-1])
    last_low = float(l.iloc[-1])
    last_close = float(c.iloc[-1])
    last_vol = float(v.iloc[-1])

    rng = float(last_high - last_low)
    rng = max(rng, 1e-12)
    body = float(abs(last_close - last_open))
    body_ratio = float(body / rng)

    atr = _atr_last(df, 14)
    impulse_atr = float((rng / atr) if atr > 0 else 0.0)

    w = df.tail(int(vol_lookback))
    avg_vol = float(w["volume"].astype(float).mean()) if len(w) else 0.0
    volume_factor = float((last_vol / avg_vol) if avg_vol > 0 else 1.0)

    range_pos = float((last_close - last_low) / rng)

    upper_wick = float(last_high - max(last_open, last_close))
    lower_wick = float(min(last_open, last_close) - last_low)
    wick_ratio = float((upper_wick + lower_wick) / rng)

    # OI slope (simple)
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

    # Liquidity raid bonus
    liquidity_raid = False
    try:
        src = df_liq if (df_liq is not None and not df_liq.empty) else df
        lv = detect_equal_levels(src.tail(220))
        eq_highs = lv.get("eq_highs", []) or []
        eq_lows = lv.get("eq_lows", []) or []
        buf = max((atr * 0.08) if atr > 0 else 0.0, abs(float(tick)) * 5.0, last_close * 0.0004)

        d = (direction or "").upper()
        if d == "UP":
            for x in eq_highs:
                lvl = float(x.get("level") or 0.0)
                if lvl > 0 and last_high > (lvl + buf) and last_close < (lvl + 0.25 * buf):
                    liquidity_raid = True
                    break
        elif d == "DOWN":
            for x in eq_lows:
                lvl = float(x.get("level") or 0.0)
                if lvl > 0 and last_low < (lvl - buf) and last_close > (lvl - 0.25 * buf):
                    liquidity_raid = True
                    break
    except Exception:
        liquidity_raid = False

    # ------------------------------------------------------------
    # SCORE
    # ------------------------------------------------------------
    score = 0.0
    reasons: List[str] = []

    # impulse / ATR
    if atr > 0 and rng < 0.25 * atr:
        score -= 1.0
        reasons.append("tiny_range")
    elif atr > 0 and impulse_atr >= 1.4:
        score += 1.0
    elif atr > 0 and impulse_atr >= 0.9:
        score += 0.5

    # volume
    if volume_factor >= 1.8:
        score += 1.5
    elif volume_factor >= 1.2:
        score += 1.0
    elif volume_factor >= 0.7:
        score += 0.0
    else:
        score -= 0.8
        reasons.append("low_volume")

    # body
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

    # close location aligned
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

    # raid bonus
    if liquidity_raid:
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
        "impulse_atr": float(impulse_atr),
        "wick_ratio": float(wick_ratio),
        "range_pos": float(range_pos),
        "oi_slope": float(oi_slope),
        "oi_used": bool(oi_used),
        "liquidity_raid": bool(liquidity_raid),
        "reasons": reasons,
    }


# =====================================================================
# Order blocks (light)
# =====================================================================

def _detect_order_blocks(df: pd.DataFrame, lookback: int = 120) -> Dict[str, Any]:
    if df is None or len(df) < 60 or not _has_cols(df, ("open", "high", "low", "close")):
        return {"bullish": None, "bearish": None}

    sub = df.tail(int(lookback)).reset_index(drop=True)
    idx_offset = len(df) - len(sub)

    atr = _atr_last(sub, 14)
    if atr <= 0:
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
        impulse = rng / atr if atr > 0 else 0.0

        if impulse >= 1.6 and body / max(rng, 1e-12) >= 0.45:
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
            elif down:
                for j in range(max(0, i - 8), i):
                    if c[j] > o[j]:
                        bearish_ob = {
                            "index": int(idx_offset + j),
                            "low": float(min(o[j], c[j])),
                            "high": float(max(o[j], c[j])),
                            "direction": "bearish",
                        }

    return {"bullish": bullish_ob, "bearish": bearish_ob}


# =====================================================================
# FVG (low/high + direction) + mitigated filter
# =====================================================================

def _detect_fvg(df: pd.DataFrame, lookback: int = 140, keep_last: int = 8) -> List[Dict[str, Any]]:
    zones: List[Dict[str, Any]] = []
    if df is None or len(df) < 10 or not _has_cols(df, ("high", "low")):
        return zones

    sub = df.tail(int(lookback)).reset_index(drop=True)
    idx_offset = len(df) - len(sub)

    H = sub["high"].astype(float).to_numpy()
    L = sub["low"].astype(float).to_numpy()

    for i in range(2, len(sub)):
        # bullish fvg: low[i] > high[i-2]
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

        # bearish fvg: high[i] < low[i-2]
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

    # mitigated detection
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
# Main entry: analyze_structure(df_h1)
# =====================================================================

def analyze_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Output keys used by analyze_signal:
      - trend: "LONG"/"SHORT"/"RANGE"
      - bos: bool
      - bos_direction: "UP"/"DOWN"|None
      - bos_type: "INTERNAL"/"EXTERNAL"|None
      - broken_level: float|None
      - fvg_zones: list[dict] with low/high + direction
      - oi_series: Series|None (if df has 'oi')
      - liquidity: detect_equal_levels() output (unpackable)
    """
    if df is None or len(df) < 50 or not _has_cols(df, ("open", "high", "low", "close")):
        return {
            "trend": "RANGE",
            "swings": {"highs": [], "lows": []},
            "liquidity": EqualLevels(eq_highs=[], eq_lows=[]),
            "bos": False,
            "choch": False,
            "cos": False,
            "bos_type": None,
            "bos_direction": None,
            "broken_level": None,
            "order_blocks": {"bullish": None, "bearish": None},
            "fvg_zones": [],
            "oi_series": None,
        }

    trend = _trend_from_ema(df["close"].astype(float))
    swings = find_swings(df, left=3, right=3)
    liquidity = detect_equal_levels(df, left=3, right=3, max_window=220)
    bos_block = _detect_bos_choch_cos(df)
    ob = _detect_order_blocks(df, lookback=120)
    fvg_zones = _detect_fvg(df, lookback=160, keep_last=8)

    oi_series = df["oi"] if "oi" in df.columns else None

    return {
        "trend": trend,
        "swings": swings,
        "liquidity": liquidity,
        "bos": bool(bos_block.get("bos")),
        "choch": bool(bos_block.get("choch")),
        "cos": bool(bos_block.get("cos")),
        "bos_type": bos_block.get("bos_type"),
        "bos_direction": bos_block.get("direction"),
        "broken_level": bos_block.get("broken_level"),
        "order_blocks": ob,
        "fvg_zones": fvg_zones,
        "oi_series": oi_series,
    }


# =====================================================================
# Commitment score (optional / kept)
# =====================================================================

def commitment_score(
    df: pd.DataFrame,
    oi_series: Optional[pd.Series] = None,
    cvd_series: Optional[pd.Series] = None,
) -> float:
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
