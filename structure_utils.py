# =====================================================================
# structure_utils.py — Institutional Structure Engine (Desk v4)
# BOS / CHOCH / COS / Internal vs External / Liquidity EQH-EQL / Sweep / OB / FVG / HTF
# =====================================================================
# ✅ FVG zones: format stable {low, high, direction, index, mitigated}
# ✅ EQH/EQL: renvoie LISTES de floats (eq_highs/eq_lows) + tol utilisé
# ✅ Liquidity sweep: fonctionne vraiment (wick sweep + reclaim)
# ✅ Trend: EMA20/50 + slope + spread threshold (moins de faux RANGE)
# ✅ BOS: buffered ATR-aware, internal/external, broken_level
# ✅ BOS quality: scoring desk (volume/body/impulse/close location/wicks/oi slope/sweep bonus)
# ✅ Fail-safe partout
#
# API utilisée par analyze_signal.py :
#   - analyze_structure(df) -> dict
#   - htf_trend_ok(df_htf, bias) -> bool
#   - bos_quality_details(...)
#   - detect_equal_levels(...)
#   - liquidity_sweep_details(...)
#   - has_liquidity_zone(...)
# =====================================================================

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd


# =====================================================================
# Utils
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


# Try to reuse true ATR from indicators if available (preferred)
_compute_atr_ext = None
try:
    from indicators import true_atr as _compute_atr_ext  # type: ignore
except Exception:
    try:
        from indicators import compute_atr as _compute_atr_ext  # type: ignore
    except Exception:
        _compute_atr_ext = None


def _atr_last(df: pd.DataFrame, length: int = 14) -> float:
    """
    Returns last ATR value as float (0 if not available).
    """
    try:
        if df is None or df.empty or len(df) < length + 3:
            return 0.0
        if not _has_cols(df, ("high", "low", "close")):
            return 0.0

        if _compute_atr_ext is not None:
            s = _compute_atr_ext(df, length=length)  # series
            if s is None or len(s) == 0:
                return 0.0
            v = float(s.iloc[-1])
            return max(0.0, v) if np.isfinite(v) else 0.0

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
    Returns: {"highs": [(idx, price), ...], "lows": [(idx, price), ...]}
    """
    highs: List[Tuple[int, float]] = []
    lows: List[Tuple[int, float]] = []

    if df is None or len(df) < left + right + 8 or not _has_cols(df, ("high", "low")):
        return {"highs": highs, "lows": lows}

    h = df["high"].astype(float).to_numpy()
    l = df["low"].astype(float).to_numpy()

    for i in range(left, len(df) - right):
        win_h = h[i - left: i + right + 1]
        win_l = l[i - left: i + right + 1]

        hi = float(h[i])
        lo = float(l[i])
        if not (np.isfinite(hi) and np.isfinite(lo)):
            continue

        if hi >= float(np.max(win_h)):
            # keep last occurrence in duplicated max window
            if i == (i - left + int(np.argmax(win_h))):
                highs.append((i, hi))

        if lo <= float(np.min(win_l)):
            if i == (i - left + int(np.argmin(win_l))):
                lows.append((i, lo))

    return {"highs": highs, "lows": lows}


# =====================================================================
# Liquidity EQH/EQL (equal highs/lows) — returns floats
# =====================================================================

def _cluster_levels(levels: List[float], tolerance: float) -> List[float]:
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
        # compare to last element (stable)
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
) -> Dict[str, Any]:
    """
    Detect EQH/EQL pools from swing points.
    Returns:
      {
        "eq_highs": [float...],
        "eq_lows": [float...],
        "tol": float
      }
    """
    if df is None or len(df) < left + right + 15 or not _has_cols(df, ("high", "low", "close")):
        return {"eq_highs": [], "eq_lows": [], "tol": 0.0}

    sub = df.tail(int(max_window)).reset_index(drop=True)
    swings = find_swings(sub, left=left, right=right)

    high_prices = [p for _, p in (swings.get("highs") or [])]
    low_prices = [p for _, p in (swings.get("lows") or [])]

    last_price = float(sub["close"].astype(float).iloc[-1])
    atr = _atr_last(sub, 14)
    base_range = _median_range(sub, 60)

    tol = 0.0
    if atr > 0:
        tol = atr * float(tol_mult_atr)
    if tol <= 0 and base_range > 0:
        tol = base_range * float(tol_mult_range)
    if tol <= 0:
        tol = max(1e-12, last_price * float(tol_fallback_pct))

    eq_highs = sorted(_cluster_levels(high_prices, tol))
    eq_lows = sorted(_cluster_levels(low_prices, tol))

    return {"eq_highs": eq_highs, "eq_lows": eq_lows, "tol": float(tol)}


def _nearest_level(levels: List[float], price: float) -> Optional[float]:
    if not levels:
        return None
    p = float(price)
    best = None
    bestd = 1e18
    for x in levels:
        d = abs(float(x) - p)
        if d < bestd:
            bestd = d
            best = float(x)
    return best


def has_liquidity_zone(df: pd.DataFrame, direction: str, lookback: int = 200) -> bool:
    """
    Quick helper: True if an EQ pool exists near current price.
      - LONG  : EQH near price (overhead liquidity)
      - SHORT : EQL near price (below liquidity)
    """
    try:
        if df is None or df.empty or len(df) < 60:
            return False
        d = (direction or "").upper()
        snap = detect_equal_levels(df.tail(int(lookback)))
        eqh = snap.get("eq_highs") or []
        eql = snap.get("eq_lows") or []
        close = float(df["close"].astype(float).iloc[-1])
        atr = _atr_last(df, 14)
        buf = max(atr * 0.15, close * 0.0008)

        if d == "LONG":
            # overhead pool near
            lvl = _nearest_level(eqh, close)
            return bool(lvl is not None and abs(lvl - close) <= buf)
        if d == "SHORT":
            lvl = _nearest_level(eql, close)
            return bool(lvl is not None and abs(lvl - close) <= buf)
        return False
    except Exception:
        return False


# =====================================================================
# Liquidity Sweep (stop-hunt reclaim) — works
# =====================================================================

def liquidity_sweep_details(
    df: pd.DataFrame,
    direction: str,
    lookback: int = 160,
    tol_atr: float = 0.12,
    wick_ratio_min: float = 0.55,
) -> Dict[str, Any]:
    """
    Desk sweep detector on the LAST candle:

    LONG:
      - sweeps below an EQL level (low < level - tol)
      - closes back above (close > level + tol*0.25)
      - has meaningful lower wick

    SHORT:
      - sweeps above an EQH level (high > level + tol)
      - closes back below (close < level - tol*0.25)
      - has meaningful upper wick
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
    try:
        if df is None or getattr(df, "empty", True) or len(df) < 80:
            return out
        if not _has_cols(df, ("open", "high", "low", "close")):
            return out

        d = str(direction).upper()
        if d not in {"LONG", "SHORT"}:
            return out

        dfw = df.tail(int(max(lookback, 80))).copy()
        last = dfw.iloc[-1]

        o = float(last["open"])
        h = float(last["high"])
        l = float(last["low"])
        c = float(last["close"])

        rng = max(h - l, 1e-12)
        body = abs(c - o)
        out["reclaim_close"] = c
        out["body_ratio"] = float(body / rng)

        atr = _atr_last(dfw, 14)
        tol = float(tol_atr) * max(atr, 1e-12)

        snap = detect_equal_levels(dfw, max_window=int(max(120, lookback)), tol_mult_atr=float(tol_atr))
        eqh = snap.get("eq_highs") or []
        eql = snap.get("eq_lows") or []

        if d == "LONG":
            if not eql:
                return out
            lvl = _nearest_level(eql, c)
            if lvl is None:
                return out

            lower_wick = max(min(o, c) - l, 0.0)
            out["wick_ratio"] = float(lower_wick / rng)
            out["level"] = float(lvl)
            out["sweep_extreme"] = float(l)

            if (l < (lvl - tol)) and (c > (lvl + tol * 0.25)) and (out["wick_ratio"] >= float(wick_ratio_min)):
                out["ok"] = True
                out["kind"] = "EQL_SWEEP_RECLAIM"
            return out

        # SHORT
        if not eqh:
            return out
        lvl = _nearest_level(eqh, c)
        if lvl is None:
            return out

        upper_wick = max(h - max(o, c), 0.0)
        out["wick_ratio"] = float(upper_wick / rng)
        out["level"] = float(lvl)
        out["sweep_extreme"] = float(h)

        if (h > (lvl + tol)) and (c < (lvl - tol * 0.25)) and (out["wick_ratio"] >= float(wick_ratio_min)):
            out["ok"] = True
            out["kind"] = "EQH_SWEEP_RECLAIM"
        return out

    except Exception:
        return out


# =====================================================================
# Trend via EMA 20/50 (desk bias)
# =====================================================================

def _trend_from_ema(close: pd.Series, fast: int = 20, slow: int = 50) -> str:
    """
    LONG if ema20>ema50 AND slope(ema20)>0 AND spread%>min_spread
    SHORT if ema20<ema50 AND slope(ema20)<0 AND spread%<-min_spread
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
# BOS / CHOCH / COS (buffered ATR-aware)
# =====================================================================

def _classify_bos(
    swings: Dict[str, List[Tuple[int, float]]],
    last_close: float,
    *,
    break_buffer: float = 0.0,
) -> Dict[str, Any]:
    """
    Determine BOS based on last swing high/low with optional buffer.
    Returns:
      {"bos": bool, "direction": "UP"/"DOWN"/None, "bos_type": "INTERNAL"/"EXTERNAL"/None, "broken_level": float|None}
    """
    res = {"bos": False, "direction": None, "bos_type": None, "broken_level": None}

    highs = swings.get("highs") or []
    lows = swings.get("lows") or []

    if not highs and not lows:
        return res

    last_hi = float(highs[-1][1]) if highs else None
    last_lo = float(lows[-1][1]) if lows else None

    # external ref: use last 3 swings (more "macro" than last pivot only)
    ext_hi = float(max([p for _, p in highs[-3:]])) if len(highs) >= 2 else last_hi
    ext_lo = float(min([p for _, p in lows[-3:]])) if len(lows) >= 2 else last_lo

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

    # if both triggered (rare), keep stronger displacement
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
      - BOS: break of last swing high/low (with buffer)
      - CHOCH: BOS opposite to EMA trend
      - COS: BOS aligned with EMA trend
    """
    if df is None or len(df) < 80 or not _has_cols(df, ("open", "high", "low", "close")):
        return {"bos": False, "choch": False, "cos": False, "direction": None, "bos_type": None, "broken_level": None}

    close = df["close"].astype(float)
    last_close = float(close.iloc[-1])

    swings = find_swings(df, left=3, right=3)
    atr = _atr_last(df, 14)
    buf = max(atr * 0.12, last_close * 0.0004)

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
    H4 veto:
      - H4 LONG  veto SHORT
      - H4 SHORT veto LONG
      - H4 RANGE => allow both
    """
    try:
        if df_htf is None or len(df_htf) < 80 or not _has_cols(df_htf, ("close",)):
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
# BOS quality (desk scoring)
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
    Non-binary BOS quality scoring.
    ok = score >= 0
    """
    try:
        if df is None or len(df) < max(int(vol_lookback), 60) or not _has_cols(df, ("open", "high", "low", "close", "volume")):
            return {"ok": True, "score": 0.0, "grade": "C", "reasons": ["not_enough_data"]}

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

        atr = _atr_last(df, 14)
        impulse = (rng / atr) if atr > 0 else 0.0

        w = df.tail(int(vol_lookback))
        avg_vol = float(w["volume"].astype(float).mean()) if len(w) else 0.0
        volume_factor = (last_vol / avg_vol) if avg_vol > 0 else 1.0

        range_pos = ((last_close - last_low) / rng) if rng > 0 else 0.5

        upper_wick = float(last_high - max(last_open, last_close))
        lower_wick = float(min(last_open, last_close) - last_low)
        wick_ratio = float((upper_wick + lower_wick) / rng) if rng > 0 else 0.0

        # direction normalize
        d = (direction or "").upper()
        if d in {"LONG", "BUY"}:
            d = "UP"
        if d in {"SHORT", "SELL"}:
            d = "DOWN"

        # oi slope
        oi_slope = 0.0
        oi_used = False
        if oi_series is not None:
            try:
                s = pd.Series(oi_series).astype(float)
                if len(s) >= 12:
                    base = float(s.iloc[-12])
                    last = float(s.iloc[-1])
                    if abs(base) > 1e-12 and np.isfinite(last) and np.isfinite(base):
                        oi_slope = (last - base) / abs(base)
                        oi_used = True
            except Exception:
                oi_slope = 0.0
                oi_used = False

        # liquidity sweep bonus (wick above/below eq pool + close back)
        liquidity_sweep = False
        try:
            if df_liq is None:
                df_liq = df
            snap = detect_equal_levels(df_liq.tail(200))
            eqh = snap.get("eq_highs") or []
            eql = snap.get("eq_lows") or []
            buf = max((atr * 0.08) if atr > 0 else 0.0, abs(float(tick)) * 5.0, last_close * 0.0004)

            if d == "UP" and eqh:
                lvl = _nearest_level(eqh, last_close)
                if lvl is not None and last_high > (lvl + buf) and last_close < (lvl + 0.25 * buf):
                    liquidity_sweep = True
            elif d == "DOWN" and eql:
                lvl = _nearest_level(eql, last_close)
                if lvl is not None and last_low < (lvl - buf) and last_close > (lvl - 0.25 * buf):
                    liquidity_sweep = True
        except Exception:
            liquidity_sweep = False

        # ------------------------------------------------------------
        # SCORE
        # ------------------------------------------------------------
        score = 0.0
        reasons: List[str] = []

        # impulse / dead market
        if rng <= 0 or (atr > 0 and rng < 0.25 * atr):
            score -= 1.0
            reasons.append("tiny_range")
        elif atr > 0 and impulse >= 1.4:
            score += 1.0
        elif atr > 0 and impulse >= 0.9:
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

        # wicky fake break
        if wick_ratio >= 0.65 and body_ratio < 0.35:
            score -= 0.8
            reasons.append("wicky_candle")

        # close location vs direction
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

        # oi support
        if oi_used:
            abs_oi = abs(float(oi_slope))
            if abs_oi >= 0.012:
                score += 1.2
            elif abs_oi >= 0.005:
                score += 0.6
            elif abs_oi < 0.0012:
                score -= 0.5
                reasons.append("weak_oi")

        # sweep bonus
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

    except Exception:
        return {"ok": True, "score": 0.0, "grade": "C", "reasons": ["exception"]}


# =====================================================================
# Order blocks (lightweight)
# =====================================================================

def _detect_order_blocks(df: pd.DataFrame, lookback: int = 120) -> Dict[str, Any]:
    if df is None or len(df) < 80 or not _has_cols(df, ("open", "high", "low", "close")):
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
# FVG — stable format
# =====================================================================

def _detect_fvg(df: pd.DataFrame, lookback: int = 140, keep_last: int = 8) -> List[Dict[str, Any]]:
    zones: List[Dict[str, Any]] = []
    if df is None or len(df) < 20 or not _has_cols(df, ("high", "low")):
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
                        "direction": "bullish",
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
                        "direction": "bearish",
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
            later = sub.iloc[zi + 1:]
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
# Main structure engine (H1)
# =====================================================================

def analyze_structure(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or len(df) < 60 or not _has_cols(df, ("open", "high", "low", "close")):
        return {
            "trend": "RANGE",
            "swings": {"highs": [], "lows": []},
            "liquidity": {"eq_highs": [], "eq_lows": [], "tol": 0.0},
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
    levels = detect_equal_levels(df, left=3, right=3, max_window=200)
    bos_block = _detect_bos_choch_cos(df)
    ob = _detect_order_blocks(df, lookback=120)
    fvg_zones = _detect_fvg(df, lookback=140, keep_last=8)

    oi_series = df["oi"] if "oi" in df.columns else None

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
    }


# =====================================================================
# Commitment score (optional)
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
