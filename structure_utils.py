# =====================================================================
# structure_utils.py — Institutional Structure Engine ++ (Desk v2)
# BOS / CHOCH / Internal vs External / Liquidity / OB / FVG / HTF
# =====================================================================

from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd


# =====================================================================
# SWINGS (pivot highs / lows)
# =====================================================================

def find_swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> Dict[str, List[Tuple[int, float]]]:
    """
    Detect basic swing highs / lows (pivot highs / lows).

    A pivot high at index i is where high[i] is the max of [i-left, i+right].
    A pivot low  at index i is where low[i]  is the min of [i-left, i+right].

    Returns:
        {
          "highs": [(idx, price), ...],
          "lows" : [(idx, price), ...],
        }
    """
    highs: List[Tuple[int, float]] = []
    lows: List[Tuple[int, float]] = []

    if df is None or len(df) < left + right + 3:
        return {"highs": highs, "lows": lows}

    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)

    for i in range(left, len(df) - right):
        window_h = h[i - left : i + right + 1]
        window_l = l[i - left : i + right + 1]
        hi = h[i]
        lo = l[i]

        if hi >= window_h.max():
            highs.append((i, float(hi)))
        if lo <= window_l.min():
            lows.append((i, float(lo)))

    return {"highs": highs, "lows": lows}


# =====================================================================
# LIQUIDITY (equal highs / equal lows)
# =====================================================================

def _cluster_levels(levels: List[float], tolerance: float) -> List[float]:
    """
    Groups nearby price levels into liquidity clusters.

    Args:
        levels: list of raw high/low prices.
        tolerance: max absolute distance between prices to group them.

    Returns:
        List of cluster representative prices (mean of each cluster)
        with at least 2 contributing points.
    """
    if not levels:
        return []

    lv_sorted = sorted(levels)
    clusters: List[List[float]] = [[lv_sorted[0]]]

    for p in lv_sorted[1:]:
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
) -> Dict[str, List[float]]:
    """
    Detect institutional liquidity pools (equal highs / equal lows) using swings.

    - Restricts to last `max_window` bars.
    - Tolerance is based on recent volatility (median range * factor).

    Returns:
        {
          "eq_highs": [price1, price2, ...],
          "eq_lows" : [price1, price2, ...],
        }
    """
    if df is None or len(df) < left + right + 3:
        return {"eq_highs": [], "eq_lows": []}

    sub = df.tail(max_window).reset_index(drop=True)
    swings = find_swings(sub, left=left, right=right)

    high_prices = [p for _, p in swings["highs"]]
    low_prices = [p for _, p in swings["lows"]]

    ranges = (sub["high"] - sub["low"]).astype(float)
    if ranges.dropna().empty:
        base_range = np.nan
    else:
        base_range = float(np.nanmedian(ranges))

    # tolerances: ~10-20% of typical bar range, or ~0.1% of price as fallback
    if not np.isfinite(base_range) or base_range <= 0:
        last_price = float(sub["close"].iloc[-1])
        tolerance = last_price * 0.001
    else:
        tolerance = base_range * 0.15

    eq_highs = _cluster_levels(high_prices, tolerance)
    eq_lows = _cluster_levels(low_prices, tolerance)

    # sort ascending for convenience
    eq_highs = sorted(eq_highs)
    eq_lows = sorted(eq_lows)

    return {"eq_highs": eq_highs, "eq_lows": eq_lows}


# =====================================================================
# TREND via EMA 20/50 (LTF & HTF)
# =====================================================================

def _trend_from_ema(close: pd.Series, fast: int = 20, slow: int = 50) -> str:
    """
    Simple institutional bias from EMA20/EMA50 and slope:

      - LONG  : ema_fast > ema_slow and ema_fast rising
      - SHORT : ema_fast < ema_slow and ema_fast falling
      - RANGE : otherwise
    """
    if close is None or len(close) < slow + 5:
        return "RANGE"

    c = close.astype(float)
    ema_fast = c.ewm(span=fast, adjust=False).mean()
    ema_slow = c.ewm(span=slow, adjust=False).mean()

    ef = ema_fast.iloc[-5:]
    es = ema_slow.iloc[-5:]
    if len(ef) < 5 or len(es) < 5:
        return "RANGE"

    slope_fast = ef.iloc[-1] - ef.iloc[0]

    if ef.iloc[-1] > es.iloc[-1] and slope_fast > 0:
        return "LONG"
    if ef.iloc[-1] < es.iloc[-1] and slope_fast < 0:
        return "SHORT"
    return "RANGE"


# =====================================================================
# BOS / CHOCH : external vs internal
# =====================================================================

def _classify_bos(
    swings: Dict[str, List[Tuple[int, float]]],
    last_close: float,
) -> Dict[str, Any]:
    """
    Classifies the most recent break as BOS/CHOCH/COS and internal vs external.

    We use the last 3-4 swing highs/lows to define:
      - "external" structure = outermost high/low in the recent window
      - "internal" structure = intermediate highs/lows inside that range
    """
    highs = swings.get("highs", []) or []
    lows = swings.get("lows", []) or []

    res = {
        "bos": False,
        "choch": False,
        "cos": False,
        "direction": None,          # "UP" / "DOWN"
        "bos_type": None,           # "INTERNAL" / "EXTERNAL"
        "broken_level": None,
    }

    if len(highs) < 2 and len(lows) < 2:
        return res

    # Determine potential upside / downside levels
    last_hi_level = None
    ext_hi_level = None
    if len(highs) >= 2:
        hi_prices = [p for _, p in highs[-3:]]
        ext_hi_level = max(hi_prices)
        last_hi_level = highs[-1][1]

    last_lo_level = None
    ext_lo_level = None
    if len(lows) >= 2:
        lo_prices = [p for _, p in lows[-3:]]
        ext_lo_level = min(lo_prices)
        last_lo_level = lows[-1][1]

    # BOS up?
    bos_up = False
    broken_up_level = None
    if last_hi_level is not None and last_close > last_hi_level:
        bos_up = True
        broken_up_level = last_hi_level
        bos_type = "EXTERNAL" if ext_hi_level is not None and last_close > ext_hi_level else "INTERNAL"
    else:
        bos_type = None

    # BOS down?
    bos_dn = False
    broken_dn_level = None
    if last_lo_level is not None and last_close < last_lo_level:
        bos_dn = True
        broken_dn_level = last_lo_level
        bos_type_dn = "EXTERNAL" if ext_lo_level is not None and last_close < ext_lo_level else "INTERNAL"
    else:
        bos_type_dn = None

    if bos_up and not bos_dn:
        res["bos"] = True
        res["direction"] = "UP"
        res["bos_type"] = bos_type
        res["broken_level"] = broken_up_level
    elif bos_dn and not bos_up:
        res["bos"] = True
        res["direction"] = "DOWN"
        res["bos_type"] = bos_type_dn
        res["broken_level"] = broken_dn_level

    return res


def _detect_bos_choch_cos(df: pd.DataFrame) -> Dict[str, Any]:
    """
    High-level BOS / CHOCH / COS classification.

    - BOS : last close breaks a significant swing high or low.
    - CHOCH : BOS in the opposite direction of EMA trend.
    - COS : BOS in the same direction as EMA trend.
    - INTERNAL vs EXTERNAL: based on which swing was broken.
    """
    if df is None or len(df) < 40:
        return {
            "bos": False,
            "choch": False,
            "cos": False,
            "direction": None,
            "bos_type": None,
            "broken_level": None,
        }

    close = df["close"].astype(float)
    last_close = float(close.iloc[-1])

    swings = find_swings(df)
    bos_info = _classify_bos(swings, last_close)
    if not bos_info["bos"]:
        return bos_info

    trend = _trend_from_ema(close)
    direction = bos_info["direction"]

    bos = True
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

    bos_info["bos"] = bos
    bos_info["choch"] = choch
    bos_info["cos"] = cos

    return bos_info


# =====================================================================
# HTF trend alignment (H4 vs H1)
# =====================================================================

def htf_trend_ok(df_htf: pd.DataFrame, bias: str) -> bool:
    """
    Ensures H4 trend does not contradict H1 bias.

    Rules:
      - If H4 is LONG, a SHORT bias is vetoed.
      - If H4 is SHORT, a LONG bias is vetoed.
      - If H4 is RANGE, both are allowed.
    """
    if df_htf is None or len(df_htf) < 40:
        return True

    bias = (bias or "").upper()
    trend_htf = _trend_from_ema(df_htf["close"])

    if trend_htf == "LONG" and bias == "SHORT":
        return False
    if trend_htf == "SHORT" and bias == "LONG":
        return False
    return True


# =====================================================================
# BOS QUALITY v2 (score-based, missing OI = neutre)
# =====================================================================

def bos_quality_details(
    df: pd.DataFrame,
    oi_series: Optional[pd.Series] = None,
    vol_lookback: int = 60,
    vol_pct: float = 0.8,
    oi_min_trend: float = 0.003,
    oi_min_squeeze: float = -0.005,
    df_liq: Optional[pd.DataFrame] = None,
    price: Optional[float] = None,
    tick: float = 0.1,
    direction: Optional[str] = None,  # "UP" / "DOWN"
) -> Dict[str, Any]:
    """
    BOS QUALITY v2 (desk ++, moins bloquante)

    Idée :
      - On calcule un SCORE global plutôt que de bloquer dès qu'un critère est
        un peu faible.
      - Composants :
          * volume_factor (last_vol / avg_vol)
          * body_ratio (|close-open| / range)
          * OI slope (si oi_series réellement fourni)
          * liquidity_sweep (equal highs/lows cassés)
          * range_pos (position du close dans la bougie) vs direction

    Behaviour :
      - Si oi_series est None -> on ne pénalise PAS "weak_oi".
      - ok = True si score >= 0
      - ok = False si score < 0

    On garde les trucs illiquides (volume_factor ~0, body_ratio ~0) en rejet
    naturel via un score très négatif.
    """
    if df is None or len(df) < max(vol_lookback, 20):
        return {"ok": True, "score": 0.0, "reason": "not_enough_data"}

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

    # Fenêtre de volume
    w = df.tail(vol_lookback)
    avg_vol = float(w["volume"].mean()) if not w["volume"].empty else 0.0

    volume_factor = last_vol / avg_vol if avg_vol > 0 else 1.0

    rng = last_high - last_low
    body = abs(last_close - last_open)
    body_ratio = body / rng if rng > 0 else 0.0

    if rng > 0:
        range_pos = (last_close - last_low) / rng
    else:
        range_pos = 0.5

    # OI slope si dispo
    oi_slope = 0.0
    oi_penalized = False
    if oi_series is not None:
        try:
            s = pd.Series(oi_series).astype(float)
            if len(s) >= 10:
                base = float(s.iloc[-10])
                if abs(base) > 1e-12:
                    oi_slope = float(s.iloc[-1] - base) / abs(base)
        except Exception:
            oi_slope = 0.0

    # Liquidity sweep ?
    liquidity_sweep = False
    if df_liq is None:
        df_liq = df
    try:
        levels = detect_equal_levels(df_liq.tail(200))
        eq_highs = levels.get("eq_highs", []) or []
        eq_lows = levels.get("eq_lows", []) or []
        ref_price = float(price) if price is not None else last_close

        for lvl in eq_highs:
            if ref_price > lvl:
                liquidity_sweep = True
                break
        if not liquidity_sweep:
            for lvl in eq_lows:
                if ref_price < lvl:
                    liquidity_sweep = True
                    break
    except Exception:
        liquidity_sweep = False

    # ------------------------------------------------------------------
    # SCORE GLOBAL
    # ------------------------------------------------------------------
    score = 0.0
    reasons: List[str] = []

    # Volume : on score par zones
    if volume_factor >= 1.5:
        score += 2.0
    elif volume_factor >= 1.0:
        score += 1.0
    elif volume_factor >= 0.5:
        score += 0.0
    else:
        score -= 1.0
        reasons.append("low_volume")

    # Body ratio
    if body_ratio >= 0.6:
        score += 2.0
    elif body_ratio >= 0.4:
        score += 1.0
    elif body_ratio >= 0.25:
        score += 0.0
    else:
        score -= 1.0
        reasons.append("small_body")

    # OI (uniquement si série fournie)
    if oi_series is not None:
        abs_oi = abs(oi_slope)
        if abs_oi >= 0.01:
            score += 2.0
        elif abs_oi >= 0.004:
            score += 1.0
        elif abs_oi < 0.001:
            score -= 1.0
            oi_penalized = True
            reasons.append("weak_oi")

    # Sweep de liquidité = bonus
    if liquidity_sweep:
        score += 1.0

    # Position de close vs direction du BOS
    if direction is not None:
        d = direction.upper()
        if d == "UP":
            if range_pos > 0.6:
                score += 1.0
            elif range_pos < 0.3:
                score -= 1.0
                reasons.append("close_not_in_upper_range")
        elif d == "DOWN":
            if range_pos < 0.4:
                score += 1.0
            elif range_pos > 0.7:
                score -= 1.0
                reasons.append("close_not_in_lower_range")

    ok = score >= 0.0

    return {
        "ok": ok,
        "score": float(score),
        "volume_factor": float(volume_factor),
        "body_ratio": float(body_ratio),
        "oi_slope": float(oi_slope),
        "liquidity_sweep": bool(liquidity_sweep),
        "range_pos": float(range_pos),
        "reasons": reasons,
        "oi_penalized": bool(oi_penalized),
    }


# =====================================================================
# ORDER BLOCKS & FAIR VALUE GAPS (lightweight)
# =====================================================================

def _detect_order_blocks(df: pd.DataFrame, lookback: int = 80) -> Dict[str, Any]:
    """Very lightweight last-impulse order block detection.

    - Bullish OB: last bearish candle before an impulsive move up.
    - Bearish OB: last bullish candle before an impulsive move down.
    """
    if df is None or len(df) < 20:
        return {"bullish": None, "bearish": None}

    sub = df.tail(lookback)
    o = sub["open"].astype(float).to_numpy()
    c = sub["close"].astype(float).to_numpy()

    closes = c
    if closes.size < 10:
        return {"bullish": None, "bearish": None}

    N = min(5, closes.size - 1)
    impulse_ret = (closes[-1] - closes[-1 - N]) / max(abs(closes[-1 - N]), 1e-8)

    bullish_ob = None
    bearish_ob = None

    idx_offset = len(df) - len(sub)

    if impulse_ret > 0.03:  # > 3% up move
        for i in range(len(sub) - N - 1, len(sub) - 1):
            if c[i] < o[i]:  # bearish
                bullish_ob = {
                    "index": int(idx_offset + i),
                    "low": float(min(o[i], c[i])),
                    "high": float(max(o[i], c[i])),
                }
    elif impulse_ret < -0.03:  # > 3% down move
        for i in range(len(sub) - N - 1, len(sub) - 1):
            if c[i] > o[i]:  # bullish
                bearish_ob = {
                    "index": int(idx_offset + i),
                    "low": float(min(o[i], c[i])),
                    "high": float(max(o[i], c[i])),
                }

    return {"bullish": bullish_ob, "bearish": bearish_ob}


def _detect_fvg(df: pd.DataFrame, lookback: int = 80) -> List[Dict[str, Any]]:
    """Detects simple Fair Value Gaps (FVG) on last `lookback` bars."""
    zones: List[Dict[str, Any]] = []

    if df is None or len(df) < 5:
        return zones

    sub = df.tail(lookback)
    h = sub["high"].astype(float).to_numpy()
    l = sub["low"].astype(float).to_numpy()

    idx_offset = len(df) - len(sub)

    for i in range(1, len(sub) - 1):
        # bullish gap
        if l[i] > h[i - 1] and l[i] > h[i + 1]:
            zones.append({
                "type": "bullish",
                "start": float(h[i - 1]),
                "end": float(l[i]),
                "index": int(idx_offset + i),
            })
        # bearish gap
        if h[i] < l[i - 1] and h[i] < l[i + 1]:
            zones.append({
                "type": "bearish",
                "start": float(h[i]),
                "end": float(l[i - 1]),
                "index": int(idx_offset + i),
            })

    return zones


# =====================================================================
# STRUCTURE ENGINE (H1)
# =====================================================================

def analyze_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Main structure analysis entrypoint for H1.
    """
    if df is None or len(df) < 30:
        return {
            "trend": "RANGE",
            "swings": {"highs": [], "lows": []},
            "liquidity": {"eq_highs": [], "eq_lows": []},
            "bos": False,
            "choch": False,
            "cos": False,
            "bos_type": None,
            "bos_direction": None,
            "order_blocks": {"bullish": None, "bearish": None},
            "fvg_zones": [],
            "oi_series": None,
        }

    trend = _trend_from_ema(df["close"])
    swings = find_swings(df)
    levels = detect_equal_levels(df)
    bos_block = _detect_bos_choch_cos(df)
    ob = _detect_order_blocks(df)
    fvg_zones = _detect_fvg(df)

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
        "order_blocks": ob,
        "fvg_zones": fvg_zones,
        "oi_series": oi_series,
    }


# =====================================================================
# COMMITMENT SCORE (OI + CVD)
# =====================================================================

def commitment_score(
    df: pd.DataFrame,
    oi_series: Optional[pd.Series] = None,
    cvd_series: Optional[pd.Series] = None,
) -> float:
    """
    Very lightweight commitment proxy combining OI & CVD behaviours.
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
