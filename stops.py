# =====================================================================
# stops.py — Desk Lead Institutional Stop Engine (PRO v2)
# Compatible with analyze_signal.py (protective_stop_long / protective_stop_short)
# =====================================================================
# Desk-lead upgrades:
# - Directional tick rounding (LONG SL rounds DOWN, SHORT SL rounds UP)
# - Uses settings.py if present (ATR_LEN, ATR_MULT_SL, buffers, caps, min ticks)
# - Structural SL candidates:
#     * Swing-based (most recent swing beyond entry)
#     * Liquidity-based (equal highs/lows) with dedicated liquidity buffer
#     * ATR-based (risk cap / fallback)
# - Guardrails:
#     * Minimum SL distance in ticks (avoid “too tight”)
#     * Maximum SL distance in % (avoid absurd SL)
#     * Always enforce SL side correct vs entry
# - Rich meta payload for logs/debug (chosen ref, buffers, ticks, pct)
# =====================================================================

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

from structure_utils import find_swings, detect_equal_levels
from indicators import true_atr


# =====================================================================
# Optional settings (safe defaults)
# =====================================================================

try:
    from settings import (
        ATR_LEN,
        ATR_MULT_SL,
        ATR_MULT_SL_CAP,
        SL_BUFFER_PCT,
        SL_BUFFER_TICKS,
        MIN_SL_TICKS,
        MAX_SL_PCT,
        LIQ_BUFFER_PCT,
        LIQ_BUFFER_TICKS,
        STRUCT_LOOKBACK,
    )
except Exception:
    ATR_LEN = 14
    ATR_MULT_SL = 2.5
    ATR_MULT_SL_CAP = 3.5

    SL_BUFFER_PCT = 0.0020
    SL_BUFFER_TICKS = 3
    MIN_SL_TICKS = 3
    MAX_SL_PCT = 0.07

    LIQ_BUFFER_PCT = 0.0008
    LIQ_BUFFER_TICKS = 3

    STRUCT_LOOKBACK = 60


REQUIRED_COLS = ("open", "high", "low", "close", "volume")


def _ensure_ohlcv(df: pd.DataFrame) -> bool:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False
    for c in REQUIRED_COLS:
        if c not in df.columns:
            return False
    return True


# =====================================================================
# Tick rounding (directional)
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
    """
    Floors price to tick grid (safe for LONG stops: ensure stop is not above intended).
    """
    t = _safe_tick(tick)
    if t <= 0:
        return float(price)
    p = float(price)
    return float(np.floor(p / t) * t)


def _ceil_to_tick(price: float, tick: float) -> float:
    """
    Ceils price to tick grid (safe for SHORT stops: ensure stop is not below intended).
    """
    t = _safe_tick(tick)
    if t <= 0:
        return float(price)
    p = float(price)
    return float(np.ceil(p / t) * t)


# =====================================================================
# ATR
# =====================================================================

def _get_atr_value(df: pd.DataFrame, length: int = 14) -> float:
    """
    Return latest true ATR value with robust fallbacks.
    """
    try:
        if not _ensure_ohlcv(df) or len(df) < max(20, length + 2):
            return 0.0

        atr_series = true_atr(df, length=int(length))
        val = float(atr_series.iloc[-1])
        if not np.isfinite(val) or val <= 0:
            raise ValueError("ATR invalid")
        return float(val)
    except Exception:
        # fallback proxy
        try:
            w = df.tail(max(int(length), 14))
            if w.empty:
                return 0.0
            approx = float((w["high"].astype(float).max() - w["low"].astype(float).min()) / max(len(w), 1))
            return float(approx) if np.isfinite(approx) and approx > 0 else 0.0
        except Exception:
            return 0.0


# =====================================================================
# Structural references (swings / liquidity)
# =====================================================================

def _get_struct_window(df: pd.DataFrame) -> pd.DataFrame:
    try:
        n = int(STRUCT_LOOKBACK) if int(STRUCT_LOOKBACK) > 20 else 60
    except Exception:
        n = 60
    return df.tail(n)


def _last_swing_low_below(df: pd.DataFrame, entry: float) -> Optional[float]:
    """
    Pick the most recent swing low that is strictly below entry.
    If none, fallback to min low in a recent window.
    """
    try:
        sub = _get_struct_window(df)
        swings = find_swings(sub)
        lows = swings.get("lows", []) or []
        # walk backwards: most recent swing low below entry
        for _, p in reversed(lows):
            if np.isfinite(p) and float(p) < float(entry):
                return float(p)
        # fallback
        v = float(sub["low"].astype(float).min())
        return float(v) if np.isfinite(v) else None
    except Exception:
        return None


def _last_swing_high_above(df: pd.DataFrame, entry: float) -> Optional[float]:
    """
    Pick the most recent swing high that is strictly above entry.
    If none, fallback to max high in a recent window.
    """
    try:
        sub = _get_struct_window(df)
        swings = find_swings(sub)
        highs = swings.get("highs", []) or []
        for _, p in reversed(highs):
            if np.isfinite(p) and float(p) > float(entry):
                return float(p)
        v = float(sub["high"].astype(float).max())
        return float(v) if np.isfinite(v) else None
    except Exception:
        return None


def _liq_low_below(df: pd.DataFrame, entry: float) -> Optional[float]:
    """
    Highest equal-low strictly below entry (closest liquidity pool under price).
    """
    try:
        sub = df.tail(max(120, int(STRUCT_LOOKBACK)))
        liq = detect_equal_levels(sub)
        eq_lows = liq.get("eq_lows", []) or []
        below = [float(x) for x in eq_lows if np.isfinite(x) and float(x) < float(entry)]
        if not below:
            return None
        return float(max(below))
    except Exception:
        return None


def _liq_high_above(df: pd.DataFrame, entry: float) -> Optional[float]:
    """
    Lowest equal-high strictly above entry (closest liquidity pool over price).
    """
    try:
        sub = df.tail(max(120, int(STRUCT_LOOKBACK)))
        liq = detect_equal_levels(sub)
        eq_highs = liq.get("eq_highs", []) or []
        above = [float(x) for x in eq_highs if np.isfinite(x) and float(x) > float(entry)]
        if not above:
            return None
        return float(min(above))
    except Exception:
        return None


# =====================================================================
# Buffers / guardrails
# =====================================================================

def _buffer_price(entry: float, tick: float, pct: float, ticks: int) -> float:
    """
    Returns a positive buffer in price units.
    Uses max(pct*entry, ticks*tick).
    """
    entry_f = float(entry)
    t = _safe_tick(tick)
    pct_buf = abs(entry_f) * float(pct) if np.isfinite(entry_f) else 0.0
    tick_buf = float(ticks) * t if t > 0 else 0.0
    out = float(max(pct_buf, tick_buf, 0.0))
    return out


def _enforce_min_distance_long(sl: float, entry: float, tick: float) -> float:
    """
    Ensure entry - sl >= MIN_SL_TICKS * tick (if tick known).
    """
    t = _safe_tick(tick)
    if t <= 0:
        return float(sl)

    min_dist = float(max(int(MIN_SL_TICKS), 0)) * t
    if min_dist <= 0:
        return float(sl)

    if (float(entry) - float(sl)) < min_dist:
        return float(entry) - min_dist
    return float(sl)


def _enforce_min_distance_short(sl: float, entry: float, tick: float) -> float:
    """
    Ensure sl - entry >= MIN_SL_TICKS * tick (if tick known).
    """
    t = _safe_tick(tick)
    if t <= 0:
        return float(sl)

    min_dist = float(max(int(MIN_SL_TICKS), 0)) * t
    if min_dist <= 0:
        return float(sl)

    if (float(sl) - float(entry)) < min_dist:
        return float(entry) + min_dist
    return float(sl)


def _cap_max_distance(sl: float, entry: float, side: str) -> float:
    """
    Hard cap SL distance in % of entry (MAX_SL_PCT) to avoid insane stops.
    If structural stop is too wide, we clamp closer using ATR_MULT_SL_CAP*ATR
    in the caller, then still enforce MAX_SL_PCT here.
    """
    entry_f = float(entry)
    sl_f = float(sl)
    if not (np.isfinite(entry_f) and np.isfinite(sl_f)) or entry_f == 0:
        return sl_f

    max_pct = float(MAX_SL_PCT)
    if max_pct <= 0:
        return sl_f

    if side == "LONG":
        # want sl <= entry, cap: entry - sl <= max_pct*entry
        max_dist = abs(entry_f) * max_pct
        dist = entry_f - sl_f
        if dist > max_dist:
            return entry_f - max_dist
        return sl_f

    # SHORT
    max_dist = abs(entry_f) * max_pct
    dist = sl_f - entry_f
    if dist > max_dist:
        return entry_f + max_dist
    return sl_f


# =====================================================================
# Candidate builder
# =====================================================================

def _build_long_candidates(df: pd.DataFrame, entry: float, tick: float) -> Dict[str, float]:
    """
    Returns dict of candidate SL raw (unrounded) prices for LONG.
    Lower is wider. We will choose the best (desk).
    """
    entry_f = float(entry)
    atr_v = _get_atr_value(df, length=int(ATR_LEN))

    gen_buf = _buffer_price(entry_f, tick, float(SL_BUFFER_PCT), int(SL_BUFFER_TICKS))
    liq_buf = _buffer_price(entry_f, tick, float(LIQ_BUFFER_PCT), int(LIQ_BUFFER_TICKS))

    swing = _last_swing_low_below(df, entry_f)
    liq = _liq_low_below(df, entry_f)

    cands: Dict[str, float] = {}

    # Swing-based
    if swing is not None and np.isfinite(swing):
        cands["SWING"] = float(swing) - gen_buf

    # Liquidity-based: go BELOW the pool (stop hunt)
    if liq is not None and np.isfinite(liq):
        cands["LIQ"] = float(liq) - liq_buf

    # ATR-based fallback / cap (not too tight by default)
    if np.isfinite(atr_v) and atr_v > 0:
        cands["ATR"] = entry_f - float(ATR_MULT_SL) * atr_v
        cands["ATR_CAP"] = entry_f - float(ATR_MULT_SL_CAP) * atr_v

    return cands


def _build_short_candidates(df: pd.DataFrame, entry: float, tick: float) -> Dict[str, float]:
    """
    Returns dict of candidate SL raw (unrounded) prices for SHORT.
    Higher is wider.
    """
    entry_f = float(entry)
    atr_v = _get_atr_value(df, length=int(ATR_LEN))

    gen_buf = _buffer_price(entry_f, tick, float(SL_BUFFER_PCT), int(SL_BUFFER_TICKS))
    liq_buf = _buffer_price(entry_f, tick, float(LIQ_BUFFER_PCT), int(LIQ_BUFFER_TICKS))

    swing = _last_swing_high_above(df, entry_f)
    liq = _liq_high_above(df, entry_f)

    cands: Dict[str, float] = {}

    if swing is not None and np.isfinite(swing):
        cands["SWING"] = float(swing) + gen_buf

    if liq is not None and np.isfinite(liq):
        cands["LIQ"] = float(liq) + liq_buf

    if np.isfinite(atr_v) and atr_v > 0:
        cands["ATR"] = entry_f + float(ATR_MULT_SL) * atr_v
        cands["ATR_CAP"] = entry_f + float(ATR_MULT_SL_CAP) * atr_v

    return cands


def _choose_long_sl(entry: float, tick: float, candidates: Dict[str, float]) -> Tuple[float, str]:
    """
    Desk rule:
      - We prefer the *tightest* SL that is still valid (below entry) and respects min ticks,
        but we also cap max distance later.
      - So we filter valid candidates and choose the one closest to entry (highest SL).
    """
    entry_f = float(entry)

    valid = []
    for k, sl in candidates.items():
        if sl is None:
            continue
        slf = float(sl)
        if np.isfinite(slf) and slf < entry_f:
            valid.append((k, slf))

    if not valid:
        # fallback 4% under entry
        return (entry_f * 0.96, "FALLBACK_PCT")

    # choose highest SL (closest to entry)
    best_k, best_sl = sorted(valid, key=lambda x: x[1], reverse=True)[0]
    return float(best_sl), str(best_k)


def _choose_short_sl(entry: float, tick: float, candidates: Dict[str, float]) -> Tuple[float, str]:
    """
    For SHORT, prefer the *tightest* valid SL above entry (lowest SL).
    """
    entry_f = float(entry)

    valid = []
    for k, sl in candidates.items():
        if sl is None:
            continue
        slf = float(sl)
        if np.isfinite(slf) and slf > entry_f:
            valid.append((k, slf))

    if not valid:
        # fallback 4% above entry
        return (entry_f * 1.04, "FALLBACK_PCT")

    # choose lowest SL (closest to entry)
    best_k, best_sl = sorted(valid, key=lambda x: x[1])[0]
    return float(best_sl), str(best_k)


# =====================================================================
# PUBLIC API — Protective Stops
# =====================================================================

def protective_stop_long(
    df: pd.DataFrame,
    entry: float,
    tick: float = 0.1,
    return_meta: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """
    Desk Lead protective stop for LONG.

    Pipeline:
      1) Build candidates (LIQ / SWING / ATR)
      2) Pick tightest valid
      3) Enforce min distance in ticks
      4) Cap max distance in % (MAX_SL_PCT)
      5) Directional rounding (FLOOR)
      6) Final sanity: SL < entry
    """
    meta: Dict[str, Any] = {}

    try:
        entry_f = float(entry)
        t = _safe_tick(tick)

        if not _ensure_ohlcv(df) or len(df) < 20 or not np.isfinite(entry_f):
            sl_raw = entry_f * 0.96
            sl_raw = _enforce_min_distance_long(sl_raw, entry_f, t)
            sl_raw = _cap_max_distance(sl_raw, entry_f, side="LONG")
            sl_final = _floor_to_tick(sl_raw, t)
            if sl_final >= entry_f and t > 0:
                sl_final = _floor_to_tick(entry_f - max(float(MIN_SL_TICKS) * t, t), t)

            meta = {
                "mode": "fallback_df",
                "chosen": "FALLBACK_PCT",
                "entry": entry_f,
                "tick": t,
                "sl_raw": float(sl_raw),
                "sl_final": float(sl_final),
            }
            return (float(sl_final), meta) if return_meta else (float(sl_final), {})

        atr_v = _get_atr_value(df, length=int(ATR_LEN))

        cands = _build_long_candidates(df, entry_f, t)
        sl_raw, chosen = _choose_long_sl(entry_f, t, cands)

        # Guardrails
        sl_raw = _enforce_min_distance_long(sl_raw, entry_f, t)
        sl_raw = _cap_max_distance(sl_raw, entry_f, side="LONG")

        # If still not below entry, force minimal safe
        if sl_raw >= entry_f:
            fallback = entry_f - max(atr_v, abs(entry_f) * 0.01, (float(MIN_SL_TICKS) * t if t > 0 else 0.0))
            sl_raw = float(fallback)
            chosen = "FORCED_BELOW_ENTRY"

        # Round (floor) for safety
        sl_final = _floor_to_tick(sl_raw, t)

        # Final sanity vs entry
        if sl_final >= entry_f:
            if t > 0:
                sl_final = _floor_to_tick(entry_f - max(float(MIN_SL_TICKS) * t, t), t)
            else:
                sl_final = float(entry_f * 0.99)

        dist_abs = float(entry_f - sl_final)
        dist_pct = float(dist_abs / abs(entry_f)) if entry_f != 0 else None
        dist_ticks = float(dist_abs / t) if t > 0 else None

        meta = {
            "mode": "ok",
            "chosen": chosen,
            "entry": entry_f,
            "tick": t,
            "atr": float(atr_v),
            "candidates": {k: float(v) for k, v in cands.items()},
            "sl_raw": float(sl_raw),
            "sl_final": float(sl_final),
            "dist_abs": dist_abs,
            "dist_pct": dist_pct,
            "dist_ticks": dist_ticks,
            "buffers": {
                "sl_buffer_price": _buffer_price(entry_f, t, float(SL_BUFFER_PCT), int(SL_BUFFER_TICKS)),
                "liq_buffer_price": _buffer_price(entry_f, t, float(LIQ_BUFFER_PCT), int(LIQ_BUFFER_TICKS)),
            },
        }

        return (float(sl_final), meta) if return_meta else (float(sl_final), {})

    except Exception as e:
        entry_f = float(entry) if np.isfinite(float(entry)) else 0.0
        t = _safe_tick(tick)
        sl_raw = entry_f * 0.95
        sl_raw = _enforce_min_distance_long(sl_raw, entry_f, t)
        sl_raw = _cap_max_distance(sl_raw, entry_f, side="LONG")
        sl_final = _floor_to_tick(sl_raw, t)
        if sl_final >= entry_f and t > 0:
            sl_final = _floor_to_tick(entry_f - max(float(MIN_SL_TICKS) * t, t), t)

        meta = {"mode": "exception", "error": str(e), "sl_raw": float(sl_raw), "sl_final": float(sl_final)}
        return (float(sl_final), meta) if return_meta else (float(sl_final), {})


def protective_stop_short(
    df: pd.DataFrame,
    entry: float,
    tick: float = 0.1,
    return_meta: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """
    Desk Lead protective stop for SHORT.

    Pipeline:
      1) Build candidates (LIQ / SWING / ATR)
      2) Pick tightest valid
      3) Enforce min distance in ticks
      4) Cap max distance in % (MAX_SL_PCT)
      5) Directional rounding (CEIL)
      6) Final sanity: SL > entry
    """
    meta: Dict[str, Any] = {}

    try:
        entry_f = float(entry)
        t = _safe_tick(tick)

        if not _ensure_ohlcv(df) or len(df) < 20 or not np.isfinite(entry_f):
            sl_raw = entry_f * 1.04
            sl_raw = _enforce_min_distance_short(sl_raw, entry_f, t)
            sl_raw = _cap_max_distance(sl_raw, entry_f, side="SHORT")
            sl_final = _ceil_to_tick(sl_raw, t)
            if sl_final <= entry_f and t > 0:
                sl_final = _ceil_to_tick(entry_f + max(float(MIN_SL_TICKS) * t, t), t)

            meta = {
                "mode": "fallback_df",
                "chosen": "FALLBACK_PCT",
                "entry": entry_f,
                "tick": t,
                "sl_raw": float(sl_raw),
                "sl_final": float(sl_final),
            }
            return (float(sl_final), meta) if return_meta else (float(sl_final), {})

        atr_v = _get_atr_value(df, length=int(ATR_LEN))

        cands = _build_short_candidates(df, entry_f, t)
        sl_raw, chosen = _choose_short_sl(entry_f, t, cands)

        # Guardrails
        sl_raw = _enforce_min_distance_short(sl_raw, entry_f, t)
        sl_raw = _cap_max_distance(sl_raw, entry_f, side="SHORT")

        if sl_raw <= entry_f:
            fallback = entry_f + max(atr_v, abs(entry_f) * 0.01, (float(MIN_SL_TICKS) * t if t > 0 else 0.0))
            sl_raw = float(fallback)
            chosen = "FORCED_ABOVE_ENTRY"

        sl_final = _ceil_to_tick(sl_raw, t)

        if sl_final <= entry_f:
            if t > 0:
                sl_final = _ceil_to_tick(entry_f + max(float(MIN_SL_TICKS) * t, t), t)
            else:
                sl_final = float(entry_f * 1.01)

        dist_abs = float(sl_final - entry_f)
        dist_pct = float(dist_abs / abs(entry_f)) if entry_f != 0 else None
        dist_ticks = float(dist_abs / t) if t > 0 else None

        meta = {
            "mode": "ok",
            "chosen": chosen,
            "entry": entry_f,
            "tick": t,
            "atr": float(atr_v),
            "candidates": {k: float(v) for k, v in cands.items()},
            "sl_raw": float(sl_raw),
            "sl_final": float(sl_final),
            "dist_abs": dist_abs,
            "dist_pct": dist_pct,
            "dist_ticks": dist_ticks,
            "buffers": {
                "sl_buffer_price": _buffer_price(entry_f, t, float(SL_BUFFER_PCT), int(SL_BUFFER_TICKS)),
                "liq_buffer_price": _buffer_price(entry_f, t, float(LIQ_BUFFER_PCT), int(LIQ_BUFFER_TICKS)),
            },
        }

        return (float(sl_final), meta) if return_meta else (float(sl_final), {})

    except Exception as e:
        entry_f = float(entry) if np.isfinite(float(entry)) else 0.0
        t = _safe_tick(tick)
        sl_raw = entry_f * 1.05
        sl_raw = _enforce_min_distance_short(sl_raw, entry_f, t)
        sl_raw = _cap_max_distance(sl_raw, entry_f, side="SHORT")
        sl_final = _ceil_to_tick(sl_raw, t)
        if sl_final <= entry_f and t > 0:
            sl_final = _ceil_to_tick(entry_f + max(float(MIN_SL_TICKS) * t, t), t)

        meta = {"mode": "exception", "error": str(e), "sl_raw": float(sl_raw), "sl_final": float(sl_final)}
        return (float(sl_final), meta) if return_meta else (float(sl_final), {})
