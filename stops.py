# =====================================================================
# stops.py — Desk Lead Institutional Stop Engine (FULL VERSION)
# Compatible with analyze_signal.py (protective_stop_long / protective_stop_short)
# =====================================================================

from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from structure_utils import find_swings, detect_equal_levels
from indicators import true_atr


# =====================================================================
# Helper — Round to Tick
# =====================================================================

def _round_to_tick(price: float, tick: float) -> float:
    """Round a price to the nearest valid tick.

    We keep it simple: nearest multiple of tick. If tick is invalid,
    return price unchanged.
    """
    try:
        tick = float(tick)
        if not np.isfinite(tick) or tick <= 0:
            return float(price)
        return float(round(float(price) / tick) * tick)
    except Exception:
        return float(price)


# =====================================================================
# Internal helpers — structural references & ATR
# =====================================================================

def _get_last_swing_low(df: pd.DataFrame) -> float:
    swings = find_swings(df)
    lows = swings.get("lows", []) or []
    if not lows:
        # fallback: recent min low
        return float(df["low"].tail(30).min())
    # last swing low (most recent)
    return float(lows[-1][1])


def _get_last_swing_high(df: pd.DataFrame) -> float:
    swings = find_swings(df)
    highs = swings.get("highs", []) or []
    if not highs:
        # fallback: recent max high
        return float(df["high"].tail(30).max())
    # last swing high (most recent)
    return float(highs[-1][1])


def _get_liquidity_low(df: pd.DataFrame, entry: float) -> float:
    """Return the most relevant equal-lows level below price.

    We look for the *highest* equal low strictly below entry. If none,
    return np.nan and let the caller fallback to swing.
    """
    liq = detect_equal_levels(df)
    eq_lows = liq.get("eq_lows", []) or []
    below = [lvl for lvl in eq_lows if lvl < entry]
    if not below:
        return np.nan
    # highest equal low below price
    return float(max(below))


def _get_liquidity_high(df: pd.DataFrame, entry: float) -> float:
    """Return the most relevant equal-highs level above price.

    We look for the *lowest* equal high strictly above entry. If none,
    return np.nan and let the caller fallback to swing.
    """
    liq = detect_equal_levels(df)
    eq_highs = liq.get("eq_highs", []) or []
    above = [lvl for lvl in eq_highs if lvl > entry]
    if not above:
        return np.nan
    # lowest equal high above price
    return float(min(above))


def _get_atr_value(df: pd.DataFrame, length: int = 14) -> float:
    """Return latest true ATR value with robust fallbacks."""
    try:
        atr_series = true_atr(df, length=length)
        val = float(atr_series.iloc[-1])
        if np.isnan(val) or not np.isfinite(val):
            raise ValueError("ATR nan")
        return val
    except Exception:
        # Fallback: simple range-based proxy on last N bars
        w = df.tail(max(length, 10))
        if len(w) == 0:
            return 0.0
        approx = float((w["high"].max() - w["low"].min()) / max(len(w), 1))
        return approx if np.isfinite(approx) else 0.0


# =====================================================================
# PUBLIC API — Protective Stops
# =====================================================================

def protective_stop_long(
    df: pd.DataFrame,
    entry: float,
    tick: float = 0.1,
    return_meta: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """Institutional protective stop for LONGs.

    Logic:
    - Take last swing low (structural reference).
    - Look for equal-lows (liquidity) below price.
    - Base reference = min(swing_low, liq_low) when liq exists, else swing_low.
    - Add small ATR buffer below that reference.
    - Ensure stop < entry.
    - Round to tick.
    """
    try:
        if df is None or len(df) < 20:
            # ultra defensive fallback: 3-4% under entry
            sl_raw = float(entry) * 0.96
            sl_final = _round_to_tick(sl_raw, tick)
            meta = {
                "mode": "fallback_len",
                "sl_raw": sl_raw,
                "atr": None,
                "last_swing_low": None,
                "liq_low": None,
            }
            return (sl_final, meta) if return_meta else (sl_final, {})

        entry_f = float(entry)

        last_swing_low = _get_last_swing_low(df)
        liq_low = _get_liquidity_low(df, entry_f)

        if np.isnan(liq_low):
            base_ref = last_swing_low
        else:
            base_ref = min(last_swing_low, liq_low)

        atr_val = _get_atr_value(df, length=14)
        # buffer: 0.2 * ATR below the structural / liquidity level
        sl_raw = base_ref - atr_val * 0.2

        # Guard rails: ensure SL is below entry
        if sl_raw >= entry_f:
            sl_raw = entry_f - max(atr_val, entry_f * 0.01)

        # Ensure SL stays positive
        if sl_raw <= 0:
            sl_raw = entry_f * 0.9

        sl_final = _round_to_tick(sl_raw, tick)

        meta = {
            "mode": "ok",
            "last_swing_low": last_swing_low,
            "liq_low": liq_low if not np.isnan(liq_low) else None,
            "atr": atr_val,
            "sl_raw": sl_raw,
        }

        return (sl_final, meta) if return_meta else (sl_final, {})

    except Exception:
        # Last-resort fallback
        sl_raw = float(entry) * 0.95
        sl_final = _round_to_tick(sl_raw, tick)
        meta = {
            "mode": "exception",
            "sl_raw": sl_raw,
            "atr": None,
            "last_swing_low": None,
            "liq_low": None,
        }
        return (sl_final, meta) if return_meta else (sl_final, {})


def protective_stop_short(
    df: pd.DataFrame,
    entry: float,
    tick: float = 0.1,
    return_meta: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """Institutional protective stop for SHORTs.

    Logic:
    - Take last swing high (structural reference).
    - Look for equal-highs (liquidity) above price.
    - Base reference = max(swing_high, liq_high) when liq exists, else swing_high.
    - Add small ATR buffer above that reference.
    - Ensure stop > entry.
    - Round to tick.
    """
    try:
        if df is None or len(df) < 20:
            # ultra defensive fallback: 3-4% au-dessus de l'entry
            sl_raw = float(entry) * 1.04
            sl_final = _round_to_tick(sl_raw, tick)
            meta = {
                "mode": "fallback_len",
                "sl_raw": sl_raw,
                "atr": None,
                "last_swing_high": None,
                "liq_high": None,
            }
            return (sl_final, meta) if return_meta else (sl_final, {})

        entry_f = float(entry)

        last_swing_high = _get_last_swing_high(df)
        liq_high = _get_liquidity_high(df, entry_f)

        if np.isnan(liq_high):
            base_ref = last_swing_high
        else:
            base_ref = max(last_swing_high, liq_high)

        atr_val = _get_atr_value(df, length=14)

        # buffer: 0.2 * ATR au-dessus du niveau structurel/liquidité
        sl_raw = base_ref + atr_val * 0.2

        # Guard rails: ensure SL is above entry
        if sl_raw <= entry_f:
            sl_raw = entry_f + max(atr_val, entry_f * 0.01)

        sl_final = _round_to_tick(sl_raw, tick)

        meta = {
            "mode": "ok",
            "last_swing_high": last_swing_high,
            "liq_high": liq_high if not np.isnan(liq_high) else None,
            "atr": atr_val,
            "sl_raw": sl_raw,
        }

        return (sl_final, meta) if return_meta else (sl_final, {})

    except Exception:
        # Last-resort fallback
        sl_raw = float(entry) * 1.05
        sl_final = _round_to_tick(sl_raw, tick)
        meta = {
            "mode": "exception",
            "sl_raw": sl_raw,
            "atr": None,
            "last_swing_high": None,
            "liq_high": None,
        }
        return (sl_final, meta) if return_meta else (sl_final, {})
