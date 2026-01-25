# =====================================================================
# stops.py — Desk Lead Institutional Stop Engine (PRO v3)
# Compatible with analyze_signal.py (protective_stop_long / protective_stop_short)
# =====================================================================
# Upgrades v3:
# - Policy-aware selection by setup/entry_type (optional inputs, backward compatible)
# - Liquidity buffer can include ATR component (institutional stop-hunt padding)
# - ATR distance cap: prevents insane structural stops
# - Optional HTF dataframe candidates (htf_df) if you want later (H4/H1 blend)
# - Rich meta for debugging (policy, chosen, buffers, caps, distances)
# =====================================================================

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional, List

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

        # Optional (v3):
        LIQ_BUFFER_ATR_MULT,     # extra buffer = ATR * this mult (institutional)
        SL_BUFFER_ATR_MULT,      # optional: general buffer can also use ATR
        SL_POLICY_BOS,           # e.g. "SWING,LIQ,ATR"
        SL_POLICY_OTE,           # e.g. "LIQ,SWING,ATR"
        SL_POLICY_INST,          # e.g. "LIQ,SWING,ATR"
        SL_POLICY_DEFAULT,       # e.g. "TIGHT" or "SWING,LIQ,ATR"
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

    # v3 defaults
    LIQ_BUFFER_ATR_MULT = 0.12
    SL_BUFFER_ATR_MULT = 0.00
    SL_POLICY_BOS = "SWING,LIQ,ATR"
    SL_POLICY_OTE = "LIQ,SWING,ATR"
    SL_POLICY_INST = "LIQ,SWING,ATR"
    SL_POLICY_DEFAULT = "TIGHT"


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
    t = _safe_tick(tick)
    if t <= 0:
        return float(price)
    p = float(price)
    return float(np.floor(p / t) * t)


def _ceil_to_tick(price: float, tick: float) -> float:
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
    Most recent swing low strictly below entry.
    """
    try:
        sub = _get_struct_window(df)
        swings = find_swings(sub)
        lows = swings.get("lows", []) or []
        for _, p in reversed(lows):
            if np.isfinite(p) and float(p) < float(entry):
                return float(p)
        v = float(sub["low"].astype(float).min())
        return float(v) if np.isfinite(v) else None
    except Exception:
        return None


def _last_swing_high_above(df: pd.DataFrame, entry: float) -> Optional[float]:
    """
    Most recent swing high strictly above entry.
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
    Highest EQL strictly below entry (closest liquidity pool under price).
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
    Lowest EQH strictly above entry (closest liquidity pool over price).
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

def _buffer_price(entry: float, tick: float, pct: float, ticks: int, atr: float = 0.0, atr_mult: float = 0.0) -> float:
    """
    Returns a positive buffer in price units.
    Uses max(pct*entry, ticks*tick, atr_mult*atr).
    """
    entry_f = float(entry)
    t = _safe_tick(tick)

    pct_buf = abs(entry_f) * float(pct) if np.isfinite(entry_f) else 0.0
    tick_buf = float(ticks) * t if t > 0 else 0.0
    atr_buf = float(atr_mult) * float(atr) if (np.isfinite(atr) and atr > 0 and atr_mult > 0) else 0.0

    out = float(max(pct_buf, tick_buf, atr_buf, 0.0))
    return out


def _enforce_min_distance_long(sl: float, entry: float, tick: float) -> float:
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
    t = _safe_tick(tick)
    if t <= 0:
        return float(sl)

    min_dist = float(max(int(MIN_SL_TICKS), 0)) * t
    if min_dist <= 0:
        return float(sl)

    if (float(sl) - float(entry)) < min_dist:
        return float(entry) + min_dist
    return float(sl)


def _cap_max_distance_pct(sl: float, entry: float, side: str) -> float:
    """
    Hard cap SL distance in % of entry (MAX_SL_PCT) to avoid insane stops.
    """
    entry_f = float(entry)
    sl_f = float(sl)
    if not (np.isfinite(entry_f) and np.isfinite(sl_f)) or entry_f == 0:
        return sl_f

    max_pct = float(MAX_SL_PCT)
    if max_pct <= 0:
        return sl_f

    max_dist = abs(entry_f) * max_pct

    if side == "LONG":
        dist = entry_f - sl_f
        if dist > max_dist:
            return entry_f - max_dist
        return sl_f

    dist = sl_f - entry_f
    if dist > max_dist:
        return entry_f + max_dist
    return sl_f


def _cap_by_atr(sl: float, entry: float, atr: float, side: str) -> Tuple[float, bool]:
    """
    If a structural SL is extremely far, cap it using ATR_MULT_SL_CAP * ATR.
    LONG: ensure SL >= entry - cap_dist
    SHORT: ensure SL <= entry + cap_dist
    Returns (sl_capped, did_cap)
    """
    try:
        a = float(atr)
        if not np.isfinite(a) or a <= 0:
            return float(sl), False
        cap_mult = float(ATR_MULT_SL_CAP)
        if cap_mult <= 0:
            return float(sl), False

        cap_dist = cap_mult * a
        entry_f = float(entry)
        sl_f = float(sl)

        if side == "LONG":
            cap_level = entry_f - cap_dist
            if sl_f < cap_level:  # too wide
                return float(cap_level), True
            return sl_f, False

        cap_level = entry_f + cap_dist
        if sl_f > cap_level:  # too wide
            return float(cap_level), True
        return sl_f, False
    except Exception:
        return float(sl), False


# =====================================================================
# Entry context helpers (FVG / OB / LEVEL)
# =====================================================================

def _ctx_bounds(entry_ctx: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[float]]:
    """
    Returns (ctx_type, low, high, level).
    """
    if not isinstance(entry_ctx, dict):
        return None, None, None, None
    ctx_type = str(entry_ctx.get("type") or "").upper()
    low = high = level = None
    try:
        if entry_ctx.get("low") is not None:
            low = float(entry_ctx.get("low"))
        if entry_ctx.get("high") is not None:
            high = float(entry_ctx.get("high"))
        if entry_ctx.get("level") is not None:
            level = float(entry_ctx.get("level"))
    except Exception:
        low = high = level = None
    return ctx_type, low, high, level


# =====================================================================
# Policy (setup/entry_type)
# =====================================================================

def _parse_policy(s: str) -> List[str]:
    parts = []
    for p in str(s or "").upper().split(","):
        p = p.strip()
        if p:
            parts.append(p)
    return parts


def _sl_policy(setup: Optional[str], entry_type: Optional[str], entry_ctx: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Returns an ordered list of preferred groups: LIQ / SWING / ATR
    """
    su = str(setup or "").upper()
    et = str(entry_type or "").upper()

    default_pol = str(SL_POLICY_DEFAULT or "TIGHT").upper().strip()

    ctx_type = str((entry_ctx or {}).get("type") or "").upper()
    if ctx_type in {"FVG", "OB", "LEVEL"}:
        base = _parse_policy(SL_POLICY_BOS if "BOS" in su else SL_POLICY_DEFAULT)
        if not base:
            base = ["LIQ", "SWING", "ATR"]
        return ["STRUCT"] + [p for p in base if p != "STRUCT"]

    if "BOS" in su:
        return _parse_policy(SL_POLICY_BOS)
    if "INST" in su:
        return _parse_policy(SL_POLICY_INST)

    if "OTE" in et:
        return _parse_policy(SL_POLICY_OTE)
    if "PULLBACK" in et:
        return _parse_policy(SL_POLICY_OTE)

    if default_pol == "TIGHT":
        return ["TIGHT"]
    return _parse_policy(default_pol)


def _group_of_candidate(key: str) -> str:
    k = str(key or "").upper()
    if "FVG" in k or "OB" in k or "LEVEL" in k:
        return "STRUCT"
    if "LIQ" in k:
        return "LIQ"
    if "SWING" in k:
        return "SWING"
    if "ATR" in k:
        return "ATR"
    return "OTHER"


def _pick_best_in_group(side: str, entry: float, candidates: Dict[str, float], group: str) -> Optional[Tuple[str, float]]:
    """
    Within a group, pick the tightest valid:
      LONG: highest SL below entry
      SHORT: lowest SL above entry
    """
    entry_f = float(entry)
    side_u = str(side).upper()
    grp_u = str(group).upper()

    valid: List[Tuple[str, float]] = []
    for k, sl in candidates.items():
        if _group_of_candidate(k) != grp_u:
            continue
        if sl is None:
            continue
        slf = float(sl)
        if not np.isfinite(slf):
            continue
        if side_u == "LONG" and slf < entry_f:
            valid.append((k, slf))
        if side_u == "SHORT" and slf > entry_f:
            valid.append((k, slf))

    if not valid:
        return ("FALLBACK_PCT", (entry_f * 0.96) if side_u == "LONG" else (entry_f * 1.04))

    if side_u == "LONG":
        return sorted(valid, key=lambda x: x[1], reverse=True)[0]
    return sorted(valid, key=lambda x: x[1])[0]


def _choose_sl(side: str, entry: float, candidates: Dict[str, float], policy: List[str]) -> Tuple[float, str]:
    """
    Choose SL according to policy.
    If policy = ["TIGHT"], use tightest-any.
    Otherwise, try groups in order then fallback to tightest-any.
    """
    if not policy:
        policy = ["TIGHT"]

    if len(policy) == 1 and policy[0] == "TIGHT":
        k, sl = _pick_tightest_any(side, entry, candidates)
        return float(sl), str(k)

    for grp in policy:
        if grp in ("STRUCT", "LIQ", "SWING", "ATR"):
            picked = _pick_best_in_group(side, entry, candidates, grp)
            if picked:
                k, sl = picked
                return float(sl), str(k)

    k, sl = _pick_tightest_any(side, entry, candidates)
    return float(sl), str(k)


# =====================================================================
# Candidate builder
# =====================================================================

def _build_long_candidates(
    df: pd.DataFrame,
    entry: float,
    tick: float,
    atr_v: float,
    htf_df: Optional[pd.DataFrame] = None,
    entry_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    entry_f = float(entry)

    gen_buf = _buffer_price(
        entry_f, tick,
        float(SL_BUFFER_PCT), int(SL_BUFFER_TICKS),
        atr=atr_v, atr_mult=float(SL_BUFFER_ATR_MULT)
    )
    liq_buf = _buffer_price(
        entry_f, tick,
        float(LIQ_BUFFER_PCT), int(LIQ_BUFFER_TICKS),
        atr=atr_v, atr_mult=float(LIQ_BUFFER_ATR_MULT)
    )

    swing = _last_swing_low_below(df, entry_f)
    liq = _liq_low_below(df, entry_f)

    cands: Dict[str, float] = {}

    if swing is not None and np.isfinite(swing):
        cands["SWING"] = float(swing) - gen_buf

    if liq is not None and np.isfinite(liq):
        cands["LIQ"] = float(liq) - liq_buf

    ctx_type, ctx_low, ctx_high, ctx_level = _ctx_bounds(entry_ctx)
    if ctx_type == "FVG" and ctx_low is not None and np.isfinite(ctx_low):
        cands["FVG_INV"] = float(ctx_low) - gen_buf
    if ctx_type == "OB" and ctx_low is not None and np.isfinite(ctx_low):
        cands["OB_INV"] = float(ctx_low) - gen_buf
    if ctx_type == "LEVEL" and ctx_level is not None and np.isfinite(ctx_level):
        cands["LEVEL_INV"] = float(ctx_level) - gen_buf

    if htf_df is not None and _ensure_ohlcv(htf_df) and len(htf_df) >= 40:
        swing_htf = _last_swing_low_below(htf_df, entry_f)
        liq_htf = _liq_low_below(htf_df, entry_f)
        if swing_htf is not None and np.isfinite(swing_htf):
            cands["HTF_SWING"] = float(swing_htf) - gen_buf
        if liq_htf is not None and np.isfinite(liq_htf):
            cands["HTF_LIQ"] = float(liq_htf) - liq_buf

    if np.isfinite(atr_v) and atr_v > 0:
        cands["ATR"] = entry_f - float(ATR_MULT_SL) * atr_v
        cands["ATR_CAP"] = entry_f - float(ATR_MULT_SL_CAP) * atr_v

    return cands


def _build_short_candidates(
    df: pd.DataFrame,
    entry: float,
    tick: float,
    atr_v: float,
    htf_df: Optional[pd.DataFrame] = None,
    entry_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    entry_f = float(entry)

    gen_buf = _buffer_price(
        entry_f, tick,
        float(SL_BUFFER_PCT), int(SL_BUFFER_TICKS),
        atr=atr_v, atr_mult=float(SL_BUFFER_ATR_MULT)
    )
    liq_buf = _buffer_price(
        entry_f, tick,
        float(LIQ_BUFFER_PCT), int(LIQ_BUFFER_TICKS),
        atr=atr_v, atr_mult=float(LIQ_BUFFER_ATR_MULT)
    )

    swing = _last_swing_high_above(df, entry_f)
    liq = _liq_high_above(df, entry_f)

    cands: Dict[str, float] = {}

    if swing is not None and np.isfinite(swing):
        cands["SWING"] = float(swing) + gen_buf

    if liq is not None and np.isfinite(liq):
        cands["LIQ"] = float(liq) + liq_buf

    ctx_type, ctx_low, ctx_high, ctx_level = _ctx_bounds(entry_ctx)
    if ctx_type == "FVG" and ctx_high is not None and np.isfinite(ctx_high):
        cands["FVG_INV"] = float(ctx_high) + gen_buf
    if ctx_type == "OB" and ctx_high is not None and np.isfinite(ctx_high):
        cands["OB_INV"] = float(ctx_high) + gen_buf
    if ctx_type == "LEVEL" and ctx_level is not None and np.isfinite(ctx_level):
        cands["LEVEL_INV"] = float(ctx_level) + gen_buf

    if htf_df is not None and _ensure_ohlcv(htf_df) and len(htf_df) >= 40:
        swing_htf = _last_swing_high_above(htf_df, entry_f)
        liq_htf = _liq_high_above(htf_df, entry_f)
        if swing_htf is not None and np.isfinite(swing_htf):
            cands["HTF_SWING"] = float(swing_htf) + gen_buf
        if liq_htf is not None and np.isfinite(liq_htf):
            cands["HTF_LIQ"] = float(liq_htf) + liq_buf

    if np.isfinite(atr_v) and atr_v > 0:
        cands["ATR"] = entry_f + float(ATR_MULT_SL) * atr_v
        cands["ATR_CAP"] = entry_f + float(ATR_MULT_SL_CAP) * atr_v

    return cands


# =====================================================================
# PUBLIC API — Protective Stops
# =====================================================================

def protective_stop_long(
    df: pd.DataFrame,
    entry: float,
    tick: float = 0.1,
    return_meta: bool = False,
    *,
    setup: Optional[str] = None,
    entry_type: Optional[str] = None,
    htf_df: Optional[pd.DataFrame] = None,
    entry_ctx: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Desk Lead protective stop for LONG.

    Pipeline:
      1) Build candidates (LIQ / SWING / ATR) + optional HTF
      2) Choose by policy (setup/entry_type if provided)
      3) Cap by ATR (avoid insane structural SL)
      4) Enforce min distance (ticks)
      5) Cap max distance (%)
      6) Directional rounding (FLOOR)
      7) Final sanity: SL < entry
    """
    meta: Dict[str, Any] = {}
    try:
        entry_f = float(entry)
        t = _safe_tick(tick)

        if not _ensure_ohlcv(df) or len(df) < 20 or not np.isfinite(entry_f):
            sl_raw = entry_f * 0.96
            sl_raw = _enforce_min_distance_long(sl_raw, entry_f, t)
            sl_raw = _cap_max_distance_pct(sl_raw, entry_f, side="LONG")
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
        policy = _sl_policy(setup, entry_type, entry_ctx)

        cands = _build_long_candidates(df, entry_f, t, atr_v, htf_df=htf_df, entry_ctx=entry_ctx)
        sl_raw, chosen = _choose_sl("LONG", entry_f, cands, policy)

        sl_raw2, did_cap = _cap_by_atr(sl_raw, entry_f, atr_v, side="LONG")
        if did_cap:
            chosen = f"{chosen}+ATR_CAP"

        sl_raw2 = _enforce_min_distance_long(sl_raw2, entry_f, t)
        sl_raw2 = _cap_max_distance_pct(sl_raw2, entry_f, side="LONG")

        if sl_raw2 >= entry_f:
            fallback = entry_f - max(atr_v, abs(entry_f) * 0.01, (float(MIN_SL_TICKS) * t if t > 0 else 0.0))
            sl_raw2 = float(fallback)
            chosen = "FORCED_BELOW_ENTRY"

        sl_final = _floor_to_tick(sl_raw2, t)

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
            "setup": str(setup or ""),
            "entry_type": str(entry_type or ""),
            "policy": policy,
            "chosen": chosen,
            "entry_ctx": entry_ctx,
            "entry": entry_f,
            "tick": t,
            "atr": float(atr_v),
            "candidates": {k: float(v) for k, v in cands.items()},
            "sl_raw": float(sl_raw),
            "sl_after_cap": float(sl_raw2),
            "sl_final": float(sl_final),
            "did_atr_cap": bool("ATR_CAP" in chosen),
            "dist_abs": dist_abs,
            "dist_pct": dist_pct,
            "dist_ticks": dist_ticks,
            "buffers": {
                "sl_buffer_price": _buffer_price(entry_f, t, float(SL_BUFFER_PCT), int(SL_BUFFER_TICKS), atr=atr_v, atr_mult=float(SL_BUFFER_ATR_MULT)),
                "liq_buffer_price": _buffer_price(entry_f, t, float(LIQ_BUFFER_PCT), int(LIQ_BUFFER_TICKS), atr=atr_v, atr_mult=float(LIQ_BUFFER_ATR_MULT)),
            },
        }

        return (float(sl_final), meta) if return_meta else (float(sl_final), {})

    except Exception as e:
        entry_f = float(entry) if np.isfinite(float(entry)) else 0.0
        t = _safe_tick(tick)
        atr_v = _get_atr_value(df, length=int(ATR_LEN)) if _ensure_ohlcv(df) else 0.0

        sl_raw = entry_f * 0.95
        sl_raw = _enforce_min_distance_long(sl_raw, entry_f, t)
        sl_raw = _cap_max_distance_pct(sl_raw, entry_f, side="LONG")
        sl_final = _floor_to_tick(sl_raw, t)
        if sl_final >= entry_f and t > 0:
            sl_final = _floor_to_tick(entry_f - max(float(MIN_SL_TICKS) * t, t), t)

        meta = {"mode": "exception", "error": str(e), "atr": float(atr_v), "sl_raw": float(sl_raw), "sl_final": float(sl_final)}
        return (float(sl_final), meta) if return_meta else (float(sl_final), {})


def protective_stop_short(
    df: pd.DataFrame,
    entry: float,
    tick: float = 0.1,
    return_meta: bool = False,
    *,
    setup: Optional[str] = None,
    entry_type: Optional[str] = None,
    htf_df: Optional[pd.DataFrame] = None,
    entry_ctx: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Desk Lead protective stop for SHORT.

    Pipeline:
      1) Build candidates (LIQ / SWING / ATR) + optional HTF
      2) Choose by policy (setup/entry_type if provided)
      3) Cap by ATR (avoid insane structural SL)
      4) Enforce min distance (ticks)
      5) Cap max distance (%)
      6) Directional rounding (CEIL)
      7) Final sanity: SL > entry
    """
    meta: Dict[str, Any] = {}
    try:
        entry_f = float(entry)
        t = _safe_tick(tick)

        if not _ensure_ohlcv(df) or len(df) < 20 or not np.isfinite(entry_f):
            sl_raw = entry_f * 1.04
            sl_raw = _enforce_min_distance_short(sl_raw, entry_f, t)
            sl_raw = _cap_max_distance_pct(sl_raw, entry_f, side="SHORT")
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
        policy = _sl_policy(setup, entry_type, entry_ctx)

        cands = _build_short_candidates(df, entry_f, t, atr_v, htf_df=htf_df, entry_ctx=entry_ctx)
        sl_raw, chosen = _choose_sl("SHORT", entry_f, cands, policy)

        sl_raw2, did_cap = _cap_by_atr(sl_raw, entry_f, atr_v, side="SHORT")
        if did_cap:
            chosen = f"{chosen}+ATR_CAP"

        sl_raw2 = _enforce_min_distance_short(sl_raw2, entry_f, t)
        sl_raw2 = _cap_max_distance_pct(sl_raw2, entry_f, side="SHORT")

        if sl_raw2 <= entry_f:
            fallback = entry_f + max(atr_v, abs(entry_f) * 0.01, (float(MIN_SL_TICKS) * t if t > 0 else 0.0))
            sl_raw2 = float(fallback)
            chosen = "FORCED_ABOVE_ENTRY"

        sl_final = _ceil_to_tick(sl_raw2, t)

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
            "setup": str(setup or ""),
            "entry_type": str(entry_type or ""),
            "policy": policy,
            "chosen": chosen,
            "entry_ctx": entry_ctx,
            "entry": entry_f,
            "tick": t,
            "atr": float(atr_v),
            "candidates": {k: float(v) for k, v in cands.items()},
            "sl_raw": float(sl_raw),
            "sl_after_cap": float(sl_raw2),
            "sl_final": float(sl_final),
            "did_atr_cap": bool("ATR_CAP" in chosen),
            "dist_abs": dist_abs,
            "dist_pct": dist_pct,
            "dist_ticks": dist_ticks,
            "buffers": {
                "sl_buffer_price": _buffer_price(entry_f, t, float(SL_BUFFER_PCT), int(SL_BUFFER_TICKS), atr=atr_v, atr_mult=float(SL_BUFFER_ATR_MULT)),
                "liq_buffer_price": _buffer_price(entry_f, t, float(LIQ_BUFFER_PCT), int(LIQ_BUFFER_TICKS), atr=atr_v, atr_mult=float(LIQ_BUFFER_ATR_MULT)),
            },
        }

        return (float(sl_final), meta) if return_meta else (float(sl_final), {})

    except Exception as e:
        entry_f = float(entry) if np.isfinite(float(entry)) else 0.0
        t = _safe_tick(tick)
        atr_v = _get_atr_value(df, length=int(ATR_LEN)) if _ensure_ohlcv(df) else 0.0

        sl_raw = entry_f * 1.05
        sl_raw = _enforce_min_distance_short(sl_raw, entry_f, t)
        sl_raw = _cap_max_distance_pct(sl_raw, entry_f, side="SHORT")
        sl_final = _ceil_to_tick(sl_raw, t)
        if sl_final <= entry_f and t > 0:
            sl_final = _ceil_to_tick(entry_f + max(float(MIN_SL_TICKS) * t, t), t)

        meta = {"mode": "exception", "error": str(e), "atr": float(atr_v), "sl_raw": float(sl_raw), "sl_final": float(sl_final)}
        return (float(sl_final), meta) if return_meta else (float(sl_final), {})
