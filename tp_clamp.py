# =====================================================================
# tp_clamp.py — Institutional TP Engine (Desk Lead v2)
# TP1 / TP2 dynamiques basés sur RR, volatilité, momentum + "liquidity aware"
# Compatible analyze_signal.py (compute_tp1) + future compute_tp2
# =====================================================================

from __future__ import annotations

from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

from indicators import true_atr

# Optional: use swings to "take liquidity" at obvious levels
try:
    from structure_utils import find_swings
except Exception:
    find_swings = None  # type: ignore


# ============================================================
# Optional settings (safe defaults)
# ============================================================

try:
    from settings import (
        ATR_LEN,
        TP1_R_CLAMP_MIN,
        TP1_R_CLAMP_MAX,
        TP2_R_TARGET,
        MIN_TP_TICKS,
        TP1_R_BY_VOL,
    )
except Exception:
    ATR_LEN = 14
    TP1_R_CLAMP_MIN = 1.40
    TP1_R_CLAMP_MAX = 1.60
    TP2_R_TARGET = 2.80
    MIN_TP_TICKS = 1
    TP1_R_BY_VOL = True


# ============================================================
# Tick helpers (directional rounding for better fills)
# ============================================================

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


def _round_tp_for_fill(tp_raw: float, bias: str, tick: float) -> float:
    """
    Desk logic:
      - LONG TP is above entry: round DOWN (easier fill, slightly less RR)
      - SHORT TP is below entry: round UP (easier fill, slightly less RR)
    """
    b = (bias or "").upper()
    if b == "LONG":
        return _floor_to_tick(tp_raw, tick)
    if b == "SHORT":
        return _ceil_to_tick(tp_raw, tick)
    # fallback nearest
    t = _safe_tick(tick)
    if t <= 0:
        return float(tp_raw)
    return float(round(float(tp_raw) / t) * t)


# ============================================================
# ATR local (cohérence true_atr)
# ============================================================

def _atr(df: pd.DataFrame, length: int = 14) -> float:
    try:
        if df is None or len(df) < length + 3:
            raise ValueError("not enough data")
        atr_series = true_atr(df, length=int(length))
        val = float(atr_series.iloc[-1])
        if not np.isfinite(val) or val <= 0:
            raise ValueError("atr invalid")
        return float(val)
    except Exception:
        if df is None or len(df) == 0:
            return 0.0
        w = df.tail(max(int(length), 10))
        approx = float((w["high"].astype(float).max() - w["low"].astype(float).min()) / max(len(w), 1))
        return float(approx) if np.isfinite(approx) and approx > 0 else 0.0


# ============================================================
# Swing target helper (optional liquidity take)
# ============================================================

def _nearest_swing_target(df: pd.DataFrame, entry: float, bias: str) -> Optional[float]:
    """
    Returns the nearest obvious swing in trade direction:
      - LONG  -> nearest swing high strictly above entry
      - SHORT -> nearest swing low strictly below entry
    """
    if find_swings is None:
        return None
    try:
        if df is None or len(df) < 20:
            return None

        sw = find_swings(df.tail(200))
        b = (bias or "").upper()

        if b == "LONG":
            highs = sw.get("highs", []) or []
            above = [float(p) for _, p in highs if np.isfinite(p) and float(p) > float(entry)]
            if not above:
                return None
            return float(min(above))  # nearest above
        if b == "SHORT":
            lows = sw.get("lows", []) or []
            below = [float(p) for _, p in lows if np.isfinite(p) and float(p) < float(entry)]
            if not below:
                return None
            return float(max(below))  # nearest below
        return None
    except Exception:
        return None


def _enforce_min_tp_distance(tp: float, entry: float, bias: str, tick: float) -> float:
    """
    Enforce TP to be at least MIN_TP_TICKS away from entry (tick-based),
    and in correct direction.
    """
    b = (bias or "").upper()
    t = _safe_tick(tick)
    min_ticks = max(int(MIN_TP_TICKS), 0)

    if t <= 0 or min_ticks <= 0:
        # just enforce direction
        if b == "LONG" and tp <= entry:
            return float(entry * 1.001)
        if b == "SHORT" and tp >= entry:
            return float(entry * 0.999)
        return float(tp)

    min_dist = float(min_ticks) * t

    if b == "LONG":
        if tp <= entry + min_dist:
            return float(entry + min_dist)
        return float(tp)

    if b == "SHORT":
        if tp >= entry - min_dist:
            return float(entry - min_dist)
        return float(tp)

    return float(tp)


# ============================================================
# TP1 — RR dynamique institutionnel (Desk)
# ============================================================

def compute_tp1(
    entry: float,
    sl: float,
    bias: str,
    df: pd.DataFrame,
    tick: float = 0.1,
) -> Tuple[float, float]:
    """
    Calcule un TP1 desk:
      - RR clampé dans [TP1_R_CLAMP_MIN, TP1_R_CLAMP_MAX]
      - Ajusté par volatilité (ATR%), risk%, momentum
      - Optionnel: "liquidity take" via swing target si ça respecte un RR minimal

    Retour:
      (tp1, rr_effective)

    Signature compatible analyze_signal.py:
      tp1, rr_used = compute_tp1(entry, sl, bias, df=df, tick=tick)
    """
    b = (bias or "").upper()
    entry_f = float(entry)
    sl_f = float(sl)
    t = _safe_tick(tick)

    risk = abs(entry_f - sl_f)
    if not np.isfinite(risk) or risk <= 0:
        return float(entry_f), 0.0

    if b not in ("LONG", "SHORT"):
        rr = 1.5
        tp_raw = entry_f + risk * rr
        tp_raw = _enforce_min_tp_distance(tp_raw, entry_f, "LONG", t)
        tp_rounded = _round_tp_for_fill(tp_raw, "LONG", t)
        rr_eff = abs(tp_rounded - entry_f) / risk
        return float(tp_rounded), float(rr_eff)

    # -----------------------------
    # 1) Inputs vol & risk%
    # -----------------------------
    atr_val = _atr(df, length=int(ATR_LEN))
    atrp = atr_val / max(abs(entry_f), 1e-8) if atr_val > 0 else 0.0
    riskp = risk / max(abs(entry_f), 1e-8)

    rr_min = float(TP1_R_CLAMP_MIN)
    rr_max = float(TP1_R_CLAMP_MAX)
    rr = float((rr_min + rr_max) / 2.0)

    # -----------------------------
    # 2) Momentum simple (returns)
    # -----------------------------
    try:
        closes = df["close"].astype(float)
        if len(closes) >= 25:
            ret_5 = float(closes.iloc[-1] / max(closes.iloc[-5], 1e-12) - 1.0)
            ret_20 = float(closes.iloc[-1] / max(closes.iloc[-20], 1e-12) - 1.0)
        else:
            ret_5 = 0.0
            ret_20 = 0.0
    except Exception:
        ret_5 = 0.0
        ret_20 = 0.0

    if b == "LONG":
        mom_aligned = (ret_5 > 0 and ret_20 > 0)
        mom_opposed = (ret_5 < 0 and ret_20 < 0)
    else:
        mom_aligned = (ret_5 < 0 and ret_20 < 0)
        mom_opposed = (ret_5 > 0 and ret_20 > 0)

    # -----------------------------
    # 3) Ajustements RR (si activé)
    # -----------------------------
    if bool(TP1_R_BY_VOL):
        # Volatility (ATR%)
        if atrp > 0.060:
            rr -= 0.12
        elif atrp > 0.035:
            rr -= 0.06
        elif atrp < 0.012:
            rr += 0.06

        # Stop width (risk%)
        if riskp > 0.060:
            rr -= 0.12
        elif riskp > 0.035:
            rr -= 0.06
        elif riskp < 0.015:
            rr += 0.08

        # Momentum alignment
        if mom_aligned:
            rr += 0.06
        elif mom_opposed:
            rr -= 0.06

    rr = float(max(rr_min, min(rr_max, rr)))

    # -----------------------------
    # 4) Build RR TP
    # -----------------------------
    if b == "LONG":
        tp_raw = entry_f + risk * rr
    else:
        tp_raw = entry_f - risk * rr

    # Enforce direction + min distance (pre rounding)
    tp_raw = _enforce_min_tp_distance(tp_raw, entry_f, b, t)

    # -----------------------------
    # 5) Optional liquidity take (nearest swing) if sane
    # -----------------------------
    # Idea: if there is a clean swing target closer than RR target,
    # take it only if RR still >= rr_min - small epsilon.
    swing_tgt = _nearest_swing_target(df, entry_f, b)

    chosen_mode = "RR"
    if swing_tgt is not None and np.isfinite(swing_tgt):
        # For LONG: swing must be above entry; for SHORT: below entry
        if (b == "LONG" and swing_tgt > entry_f) or (b == "SHORT" and swing_tgt < entry_f):
            # Compute swing RR
            swing_rr = abs(float(swing_tgt) - entry_f) / risk
            # If swing is nearer than rr target but still acceptable, use it
            rr_target_dist = abs(tp_raw - entry_f)
            swing_dist = abs(float(swing_tgt) - entry_f)

            if swing_dist < rr_target_dist and swing_rr >= max(0.0, rr_min - 0.08):
                tp_raw = float(swing_tgt)
                tp_raw = _enforce_min_tp_distance(tp_raw, entry_f, b, t)
                chosen_mode = "SWING"

    # -----------------------------
    # 6) Directional rounding for fill
    # -----------------------------
    tp_rounded = _round_tp_for_fill(tp_raw, b, t)

    # Enforce again after rounding (sometimes rounding crosses min-dist)
    tp_rounded = _enforce_min_tp_distance(tp_rounded, entry_f, b, t)
    tp_rounded = _round_tp_for_fill(tp_rounded, b, t)

    if not np.isfinite(tp_rounded) or tp_rounded <= 0:
        tp_rounded = float(tp_raw)

    rr_effective = float(abs(tp_rounded - entry_f) / risk) if risk > 0 else 0.0

    # safety: ensure TP in correct direction
    if b == "LONG" and tp_rounded <= entry_f:
        tp_rounded = float(entry_f + max((float(MIN_TP_TICKS) * t if t > 0 else 0.0), abs(entry_f) * 0.001))
        tp_rounded = _round_tp_for_fill(tp_rounded, "LONG", t)
        rr_effective = float(abs(tp_rounded - entry_f) / risk)

    if b == "SHORT" and tp_rounded >= entry_f:
        tp_rounded = float(entry_f - max((float(MIN_TP_TICKS) * t if t > 0 else 0.0), abs(entry_f) * 0.001))
        tp_rounded = _round_tp_for_fill(tp_rounded, "SHORT", t)
        rr_effective = float(abs(tp_rounded - entry_f) / risk)

    return float(tp_rounded), float(rr_effective)


# ============================================================
# TP2 — Runner institutionnel (Desk)
# ============================================================

def compute_tp2(
    entry: float,
    sl: float,
    bias: str,
    df: pd.DataFrame,
    tick: float = 0.1,
    rr1: Optional[float] = None,
) -> float:
    """
    TP2 runner:
      - Base RR2 = max(TP2_R_TARGET, rr1*1.6, rr1+0.8) si rr1 fourni
      - Ajusté par volatilité (ATR%) (faible vol => on vise plus loin)
      - Directional rounding for fill (LONG down, SHORT up)
      - Enforce min distance

    Retourne uniquement le prix TP2.
    """
    b = (bias or "").upper()
    if b not in ("LONG", "SHORT"):
        return float(entry)

    entry_f = float(entry)
    sl_f = float(sl)
    t = _safe_tick(tick)

    risk = abs(entry_f - sl_f)
    if not np.isfinite(risk) or risk <= 0:
        return float(entry_f)

    atr_val = _atr(df, length=int(ATR_LEN))
    atrp = atr_val / max(abs(entry_f), 1e-8) if atr_val > 0 else 0.0

    rr2 = float(TP2_R_TARGET)

    if rr1 is not None and np.isfinite(rr1) and rr1 > 0:
        rr2 = max(rr2, float(rr1) * 1.6, float(rr1) + 0.8)
    else:
        rr2 = max(rr2, 2.0)

    # Volatility adjustment
    if atrp < 0.012:
        rr2 += 0.20
    elif atrp > 0.060:
        rr2 -= 0.20

    # clamp runner RR2
    rr2 = float(max(2.0, min(3.8, rr2)))

    if b == "LONG":
        tp2_raw = entry_f + risk * rr2
    else:
        tp2_raw = entry_f - risk * rr2

    tp2_raw = _enforce_min_tp_distance(tp2_raw, entry_f, b, t)
    tp2_rounded = _round_tp_for_fill(tp2_raw, b, t)

    tp2_rounded = _enforce_min_tp_distance(tp2_rounded, entry_f, b, t)
    tp2_rounded = _round_tp_for_fill(tp2_rounded, b, t)

    if not np.isfinite(tp2_rounded) or tp2_rounded <= 0:
        tp2_rounded = float(tp2_raw)

    # safety direction
    if b == "LONG" and tp2_rounded <= entry_f:
        tp2_rounded = float(entry_f + max((float(MIN_TP_TICKS) * t if t > 0 else 0.0), abs(entry_f) * 0.001))
        tp2_rounded = _round_tp_for_fill(tp2_rounded, "LONG", t)

    if b == "SHORT" and tp2_rounded >= entry_f:
        tp2_rounded = float(entry_f - max((float(MIN_TP_TICKS) * t if t > 0 else 0.0), abs(entry_f) * 0.001))
        tp2_rounded = _round_tp_for_fill(tp2_rounded, "SHORT", t)

    return float(tp2_rounded)
