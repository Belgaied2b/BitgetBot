# =====================================================================
# tp_utils.py — Institutional TP Engine (Desk Lead v2) (RR-based)
# Compatible legacy: compute_tp1(entry, sl, bias, df, tick, min_rr, max_rr)
# =====================================================================

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import pandas as pd

from indicators import true_atr

# Optional settings (safe defaults)
try:
    from settings import ATR_LEN, MIN_TP_TICKS
except Exception:
    ATR_LEN = 14
    MIN_TP_TICKS = 1


# ============================================================
# Tick helpers (desk: round for better fills)
# ============================================================

def _safe_tick(tick: float) -> float:
    try:
        t = float(tick)
        if not np.isfinite(t) or t <= 0:
            return 0.0
        return t
    except Exception:
        return 0.0


def _floor_to_tick(x: float, tick: float) -> float:
    t = _safe_tick(tick)
    if t <= 0:
        return float(x)
    return float(np.floor(float(x) / t) * t)


def _ceil_to_tick(x: float, tick: float) -> float:
    t = _safe_tick(tick)
    if t <= 0:
        return float(x)
    return float(np.ceil(float(x) / t) * t)


def _round_tp_for_fill(tp_raw: float, bias: str, tick: float) -> float:
    """
    Desk fill logic:
      - LONG TP (above entry): round DOWN a bit -> higher fill probability
      - SHORT TP (below entry): round UP a bit  -> higher fill probability
    If we must enforce RR min, we’ll override with ceil/floor later.
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


def _ensure_min_distance(tp: float, entry: float, bias: str, tick: float) -> float:
    """
    Enforce TP to be at least MIN_TP_TICKS away from entry (tick-based),
    and in the correct direction.
    """
    b = (bias or "").upper()
    t = _safe_tick(tick)
    min_ticks = max(int(MIN_TP_TICKS), 0)

    if t <= 0 or min_ticks <= 0:
        # direction only
        if b == "LONG" and tp <= entry:
            return float(entry * 1.001)
        if b == "SHORT" and tp >= entry:
            return float(entry * 0.999)
        return float(tp)

    min_dist = float(min_ticks) * t

    if b == "LONG":
        return float(max(tp, entry + min_dist))
    if b == "SHORT":
        return float(min(tp, entry - min_dist))
    return float(tp)


# ============================================================
# ATR helper (cohérence true_atr)
# ============================================================

def _atr_last(df: pd.DataFrame, length: int = 14) -> float:
    try:
        if df is None or len(df) < length + 3:
            raise ValueError("not_enough_data")
        s = true_atr(df, length=int(length))
        v = float(s.iloc[-1])
        if not np.isfinite(v) or v <= 0:
            raise ValueError("atr_invalid")
        return float(v)
    except Exception:
        if df is None or len(df) == 0:
            return 0.0
        w = df.tail(max(int(length), 10))
        approx = float((w["high"].astype(float).max() - w["low"].astype(float).min()) / max(len(w), 1))
        return float(approx) if np.isfinite(approx) and approx > 0 else 0.0


def _momentum_alignment(df: pd.DataFrame, bias: str) -> int:
    """
    Returns:
      +1 aligned
       0 mixed
      -1 opposed
    """
    try:
        closes = df["close"].astype(float)
        if len(closes) < 25:
            return 0
        ret_5 = float(closes.iloc[-1] / max(closes.iloc[-5], 1e-12) - 1.0)
        ret_20 = float(closes.iloc[-1] / max(closes.iloc[-20], 1e-12) - 1.0)

        b = (bias or "").upper()
        if b == "LONG":
            if ret_5 > 0 and ret_20 > 0:
                return +1
            if ret_5 < 0 and ret_20 < 0:
                return -1
            return 0

        if b == "SHORT":
            if ret_5 < 0 and ret_20 < 0:
                return +1
            if ret_5 > 0 and ret_20 > 0:
                return -1
            return 0

        return 0
    except Exception:
        return 0


# ============================================================
# TP1 — RR dynamique desk (legacy signature)
# ============================================================

def compute_tp1(
    entry: float,
    sl: float,
    bias: str,
    df: pd.DataFrame,
    tick: float,
    min_rr: float = 1.6,
    max_rr: float = 3.0,
) -> Tuple[float, float]:
    """
    Retourne :
        TP1 final (arrondi au tick),
        RR réel utilisé.

    Desk logic:
      - RR de base = min_rr
      - Ajustements selon:
          * ATR% (vol regime)
          * risk% (stop trop large -> TP1 plus proche)
          * momentum (aligned -> TP1 un peu plus loin)
      - Clamp RR dans [min_rr, max_rr]
      - Rounding orienté fill (LONG floor, SHORT ceil)
      - Ensuite enforce RR >= min_rr même après rounding (via ceil/floor opposé si besoin)
    """
    b = (bias or "").upper()
    entry_f = float(entry)
    sl_f = float(sl)
    t = _safe_tick(tick)

    risk = abs(entry_f - sl_f)
    if not np.isfinite(risk) or risk <= 0:
        # fallback mini
        if b == "SHORT":
            tp_f = entry_f * 0.99
        else:
            tp_f = entry_f * 1.01
        tp_f = _round_tp_for_fill(tp_f, b if b in ("LONG", "SHORT") else "LONG", t)
        rr_eff = abs(tp_f - entry_f) / max(risk, 1e-12)
        return float(tp_f), float(rr_eff)

    if b not in ("LONG", "SHORT"):
        # fallback neutral
        rr = float(max(min_rr, min(1.6, max_rr)))
        tp_raw = entry_f + risk * rr
        tp_raw = _ensure_min_distance(tp_raw, entry_f, "LONG", t)
        tp = _round_tp_for_fill(tp_raw, "LONG", t)
        rr_eff = abs(tp - entry_f) / risk
        return float(tp), float(rr_eff)

    min_rr = float(min_rr)
    max_rr = float(max_rr)
    if not np.isfinite(min_rr) or min_rr <= 0:
        min_rr = 1.3
    if not np.isfinite(max_rr) or max_rr <= min_rr:
        max_rr = max(min_rr + 0.5, 2.0)

    rr = float(min_rr)

    # -----------------------------
    # Volatility / risk%
    # -----------------------------
    atr_val = _atr_last(df, length=int(ATR_LEN))
    last_close = float(df["close"].astype(float).iloc[-1]) if df is not None and len(df) else entry_f
    denom = max(abs(last_close), 1e-8)

    atrp = (atr_val / denom) if atr_val > 0 else 0.0
    riskp = (risk / max(abs(entry_f), 1e-8))

    # Volatility: très volatile -> TP1 plus "grab" (RR un peu plus bas)
    if atrp > 0.060:
        rr -= 0.15
    elif atrp > 0.035:
        rr -= 0.08
    elif atrp < 0.012:
        rr += 0.08

    # Stop trop large -> TP1 plus proche (sinon TP1 devient irréaliste)
    if riskp > 0.060:
        rr -= 0.15
    elif riskp > 0.035:
        rr -= 0.08
    elif riskp < 0.015:
        rr += 0.10

    # Momentum alignment
    mom = _momentum_alignment(df, b)
    if mom > 0:
        rr += 0.08
    elif mom < 0:
        rr -= 0.08

    # Clamp RR
    rr = float(max(min_rr, min(rr, max_rr)))

    # -----------------------------
    # TP raw
    # -----------------------------
    if b == "LONG":
        tp_raw = entry_f + risk * rr
    else:
        tp_raw = entry_f - risk * rr

    tp_raw = _ensure_min_distance(tp_raw, entry_f, b, t)

    # Fill rounding
    tp = _round_tp_for_fill(tp_raw, b, t)
    tp = _ensure_min_distance(tp, entry_f, b, t)
    tp = _round_tp_for_fill(tp, b, t)

    # -----------------------------
    # Ensure RR >= min_rr AFTER rounding
    # (override rounding direction if needed)
    # -----------------------------
    rr_eff = abs(tp - entry_f) / risk

    if rr_eff + 1e-9 < min_rr:
        if b == "LONG":
            # use CEIL to guarantee >= min_rr
            tp_need = entry_f + risk * min_rr
            tp_need = _ensure_min_distance(tp_need, entry_f, "LONG", t)
            tp = _ceil_to_tick(tp_need, t) if t > 0 else float(tp_need)
        else:
            # use FLOOR to guarantee >= min_rr (more reward)
            tp_need = entry_f - risk * min_rr
            tp_need = _ensure_min_distance(tp_need, entry_f, "SHORT", t)
            tp = _floor_to_tick(tp_need, t) if t > 0 else float(tp_need)

        rr_eff = abs(tp - entry_f) / risk

    # Safety direction
    if b == "LONG" and tp <= entry_f:
        tp = _ceil_to_tick(entry_f + max((float(MIN_TP_TICKS) * t if t > 0 else 0.0), abs(entry_f) * 0.001), t) if t > 0 else float(entry_f * 1.001)
        rr_eff = abs(tp - entry_f) / risk

    if b == "SHORT" and tp >= entry_f:
        tp = _floor_to_tick(entry_f - max((float(MIN_TP_TICKS) * t if t > 0 else 0.0), abs(entry_f) * 0.001), t) if t > 0 else float(entry_f * 0.999)
        rr_eff = abs(tp - entry_f) / risk

    return float(tp), float(rr_eff)


# ============================================================
# Optional: TP2 runner (useful later)
# ============================================================

def compute_tp2(
    entry: float,
    sl: float,
    bias: str,
    df: pd.DataFrame,
    tick: float,
    rr1: Optional[float] = None,
    min_rr2: float = 2.0,
    max_rr2: float = 3.8,
) -> float:
    """
    TP2 runner basé sur le même risk:
      - rr2 = max(min_rr2, rr1*1.6, rr1+0.8) si rr1 dispo
      - ajuste volatilité
      - rounding fill-friendly
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

    rr2 = float(min_rr2)
    if rr1 is not None and np.isfinite(rr1) and rr1 > 0:
        rr2 = max(rr2, float(rr1) * 1.6, float(rr1) + 0.8)

    atr_val = _atr_last(df, length=int(ATR_LEN))
    last_close = float(df["close"].astype(float).iloc[-1]) if df is not None and len(df) else entry_f
    atrp = (atr_val / max(abs(last_close), 1e-8)) if atr_val > 0 else 0.0

    if atrp < 0.012:
        rr2 += 0.20
    elif atrp > 0.060:
        rr2 -= 0.20

    rr2 = float(max(min_rr2, min(rr2, max_rr2)))

    if b == "LONG":
        tp_raw = entry_f + risk * rr2
    else:
        tp_raw = entry_f - risk * rr2

    tp_raw = _ensure_min_distance(tp_raw, entry_f, b, t)
    tp = _round_tp_for_fill(tp_raw, b, t)

    # enforce again post rounding
    tp = _ensure_min_distance(tp, entry_f, b, t)
    tp = _round_tp_for_fill(tp, b, t)

    # safety direction
    if b == "LONG" and tp <= entry_f:
        tp = _ceil_to_tick(tp_raw, t) if t > 0 else float(tp_raw)
    if b == "SHORT" and tp >= entry_f:
        tp = _floor_to_tick(tp_raw, t) if t > 0 else float(tp_raw)

    return float(tp)
