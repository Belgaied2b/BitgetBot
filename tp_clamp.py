# =====================================================================
# tp_clamp.py — Institutional TP Engine (Desk Lead v3)
# TP1 / TP2 dynamiques basés sur RR, volatilité, momentum + "liquidity aware"
# Upgrades v3:
# - TP1 peut viser des pools de liquidité EQH/EQL (equal highs/lows) en priorité
# - "Front-run liquidity": TP placé légèrement AVANT le niveau (buffer ticks / % / ATR)
# - Garde le clamp RR [min,max] + ajustements vol/risk/momentum
# - Fallback robuste si detect_equal_levels/find_swings indispo
# Compatible analyze_signal.py: tp1, rr_used = compute_tp1(entry, sl, bias, df=df, tick=tick)
# =====================================================================

from __future__ import annotations

from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import pandas as pd

from indicators import true_atr

# Optional: use swings / equal-levels to "take liquidity"
try:
    from structure_utils import find_swings  # type: ignore
except Exception:
    find_swings = None  # type: ignore

try:
    from structure_utils import detect_equal_levels  # type: ignore
except Exception:
    detect_equal_levels = None  # type: ignore


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

        # v3 (optional)
        TP1_USE_EQUAL_LEVELS,         # True/False
        TP1_USE_SWINGS,               # True/False
        TP1_LIQ_MAX_DIST_FACTOR,      # e.g. 1.00 => only if closer than RR target
        TP1_LIQ_MIN_RR_EPS,           # e.g. 0.08 => accept liq rr >= rr_min - eps

        TP_LIQ_BUFFER_PCT,            # e.g. 0.0003 (front-run)
        TP_LIQ_BUFFER_TICKS,          # e.g. 1
        TP_LIQ_BUFFER_ATR_MULT,       # e.g. 0.06 (front-run via ATR)
    )
except Exception:
    ATR_LEN = 14
    TP1_R_CLAMP_MIN = 1.40
    TP1_R_CLAMP_MAX = 1.60
    TP2_R_TARGET = 2.80
    MIN_TP_TICKS = 1
    TP1_R_BY_VOL = True

    # v3 defaults
    TP1_USE_EQUAL_LEVELS = True
    TP1_USE_SWINGS = True
    TP1_LIQ_MAX_DIST_FACTOR = 1.00     # only if closer than RR target
    TP1_LIQ_MIN_RR_EPS = 0.08          # allow slightly under rr_min

    # front-run buffer (small + safe)
    TP_LIQ_BUFFER_PCT = 0.0000
    TP_LIQ_BUFFER_TICKS = 1
    TP_LIQ_BUFFER_ATR_MULT = 0.06


REQUIRED_COLS = ("open", "high", "low", "close", "volume")


def _ensure_ohlcv(df: pd.DataFrame) -> bool:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False
    for c in REQUIRED_COLS:
        if c not in df.columns:
            return False
    return True


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
      - LONG TP is above entry: round DOWN (easier fill)
      - SHORT TP is below entry: round UP (easier fill)
    """
    b = (bias or "").upper()
    if b == "LONG":
        return _floor_to_tick(tp_raw, tick)
    if b == "SHORT":
        return _ceil_to_tick(tp_raw, tick)

    t = _safe_tick(tick)
    if t <= 0:
        return float(tp_raw)
    return float(round(float(tp_raw) / t) * t)


# ============================================================
# ATR local (cohérence true_atr)
# ============================================================

def _atr(df: pd.DataFrame, length: int = 14) -> float:
    try:
        if not _ensure_ohlcv(df) or len(df) < length + 3:
            raise ValueError("not enough data")
        atr_series = true_atr(df, length=int(length))
        val = float(atr_series.iloc[-1])
        if not np.isfinite(val) or val <= 0:
            raise ValueError("atr invalid")
        return float(val)
    except Exception:
        if df is None or len(df) == 0 or (not _ensure_ohlcv(df)):
            return 0.0
        w = df.tail(max(int(length), 10))
        approx = float((w["high"].astype(float).max() - w["low"].astype(float).min()) / max(len(w), 1))
        return float(approx) if np.isfinite(approx) and approx > 0 else 0.0


# ============================================================
# Liquidity buffers (front-run)
# ============================================================

def _liq_buffer(entry: float, tick: float, atr: float) -> float:
    """
    Buffer in price units for front-running liquidity.
    Uses max(pct*entry, ticks*tick, atr_mult*atr).
    """
    e = float(entry)
    t = _safe_tick(tick)

    pct_buf = abs(e) * float(TP_LIQ_BUFFER_PCT) if np.isfinite(e) else 0.0
    tick_buf = float(max(int(TP_LIQ_BUFFER_TICKS), 0)) * t if t > 0 else 0.0
    atr_buf = float(TP_LIQ_BUFFER_ATR_MULT) * float(atr) if (np.isfinite(atr) and atr > 0 and float(TP_LIQ_BUFFER_ATR_MULT) > 0) else 0.0

    return float(max(pct_buf, tick_buf, atr_buf, 0.0))


# ============================================================
# Swing / Equal-level target helpers
# ============================================================

def _nearest_swing_target(df: pd.DataFrame, entry: float, bias: str) -> Optional[float]:
    """
    Nearest obvious swing in trade direction:
      - LONG  -> nearest swing high strictly above entry
      - SHORT -> nearest swing low strictly below entry
    """
    if find_swings is None:
        return None
    try:
        if not _ensure_ohlcv(df) or len(df) < 20:
            return None

        sw = find_swings(df.tail(250))
        b = (bias or "").upper()

        if b == "LONG":
            highs = sw.get("highs", []) or []
            above = [float(p) for _, p in highs if np.isfinite(p) and float(p) > float(entry)]
            return float(min(above)) if above else None

        if b == "SHORT":
            lows = sw.get("lows", []) or []
            below = [float(p) for _, p in lows if np.isfinite(p) and float(p) < float(entry)]
            return float(max(below)) if below else None

        return None
    except Exception:
        return None


def _nearest_equal_level_target(df: pd.DataFrame, entry: float, bias: str) -> Optional[float]:
    """
    Nearest EQH/EQL pool in trade direction:
      - LONG  -> nearest EQH strictly above entry
      - SHORT -> nearest EQL strictly below entry
    """
    if detect_equal_levels is None:
        return None
    try:
        if not _ensure_ohlcv(df) or len(df) < 50:
            return None

        # Try "new" signature first (as used in your analyze_signal)
        try:
            lv = detect_equal_levels(df.tail(250), max_window=200, tol_mult_atr=0.10)
        except TypeError:
            # Fallback if signature is simpler
            lv = detect_equal_levels(df.tail(250))

        eq_highs = (lv.get("eq_highs", []) or []) if isinstance(lv, dict) else []
        eq_lows = (lv.get("eq_lows", []) or []) if isinstance(lv, dict) else []

        b = (bias or "").upper()
        if b == "LONG":
            above = [float(x) for x in eq_highs if np.isfinite(x) and float(x) > float(entry)]
            return float(min(above)) if above else None

        if b == "SHORT":
            below = [float(x) for x in eq_lows if np.isfinite(x) and float(x) < float(entry)]
            return float(max(below)) if below else None

        return None
    except Exception:
        return None


# ============================================================
# TP distance guards
# ============================================================

def _enforce_min_tp_distance(tp: float, entry: float, bias: str, tick: float) -> float:
    """
    Enforce TP to be at least MIN_TP_TICKS away from entry (tick-based),
    and in correct direction.
    """
    b = (bias or "").upper()
    t = _safe_tick(tick)
    min_ticks = max(int(MIN_TP_TICKS), 0)

    if t <= 0 or min_ticks <= 0:
        if b == "LONG" and tp <= entry:
            return float(entry * 1.001)
        if b == "SHORT" and tp >= entry:
            return float(entry * 0.999)
        return float(tp)

    min_dist = float(min_ticks) * t

    if b == "LONG":
        return float(entry + min_dist) if tp <= entry + min_dist else float(tp)

    if b == "SHORT":
        return float(entry - min_dist) if tp >= entry - min_dist else float(tp)

    return float(tp)


def _rr_of_target(entry: float, sl: float, tp: float, bias: str) -> Optional[float]:
    try:
        e = float(entry)
        s = float(sl)
        t = float(tp)
        if not (np.isfinite(e) and np.isfinite(s) and np.isfinite(t)):
            return None
        risk = abs(e - s)
        if risk <= 0:
            return None
        b = (bias or "").upper()
        if b == "LONG":
            if t <= e:
                return None
            return float((t - e) / risk)
        if b == "SHORT":
            if t >= e:
                return None
            return float((e - t) / risk)
        return None
    except Exception:
        return None


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
      - Liquidity-aware:
          * peut viser EQH/EQL (equal highs/lows)
          * peut viser swing (high/low)
          * TP "front-run" le niveau via un buffer (ticks/%/ATR)
        => uniquement si plus proche que le TP RR (par défaut) et RR reste acceptable

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

    # If bias unknown, fallback LONG RR median
    if b not in ("LONG", "SHORT"):
        rr = float((float(TP1_R_CLAMP_MIN) + float(TP1_R_CLAMP_MAX)) / 2.0)
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
        closes = df["close"].astype(float) if _ensure_ohlcv(df) else pd.Series([], dtype=float)
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
    # 4) Base RR TP (raw)
    # -----------------------------
    if b == "LONG":
        tp_rr = entry_f + risk * rr
    else:
        tp_rr = entry_f - risk * rr

    tp_rr = _enforce_min_tp_distance(tp_rr, entry_f, b, t)

    rr_target_dist = abs(tp_rr - entry_f)

    # -----------------------------
    # 5) Liquidity candidates (EQH/EQL, SWING) — optional
    # -----------------------------
    candidates: List[Tuple[str, float]] = []

    if bool(TP1_USE_EQUAL_LEVELS):
        eq_tgt = _nearest_equal_level_target(df, entry_f, b)
        if eq_tgt is not None and np.isfinite(eq_tgt):
            candidates.append(("EQ", float(eq_tgt)))

    if bool(TP1_USE_SWINGS):
        sw_tgt = _nearest_swing_target(df, entry_f, b)
        if sw_tgt is not None and np.isfinite(sw_tgt):
            candidates.append(("SWING", float(sw_tgt)))

    # Apply front-run buffer to liquidity targets (not to RR)
    buf = _liq_buffer(entry_f, t, atr_val)

    best_mode = "RR"
    tp_raw = float(tp_rr)

    # Accept a liquidity target only if:
    # - it is closer than RR target (by factor), AND
    # - its RR is >= rr_min - eps
    max_factor = float(TP1_LIQ_MAX_DIST_FACTOR)
    eps_rr = float(TP1_LIQ_MIN_RR_EPS)

    best_dist = rr_target_dist  # start from RR target dist
    best_tp = tp_raw

    for mode, lvl in candidates:
        # ensure direction
        if b == "LONG" and lvl <= entry_f:
            continue
        if b == "SHORT" and lvl >= entry_f:
            continue

        # front-run
        if b == "LONG":
            tp_liq = float(lvl - buf)
        else:
            tp_liq = float(lvl + buf)

        tp_liq = _enforce_min_tp_distance(tp_liq, entry_f, b, t)

        dist = abs(tp_liq - entry_f)
        if dist <= 0:
            continue

        # closer-than-RR check (default strict)
        if max_factor > 0:
            if dist > (rr_target_dist * max_factor):
                continue

        rr_liq = _rr_of_target(entry_f, sl_f, tp_liq, b)
        if rr_liq is None:
            continue

        if rr_liq >= max(0.0, rr_min - eps_rr):
            # choose nearest acceptable liquidity
            if dist < best_dist:
                best_dist = dist
                best_tp = tp_liq
                best_mode = mode

    tp_raw = float(best_tp)

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

    if atrp < 0.012:
        rr2 += 0.20
    elif atrp > 0.060:
        rr2 -= 0.20

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

    if b == "LONG" and tp2_rounded <= entry_f:
        tp2_rounded = float(entry_f + max((float(MIN_TP_TICKS) * t if t > 0 else 0.0), abs(entry_f) * 0.001))
        tp2_rounded = _round_tp_for_fill(tp2_rounded, "LONG", t)

    if b == "SHORT" and tp2_rounded >= entry_f:
        tp2_rounded = float(entry_f - max((float(MIN_TP_TICKS) * t if t > 0 else 0.0), abs(entry_f) * 0.001))
        tp2_rounded = _round_tp_for_fill(tp2_rounded, "SHORT", t)

    return float(tp2_rounded)
