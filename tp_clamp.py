# =====================================================================
# tp_clamp.py — Institutional TP Engine (Desk Lead)
# TP1 / TP2 dynamiques basés sur RR, volatilité et momentum
# Compatible analyze_signal.py (compute_tp1) + future compute_tp2
# =====================================================================

from typing import Tuple, Optional

import numpy as np
import pandas as pd

from indicators import true_atr


# ============================================================
# Helper — round price to tick
# ============================================================

def _round_to_tick(price: float, tick: float) -> float:
    """
    Arrondit un prix au tick le plus proche.

    Si le tick est invalide (<= 0 ou NaN), retourne le prix brut.
    """
    try:
        tick = float(tick)
        if not np.isfinite(tick) or tick <= 0:
            return float(price)
        return float(round(float(price) / tick) * tick)
    except Exception:
        return float(price)


# ============================================================
# Helper — ATR local (en utilisant true_atr pour cohérence)
# ============================================================

def _atr(df: pd.DataFrame, length: int = 14) -> float:
    """
    Renvoie la dernière valeur d'ATR (true range lissé).
    Fallback range simple si les données sont insuffisantes.
    """
    try:
        if df is None or len(df) < length + 3:
            raise ValueError("not enough data")
        atr_series = true_atr(df, length=length)
        val = float(atr_series.iloc[-1])
        if not np.isfinite(val) or val <= 0:
            raise ValueError("atr nan")
        return val
    except Exception:
        if df is None or len(df) == 0:
            return 0.0
        w = df.tail(max(length, 10))
        approx = float((w["high"].max() - w["low"].min()) / max(len(w), 1))
        return approx if np.isfinite(approx) else 0.0


# ============================================================
# TP1 — RR dynamique institutionnel
# ============================================================

def compute_tp1(
    entry: float,
    sl: float,
    bias: str,
    df: pd.DataFrame,
    tick: float = 0.1,
) -> Tuple[float, float]:
    """
    Calcule un TP1 institutionnel basé sur:
      - Risk / Reward de base
      - Volatilité (ATR%)
      - Largeur du stop (risk%)
      - Momentum récent du prix

    Retourne:
      (tp1, rr_effective)

    Signature compatible avec analyze_signal.py qui fait:
      tp1, rr_used = compute_tp1(entry, sl, bias, df=df, tick=tick)
    """
    bias = (bias or "").upper()
    entry = float(entry)
    sl = float(sl)

    risk = abs(entry - sl)
    if risk <= 0:
        return float(entry), 0.0

    # Si bias invalide, fallback neutre: RR fixe ~ 1.5
    if bias not in ("LONG", "SHORT"):
        rr_base = 1.5
        tp_raw = entry + risk * rr_base if entry >= sl else entry - risk * rr_base
        tp_rounded = _round_to_tick(tp_raw, tick)
        rr_effective = abs(tp_rounded - entry) / risk
        return float(tp_rounded), float(rr_effective)

    # -----------------------------
    # 1) Volatilité & risk%
    # -----------------------------
    atr_val = _atr(df)
    atrp = atr_val / max(abs(entry), 1e-8)   # ATR en % relatif
    riskp = risk / max(abs(entry), 1e-8)    # taille du stop en %

    # RR base institutionnel
    rr_min = 1.35
    rr_max = 1.80
    rr_base = 1.50

    # -----------------------------
    # 2) Momentum de prix
    # -----------------------------
    closes = df["close"].astype(float)
    if len(closes) >= 25:
        ret_5 = closes.iloc[-1] / closes.iloc[-5] - 1.0
        ret_20 = closes.iloc[-1] / closes.iloc[-20] - 1.0
    else:
        ret_5 = 0.0
        ret_20 = 0.0

    # Momentum aligné ?
    if bias == "LONG":
        mom_sign = 1 if (ret_5 > 0 and ret_20 > 0) else -1 if (ret_5 < 0 and ret_20 < 0) else 0
    else:
        mom_sign = 1 if (ret_5 < 0 and ret_20 < 0) else -1 if (ret_5 > 0 and ret_20 > 0) else 0

    # -----------------------------
    # 3) Ajustements RR
    # -----------------------------
    rr = rr_base

    # a) Volatilité (ATR%) : plus c'est volatile, plus on réduit un peu le RR
    if atrp > 0.06:
        rr -= 0.25
    elif atrp > 0.03:
        rr -= 0.10
    elif atrp < 0.01:
        rr += 0.10

    # b) Largeur du stop (risk%)
    if riskp > 0.06:
        rr -= 0.25
    elif riskp > 0.03:
        rr -= 0.10
    elif riskp < 0.015:
        rr += 0.15

    # c) Momentum directionnel
    if mom_sign > 0:
        rr += 0.15
    elif mom_sign < 0:
        rr -= 0.10

    # Clamp final
    rr = max(rr_min, min(rr_max, rr))

    # -----------------------------
    # 4) Construction TP1
    # -----------------------------
    if bias == "LONG":
        tp_raw = entry + risk * rr
    else:
        tp_raw = entry - risk * rr

    tp_rounded = _round_to_tick(tp_raw, tick)
    if tp_rounded <= 0:
        tp_rounded = tp_raw

    rr_effective = abs(tp_rounded - entry) / risk

    return float(tp_rounded), float(rr_effective)


# ============================================================
# TP2 — Runner institutionnel
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
    Calcule un TP2 de type "runner" basé sur le même risk que TP1.

    Logique:
      - Si rr1 est fourni (RR utilisé pour TP1), construit un RR2 > rr1,
        mais clampé dans une zone réaliste (2.0–3.5).
      - Sinon, part sur un RR2 par défaut (~2.0) ajusté par la volatilité.

    Retourne uniquement le prix de TP2 (float).
    """
    bias = (bias or "").upper()
    if bias not in ("LONG", "SHORT"):
        return float(entry)

    entry = float(entry)
    sl = float(sl)
    risk = abs(entry - sl)
    if risk <= 0:
        return float(entry)

    # Volatilité pour calibrer jusqu'où on peut viser
    atr_val = _atr(df)
    atrp = atr_val / max(abs(entry), 1e-8)

    # Base RR2 à partir de rr1 ou de 2.0
    if rr1 is None or rr1 <= 0:
        rr2 = 2.0
    else:
        # On veut quelque chose de sensiblement plus loin que TP1
        rr2 = max(2.0, rr1 * 1.6, rr1 + 0.7)

    # Ajustement avec volatilité :
    # - Si ATR% faible → on peut viser plus loin
    # - Si ATR% très fort → on réduit un peu
    if atrp < 0.01:
        rr2 += 0.3
    elif atrp > 0.06:
        rr2 -= 0.3

    # Clamp final entre 2.0 et 3.5
    rr2 = max(2.0, min(3.5, rr2))

    if bias == "LONG":
        tp2_raw = entry + risk * rr2
    else:
        tp2_raw = entry - risk * rr2

    tp2_rounded = _round_to_tick(tp2_raw, tick)
    if tp2_rounded <= 0:
        tp2_rounded = tp2_raw

    return float(tp2_rounded)
