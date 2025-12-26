# =====================================================================
# tp_utils.py — TP institutionnel dynamique (RR-based)
# =====================================================================
import pandas as pd
from typing import Tuple
from indicators import true_atr


def _round_to_tick(x: float, tick: float) -> float:
    if tick <= 0:
        return float(x)
    steps = round(x / tick)
    return round(steps * tick, 12)


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
        TP1 final arrondi au tick,
        RR réel utilisé.

    Logic institutionnelle :
        - On calcule la distance SL.
        - On applique un RR cible.
        - On ajuste selon ATR (régime volatilité).
        - On clamp dans une fourchette réaliste : [min_rr, max_rr].
        - Retourne (tp1, rr_effective)

    """
    risk = abs(entry - sl)

    if risk <= 0:
        # impossible → fallback mini
        if bias.upper() == "LONG":
            return entry + entry * 0.01, 0.01
        else:
            return entry - entry * 0.01, 0.01

    # Base : RR minimal institutionnel
    rr_base = min_rr

    # ---------------------------------------------------------
    # Ajustement ATR : si marché très volatile → on élargit TP1
    # ---------------------------------------------------------
    atr = true_atr(df)
    if len(atr) > 20:
        atrp = float(atr.iloc[-1]) / max(1e-9, float(df["close"].iloc[-1]))
        if atrp > 0.02:     # volatilité très haute
            rr_base *= 1.2
        elif atrp < 0.008:  # faible volatilité
            rr_base *= 0.9

    rr_base = max(min_rr, min(rr_base, max_rr))

    # ---------------------------------------------------------
    # TP théorique
    # ---------------------------------------------------------
    if bias.upper() == "LONG":
        tp = entry + risk * rr_base
    else:
        tp = entry - risk * rr_base

    # ---------------------------------------------------------
    # Arrondi au tickSize Bitget
    # ---------------------------------------------------------
    tp = _round_to_tick(tp, tick)

    # ---------------------------------------------------------
    # RR réel recalculé
    # ---------------------------------------------------------
    rr_effective = abs((tp - entry) / risk)

    # clamp final si arrondi trop modifié le RR
    if rr_effective < min_rr:
        rr_effective = min_rr

    return float(tp), float(rr_effective)
