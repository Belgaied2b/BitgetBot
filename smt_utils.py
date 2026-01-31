# =====================================================================
# smt_utils.py — SMT Divergence (cross-symbol) lightweight
# =====================================================================

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import pandas as pd


def _has_cols(df: pd.DataFrame, cols: Tuple[str, ...]) -> bool:
    return isinstance(df, pd.DataFrame) and (not df.empty) and all(c in df.columns for c in cols)


def _swings_from_series(s: pd.Series, left: int = 3, right: int = 3) -> Dict[str, List[Tuple[int, float]]]:
    """
    Simple pivot swings on a 1D series.
    """
    highs: List[Tuple[int, float]] = []
    lows: List[Tuple[int, float]] = []

    x = pd.Series(s).astype(float).to_numpy()
    n = len(x)
    if n < left + right + 10:
        return {"highs": highs, "lows": lows}

    for i in range(left, n - right):
        win = x[i - left : i + right + 1]
        v = float(x[i])
        if not np.isfinite(v):
            continue
        if v >= float(np.max(win)):
            highs.append((i, v))
        if v <= float(np.min(win)):
            lows.append((i, v))

    return {"highs": highs, "lows": lows}


def _last_two(sw: List[Tuple[int, float]]) -> Optional[Tuple[Tuple[int, float], Tuple[int, float]]]:
    if not sw or len(sw) < 2:
        return None
    return sw[-2], sw[-1]


def compute_smt_divergence(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    lookback: int = 160,
    left: int = 3,
    right: int = 3,
    col: str = "close",
) -> Dict[str, Any]:
    """
    SMT divergence logic (2-swing):
      - Bearish SMT: A makes Higher High but B makes Lower High (or inverse).
      - Bullish SMT: A makes Lower Low  but B makes Higher Low (or inverse).

    This is "cross-symbol" divergence. You typically:
      - veto LONG if bearish_smt=True
      - veto SHORT if bullish_smt=True
    """
    out: Dict[str, Any] = {
        "available": False,
        "bearish_smt": False,
        "bullish_smt": False,
        "details": {},
    }

    try:
        if not _has_cols(df_a, (col,)) or not _has_cols(df_b, (col,)):
            return out

        a = df_a.tail(int(max(80, lookback)))[col].astype(float).reset_index(drop=True)
        b = df_b.tail(int(max(80, lookback)))[col].astype(float).reset_index(drop=True)
        if len(a) < 30 or len(b) < 30:
            return out

        swa = _swings_from_series(a, left=left, right=right)
        swb = _swings_from_series(b, left=left, right=right)

        a_h2 = _last_two(swa["highs"])
        b_h2 = _last_two(swb["highs"])
        a_l2 = _last_two(swa["lows"])
        b_l2 = _last_two(swb["lows"])

        bearish = False
        bullish = False

        # HH vs LH (bearish SMT)
        if a_h2 and b_h2:
            (_, a_h1), (_, a_h2v) = a_h2
            (_, b_h1), (_, b_h2v) = b_h2
            a_makes_hh = a_h2v > a_h1
            b_makes_lh = b_h2v < b_h1
            b_makes_hh = b_h2v > b_h1
            a_makes_lh = a_h2v < a_h1
            bearish = (a_makes_hh and b_makes_lh) or (b_makes_hh and a_makes_lh)

        # LL vs HL (bullish SMT)
        if a_l2 and b_l2:
            (_, a_l1), (_, a_l2v) = a_l2
            (_, b_l1), (_, b_l2v) = b_l2
            a_makes_ll = a_l2v < a_l1
            b_makes_hl = b_l2v > b_l1
            b_makes_ll = b_l2v < b_l1
            a_makes_hl = a_l2v > a_l1
            bullish = (a_makes_ll and b_makes_hl) or (b_makes_ll and a_makes_hl)

        out["available"] = True
        out["bearish_smt"] = bool(bearish)
        out["bullish_smt"] = bool(bullish)
        out["details"] = {
            "a_highs": swa["highs"][-2:] if len(swa["highs"]) >= 2 else swa["highs"],
            "b_highs": swb["highs"][-2:] if len(swb["highs"]) >= 2 else swb["highs"],
            "a_lows": swa["lows"][-2:] if len(swa["lows"]) >= 2 else swa["lows"],
            "b_lows": swb["lows"][-2:] if len(swb["lows"]) >= 2 else swb["lows"],
        }
        return out

    except Exception as e:
        out["available"] = False
        out["details"] = {"error": str(e)}
        return out
