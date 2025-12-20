print("🔥 analyze_signal.py LOADED (DESK SUPREME BOS v2)")

import logging
import pandas as pd
from typing import Dict, Any, Optional

from structure_utils import (
    analyze_structure,
    htf_trend_ok,
    bos_quality_details,
    commitment_score,
)

from indicators import (
    rsi,
    macd,
    ema,
    institutional_momentum,
    compute_ote,
    volatility_regime,
    extension_signal,
    composite_momentum,
)

from stops import protective_stop_long, protective_stop_short
from tp_clamp import compute_tp1

from institutional_data import compute_full_institutional_analysis

LOGGER = logging.getLogger(__name__)


def compute_premium_discount(df: pd.DataFrame, lookback: int = 80):
    if len(df) < lookback:
        return False, False

    window = df.tail(lookback)
    high = float(window["high"].max())
    low = float(window["low"].min())
    last = float(window["close"].iloc[-1])

    if high <= low:
        return False, False

    mid = (high + low) / 2.0
    in_premium = last > mid
    in_discount = last < mid
    return in_discount, in_premium


def _safe_rr(entry: float, sl: float, tp1: float, bias: str) -> Optional[float]:
    try:
        entry = float(entry)
        sl = float(sl)
        tp1 = float(tp1)
        if bias == "LONG":
            risk = entry - sl
            reward = tp1 - entry
        else:
            risk = sl - entry
            reward = entry - tp1
        if risk <= 0:
            return None
        return reward / risk
    except Exception:
        return None


def _compute_exits(df: pd.DataFrame, entry: float, bias: str, tick: float):
    if bias == "LONG":
        sl, meta = protective_stop_long(df, entry, tick, return_meta=True)
    else:
        sl, meta = protective_stop_short(df, entry, tick, return_meta=True)
    tp1, rr_used = compute_tp1(entry, sl, bias, df=df, tick=tick)
    return {"sl": sl, "tp1": tp1, "rr_used": rr_used, "sl_meta": meta}


class SignalAnalyzer:

    def __init__(self, api_key: str, api_secret: str, api_passphrase: str):
        # RR min de base (ajusté dynamiquement ensuite)
        self.rr_min_inst: float = 1.3

    async def analyze(self, symbol: str, df_h1: pd.DataFrame, df_h4: pd.DataFrame, macro: Any = None):
        LOGGER.info(f"[EVAL] ▶ START {symbol}")

        entry = float(df_h1["close"].iloc[-1])

        # 1 — STRUCTURE H1
        struct = analyze_structure(df_h1)
        bias = struct.get("trend", "").upper()
        LOGGER.info(f"[EVAL_PRE] STRUCT={struct}")

        oi_series = struct.get("oi_series", None)
        has_oi = oi_series is not None
        LOGGER.info(f"[EVAL_PRE] OI_STRUCT symbol={symbol} has_oi={has_oi}")

        if bias not in ("LONG", "SHORT"):
            LOGGER.info("[EVAL_REJECT] No clear trend (RANGE)")
            return None

        # 2 — ALIGNEMENT H4
        if not htf_trend_ok(df_h4, bias):
            LOGGER.info("[EVAL_REJECT] HTF trend veto")
            return None

        # 3 — BOS / BOS_QUALITY
        bos_flag = struct.get("bos", False)
        bos_dir = struct.get("bos_direction", None)
        bos_type = struct.get("bos_type", None)

        bos_q = bos_quality_details(
            df_h1,
            oi_series=oi_series,
            df_liq=df_h1,
            price=entry,
            direction=bos_dir,
        )
        LOGGER.info(
            f"[EVAL_PRE] BOS_QUALITY={bos_q} bos_flag={bos_flag} bos_type={bos_type}"
        )

        if not bos_flag:
            LOGGER.info("[EVAL_REJECT] No BOS on LTF")
            return None

        if not bos_q.get("ok", False):
            LOGGER.info(
                "[EVAL_REJECT] BOS quality too weak "
                f"(score={bos_q.get('score')}, reasons={bos_q.get('reasons')})"
            )
            return None

        # 4 — INSTITUTIONNEL (Binance / autres)
        inst = await compute_full_institutional_analysis(symbol, bias)
        inst_score = inst.get("institutional_score", 0)
        LOGGER.info(f"[INST_RAW] score={inst_score} details={inst}")

        if inst_score < 2:
            LOGGER.info("[EVAL_REJECT] Institutional score < 2")
            return None

        # 5 — MOMENTUM / COMPOSITE / EXTENSION / VOL REGIME
        mom = institutional_momentum(df_h1)
        comp = composite_momentum(df_h1)
        vol_regime = volatility_regime(df_h1)
        ext_sig = extension_signal(df_h1)

        LOGGER.info(f"[EVAL_PRE] MOMENTUM={mom}")
        LOGGER.info(
            f"[EVAL_PRE] MOMENTUM_COMPOSITE score={comp.get('score')} "
            f"label={comp.get('label')} components={comp.get('components')}"
        )
        LOGGER.info(f"[EVAL_PRE] VOL_REGIME={vol_regime} EXTENSION={ext_sig}")

        # Directionnel : momentum doit confirmer le biais
        if bias == "LONG" and mom not in ("BULLISH", "STRONG_BULLISH"):
            LOGGER.info("[EVAL_REJECT] Momentum not bullish for LONG")
            return None
        if bias == "SHORT" and mom not in ("BEARISH", "STRONG_BEARISH"):
            LOGGER.info("[EVAL_REJECT] Momentum not bearish for SHORT")
            return None

        # Filtre d'extension : évite d'acheter les extrêmes
        if ext_sig == "OVEREXTENDED_LONG" and bias == "LONG":
            LOGGER.info("[EVAL_REJECT] Extension signal OVEREXTENDED_LONG for LONG bias")
            return None
        if ext_sig == "OVEREXTENDED_SHORT" and bias == "SHORT":
            LOGGER.info("[EVAL_REJECT] Extension signal OVEREXTENDED_SHORT for SHORT bias")
            return None

        # 6 — PREMIUM / DISCOUNT
        discount, premium = compute_premium_discount(df_h1)
        LOGGER.info(f"[EVAL_PRE] PREMIUM={premium} DISCOUNT={discount}")

        # 7 — SL / TP1 / RR
        # NOTE: tick=0.1 ici reste un proxy; pour la vraie microstructure,
        # tu as le BitgetTrader qui quantize, donc ça va.
        exits = _compute_exits(df_h1, entry, bias, tick=0.1)
        rr = _safe_rr(entry, exits["sl"], exits["tp1"], bias)
        LOGGER.info(
            f"[EVAL_PRE] RR={rr} raw_rr={exits['rr_used']} "
            f"sl={exits['sl']} tp1={exits['tp1']}"
        )

        comp_score = float(comp.get("score", 50.0)) if isinstance(comp, dict) else 50.0
        rr_min = float(self.rr_min_inst)

        # RR dynamique en fonction du régime + momentum composite
        if vol_regime == "HIGH":
            rr_min = max(rr_min, 1.6)
        elif vol_regime == "LOW" and comp_score >= 70:
            rr_min = max(rr_min - 0.1, 1.1)

        if comp_score <= 40:
            rr_min = max(rr_min, 1.5)

        LOGGER.info(
            f"[EVAL_PRE] RR_DYNAMIC rr={rr} rr_min={rr_min} "
            f"vol_regime={vol_regime} comp_score={comp_score}"
        )

        if rr is None or rr < rr_min:
            LOGGER.info("[EVAL_REJECT] RR < dynamic minimum")
            return None

        # 8 — VALIDATION
        LOGGER.info(f"[EVAL] VALID {symbol} RR={rr}")

        return {
            "valid": True,
            "symbol": symbol,
            "side": "BUY" if bias == "LONG" else "SELL",
            "bias": bias,
            "entry": entry,
            "sl": exits["sl"],
            "tp1": exits["tp1"],
            "tp2": None,
            "rr": rr,
            "qty": 1,

            "structure": struct,
            "bos_quality": bos_q,
            "institutional": inst,
            "momentum": mom,
            "premium": premium,
            "discount": discount,
        }
