from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd

from structure_utils import (
    analyze_structure,
    htf_trend_ok,
    bos_quality_details,
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

from settings import (
    MIN_INST_SCORE,
    RR_MIN_STRICT,
    RR_MIN_DESK_PRIORITY,
    INST_SCORE_DESK_PRIORITY,
    DESK_EV_MODE,
    REQUIRE_STRUCTURE,
    REQUIRE_MOMENTUM,
    REQUIRE_HTF_ALIGN,
    REQUIRE_BOS_QUALITY,
)

LOGGER = logging.getLogger(__name__)


# =====================================================================
# Helpers
# =====================================================================

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


# =====================================================================
# Signal Analyzer (Desk)
# =====================================================================

class SignalAnalyzer:
    """
    Deux chemins d'entrée :

      1) BOS_STRICT  (setup principal, A+)
         - Trend H1 clair (LONG / SHORT)
         - BOS flag True
         - BOS_QUALITY ok si REQUIRE_BOS_QUALITY
         - H4 aligné si REQUIRE_HTF_ALIGN
         - inst_score >= MIN_INST_SCORE
         - Momentum aligné si REQUIRE_MOMENTUM
         - RR >= RR_MIN_STRICT
         -> setup_type = "BOS_STRICT"

      2) INST_CONTINUATION  (optionnel, si DESK_EV_MODE=True)
         - Trend H1 clair
         - BOS non requis (ou pas assez propre pour BOS_STRICT)
         - inst_score >= INST_SCORE_DESK_PRIORITY
         - composite_momentum.score >= 70
         - momentum institutionnel aligné
         - pas d'extension extrême contre le biais
         - RR >= RR_MIN_DESK_PRIORITY
         -> setup_type = "INST_CONTINUATION"

    Dans les deux cas, la gestion SL/TP et le risk manager restent les mêmes.
    """

    def __init__(self, api_key: str, api_secret: str, api_passphrase: str):
        # plus de rr_min interne : tout est piloté via settings
        pass

    async def analyze(
        self,
        symbol: str,
        df_h1: pd.DataFrame,
        df_h4: pd.DataFrame,
        macro: Any = None,
    ) -> Optional[Dict[str, Any]]:

        LOGGER.info(f"[EVAL] ▶ START {symbol}")

        entry = float(df_h1["close"].iloc[-1])

        # ------------------------------------------------------------------
        # 1 — STRUCTURE H1
        # ------------------------------------------------------------------
        struct = analyze_structure(df_h1)
        bias = struct.get("trend", "").upper()
        LOGGER.info(f"[EVAL_PRE] STRUCT={struct}")

        oi_series = struct.get("oi_series", None)
        has_oi_struct = oi_series is not None
        LOGGER.info(f"[EVAL_PRE] OI_STRUCT symbol={symbol} has_oi={has_oi_struct}")

        if REQUIRE_STRUCTURE:
            if bias not in ("LONG", "SHORT"):
                LOGGER.info("[EVAL_REJECT] No clear trend (RANGE) and REQUIRE_STRUCTURE=True")
                return None
        else:
            if bias not in ("LONG", "SHORT"):
                # On peut laisser une chance plus tard, mais c'est rarement pertinent
                LOGGER.info("[EVAL_PRE] Trend RANGE but REQUIRE_STRUCTURE=False")

        # ------------------------------------------------------------------
        # 2 — H4 ALIGNMENT (HTF veto)
        # ------------------------------------------------------------------
        if REQUIRE_HTF_ALIGN:
            if not htf_trend_ok(df_h4, bias):
                LOGGER.info("[EVAL_REJECT] HTF trend veto (REQUIRE_HTF_ALIGN=True)")
                return None

        # ------------------------------------------------------------------
        # 3 — BOS / BOS_QUALITY (on NE rejette pas encore, on stocke l'info)
        #     → BOS_STRICT demandera bos_flag=True + quality OK
        # ------------------------------------------------------------------
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

        bos_quality_ok = bos_q.get("ok", True)

        # ------------------------------------------------------------------
        # 4 — INSTITUTIONNEL (Binance)
        # ------------------------------------------------------------------
        inst = await compute_full_institutional_analysis(symbol, bias)
        inst_score = inst.get("institutional_score", 0)
        LOGGER.info(f"[INST_RAW] score={inst_score} details={inst}")

        if inst_score < MIN_INST_SCORE:
            LOGGER.info(
                f"[EVAL_REJECT] Institutional score < MIN_INST_SCORE "
                f"({inst_score} < {MIN_INST_SCORE})"
            )
            return None

        # ------------------------------------------------------------------
        # 5 — MOMENTUM / VOL / EXTENSION
        # ------------------------------------------------------------------
        mom = institutional_momentum(df_h1)
        comp = composite_momentum(df_h1)
        vol_regime = volatility_regime(df_h1)
        ext_sig = extension_signal(df_h1)

        comp_score = float(comp.get("score", 50.0)) if isinstance(comp, dict) else 50.0
        comp_label = str(comp.get("label", "NEUTRAL")) if isinstance(comp, dict) else "NEUTRAL"

        LOGGER.info(f"[EVAL_PRE] MOMENTUM={mom}")
        LOGGER.info(
            f"[EVAL_PRE] MOMENTUM_COMPOSITE score={comp_score} label={comp_label} "
            f"components={comp.get('components') if isinstance(comp, dict) else {}}"
        )
        LOGGER.info(f"[EVAL_PRE] VOL_REGIME={vol_regime} EXTENSION={ext_sig}")

        if REQUIRE_MOMENTUM:
            if bias == "LONG" and mom not in ("BULLISH", "STRONG_BULLISH"):
                LOGGER.info(
                    "[EVAL_REJECT] Momentum not bullish for LONG "
                    f"(REQUIRE_MOMENTUM=True, mom={mom})"
                )
                return None
            if bias == "SHORT" and mom not in ("BEARISH", "STRONG_BEARISH"):
                LOGGER.info(
                    "[EVAL_REJECT] Momentum not bearish for SHORT "
                    f"(REQUIRE_MOMENTUM=True, mom={mom})"
                )
                return None

        # Extension : on évite de rentrer pile dans un squeeze extrême
        if ext_sig == "OVEREXTENDED_LONG" and bias == "LONG":
            LOGGER.info(
                "[EVAL_REJECT] Extension signal OVEREXTENDED_LONG for LONG bias (take-profit zone)"
            )
            return None
        if ext_sig == "OVEREXTENDED_SHORT" and bias == "SHORT":
            LOGGER.info(
                "[EVAL_REJECT] Extension signal OVEREXTENDED_SHORT for SHORT bias (take-profit zone)"
            )
            return None

        # ------------------------------------------------------------------
        # 6 — PREMIUM / DISCOUNT (info)
        # ------------------------------------------------------------------
        discount, premium = compute_premium_discount(df_h1)
        LOGGER.info(f"[EVAL_PRE] PREMIUM={premium} DISCOUNT={discount}")

        # ------------------------------------------------------------------
        # 7 — SL / TP1 / RR
        # ------------------------------------------------------------------
        exits = _compute_exits(df_h1, entry, bias, tick=0.1)
        rr = _safe_rr(entry, exits["sl"], exits["tp1"], bias)
        LOGGER.info(
            f"[EVAL_PRE] RR={rr} raw_rr={exits['rr_used']} "
            f"sl={exits['sl']} tp1={exits['tp1']}"
        )

        if rr is None or rr <= 0:
            LOGGER.info("[EVAL_REJECT] Invalid RR (<= 0)")
            return None

        # ------------------------------------------------------------------
        # 8 — PATH 1 : BOS_STRICT (setup principal)
        # ------------------------------------------------------------------
        bos_strict_ok = False

        if bos_flag:
            if (not REQUIRE_BOS_QUALITY) or bos_quality_ok:
                if rr >= RR_MIN_STRICT and inst_score >= MIN_INST_SCORE:
                    bos_strict_ok = True

        LOGGER.info(
            "[EVAL_PRE] BOS_STRICT_CHECK "
            f"bos_flag={bos_flag}, bos_quality_ok={bos_quality_ok}, "
            f"rr={rr} vs RR_MIN_STRICT={RR_MIN_STRICT}, "
            f"inst_score={inst_score} vs MIN_INST_SCORE={MIN_INST_SCORE}, "
            f"bos_strict_ok={bos_strict_ok}"
        )

        if bos_strict_ok:
            LOGGER.info(
                f"[EVAL] VALID {symbol} RR={rr} SETUP=BOS_STRICT "
                f"(DESK_EV_MODE={DESK_EV_MODE})"
            )
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
                "setup_type": "BOS_STRICT",
                "structure": struct,
                "bos_quality": bos_q,
                "institutional": inst,
                "momentum": mom,
                "premium": premium,
                "discount": discount,
            }

        # ------------------------------------------------------------------
        # 9 — PATH 2 : INST_CONTINUATION (si DESK_EV_MODE=True)
        # ------------------------------------------------------------------
        inst_continuation_ok = False

        if DESK_EV_MODE:
            # Conditions continuation :
            #  - inst_score >= INST_SCORE_DESK_PRIORITY
            #  - RR >= RR_MIN_DESK_PRIORITY
            #  - composite momentum bien orienté
            #  - momentum institutionnel aligné (déjà vérifié si REQUIRE_MOMENTUM=True)
            #  - pas de signal d'extension contre le biais (déjà géré)
            good_inst = inst_score >= INST_SCORE_DESK_PRIORITY
            good_rr = rr >= RR_MIN_DESK_PRIORITY

            if bias == "LONG":
                good_comp = comp_score >= 70 and comp_label in ("BULLISH", "SLIGHT_BULLISH")
            else:
                good_comp = comp_score >= 70 and comp_label in ("BEARISH", "SLIGHT_BEARISH")

            inst_continuation_ok = good_inst and good_rr and good_comp

            LOGGER.info(
                "[EVAL_PRE] INST_CONTINUATION_CHECK "
                f"inst_score={inst_score} vs INST_SCORE_DESK_PRIORITY={INST_SCORE_DESK_PRIORITY}, "
                f"rr={rr} vs RR_MIN_DESK_PRIORITY={RR_MIN_DESK_PRIORITY}, "
                f"comp_score={comp_score}, comp_label={comp_label}, "
                f"inst_continuation_ok={inst_continuation_ok}"
            )

        if DESK_EV_MODE and inst_continuation_ok:
            LOGGER.info(
                f"[EVAL] VALID {symbol} RR={rr} SETUP=INST_CONTINUATION "
                f"(DESK_EV_MODE=True)"
            )
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
                "setup_type": "INST_CONTINUATION",
                "structure": struct,
                "bos_quality": bos_q,
                "institutional": inst,
                "momentum": mom,
                "premium": premium,
                "discount": discount,
            }

        # ------------------------------------------------------------------
        # 10 — Aucun des setups n'est validé
        # ------------------------------------------------------------------
        LOGGER.info(
            "[EVAL_REJECT] No setup validated "
            f"(BOS_STRICT_OK={bos_strict_ok}, "
            f"INST_CONTINUATION_OK={inst_continuation_ok}, "
            f"DESK_EV_MODE={DESK_EV_MODE})"
        )
        return None
