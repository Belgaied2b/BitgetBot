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

def _reject(reason: str, **extra: Any) -> Dict[str, Any]:
    d: Dict[str, Any] = {"valid": False, "reject_reason": reason}
    d.update(extra)
    return d


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


def _safe_rr(entry: float, sl: float, tp: float, bias: str) -> Optional[float]:
    try:
        entry = float(entry)
        sl = float(sl)
        tp = float(tp)
        if bias == "LONG":
            risk = entry - sl
            reward = tp - entry
        else:
            risk = sl - entry
            reward = entry - tp
        if risk <= 0:
            return None
        return reward / risk
    except Exception:
        return None


def estimate_tick_from_price(price: float) -> float:
    """
    Estimation tick (fallback). Le vrai tick est pris côté trader/scanner,
    mais on en a besoin ici pour les stops/tp clamp.
    """
    p = abs(float(price))
    if p >= 10000:
        return 1.0
    elif p >= 1000:
        return 0.1
    elif p >= 100:
        return 0.01
    elif p >= 10:
        return 0.001
    elif p >= 1:
        return 0.0001
    elif p >= 0.1:
        return 0.00001
    elif p >= 0.01:
        return 0.000001
    else:
        return 0.0000001


def _compute_exits(df: pd.DataFrame, entry: float, bias: str, tick: float):
    if bias == "LONG":
        sl, meta = protective_stop_long(df, entry, tick, return_meta=True)
    else:
        sl, meta = protective_stop_short(df, entry, tick, return_meta=True)

    tp1, rr_used = compute_tp1(entry, sl, bias, df=df, tick=tick)

    # TP2 runner: 2R (propre, simple, efficace)
    # LONG: entry + 2*(entry - sl)
    # SHORT: entry - 2*(sl - entry)
    risk = abs(entry - sl)
    if risk <= 0:
        tp2 = None
    else:
        if bias == "LONG":
            tp2 = entry + 2.0 * risk
        else:
            tp2 = entry - 2.0 * risk

    return {"sl": sl, "tp1": tp1, "tp2": tp2, "rr_used": rr_used, "sl_meta": meta}


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
         - inst_score >= MIN_INST_SCORE (ou bypass Option 2 si unmapped)
         - Momentum aligné si REQUIRE_MOMENTUM
         - RR >= RR_MIN_STRICT
         -> setup_type = "BOS_STRICT"

      2) INST_CONTINUATION  (optionnel, si DESK_EV_MODE=True)
         - Trend H1 clair
         - BOS non requis
         - inst_score >= INST_SCORE_DESK_PRIORITY (ou bypass Option 2 si unmapped)
         - composite_momentum.score >= 70
         - pas d'extension extrême contre le biais
         - RR >= RR_MIN_DESK_PRIORITY
         -> setup_type = "INST_CONTINUATION"
    """

    def __init__(self, *args, **kwargs):
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
                return _reject("no_clear_trend_range", structure=struct)
        else:
            if bias not in ("LONG", "SHORT"):
                LOGGER.info("[EVAL_PRE] Trend RANGE but REQUIRE_STRUCTURE=False")
                return _reject("trend_range_but_allowed", structure=struct)

        # ------------------------------------------------------------------
        # 2 — H4 ALIGNMENT (HTF veto)
        # ------------------------------------------------------------------
        if REQUIRE_HTF_ALIGN:
            if not htf_trend_ok(df_h4, bias):
                LOGGER.info("[EVAL_REJECT] HTF trend veto (REQUIRE_HTF_ALIGN=True)")
                return _reject("htf_veto", bias=bias, structure=struct)

        # ------------------------------------------------------------------
        # 3 — BOS / BOS_QUALITY
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
        LOGGER.info(f"[EVAL_PRE] BOS_QUALITY={bos_q} bos_flag={bos_flag} bos_type={bos_type}")

        bos_quality_ok = bos_q.get("ok", True)

        # ------------------------------------------------------------------
        # 4 — INSTITUTIONNEL (Binance) — OPTION 2
        # ------------------------------------------------------------------
        inst = await compute_full_institutional_analysis(symbol, bias)
        inst_score = int(inst.get("institutional_score", 0) or 0)
        inst_available = bool(inst.get("available"))
        warnings = inst.get("warnings") or []

        LOGGER.info(f"[INST_RAW] score={inst_score} details={inst}")

        bypass_min_inst = (not inst_available) and ("symbol_not_mapped_to_binance" in warnings)
        if bypass_min_inst:
            inst["bypass_min_inst"] = True
            LOGGER.info("[EVAL_WARN] bypass MIN_INST_SCORE (binance unmapped) symbol=%s", symbol)

        # Gate MIN_INST_SCORE : strict sauf si bypass Option 2
        if inst_score < MIN_INST_SCORE and not bypass_min_inst:
            LOGGER.info(f"[EVAL_REJECT] Institutional score < MIN_INST_SCORE ({inst_score} < {MIN_INST_SCORE})")
            return _reject("inst_score_low", institutional=inst, inst_score=inst_score)

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
                LOGGER.info("[EVAL_REJECT] Momentum not bullish for LONG (REQUIRE_MOMENTUM=True, mom=%s)", mom)
                return _reject("momentum_not_bullish", bias=bias, momentum=mom, institutional=inst)
            if bias == "SHORT" and mom not in ("BEARISH", "STRONG_BEARISH"):
                LOGGER.info("[EVAL_REJECT] Momentum not bearish for SHORT (REQUIRE_MOMENTUM=True, mom=%s)", mom)
                return _reject("momentum_not_bearish", bias=bias, momentum=mom, institutional=inst)

        # Extension : pas rentrer dans une extension extrême
        if ext_sig == "OVEREXTENDED_LONG" and bias == "LONG":
            LOGGER.info("[EVAL_REJECT] Extension signal OVEREXTENDED_LONG for LONG bias (take-profit zone)")
            return _reject("overextended_long", bias=bias, extension=ext_sig, institutional=inst)

        if ext_sig == "OVEREXTENDED_SHORT" and bias == "SHORT":
            LOGGER.info("[EVAL_REJECT] Extension signal OVEREXTENDED_SHORT for SHORT bias (take-profit zone)")
            return _reject("overextended_short", bias=bias, extension=ext_sig, institutional=inst)

        # ------------------------------------------------------------------
        # 6 — PREMIUM / DISCOUNT (info)
        # ------------------------------------------------------------------
        discount, premium = compute_premium_discount(df_h1)
        LOGGER.info(f"[EVAL_PRE] PREMIUM={premium} DISCOUNT={discount}")

        # ------------------------------------------------------------------
        # 7 — SL / TP1 / TP2 / RR
        # ------------------------------------------------------------------
        tick = estimate_tick_from_price(entry)
        exits = _compute_exits(df_h1, entry, bias, tick=tick)

        sl = exits["sl"]
        tp1 = exits["tp1"]
        tp2 = exits.get("tp2")

        rr = _safe_rr(entry, sl, tp1, bias)
        rr2 = _safe_rr(entry, sl, tp2, bias) if tp2 is not None else None

        LOGGER.info(
            f"[EVAL_PRE] RR={rr} raw_rr={exits['rr_used']} rr2={rr2} "
            f"sl={sl} tp1={tp1} tp2={tp2} tick={tick}"
        )

        if rr is None or rr <= 0:
            LOGGER.info("[EVAL_REJECT] Invalid RR (<= 0)")
            return _reject("rr_invalid", entry=entry, sl=sl, tp1=tp1, bias=bias, institutional=inst)

        # TP2 sanity (évite missing_tp côté scanner)
        if tp2 is None or float(tp2) <= 0:
            LOGGER.info("[EVAL_REJECT] TP2 invalid/missing")
            return _reject("tp2_missing", entry=entry, sl=sl, tp1=tp1, tp2=tp2, bias=bias, institutional=inst)

        # ------------------------------------------------------------------
        # 8 — PATH 1 : BOS_STRICT (setup principal)
        # ------------------------------------------------------------------
        bos_strict_ok = False
        inst_gate_ok = (inst_score >= MIN_INST_SCORE) or bypass_min_inst

        if bos_flag:
            if (not REQUIRE_BOS_QUALITY) or bos_quality_ok:
                if rr >= RR_MIN_STRICT and inst_gate_ok:
                    bos_strict_ok = True

        LOGGER.info(
            "[EVAL_PRE] BOS_STRICT_CHECK "
            f"bos_flag={bos_flag}, bos_quality_ok={bos_quality_ok}, "
            f"rr={rr} vs RR_MIN_STRICT={RR_MIN_STRICT}, "
            f"inst_score={inst_score} vs MIN_INST_SCORE={MIN_INST_SCORE}, "
            f"bypass_min_inst={bypass_min_inst}, "
            f"bos_strict_ok={bos_strict_ok}"
        )

        if bos_strict_ok:
            LOGGER.info(f"[EVAL] VALID {symbol} RR={rr} SETUP=BOS_STRICT (DESK_EV_MODE={DESK_EV_MODE})")
            return {
                "valid": True,
                "symbol": symbol,
                "side": "BUY" if bias == "LONG" else "SELL",
                "bias": bias,
                "entry": entry,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
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
            good_inst = (inst_score >= INST_SCORE_DESK_PRIORITY) or bypass_min_inst
            good_rr = rr >= RR_MIN_DESK_PRIORITY

            if bias == "LONG":
                good_comp = comp_score >= 70 and comp_label in ("BULLISH", "SLIGHT_BULLISH")
            else:
                good_comp = comp_score >= 70 and comp_label in ("BEARISH", "SLIGHT_BEARISH")

            inst_continuation_ok = good_inst and good_rr and good_comp

            LOGGER.info(
                "[EVAL_PRE] INST_CONTINUATION_CHECK "
                f"inst_score={inst_score} vs INST_SCORE_DESK_PRIORITY={INST_SCORE_DESK_PRIORITY}, "
                f"bypass_min_inst={bypass_min_inst}, "
                f"rr={rr} vs RR_MIN_DESK_PRIORITY={RR_MIN_DESK_PRIORITY}, "
                f"comp_score={comp_score}, comp_label={comp_label}, "
                f"inst_continuation_ok={inst_continuation_ok}"
            )

        if DESK_EV_MODE and inst_continuation_ok:
            LOGGER.info(f"[EVAL] VALID {symbol} RR={rr} SETUP=INST_CONTINUATION (DESK_EV_MODE=True)")
            return {
                "valid": True,
                "symbol": symbol,
                "side": "BUY" if bias == "LONG" else "SELL",
                "bias": bias,
                "entry": entry,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
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
        return _reject(
            "no_setup_validated",
            bos_strict_ok=bos_strict_ok,
            inst_continuation_ok=inst_continuation_ok,
            desk_ev_mode=DESK_EV_MODE,
            institutional=inst,
            structure=struct,
            bos_quality=bos_q,
            rr=rr,
            tp1=tp1,
            tp2=tp2,
        )
