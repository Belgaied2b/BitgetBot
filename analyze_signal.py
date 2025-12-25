from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from structure_utils import (
    analyze_structure,
    htf_trend_ok,
    bos_quality_details,
)

from indicators import (
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

def _atr(df: pd.DataFrame, n: int = 14) -> float:
    try:
        if df is None or df.empty or len(df) < n + 2:
            return 0.0
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)
        tr = (high - low).abs().to_frame("hl")
        tr["hc"] = (high - prev_close).abs()
        tr["lc"] = (low - prev_close).abs()
        true_range = tr.max(axis=1)
        v = float(true_range.rolling(n).mean().iloc[-1])
        return max(0.0, v)
    except Exception:
        return 0.0


def compute_premium_discount(df: pd.DataFrame, lookback: int = 80) -> Tuple[bool, bool]:
    if df is None or df.empty or len(df) < lookback:
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


def estimate_tick_from_price(price: float) -> float:
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


def _compute_exits(df: pd.DataFrame, entry: float, bias: str, tick: float) -> Dict[str, Any]:
    if bias == "LONG":
        sl, meta = protective_stop_long(df, entry, tick, return_meta=True)
    else:
        sl, meta = protective_stop_short(df, entry, tick, return_meta=True)
    tp1, rr_used = compute_tp1(entry, sl, bias, df=df, tick=tick)
    return {"sl": sl, "tp1": tp1, "rr_used": rr_used, "sl_meta": meta}


def _pick_fvg_entry(struct: Dict[str, Any], entry_mkt: float, bias: str, atr: float) -> Tuple[Optional[float], str]:
    """
    FVG zones (from structure_utils) are like:
      {type: bullish/bearish, start/end, low/high, mid}

    Desk rule:
      - LONG: use bullish FVG, prefer low boundary on first touch (price above zone),
              else midpoint if price inside.
      - SHORT: use bearish FVG, prefer high boundary on first touch (price below zone),
               else midpoint if price inside.

    Reject if too far (dist > 1.5 ATR) when ATR is available.
    """
    zones = struct.get("fvg_zones") or []
    if not zones:
        return None, "no_fvg"

    bias = (bias or "").upper()
    best_price = None
    best_dist = 1e18
    best_note = "no_fvg_candidate"

    for z in zones:
        try:
            ztype = str(z.get("type") or z.get("direction") or "").lower()

            # bounds
            low_b = z.get("low")
            high_b = z.get("high")
            if low_b is None or high_b is None:
                s0 = z.get("start")
                e0 = z.get("end")
                if s0 is None or e0 is None:
                    continue
                low_b = float(min(float(s0), float(e0)))
                high_b = float(max(float(s0), float(e0)))

            low_b = float(low_b)
            high_b = float(high_b)
            if high_b <= low_b:
                continue

            # directional filter
            if bias == "LONG" and "bull" not in ztype:
                continue
            if bias == "SHORT" and "bear" not in ztype:
                continue

            in_zone = low_b <= entry_mkt <= high_b

            if bias == "LONG":
                cand = float((low_b + high_b) / 2.0) if in_zone else float(low_b)
            else:
                cand = float((low_b + high_b) / 2.0) if in_zone else float(high_b)

            dist = abs(entry_mkt - cand)

            if dist < best_dist:
                best_dist = dist
                best_price = cand
                best_note = f"fvg_ok z=[{low_b:.6g},{high_b:.6g}] in_zone={in_zone} dist={dist:.6g} atr={atr:.6g}"

        except Exception:
            continue

    if best_price is None:
        return None, "no_fvg_parse"

    if atr > 0 and best_dist > 1.5 * atr:
        return None, f"fvg_too_far dist={best_dist:.6g} atr={atr:.6g}"

    return float(best_price), best_note



def _pick_entry(df_h1: pd.DataFrame, struct: Dict[str, Any], bias: str) -> Dict[str, Any]:
    """
    Desk entry selection.

    Priority:
      1) OTE (LIMIT) if in zone OR close to zone (distance <= OTE_NEAR_ATR_MULT * ATR)
      2) FVG (LIMIT) if valid & close
      3) EMA pullback (LIMIT) if close (distance <= EMA_PB_ATR_MULT * ATR)
      4) MARKET fallback only if ALLOW_MARKET_FALLBACK=1

    Returns:
      entry_used, entry_type, order_type, in_zone, note, entry_mkt, atr
    """
    import os

    entry_mkt = float(df_h1["close"].iloc[-1])
    atr = _atr(df_h1, 14)
    bias = (bias or "").upper()

    # OTE
    try:
        ote = compute_ote(df_h1, bias=bias, lookback=50) or {}
    except Exception:
        ote = {}
    in_ote = bool(ote.get("in_ote", False))
    ote_low = ote.get("ote_low")
    ote_high = ote.get("ote_high")

    OTE_NEAR_ATR_MULT = float(os.getenv("OTE_NEAR_ATR_MULT", "0.35"))

    if ote_low is not None and ote_high is not None:
        ote_low = float(ote_low)
        ote_high = float(ote_high)

        if in_ote:
            return {
                "entry_used": entry_mkt,
                "entry_type": "OTE",
                "order_type": "LIMIT",
                "in_zone": True,
                "note": f"ote in_ote=True zone=[{ote_low:.6g},{ote_high:.6g}] atr={atr:.6g}",
                "entry_mkt": entry_mkt,
                "atr": atr,
                "ote_low": ote_low,
                "ote_high": ote_high,
            }

        if atr > 0:
            if bias == "LONG" and entry_mkt > ote_high:
                dist = entry_mkt - ote_high
                if dist <= OTE_NEAR_ATR_MULT * atr:
                    return {
                        "entry_used": float(ote_high),
                        "entry_type": "OTE",
                        "order_type": "LIMIT",
                        "in_zone": False,
                        "note": f"ote near dist={dist:.6g} mult={OTE_NEAR_ATR_MULT} zone=[{ote_low:.6g},{ote_high:.6g}] atr={atr:.6g}",
                        "entry_mkt": entry_mkt,
                        "atr": atr,
                        "ote_low": ote_low,
                        "ote_high": ote_high,
                    }

            if bias == "SHORT" and entry_mkt < ote_low:
                dist = ote_low - entry_mkt
                if dist <= OTE_NEAR_ATR_MULT * atr:
                    return {
                        "entry_used": float(ote_low),
                        "entry_type": "OTE",
                        "order_type": "LIMIT",
                        "in_zone": False,
                        "note": f"ote near dist={dist:.6g} mult={OTE_NEAR_ATR_MULT} zone=[{ote_low:.6g},{ote_high:.6g}] atr={atr:.6g}",
                        "entry_mkt": entry_mkt,
                        "atr": atr,
                        "ote_low": ote_low,
                        "ote_high": ote_high,
                    }

    # FVG
    fvg_entry, fvg_note = _pick_fvg_entry(struct, entry_mkt, bias, atr)
    if fvg_entry is not None:
        return {
            "entry_used": float(fvg_entry),
            "entry_type": "FVG",
            "order_type": "LIMIT",
            "in_zone": False,
            "note": fvg_note,
            "entry_mkt": entry_mkt,
            "atr": atr,
            "ote_low": ote_low,
            "ote_high": ote_high,
        }

    # EMA pullback
    try:
        close = df_h1["close"].astype(float)
        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])

        EMA_PB_ATR_MULT = float(os.getenv("EMA_PB_ATR_MULT", "0.50"))

        if bias == "LONG":
            pb = min(ema20, ema50)
            dist = abs(entry_mkt - pb)
            if atr > 0 and dist <= EMA_PB_ATR_MULT * atr and pb < entry_mkt:
                return {
                    "entry_used": float(pb),
                    "entry_type": "EMA_PB",
                    "order_type": "LIMIT",
                    "in_zone": False,
                    "note": f"ema_pb long pb={pb:.6g} dist={dist:.6g} atr={atr:.6g}",
                    "entry_mkt": entry_mkt,
                    "atr": atr,
                    "ote_low": ote_low,
                    "ote_high": ote_high,
                }
        else:
            pb = max(ema20, ema50)
            dist = abs(entry_mkt - pb)
            if atr > 0 and dist <= EMA_PB_ATR_MULT * atr and pb > entry_mkt:
                return {
                    "entry_used": float(pb),
                    "entry_type": "EMA_PB",
                    "order_type": "LIMIT",
                    "in_zone": False,
                    "note": f"ema_pb short pb={pb:.6g} dist={dist:.6g} atr={atr:.6g}",
                    "entry_mkt": entry_mkt,
                    "atr": atr,
                    "ote_low": ote_low,
                    "ote_high": ote_high,
                }
    except Exception:
        pass

    allow_market = os.getenv("ALLOW_MARKET_FALLBACK", "1").strip().lower() in ("1", "true", "yes", "on")
    if allow_market:
        return {
            "entry_used": float(entry_mkt),
            "entry_type": "MARKET",
            "order_type": "LIMIT",
            "in_zone": False,
            "note": "fallback_market",
            "entry_mkt": entry_mkt,
            "atr": atr,
            "ote_low": ote_low,
            "ote_high": ote_high,
        }

    return {
        "entry_used": float(entry_mkt),
        "entry_type": "NO_ENTRY",
        "order_type": "NONE",
        "in_zone": False,
        "note": "no_zone_no_market_fallback",
        "entry_mkt": entry_mkt,
        "atr": atr,
        "ote_low": ote_low,
        "ote_high": ote_high,
    }




# =====================================================================
# Signal Analyzer (Desk)
# =====================================================================

class SignalAnalyzer:
    """
    Chemins :

    1) BOS_STRICT (setup principal)
    2) INST_CONTINUATION (optionnel si DESK_EV_MODE=True)

    Ajout desk :
    - Entrées OTE/FVG = LIMIT uniquement (si OTE/FVG choisi)
    - Momentum composite un peu moins strict pour la continuation (>=65)
    """

    def __init__(self, *args, **kwargs):
        pass

    async def analyze(
        self,
        symbol: str,
        df_h1: pd.DataFrame,
        df_h4: pd.DataFrame,
        macro: Any = None,
    ) -> Dict[str, Any]:

        LOGGER.info(f"[EVAL] ▶ START {symbol}")

        # ------------------------------------------------------------------
        # 1 — STRUCTURE H1
        # ------------------------------------------------------------------
        struct = analyze_structure(df_h1)
        bias = str(struct.get("trend", "")).upper()

        # Data sanity (desk): avoid dead markets / broken OHLCV
        try:
            sub = df_h1.tail(40)
            rng_med = float((sub["high"].astype(float) - sub["low"].astype(float)).median())
            vol_med = float(sub["volume"].astype(float).median())
            if (not pd.isfinite(rng_med)) or rng_med <= 0 or (not pd.isfinite(vol_med)) or vol_med <= 0:
                LOGGER.info("[EVAL_REJECT] bad_ohlcv_data rng_med=%s vol_med=%s", rng_med, vol_med)
                return {"valid": False, "reject_reason": "bad_ohlcv_data", "structure": struct}
        except Exception:
            pass
        LOGGER.info(f"[EVAL_PRE] STRUCT={struct}")

        oi_series = struct.get("oi_series", None)
        LOGGER.info(f"[EVAL_PRE] OI_STRUCT symbol={symbol} has_oi={oi_series is not None}")

        if REQUIRE_STRUCTURE:
            if bias not in ("LONG", "SHORT"):
                LOGGER.info("[EVAL_REJECT] no_clear_trend_range")
                return {"valid": False, "reject_reason": "no_clear_trend_range", "structure": struct}
        else:
            if bias not in ("LONG", "SHORT"):
                LOGGER.info("[EVAL_PRE] Trend RANGE but REQUIRE_STRUCTURE=False")

        # ------------------------------------------------------------------
        # 2 — H4 ALIGNMENT
        # ------------------------------------------------------------------
        if REQUIRE_HTF_ALIGN:
            if not htf_trend_ok(df_h4, bias):
                LOGGER.info("[EVAL_REJECT] htf_veto")
                return {"valid": False, "reject_reason": "htf_veto", "structure": struct}

        # ------------------------------------------------------------------
        # 3 — BOS / BOS_QUALITY
        # ------------------------------------------------------------------
        entry_mkt = float(df_h1["close"].iloc[-1])

        bos_flag = bool(struct.get("bos", False))
        bos_dir = struct.get("bos_direction", None)
        bos_type = struct.get("bos_type", None)

        bos_q = bos_quality_details(
            df_h1,
            oi_series=oi_series,
            df_liq=df_h1,
            price=entry_mkt,
            direction=bos_dir,
        )
        bos_quality_ok = bool(bos_q.get("ok", True))

        LOGGER.info(f"[EVAL_PRE] BOS_QUALITY={bos_q} bos_flag={bos_flag} bos_type={bos_type}")

        # ------------------------------------------------------------------
        # 4 — INSTITUTIONNEL (Binance)
        # ------------------------------------------------------------------
        inst = await compute_full_institutional_analysis(symbol, bias)

        # If structure did not provide an OI series, reuse institutional OI slope to enrich BOS quality scoring.
        try:
            if oi_series is None and isinstance(inst, dict) and inst.get("available", False):
                bos_q = bos_quality_details(
                    df_h1,
                    oi_series=None,
                    df_liq=df_h1,
                    price=entry_mkt,
                    direction=bos_dir,
                    oi_slope_override=inst.get("oi_slope"),
                )
                bos_quality_ok = bool(bos_q.get("ok", True))
                LOGGER.info(f"[EVAL_PRE] BOS_QUALITY(inst_oi_override)={bos_q} bos_flag={bos_flag} bos_type={bos_type}")
        except Exception:
            pass
        inst_score = int(inst.get("institutional_score") or 0)

        binance_symbol = inst.get("binance_symbol")
        available = bool(inst.get("available", False))

        # Bypass si non mappé Binance (desk : on autorise seulement BOS_STRICT A+)
        bypass_inst = (not available) or (binance_symbol is None)

        LOGGER.info(f"[INST_RAW] score={inst_score} details={inst}")

        if not bypass_inst and inst_score < MIN_INST_SCORE:
            LOGGER.info(f"[EVAL_REJECT] inst_score_low ({inst_score} < {MIN_INST_SCORE})")
            return {"valid": False, "reject_reason": "inst_score_low", "institutional": inst, "structure": struct}

        if bypass_inst:
            LOGGER.warning(f"[EVAL_WARN] bypass MIN_INST_SCORE (binance unmapped) symbol={symbol}")

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
        LOGGER.info(f"[EVAL_PRE] MOMENTUM_COMPOSITE score={comp_score} label={comp_label} components={comp.get('components') if isinstance(comp, dict) else {}}")
        LOGGER.info(f"[EVAL_PRE] VOL_REGIME={vol_regime} EXTENSION={ext_sig}")

        if REQUIRE_MOMENTUM:
            if bias == "LONG" and mom not in ("BULLISH", "STRONG_BULLISH"):
                LOGGER.info("[EVAL_REJECT] momentum_not_bullish")
                return {"valid": False, "reject_reason": "momentum_not_bullish", "institutional": inst, "structure": struct}
            if bias == "SHORT" and mom not in ("BEARISH", "STRONG_BEARISH"):
                LOGGER.info("[EVAL_REJECT] momentum_not_bearish")
                return {"valid": False, "reject_reason": "momentum_not_bearish", "institutional": inst, "structure": struct}

        if ext_sig == "OVEREXTENDED_LONG" and bias == "LONG":
            LOGGER.info("[EVAL_REJECT] overextended_long")
            return {"valid": False, "reject_reason": "overextended_long", "institutional": inst, "structure": struct}
        if ext_sig == "OVEREXTENDED_SHORT" and bias == "SHORT":
            LOGGER.info("[EVAL_REJECT] overextended_short")
            return {"valid": False, "reject_reason": "overextended_short", "institutional": inst, "structure": struct}

        # ------------------------------------------------------------------
        # 6 — PREMIUM / DISCOUNT (info)
        # ------------------------------------------------------------------
        discount, premium = compute_premium_discount(df_h1)
        LOGGER.info(f"[EVAL_PRE] PREMIUM={premium} DISCOUNT={discount}")

        # ------------------------------------------------------------------
        # 7 — ENTRY PICK (OTE/FVG/MARKET)
        # ------------------------------------------------------------------
        entry_pick = _pick_entry(df_h1, struct, bias)
        entry = float(entry_pick["entry_used"])
        entry_type = str(entry_pick["entry_type"])
        note = str(entry_pick.get("note"))

        LOGGER.info(
            f"[EVAL_PRE] ENTRY_PICK type={entry_type} entry_mkt={entry_pick.get('entry_mkt')} "
            f"entry_used={entry} in_zone={entry_pick.get('in_zone')} note={note} atr={entry_pick.get('atr')}"
        )

        # Desk rule : si entry_type = OTE/FVG, on refuse si le setup n'est PAS "zone compatible".
        # Ici, on autorise OTE/FVG uniquement sur BOS_STRICT (retest) ou continuation,
        # MAIS on ne force pas : c'est déjà choisi uniquement si exploitable.
        # Donc pas besoin de reject ici.

        # ------------------------------------------------------------------
        # 8 — SL / TP1 / RR (TP2 supprimé)
        # ------------------------------------------------------------------
        tick = estimate_tick_from_price(entry)
        exits = _compute_exits(df_h1, entry, bias, tick=tick)
        sl = float(exits["sl"])
        tp1 = float(exits["tp1"])
        rr = _safe_rr(entry, sl, tp1, bias)

        LOGGER.info(
            f"[EVAL_PRE] EXITS entry={entry} sl={sl} tp1={tp1} tick={tick} RR={rr} raw_rr={exits['rr_used']} entry_type={entry_type}"
        )

        if rr is None or rr <= 0:
            LOGGER.info("[EVAL_REJECT] rr_invalid")
            return {"valid": False, "reject_reason": "rr_invalid", "institutional": inst, "structure": struct}

        
        # ------------------------------------------------------------------
        # Desk setup scoring (filter noise without paid data)
        # ------------------------------------------------------------------
        setup_score = None
        try:
            import os
            bos_score = float(bos_q.get("score", 0.0)) if isinstance(bos_q, dict) else 0.0
            inst_s = float(inst_score)
            comp_s = float(comp_score)

            # premium/discount alignment (cheap proxy)
            pd_bonus = 0.5 if ((premium and bias == "SHORT") or (discount and bias == "LONG")) else 0.0

            # entry quality
            if entry_type in ("OTE", "FVG"):
                entry_bonus = 1.0
            elif entry_type == "EMA_PB":
                entry_bonus = 0.5
            elif entry_type == "MARKET":
                entry_bonus = -0.5
            else:
                entry_bonus = 0.0

            mom_bonus = 0.5 if comp_s >= 70.0 else 0.0

            setup_score = inst_s + bos_score + entry_bonus + pd_bonus + mom_bonus

            min_setup_score = float(os.getenv("MIN_SETUP_SCORE", "2.0"))
            if setup_score < min_setup_score:
                LOGGER.info(
                    "[EVAL_REJECT] setup_score_low score=%s < %s (inst=%s bos=%s entry=%s pd=%s mom=%s)",
                    setup_score, min_setup_score, inst_s, bos_score, entry_bonus, pd_bonus, mom_bonus,
                )
                return {
                    "valid": False,
                    "reject_reason": "setup_score_low",
                    "setup_score": setup_score,
                    "institutional": inst,
                    "structure": struct,
                    "bos_quality": bos_q,
                }
        except Exception:
            setup_score = None

# ------------------------------------------------------------------
        # 9 — PATH 1 : BOS_STRICT
        # ------------------------------------------------------------------
        bos_strict_ok = False
        if bos_flag:
            if (not REQUIRE_BOS_QUALITY) or bos_quality_ok:
                # si bypass inst, on ne valide que du BOS_STRICT (A+) avec RR strict
                if bypass_inst:
                    if rr >= RR_MIN_STRICT:
                        bos_strict_ok = True
                else:
                    if rr >= RR_MIN_STRICT and inst_score >= MIN_INST_SCORE:
                        bos_strict_ok = True

        LOGGER.info(
            "[EVAL_PRE] BOS_STRICT_CHECK "
            f"bos_flag={bos_flag}, bos_quality_ok={bos_quality_ok}, rr={rr} vs RR_MIN_STRICT={RR_MIN_STRICT}, "
            f"inst_score={inst_score} vs MIN_INST_SCORE={MIN_INST_SCORE}, bypass_inst={bypass_inst}, bos_strict_ok={bos_strict_ok}"
        )

        if bos_strict_ok:
            LOGGER.info(f"[EVAL] VALID {symbol} RR={rr} SETUP=BOS_STRICT (DESK_EV_MODE={DESK_EV_MODE})")
            return {
                "valid": True,
                "symbol": symbol,
                "side": "BUY" if bias == "LONG" else "SELL",
                "bias": bias,
                "entry": entry,
                "entry_type": entry_type,
                "sl": sl,
                "tp1": tp1,
                "tp2": None,
                "rr": rr,
                "qty": 1,
                "setup_type": "BOS_STRICT",
                "setup_score": setup_score,
                "structure": struct,
                "bos_quality": bos_q,
                "institutional": inst,
                "momentum": mom,
                "composite": comp,
                "premium": premium,
                "discount": discount,
            }

        # ------------------------------------------------------------------
        # 10 — PATH 2 : INST_CONTINUATION (si DESK_EV_MODE=True)
        # ------------------------------------------------------------------
        inst_continuation_ok = False

        if DESK_EV_MODE and (not bypass_inst):
            good_inst = inst_score >= INST_SCORE_DESK_PRIORITY
            good_rr = rr >= RR_MIN_DESK_PRIORITY

            # desk : un peu moins strict que 70
            comp_thr = 65.0

            if bias == "LONG":
                good_comp = comp_score >= comp_thr and comp_label in ("BULLISH", "SLIGHT_BULLISH")
            else:
                good_comp = comp_score >= comp_thr and comp_label in ("BEARISH", "SLIGHT_BEARISH")

            inst_continuation_ok = bool(good_inst and good_rr and good_comp)

            LOGGER.info(
                "[EVAL_PRE] INST_CONTINUATION_CHECK "
                f"inst_score={inst_score} vs INST_SCORE_DESK_PRIORITY={INST_SCORE_DESK_PRIORITY}, "
                f"rr={rr} vs RR_MIN_DESK_PRIORITY={RR_MIN_DESK_PRIORITY}, "
                f"comp_score={comp_score}, comp_label={comp_label}, comp_thr={comp_thr}, "
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
                "entry_type": entry_type,
                "sl": sl,
                "tp1": tp1,
                "tp2": None,
                "rr": rr,
                "qty": 1,
                "setup_type": "INST_CONTINUATION",
                "setup_score": setup_score,
                "structure": struct,
                "bos_quality": bos_q,
                "institutional": inst,
                "momentum": mom,
                "composite": comp,
                "premium": premium,
                "discount": discount,
            }

        # ------------------------------------------------------------------
        # 11 — Aucun setup validé
        # ------------------------------------------------------------------
        LOGGER.info(
            "[EVAL_REJECT] no_setup_validated "
            f"(BOS_STRICT_OK={bos_strict_ok}, INST_CONTINUATION_OK={inst_continuation_ok}, DESK_EV_MODE={DESK_EV_MODE})"
        )
        return {
            "valid": False,
            "reject_reason": "no_setup_validated",
            "structure": struct,
            "institutional": inst,
            "bos_quality": bos_q,
            "entry_pick": entry_pick,
            "rr": rr,
        }
