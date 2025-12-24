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
    Retourne (entry_price, note). None si pas exploitable.
    On prend une FVG proche (<= 1.5 ATR) et alignée directionnellement.
    """
    zones = struct.get("fvg_zones") or []
    if not zones:
        return None, "no_fvg"

    best = None
    best_dist = 1e18

    for z in zones:
        try:
            zl = float(z.get("low") if z.get("low") is not None else z.get("bottom"))
            zh = float(z.get("high") if z.get("high") is not None else z.get("top"))
            if zh <= zl:
                continue

            # direction (si présent)
            zdir = (z.get("direction") or z.get("dir") or "").lower()
            if bias == "LONG" and zdir and "bear" in zdir:
                continue
            if bias == "SHORT" and zdir and "bull" in zdir:
                continue

            mid = (zl + zh) / 2.0
            dist = abs(entry_mkt - mid)

            # on accepte si proche d'une zone (ou dedans)
            in_zone = zl <= entry_mkt <= zh
            if in_zone:
                dist = 0.0

            if dist < best_dist:
                best_dist = dist
                best = mid

        except Exception:
            continue

    if best is None:
        return None, "no_fvg_parse"

    if atr > 0 and best_dist > 1.5 * atr:
        return None, f"fvg_too_far dist={best_dist:.6g} atr={atr:.6g}"

    return float(best), f"fvg_ok dist={best_dist:.6g} atr={atr:.6g}"


def _pick_entry(df_h1: pd.DataFrame, struct: Dict[str, Any], bias: str) -> Dict[str, Any]:
    """
    Logique desk :
    - Si OTE valide (in_zone=True) => entry_type=OTE (LIMIT)
    - Sinon si FVG exploitable proche => entry_type=FVG (LIMIT)
    - Sinon => entry_type=MARKET (limit au prix spot, execution "agressive")
    """
    entry_mkt = float(df_h1["close"].iloc[-1])
    atr = _atr(df_h1, 14)

    # 1) OTE
    ote_entry = None
    ote_in_zone = False
    ote_note = "ote_unavailable"
    try:
        ote = compute_ote(df_h1, bias)
        if isinstance(ote, dict):
            ote_entry = ote.get("entry") or ote.get("entry_price") or ote.get("price")
            ote_in_zone = bool(ote.get("in_zone") or ote.get("ok") or False)
            dist = ote.get("dist") or ote.get("distance")
            ote_note = f"ote in_zone={ote_in_zone} dist={dist} atr={atr:.6g}"
        elif isinstance(ote, (tuple, list)) and len(ote) >= 2:
            # fallback (entry, in_zone)
            ote_entry = ote[0]
            ote_in_zone = bool(ote[1])
            ote_note = f"ote tuple in_zone={ote_in_zone} atr={atr:.6g}"
    except Exception as e:
        ote_note = f"ote_error {e}"

    if ote_entry is not None and ote_in_zone:
        return {
            "entry_used": float(ote_entry),
            "entry_type": "OTE",
            "order_type": "LIMIT",
            "in_zone": True,
            "note": ote_note,
            "entry_mkt": entry_mkt,
            "atr": atr,
        }

    # 2) FVG
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
        }

    # 3) MARKET fallback (limit au prix spot)
    return {
        "entry_used": entry_mkt,
        "entry_type": "MARKET",
        "order_type": "LIMIT",
        "in_zone": False,
        "note": "no_zone_entry",
        "entry_mkt": entry_mkt,
        "atr": atr,
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
