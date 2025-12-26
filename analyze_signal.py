from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
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
    true_atr,
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
    RR_MIN_TOLERATED_WITH_INST,  # <— déjà dans settings.py
)

LOGGER = logging.getLogger(__name__)


# =====================================================================
# Helpers (desk-grade)
# =====================================================================

REQUIRED_COLS = ("open", "high", "low", "close", "volume")


def _ensure_ohlcv(df: pd.DataFrame) -> bool:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False
    for c in REQUIRED_COLS:
        if c not in df.columns:
            return False
    return True


def _last_close(df: pd.DataFrame) -> float:
    try:
        return float(df["close"].astype(float).iloc[-1])
    except Exception:
        return 0.0


def _atr(df: pd.DataFrame, n: int = 14) -> float:
    """
    ATR cohérent avec le reste du code (stops.py utilise true_atr).
    """
    try:
        if not _ensure_ohlcv(df) or len(df) < n + 2:
            return 0.0
        s = true_atr(df, length=n)
        v = float(s.iloc[-1]) if s is not None and len(s) else 0.0
        return float(max(0.0, v)) if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


def compute_premium_discount(df: pd.DataFrame, lookback: int = 80) -> Tuple[bool, bool]:
    """
    Retourne (in_discount, in_premium) basé sur mid-range du lookback.
    """
    if not _ensure_ohlcv(df) or len(df) < lookback:
        return False, False

    w = df.tail(lookback)
    hi = float(w["high"].astype(float).max())
    lo = float(w["low"].astype(float).min())
    last = float(w["close"].astype(float).iloc[-1])

    if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
        return False, False

    mid = (hi + lo) / 2.0
    return (last < mid), (last > mid)


def _safe_rr(entry: float, sl: float, tp1: float, bias: str) -> Optional[float]:
    try:
        entry = float(entry)
        sl = float(sl)
        tp1 = float(tp1)
        if not (np.isfinite(entry) and np.isfinite(sl) and np.isfinite(tp1)):
            return None

        bias = (bias or "").upper()
        if bias == "LONG":
            risk = entry - sl
            reward = tp1 - entry
        elif bias == "SHORT":
            risk = sl - entry
            reward = entry - tp1
        else:
            return None

        if risk <= 0:
            return None
        return float(reward / risk)
    except Exception:
        return None


def estimate_tick_from_price(price: float) -> float:
    """
    Heuristique (OK pour logs / rounding soft). Si tu as la vraie tickSize exchange,
    utilise-la plutôt côté trader.
    """
    p = abs(float(price))
    if p >= 10000:
        return 1.0
    if p >= 1000:
        return 0.1
    if p >= 100:
        return 0.01
    if p >= 10:
        return 0.001
    if p >= 1:
        return 0.0001
    if p >= 0.1:
        return 0.00001
    if p >= 0.01:
        return 0.000001
    return 0.0000001


def _compute_exits(df: pd.DataFrame, entry: float, bias: str, tick: float) -> Dict[str, Any]:
    if bias == "LONG":
        sl, meta = protective_stop_long(df, entry, tick, return_meta=True)
    else:
        sl, meta = protective_stop_short(df, entry, tick, return_meta=True)
    tp1, rr_used = compute_tp1(entry, sl, bias, df=df, tick=tick)
    return {"sl": float(sl), "tp1": float(tp1), "rr_used": float(rr_used), "sl_meta": meta}


def _entry_pullback_ok(entry: float, entry_mkt: float, bias: str, atr: float) -> bool:
    """
    Desk sanity: si un signal est "overextended", on accepte encore une LIMIT
    seulement si l'entry implique un vrai pullback (>= ~0.25 ATR).
    """
    try:
        if atr <= 0:
            return True
        entry = float(entry)
        entry_mkt = float(entry_mkt)
        bias = (bias or "").upper()

        if bias == "LONG":
            # pullback = entry < mkt
            return (entry_mkt - entry) >= 0.25 * atr
        if bias == "SHORT":
            return (entry - entry_mkt) >= 0.25 * atr
        return True
    except Exception:
        return True


def _parse_zone_bounds(z: Dict[str, Any]) -> Optional[Tuple[float, float, str]]:
    """
    Supporte plusieurs formats:
      - {low/high} / {bottom/top}
      - {start/end} (structure_utils)
    Retourne (zl, zh, zdir_str)
    """
    try:
        a = z.get("low")
        b = z.get("high")
        if a is None or b is None:
            a = z.get("bottom")
            b = z.get("top")
        if a is None or b is None:
            a = z.get("start")
            b = z.get("end")
        if a is None or b is None:
            return None

        zl = float(min(a, b))
        zh = float(max(a, b))
        if not (np.isfinite(zl) and np.isfinite(zh)) or zh <= zl:
            return None

        zdir = (z.get("direction") or z.get("dir") or z.get("type") or "").lower()
        return zl, zh, zdir
    except Exception:
        return None


def _pick_fvg_entry(struct: Dict[str, Any], entry_mkt: float, bias: str, atr: float) -> Tuple[Optional[float], str]:
    """
    Retourne (entry_price, note). None si pas exploitable.
    Desk : on prend une FVG alignée directionnellement et proche (<= 1.5 ATR).
    """
    zones = struct.get("fvg_zones") or []
    if not zones:
        return None, "no_fvg"

    best_mid = None
    best_dist = 1e18

    bias = (bias or "").upper()

    for z in zones:
        parsed = _parse_zone_bounds(z)
        if not parsed:
            continue
        zl, zh, zdir = parsed

        # Direction filter (si zdir informé)
        if bias == "LONG" and zdir and "bear" in zdir:
            continue
        if bias == "SHORT" and zdir and "bull" in zdir:
            continue

        mid = (zl + zh) / 2.0
        dist = abs(float(entry_mkt) - mid)
        if zl <= float(entry_mkt) <= zh:
            dist = 0.0

        if dist < best_dist:
            best_dist = dist
            best_mid = mid

    if best_mid is None:
        return None, "no_fvg_parse"

    if atr > 0 and best_dist > 1.5 * atr:
        return None, f"fvg_too_far dist={best_dist:.6g} atr={atr:.6g}"

    return float(best_mid), f"fvg_ok dist={best_dist:.6g} atr={atr:.6g}"


def _pick_entry(df_h1: pd.DataFrame, struct: Dict[str, Any], bias: str) -> Dict[str, Any]:
    """
    Desk entry:
      1) OTE (si dans la zone) -> LIMIT
      2) FVG (si proche) -> LIMIT
      3) fallback -> MARKET-like (limit au spot)
    """
    entry_mkt = _last_close(df_h1)
    atr = _atr(df_h1, 14)

    # 1) OTE (compat indicators.compute_ote: in_ote/ote_low/ote_high)
    ote_entry = None
    ote_in_zone = False
    ote_note = "ote_unavailable"
    try:
        ote = compute_ote(df_h1, bias)

        # Format actuel indicators.py
        if isinstance(ote, dict) and ("in_ote" in ote or "ote_low" in ote or "ote_high" in ote):
            in_ote = bool(ote.get("in_ote", False))
            lo = ote.get("ote_low")
            hi = ote.get("ote_high")
            if lo is not None and hi is not None:
                lo = float(lo)
                hi = float(hi)
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    ote_entry = (lo + hi) / 2.0
                    ote_in_zone = in_ote
                    dist = abs(entry_mkt - ote_entry)
                    ote_note = f"ote(in_ote={ote_in_zone}) low={lo} high={hi} dist={dist:.6g} atr={atr:.6g}"
                else:
                    ote_note = f"ote_bad_bounds low={lo} high={hi} atr={atr:.6g}"
            else:
                ote_note = f"ote_missing_bounds atr={atr:.6g}"

        # Ancien format (si jamais)
        elif isinstance(ote, dict):
            ote_entry = ote.get("entry") or ote.get("entry_price") or ote.get("price")
            ote_in_zone = bool(ote.get("in_zone") or ote.get("ok") or False)
            dist = ote.get("dist") or ote.get("distance")
            ote_note = f"ote(oldfmt in_zone={ote_in_zone}) dist={dist} atr={atr:.6g}"

        elif isinstance(ote, (tuple, list)) and len(ote) >= 2:
            ote_entry = float(ote[0])
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

    # 3) MARKET fallback (limit au spot)
    return {
        "entry_used": float(entry_mkt),
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
    Desk logic (2 paths):

      1) BOS_STRICT  : A+ structure, RR >= RR_MIN_STRICT (ou relax si inst OK)
      2) INST_CONTINUATION (si DESK_EV_MODE=True)
         - inst_score >= INST_SCORE_DESK_PRIORITY
         - RR >= RR_MIN_DESK_PRIORITY
         - composite momentum >= 65 et aligné

    Améliorations desk lead:
      - Fix compat OTE (compute_ote) + FVG (start/end/type)
      - Ne tue pas un LIMIT retest sur "overextended" si l'entry est un vrai pullback
      - Early reject "no_bos" si DESK_EV_MODE off (évite appels institutionnels inutiles)
      - RR relax avec inst: RR >= RR_MIN_TOLERATED_WITH_INST si inst_score OK
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
        # 0 — Sanity
        # ------------------------------------------------------------------
        if not _ensure_ohlcv(df_h1) or len(df_h1) < 60:
            LOGGER.info("[EVAL_REJECT] bad_df_h1")
            return {"valid": False, "reject_reason": "bad_df_h1"}

        if not _ensure_ohlcv(df_h4) or len(df_h4) < 60:
            LOGGER.info("[EVAL_REJECT] bad_df_h4")
            return {"valid": False, "reject_reason": "bad_df_h4"}

        # ------------------------------------------------------------------
        # 1 — STRUCTURE H1
        # ------------------------------------------------------------------
        struct = analyze_structure(df_h1)
        bias = str(struct.get("trend", "")).upper()
        LOGGER.info(f"[EVAL_PRE] STRUCT trend={bias} bos={struct.get('bos')} choch={struct.get('choch')} cos={struct.get('cos')}")

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
        if REQUIRE_HTF_ALIGN and bias in ("LONG", "SHORT"):
            if not htf_trend_ok(df_h4, bias):
                LOGGER.info("[EVAL_REJECT] htf_veto")
                return {"valid": False, "reject_reason": "htf_veto", "structure": struct}

        # ------------------------------------------------------------------
        # 3 — BOS / BOS_QUALITY
        # ------------------------------------------------------------------
        entry_mkt = _last_close(df_h1)

        bos_flag = bool(struct.get("bos", False))
        bos_dir = struct.get("bos_direction", None)
        bos_type = struct.get("bos_type", None)

        # Desk: si DESK_EV_MODE off et pas de BOS => on coupe tôt (sinon tu brûles l’API inst pour rien)
        if (not DESK_EV_MODE) and (not bos_flag):
            LOGGER.info("[EVAL_REJECT] no_bos")
            return {"valid": False, "reject_reason": "no_bos", "structure": struct}

        bos_q = bos_quality_details(
            df_h1,
            oi_series=oi_series,
            df_liq=df_h1,
            price=entry_mkt,
            direction=bos_dir,
        )
        bos_quality_ok = bool(bos_q.get("ok", True))
        LOGGER.info(f"[EVAL_PRE] BOS_QUALITY ok={bos_quality_ok} score={bos_q.get('score')} reasons={bos_q.get('reasons')} bos_flag={bos_flag} bos_type={bos_type} bos_dir={bos_dir}")

        # ------------------------------------------------------------------
        # 4 — MOMENTUM / VOL / EXTENSION
        # ------------------------------------------------------------------
        mom = institutional_momentum(df_h1)
        comp = composite_momentum(df_h1)
        vol_regime = volatility_regime(df_h1)
        ext_sig = extension_signal(df_h1)

        comp_score = float(comp.get("score", 50.0)) if isinstance(comp, dict) else 50.0
        comp_label = str(comp.get("label", "NEUTRAL")) if isinstance(comp, dict) else "NEUTRAL"

        LOGGER.info(f"[EVAL_PRE] MOMENTUM={mom}")
        LOGGER.info(f"[EVAL_PRE] MOMENTUM_COMPOSITE score={comp_score:.2f} label={comp_label} components={(comp.get('components') if isinstance(comp, dict) else {})}")
        LOGGER.info(f"[EVAL_PRE] VOL_REGIME={vol_regime} EXTENSION={ext_sig}")

        if REQUIRE_MOMENTUM and bias in ("LONG", "SHORT"):
            if bias == "LONG" and mom not in ("BULLISH", "STRONG_BULLISH"):
                LOGGER.info("[EVAL_REJECT] momentum_not_bullish")
                return {"valid": False, "reject_reason": "momentum_not_bullish", "structure": struct}
            if bias == "SHORT" and mom not in ("BEARISH", "STRONG_BEARISH"):
                LOGGER.info("[EVAL_REJECT] momentum_not_bearish")
                return {"valid": False, "reject_reason": "momentum_not_bearish", "structure": struct}

        # ------------------------------------------------------------------
        # 5 — PREMIUM / DISCOUNT (info + micro filter)
        # ------------------------------------------------------------------
        discount, premium = compute_premium_discount(df_h1)
        LOGGER.info(f"[EVAL_PRE] PREMIUM={premium} DISCOUNT={discount}")

        # ------------------------------------------------------------------
        # 6 — ENTRY PICK (OTE/FVG/MARKET)
        # ------------------------------------------------------------------
        entry_pick = _pick_entry(df_h1, struct, bias)
        entry = float(entry_pick["entry_used"])
        entry_type = str(entry_pick["entry_type"])
        note = str(entry_pick.get("note"))
        atr14 = float(entry_pick.get("atr") or _atr(df_h1, 14))

        LOGGER.info(
            f"[EVAL_PRE] ENTRY_PICK type={entry_type} entry_mkt={entry_pick.get('entry_mkt')} "
            f"entry_used={entry} in_zone={entry_pick.get('in_zone')} note={note} atr={atr14:.6g}"
        )

        # Extension: desk nuance
        # - On rejette l'OVEREXTENDED dans le sens du trade uniquement si on est en MARKET-like
        # - Si LIMIT retest, on l’accepte seulement si c’est un vrai pullback
        if ext_sig == "OVEREXTENDED_LONG" and bias == "LONG":
            if entry_type == "MARKET":
                LOGGER.info("[EVAL_REJECT] overextended_long_market")
                return {"valid": False, "reject_reason": "overextended_long_market", "structure": struct, "entry_pick": entry_pick}
            if not _entry_pullback_ok(entry, entry_mkt, bias, atr14):
                LOGGER.info("[EVAL_REJECT] overextended_long_no_pullback")
                return {"valid": False, "reject_reason": "overextended_long_no_pullback", "structure": struct, "entry_pick": entry_pick}

        if ext_sig == "OVEREXTENDED_SHORT" and bias == "SHORT":
            if entry_type == "MARKET":
                LOGGER.info("[EVAL_REJECT] overextended_short_market")
                return {"valid": False, "reject_reason": "overextended_short_market", "structure": struct, "entry_pick": entry_pick}
            if not _entry_pullback_ok(entry, entry_mkt, bias, atr14):
                LOGGER.info("[EVAL_REJECT] overextended_short_no_pullback")
                return {"valid": False, "reject_reason": "overextended_short_no_pullback", "structure": struct, "entry_pick": entry_pick}

        # Premium/Discount: micro filter uniquement sur MARKET (desk: éviter de chase)
        if bias == "LONG" and premium and entry_type == "MARKET":
            LOGGER.info("[EVAL_REJECT] long_in_premium_market")
            return {"valid": False, "reject_reason": "long_in_premium_market", "structure": struct, "entry_pick": entry_pick}
        if bias == "SHORT" and discount and entry_type == "MARKET":
            LOGGER.info("[EVAL_REJECT] short_in_discount_market")
            return {"valid": False, "reject_reason": "short_in_discount_market", "structure": struct, "entry_pick": entry_pick}

        # ------------------------------------------------------------------
        # 7 — INSTITUTIONNEL (Binance) (après qu’on ait un setup viable)
        # ------------------------------------------------------------------
        inst = await compute_full_institutional_analysis(symbol, bias)
        inst_score = int(inst.get("institutional_score") or 0)
        binance_symbol = inst.get("binance_symbol")
        available = bool(inst.get("available", False))

        bypass_inst = (not available) or (binance_symbol is None)

        LOGGER.info(f"[INST_RAW] score={inst_score} available={available} binance_symbol={binance_symbol} details={inst}")

        if not bypass_inst and inst_score < MIN_INST_SCORE:
            LOGGER.info(f"[EVAL_REJECT] inst_score_low ({inst_score} < {MIN_INST_SCORE})")
            return {"valid": False, "reject_reason": "inst_score_low", "institutional": inst, "structure": struct, "entry_pick": entry_pick}

        if bypass_inst:
            LOGGER.warning(f"[EVAL_WARN] bypass MIN_INST_SCORE (binance unmapped) symbol={symbol}")

        # ------------------------------------------------------------------
        # 8 — SL / TP1 / RR
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
            return {"valid": False, "reject_reason": "rr_invalid", "institutional": inst, "structure": struct, "entry_pick": entry_pick}

        # ------------------------------------------------------------------
        # 9 — PATH 1 : BOS_STRICT (A+)
        # ------------------------------------------------------------------
        bos_strict_ok = False
        bos_strict_variant = "RR_STRICT"

        if bos_flag and ((not REQUIRE_BOS_QUALITY) or bos_quality_ok):
            # Case 1: bypass inst -> uniquement RR strict
            if bypass_inst:
                if rr >= RR_MIN_STRICT:
                    bos_strict_ok = True
                    bos_strict_variant = "BYPASS_RR_STRICT"
            else:
                # Case 2: inst OK + RR strict
                if rr >= RR_MIN_STRICT and inst_score >= MIN_INST_SCORE:
                    bos_strict_ok = True
                    bos_strict_variant = "RR_STRICT"
                # Case 3 (desk): RR relax si inst OK (style “desk EV filter”)
                elif rr >= RR_MIN_TOLERATED_WITH_INST and inst_score >= MIN_INST_SCORE:
                    bos_strict_ok = True
                    bos_strict_variant = "RR_RELAX_WITH_INST"

        LOGGER.info(
            "[EVAL_PRE] BOS_STRICT_CHECK "
            f"bos_flag={bos_flag}, bos_quality_ok={bos_quality_ok}, rr={rr:.3f} "
            f"RR_MIN_STRICT={RR_MIN_STRICT}, RR_MIN_TOLERATED_WITH_INST={RR_MIN_TOLERATED_WITH_INST}, "
            f"inst_score={inst_score}, MIN_INST_SCORE={MIN_INST_SCORE}, bypass_inst={bypass_inst}, "
            f"bos_strict_ok={bos_strict_ok} variant={bos_strict_variant}"
        )

        if bos_strict_ok:
            LOGGER.info(f"[EVAL] VALID {symbol} RR={rr:.3f} SETUP=BOS_STRICT ({bos_strict_variant}) (DESK_EV_MODE={DESK_EV_MODE})")
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
                "setup_variant": bos_strict_variant,
                "structure": struct,
                "bos_quality": bos_q,
                "institutional": inst,
                "momentum": mom,
                "composite": comp,
                "premium": premium,
                "discount": discount,
                "entry_pick": entry_pick,
                "sl_meta": exits.get("sl_meta"),
            }

        # ------------------------------------------------------------------
        # 10 — PATH 2 : INST_CONTINUATION (si DESK_EV_MODE=True)
        # ------------------------------------------------------------------
        inst_continuation_ok = False
        inst_continuation_reason = ""

        if DESK_EV_MODE and (not bypass_inst) and bias in ("LONG", "SHORT"):
            good_inst = inst_score >= INST_SCORE_DESK_PRIORITY
            good_rr = rr >= RR_MIN_DESK_PRIORITY

            comp_thr = 65.0  # desk: moins strict
            if bias == "LONG":
                good_comp = (comp_score >= comp_thr) and (comp_label in ("STRONG_BULLISH", "BULLISH", "SLIGHT_BULLISH"))
            else:
                good_comp = (comp_score >= comp_thr) and (comp_label in ("STRONG_BEARISH", "BEARISH", "SLIGHT_BEARISH"))

            inst_continuation_ok = bool(good_inst and good_rr and good_comp)

            inst_continuation_reason = (
                f"good_inst={good_inst}({inst_score}>={INST_SCORE_DESK_PRIORITY}) "
                f"good_rr={good_rr}({rr:.3f}>={RR_MIN_DESK_PRIORITY}) "
                f"good_comp={good_comp}({comp_score:.1f},{comp_label},thr={comp_thr})"
            )

            LOGGER.info(f"[EVAL_PRE] INST_CONTINUATION_CHECK {inst_continuation_reason} ok={inst_continuation_ok}")

        if DESK_EV_MODE and inst_continuation_ok:
            LOGGER.info(f"[EVAL] VALID {symbol} RR={rr:.3f} SETUP=INST_CONTINUATION (DESK_EV_MODE=True)")
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
                "entry_pick": entry_pick,
                "sl_meta": exits.get("sl_meta"),
                "inst_continuation_reason": inst_continuation_reason,
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
