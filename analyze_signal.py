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
    RR_MIN_TOLERATED_WITH_INST,
)

LOGGER = logging.getLogger(__name__)

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
    try:
        if not _ensure_ohlcv(df) or len(df) < n + 2:
            return 0.0
        s = true_atr(df, length=n)
        v = float(s.iloc[-1]) if s is not None and len(s) else 0.0
        return float(max(0.0, v)) if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


def compute_premium_discount(df: pd.DataFrame, lookback: int = 80) -> Tuple[bool, bool]:
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
    try:
        if atr <= 0:
            return True
        entry = float(entry)
        entry_mkt = float(entry_mkt)
        bias = (bias or "").upper()

        if bias == "LONG":
            return (entry_mkt - entry) >= 0.25 * atr
        if bias == "SHORT":
            return (entry - entry_mkt) >= 0.25 * atr
        return True
    except Exception:
        return True


def _parse_zone_bounds(z: Dict[str, Any]) -> Optional[Tuple[float, float, str]]:
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
    entry_mkt = _last_close(df_h1)
    atr = _atr(df_h1, 14)

    ote_entry = None
    ote_in_zone = False
    ote_note = "ote_unavailable"
    try:
        ote = compute_ote(df_h1, bias)

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

    return {
        "entry_used": float(entry_mkt),
        "entry_type": "MARKET",
        "order_type": "LIMIT",
        "in_zone": False,
        "note": "no_zone_entry",
        "entry_mkt": entry_mkt,
        "atr": atr,
    }


def _desk_inst_score_eff(inst: Dict[str, Any], bias: str, base_score: int) -> Tuple[int, Optional[str]]:
    """
    Desk override: évite inst_score=0 quand le flow est clairement fort.
    On ne force PAS si régime crowded.
    """
    try:
        if base_score >= MIN_INST_SCORE:
            return base_score, None

        bias = (bias or "").upper()
        flow = str(inst.get("flow_regime") or "").lower()
        crowd = str(inst.get("crowding_regime") or "").lower()
        cvd = float(inst.get("cvd_slope") or 0.0)

        # si déjà crowded, on n'override pas
        if "crowd" in crowd or "over" in crowd:
            return base_score, None

        if bias == "LONG" and flow == "strong_buy" and cvd >= 1.5:
            return MIN_INST_SCORE, "override_flow_cvd_strong_buy"
        if bias == "SHORT" and flow == "strong_sell" and cvd <= -1.5:
            return MIN_INST_SCORE, "override_flow_cvd_strong_sell"

        return base_score, None
    except Exception:
        return base_score, None


class SignalAnalyzer:
    def __init__(self, *args, **kwargs):
        pass

    async def analyze(
        self,
        symbol: str,
        df_h1: pd.DataFrame,
        df_h4: pd.DataFrame,
        macro: Any = None,
    ) -> Dict[str, Any]:

        LOGGER.info("[EVAL] ▶ START %s", symbol)

        if not _ensure_ohlcv(df_h1) or len(df_h1) < 60:
            LOGGER.info("[EVAL_REJECT] %s bad_df_h1", symbol)
            return {"valid": False, "reject_reason": "bad_df_h1"}

        if not _ensure_ohlcv(df_h4) or len(df_h4) < 60:
            LOGGER.info("[EVAL_REJECT] %s bad_df_h4", symbol)
            return {"valid": False, "reject_reason": "bad_df_h4"}

        struct = analyze_structure(df_h1)
        bias = str(struct.get("trend", "")).upper()
        liq_sweep = {"ok": False}
        LOGGER.info("[EVAL_PRE] %s STRUCT trend=%s bos=%s choch=%s cos=%s",
                    symbol, bias, struct.get("bos"), struct.get("choch"), struct.get("cos"))

        oi_series = struct.get("oi_series", None)
        LOGGER.info("[EVAL_PRE] %s OI_STRUCT has_oi=%s", symbol, oi_series is not None)

        if REQUIRE_STRUCTURE:
            if bias not in ("LONG", "SHORT"):
                LOGGER.info("[EVAL_REJECT] %s no_clear_trend_range", symbol)
                return {"valid": False, "reject_reason": "no_clear_trend_range", "structure": struct}
        else:
            if bias not in ("LONG", "SHORT"):
                LOGGER.info("[EVAL_PRE] %s Trend RANGE but REQUIRE_STRUCTURE=False", symbol)

        if REQUIRE_HTF_ALIGN and bias in ("LONG", "SHORT"):
            if not htf_trend_ok(df_h4, bias):
                LOGGER.info("[EVAL_REJECT] %s htf_veto", symbol)
                return {"valid": False, "reject_reason": "htf_veto", "structure": struct}

        entry_mkt = _last_close(df_h1)

        bos_flag = bool(struct.get("bos", False))
        bos_dir = struct.get("bos_direction", None)
        bos_type = struct.get("bos_type", None)

        if (not DESK_EV_MODE) and (not bos_flag):
            LOGGER.info("[EVAL_REJECT] %s no_bos", symbol)
            return {"valid": False, "reject_reason": "no_bos", "structure": struct}

        bos_q = bos_quality_details(
            df_h1,
            oi_series=oi_series,
            df_liq=df_h1,
            price=entry_mkt,
            direction=bos_dir,
        )
        bos_quality_ok = bool(bos_q.get("ok", True))
        LOGGER.info("[EVAL_PRE] %s BOS_QUALITY ok=%s score=%s reasons=%s bos_flag=%s bos_type=%s bos_dir=%s",
                    symbol, bos_quality_ok, bos_q.get("score"), bos_q.get("reasons"),
                    bos_flag, bos_type, bos_dir)

        mom = institutional_momentum(df_h1)
        comp = composite_momentum(df_h1)
        vol_regime = volatility_regime(df_h1)
        ext_sig = extension_signal(df_h1)

        comp_score = float(comp.get("score", 50.0)) if isinstance(comp, dict) else 50.0
        comp_label = str(comp.get("label", "NEUTRAL")) if isinstance(comp, dict) else "NEUTRAL"

        LOGGER.info("[EVAL_PRE] %s MOMENTUM=%s", symbol, mom)
        LOGGER.info("[EVAL_PRE] %s MOMENTUM_COMPOSITE score=%.2f label=%s", symbol, comp_score, comp_label)
        LOGGER.info("[EVAL_PRE] %s VOL_REGIME=%s EXTENSION=%s", symbol, vol_regime, ext_sig)

        if REQUIRE_MOMENTUM and bias in ("LONG", "SHORT"):
            if bias == "LONG" and mom not in ("BULLISH", "STRONG_BULLISH"):
                LOGGER.info("[EVAL_REJECT] %s momentum_not_bullish", symbol)
                return {"valid": False, "reject_reason": "momentum_not_bullish", "structure": struct}
            if bias == "SHORT" and mom not in ("BEARISH", "STRONG_BEARISH"):
                LOGGER.info("[EVAL_REJECT] %s momentum_not_bearish", symbol)
                return {"valid": False, "reject_reason": "momentum_not_bearish", "structure": struct}

        discount, premium = compute_premium_discount(df_h1)
        LOGGER.info("[EVAL_PRE] %s PREMIUM=%s DISCOUNT=%s", symbol, premium, discount)

        # Liquidity sweep (stop hunt) context — can unlock early entries like a desk
        liq_sweep = liquidity_sweep_details(df_h1, bias, lookback=180)
        if isinstance(liq_sweep, dict) and liq_sweep.get("ok"):
            LOGGER.info(
                "[EVAL_PRE] %s LIQ_SWEEP ok=True kind=%s level=%s wick=%.2f body=%.2f",
                symbol,
                liq_sweep.get("kind"),
                liq_sweep.get("level"),
                float(liq_sweep.get("wick_ratio") or 0.0),
                float(liq_sweep.get("body_ratio") or 0.0),
            )

        entry_pick = _pick_entry(df_h1, struct, bias)
        entry = float(entry_pick["entry_used"])
        entry_type = str(entry_pick["entry_type"])
        note = str(entry_pick.get("note"))
        atr14 = float(entry_pick.get("atr") or _atr(df_h1, 14))

        LOGGER.info("[EVAL_PRE] %s ENTRY_PICK type=%s entry_mkt=%s entry_used=%s in_zone=%s note=%s atr=%.6g",
                    symbol, entry_type, entry_pick.get("entry_mkt"), entry, entry_pick.get("in_zone"), note, atr14)

        if ext_sig == "OVEREXTENDED_LONG" and bias == "LONG":
            if entry_type == "MARKET":
                LOGGER.info("[EVAL_REJECT] %s overextended_long_market", symbol)
                return {"valid": False, "reject_reason": "overextended_long_market", "structure": struct, "entry_pick": entry_pick}
            if not _entry_pullback_ok(entry, entry_mkt, bias, atr14):
                LOGGER.info("[EVAL_REJECT] %s overextended_long_no_pullback", symbol)
                return {"valid": False, "reject_reason": "overextended_long_no_pullback", "structure": struct, "entry_pick": entry_pick}

        if ext_sig == "OVEREXTENDED_SHORT" and bias == "SHORT":
            if entry_type == "MARKET":
                LOGGER.info("[EVAL_REJECT] %s overextended_short_market", symbol)
                return {"valid": False, "reject_reason": "overextended_short_market", "structure": struct, "entry_pick": entry_pick}
            if not _entry_pullback_ok(entry, entry_mkt, bias, atr14):
                LOGGER.info("[EVAL_REJECT] %s overextended_short_no_pullback", symbol)
                return {"valid": False, "reject_reason": "overextended_short_no_pullback", "structure": struct, "entry_pick": entry_pick}

        if bias == "LONG" and premium and entry_type == "MARKET":
            LOGGER.info("[EVAL_REJECT] %s long_in_premium_market", symbol)
            return {"valid": False, "reject_reason": "long_in_premium_market", "structure": struct, "entry_pick": entry_pick}
        if bias == "SHORT" and discount and entry_type == "MARKET":
            LOGGER.info("[EVAL_REJECT] %s short_in_discount_market", symbol)
            return {"valid": False, "reject_reason": "short_in_discount_market", "structure": struct, "entry_pick": entry_pick}

        inst = await compute_full_institutional_analysis(symbol, bias)
        inst_score = int(inst.get("institutional_score") or 0)
        binance_symbol = inst.get("binance_symbol")
        available = bool(inst.get("available", False))

        bypass_inst = (not available) or (binance_symbol is None)

        inst_score_eff, inst_override = _desk_inst_score_eff(inst, bias, inst_score)

        LOGGER.info("[INST_RAW] %s score=%s eff=%s override=%s available=%s binance_symbol=%s",
                    symbol, inst_score, inst_score_eff, inst_override, available, binance_symbol)

        if not bypass_inst and inst_override:
            LOGGER.warning("[INST_OVERRIDE] %s %s", symbol, inst_override)

        if not bypass_inst and inst_score_eff < MIN_INST_SCORE:
            LOGGER.info("[EVAL_REJECT] %s inst_score_low (%s < %s)", symbol, inst_score_eff, MIN_INST_SCORE)
            return {"valid": False, "reject_reason": "inst_score_low", "institutional": inst, "structure": struct, "entry_pick": entry_pick}

        if bypass_inst:
            LOGGER.warning("[EVAL_WARN] %s bypass MIN_INST_SCORE (binance unmapped)", symbol)

        tick = estimate_tick_from_price(entry)
        exits = _compute_exits(df_h1, entry, bias, tick=tick)
        sl = float(exits["sl"])
        tp1 = float(exits["tp1"])
        rr = _safe_rr(entry, sl, tp1, bias)

        # Desk SL hygiene: reduce stop-hunts around clear liquidity (EQ levels / sweeps)
        try:
            atr14 = float(true_atr(df_h1, 14).iloc[-1]) if len(df_h1) >= 20 else 0.0
        except Exception:
            atr14 = 0.0
        buf = max((atr14 * 0.08) if atr14 > 0 else 0.0, float(tick) * 2.0)

        # Sweep-based adjustment (if present)
        try:
            if isinstance(liq_sweep, dict) and liq_sweep.get("ok"):
                ext = float(liq_sweep.get("sweep_extreme") or 0.0)
                if ext > 0 and buf > 0:
                    if bias == "LONG":
                        sl = min(sl, ext - buf)
                    else:
                        sl = max(sl, ext + buf)
        except Exception:
            pass

        # Equal-level liquidity adjustment (push SL beyond nearest EQ pool)
        try:
            eq_highs, eq_lows = detect_equal_levels(df_h1, lookback=180, tol_atr=0.10)
            if bias == "LONG" and eq_lows:
                lvl = float(eq_lows[0].get("level") or 0.0)
                if lvl > 0 and lvl < entry and sl > (lvl - buf):
                    sl = min(sl, lvl - buf)
            elif bias == "SHORT" and eq_highs:
                lvl = float(eq_highs[0].get("level") or 0.0)
                if lvl > 0 and lvl > entry and sl < (lvl + buf):
                    sl = max(sl, lvl + buf)
        except Exception:
            pass

        if sl <= 0:
            LOGGER.info("[EVAL_REJECT] %s sl_invalid_after_liq_adj sl=%s", symbol, sl)
            return {"valid": False, "reject_reason": "sl_invalid_after_liq_adj"}

        exits["sl"] = sl
        rr = _safe_rr(entry, sl, tp1, bias)

        LOGGER.info("[EVAL_PRE] %s EXITS entry=%s sl=%s tp1=%s tick=%s RR=%s raw_rr=%s entry_type=%s",
                    symbol, entry, sl, tp1, tick, rr, exits["rr_used"], entry_type)

        if rr is None or rr <= 0:
            LOGGER.info("[EVAL_REJECT] %s rr_invalid", symbol)
            return {"valid": False, "reject_reason": "rr_invalid", "institutional": inst, "structure": struct, "entry_pick": entry_pick}

        bos_strict_ok = False
        bos_strict_variant = "RR_STRICT"

        if bos_flag and ((not REQUIRE_BOS_QUALITY) or bos_quality_ok):
            if bypass_inst:
                if rr >= RR_MIN_STRICT:
                    bos_strict_ok = True
                    bos_strict_variant = "BYPASS_RR_STRICT"
            else:
                if rr >= RR_MIN_STRICT and inst_score_eff >= MIN_INST_SCORE:
                    bos_strict_ok = True
                    bos_strict_variant = "RR_STRICT"
                elif rr >= RR_MIN_TOLERATED_WITH_INST and inst_score_eff >= MIN_INST_SCORE:
                    bos_strict_ok = True
                    bos_strict_variant = "RR_RELAX_WITH_INST"

        LOGGER.info("[EVAL_PRE] %s BOS_STRICT_CHECK bos_flag=%s bos_quality_ok=%s rr=%.3f inst_eff=%s bypass_inst=%s ok=%s variant=%s",
                    symbol, bos_flag, bos_quality_ok, float(rr), inst_score_eff, bypass_inst, bos_strict_ok, bos_strict_variant)

        if bos_strict_ok:
            setup_type = "BOS_STRICT"
            LOGGER.info("[EVAL] %s VALID RR=%.3f SETUP=%s (%s) DESK_EV_MODE=%s",
                        symbol, float(rr), setup_type, bos_strict_variant, DESK_EV_MODE)
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
                "rr": float(rr),
                "qty": 1,
                "setup_type": setup_type,
                "setup_variant": bos_strict_variant,
                "structure": struct,
                "bos_quality": bos_q,
                "institutional": inst,
                "inst_score_eff": inst_score_eff,
                "inst_override": inst_override,
                "momentum": mom,
                "composite": comp,
                "composite_score": comp_score,
                "composite_label": comp_label,
                "premium": premium,
                "discount": discount,
                "entry_pick": entry_pick,
                "sl_meta": exits.get("sl_meta"),
            }

        inst_continuation_ok = False
        inst_continuation_reason = ""

        if DESK_EV_MODE and (not bypass_inst) and bias in ("LONG", "SHORT"):
            good_inst = inst_score_eff >= INST_SCORE_DESK_PRIORITY
            good_rr = float(rr) >= RR_MIN_DESK_PRIORITY

            comp_thr = 65.0
            if bias == "LONG":
                good_comp = (comp_score >= comp_thr) and (comp_label in ("STRONG_BULLISH", "BULLISH", "SLIGHT_BULLISH"))
            else:
                good_comp = (comp_score >= comp_thr) and (comp_label in ("STRONG_BEARISH", "BEARISH", "SLIGHT_BEARISH"))

            inst_continuation_ok = bool(good_inst and good_rr and good_comp)

            inst_continuation_reason = (
                f"good_inst={good_inst}({inst_score_eff}>={INST_SCORE_DESK_PRIORITY}) "
                f"good_rr={good_rr}({float(rr):.3f}>={RR_MIN_DESK_PRIORITY}) "
                f"good_comp={good_comp}({comp_score:.1f},{comp_label},thr={comp_thr})"
            )

            
        # --- Setup 1.5: LIQ_SWEEP (stop-hunt reclaim) ---
        liq_sweep_ok = bool(isinstance(liq_sweep, dict) and liq_sweep.get("ok")) and (inst_eff >= MIN_INST_SCORE) and (rr >= MIN_RR_DESK)
        if liq_sweep_ok:
            # keep it simple: don't take a sweep against clear composite momentum
            if bias == "LONG":
                liq_sweep_ok = ("BEAR" not in str(comp_label).upper())
            else:
                liq_sweep_ok = ("BULL" not in str(comp_label).upper())

        LOGGER.info(
            "[EVAL_PRE] %s LIQ_SWEEP_CHECK ok=%s inst_eff=%s rr=%.3f comp=%s",
            symbol,
            liq_sweep_ok,
            inst_eff,
            rr,
            comp_label,
        )

        if DESK_EV_MODE and liq_sweep_ok:
            setup_type = "LIQ_SWEEP"
            side = "BUY" if bias == "LONG" else "SELL"
            entry_type = str(entry_pick.get("entry_type") or entry_pick.get("type") or "LIMIT")

            LOGGER.info("[EVAL] %s VALID RR=%s SETUP=%s bias=%s inst_eff=%s", symbol, rr, setup_type, bias, inst_eff)

            return {
                "symbol": symbol,
                "valid": True,
                "side": side,
                "setup_type": setup_type,
                "entry_type": entry_type,
                "entry": entry,
                "sl": sl,
                "tp1": tp1,
                "rr": rr,
                "inst_score_eff": int(inst_eff),
                "institutional": inst,
                "structure": struct,
                "entry_pick": entry_pick,
                "liquidity_sweep": liq_sweep,
            }

        LOGGER.info("[EVAL_PRE] %s INST_CONTINUATION_CHECK %s ok=%s",
                        symbol, inst_continuation_reason, inst_continuation_ok)

        if DESK_EV_MODE and inst_continuation_ok:
            setup_type = "INST_CONTINUATION"
            LOGGER.info("[EVAL] %s VALID RR=%.3f SETUP=%s (DESK_EV_MODE=True)", symbol, float(rr), setup_type)
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
                "rr": float(rr),
                "qty": 1,
                "setup_type": setup_type,
                "structure": struct,
                "bos_quality": bos_q,
                "institutional": inst,
                "inst_score_eff": inst_score_eff,
                "inst_override": inst_override,
                "momentum": mom,
                "composite": comp,
                "composite_score": comp_score,
                "composite_label": comp_label,
                "premium": premium,
                "discount": discount,
                "entry_pick": entry_pick,
                "sl_meta": exits.get("sl_meta"),
                "inst_continuation_reason": inst_continuation_reason,
            }

        LOGGER.info("[EVAL_REJECT] %s no_setup_validated (BOS_STRICT_OK=%s INST_CONTINUATION_OK=%s DESK_EV_MODE=%s)",
                    symbol, bos_strict_ok, inst_continuation_ok, DESK_EV_MODE)
        return {
            "valid": False,
            "reject_reason": "no_setup_validated",
            "structure": struct,
            "institutional": inst,
            "bos_quality": bos_q,
            "entry_pick": entry_pick,
            "rr": float(rr),
            "inst_score_eff": inst_score_eff,
            "composite_score": comp_score,
            "composite_label": comp_label,
        }
