from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from structure_utils import (
    analyze_structure,
    htf_trend_ok,
    bos_quality_details,
    detect_equal_levels,
    liquidity_sweep_details,
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


# =====================================================================
# Base helpers
# =====================================================================

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
    """
    Returns: (premium, discount)
      premium  = last > mid
      discount = last < mid
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
    premium = bool(last > mid)
    discount = bool(last < mid)
    return premium, discount


# =====================================================================
# Composite momentum helpers (score direction-aware)
#   - We assume composite_score is in [0..100] with 0=strong bearish, 100=strong bullish.
# =====================================================================

def _composite_bias_ok(bias: str, score: float, label: str, thr: float = 65.0) -> bool:
    try:
        b = (bias or "").upper()
        s = float(score)
        lab = str(label or "").upper()

        if b == "LONG":
            # bullish = high score
            return (s >= thr) or ("BULL" in lab and s >= (thr - 5.0))
        if b == "SHORT":
            # bearish = low score
            return (s <= (100.0 - thr)) or ("BEAR" in lab and s <= (100.0 - thr + 5.0))
        return False
    except Exception:
        return False


def _composite_bias_fallback(score: float, label: str) -> Optional[str]:
    """Fallback bias when structure trend is RANGE and REQUIRE_STRUCTURE=False."""
    try:
        s = float(score)
        lab = str(label or "").upper()

        if (s >= 60.0) or ("BULL" in lab and s >= 55.0):
            return "LONG"
        if (s <= 40.0) or ("BEAR" in lab and s <= 45.0):
            return "SHORT"
        return None
    except Exception:
        return None


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
    """
    Si marché overextended, on veut que le LIMIT soit "loin" d'au moins 0.25 ATR.
    """
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


# =====================================================================
# Zone parsing / FVG entry
# =====================================================================

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


def _pick_fvg_entry(
    struct: Dict[str, Any],
    entry_mkt: float,
    bias: str,
    atr: float,
    max_dist_atr: float = 4.0,
) -> Tuple[Optional[float], str]:
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

        # Filter direction if present
        if bias == "LONG" and zdir and "bear" in zdir:
            continue
        if bias == "SHORT" and zdir and "bull" in zdir:
            continue

        mid = (zl + zh) / 2.0
        dist = abs(float(entry_mkt) - mid)

        # If price currently inside zone, dist=0
        if zl <= float(entry_mkt) <= zh:
            dist = 0.0

        if dist < best_dist:
            best_dist = dist
            best_mid = mid

    if best_mid is None:
        return None, "no_fvg_parse"

    # Prefer "better side" for pullback limits:
    #  - LONG  -> zone midpoint should be <= market (or inside zone)
    #  - SHORT -> zone midpoint should be >= market (or inside zone)
    try:
        if atr > 0 and best_dist > 0:
            if bias == "LONG" and float(best_mid) > float(entry_mkt) + 0.1 * atr:
                return None, "fvg_not_better_side_long"
            if bias == "SHORT" and float(best_mid) < float(entry_mkt) - 0.1 * atr:
                return None, "fvg_not_better_side_short"
    except Exception:
        pass

    if atr > 0 and best_dist > float(max_dist_atr) * atr:
        return None, f"fvg_too_far dist={best_dist:.6g} atr={atr:.6g} max_atr={max_dist_atr}"

    return float(best_mid), f"fvg_ok dist={best_dist:.6g} atr={atr:.6g} max_atr={max_dist_atr}"


def _pick_entry(df_h1: pd.DataFrame, struct: Dict[str, Any], bias: str) -> Dict[str, Any]:
    """
    Desk entry priority:
      1) RAID→DISPLACEMENT→FVG (si dispo)
      2) OTE (si in-zone)
      3) OTE pullback (même si pas in-zone)
      4) FVG general (best)
      5) fallback = market close
    """
    entry_mkt = _last_close(df_h1)
    atr = _atr(df_h1, 14)

    # 1) RAID->DISPLACEMENT->FVG (from structure_utils.analyze_structure)
    raid = struct.get("raid_displacement") or {}
    try:
        if isinstance(raid, dict) and raid.get("ok") and raid.get("entry") is not None:
            raid_entry = float(raid["entry"])
            dist = abs(entry_mkt - raid_entry)
            if atr <= 0 or dist <= 2.8 * atr:
                return {
                    "entry_used": raid_entry,
                    "entry_type": "RAID_FVG",
                    "order_type": "LIMIT",
                    "in_zone": True,
                    "note": f"raid_ok note={raid.get('note')} dist={dist:.6g} atr={atr:.6g}",
                    "entry_mkt": entry_mkt,
                    "atr": atr,
                    "raid": raid,
                }
    except Exception:
        pass

    # 2) OTE
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

    # OTE pullback (not yet in-zone): if we have valid bounds, place the LIMIT at OTE mid
    # to avoid market-chasing in premium/discount.
    if ote_entry is not None and (not ote_in_zone):
        try:
            dist = abs(float(entry_mkt) - float(ote_entry))
            # only if OTE is on "better side" (buy lower / sell higher)
            if bias == "LONG" and float(ote_entry) <= float(entry_mkt) and (atr <= 0 or dist <= 4.0 * atr):
                return {
                    "entry_used": float(ote_entry),
                    "entry_type": "OTE_PULLBACK",
                    "order_type": "LIMIT",
                    "in_zone": False,
                    "note": f"{ote_note} (pullback dist={dist:.6g} atr={atr:.6g})",
                    "entry_mkt": entry_mkt,
                    "atr": atr,
                }
            if bias == "SHORT" and float(ote_entry) >= float(entry_mkt) and (atr <= 0 or dist <= 4.0 * atr):
                return {
                    "entry_used": float(ote_entry),
                    "entry_type": "OTE_PULLBACK",
                    "order_type": "LIMIT",
                    "in_zone": False,
                    "note": f"{ote_note} (pullback dist={dist:.6g} atr={atr:.6g})",
                    "entry_mkt": entry_mkt,
                    "atr": atr,
                }
        except Exception:
            pass

    # 3) FVG
    fvg_entry, fvg_note = _pick_fvg_entry(struct, entry_mkt, bias, atr, max_dist_atr=4.0)
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

    # 4) fallback
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
# Institutional soft override (desk)
# =====================================================================

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

        if "crowd" in crowd or "over" in crowd or "risky" in crowd:
            return base_score, None

        if bias == "LONG" and flow == "strong_buy" and cvd >= 1.0:
            return MIN_INST_SCORE, "override_flow_cvd_strong_buy"
        if bias == "SHORT" and flow == "strong_sell" and cvd <= -1.0:
            return MIN_INST_SCORE, "override_flow_cvd_strong_sell"

        return base_score, None
    except Exception:
        return base_score, None


# =====================================================================
# Analyzer
# =====================================================================

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

        if not _ensure_ohlcv(df_h1) or len(df_h1) < 80:
            LOGGER.info("[EVAL_REJECT] %s bad_df_h1", symbol)
            return {"valid": False, "reject_reason": "bad_df_h1"}

        if not _ensure_ohlcv(df_h4) or len(df_h4) < 80:
            LOGGER.info("[EVAL_REJECT] %s bad_df_h4", symbol)
            return {"valid": False, "reject_reason": "bad_df_h4"}

        struct = analyze_structure(df_h1)
        bias = str(struct.get("trend", "")).upper()

        # Momentum (used as fallback direction if structure range and REQUIRE_STRUCTURE=False)
        mom = institutional_momentum(df_h1)
        comp = composite_momentum(df_h1)
        vol_reg = volatility_regime(df_h1)
        ext_sig = extension_signal(df_h1)

        comp_score = float(comp.get("score", 50.0)) if isinstance(comp, dict) else 50.0
        comp_label = str(comp.get("label", "NEUTRAL")) if isinstance(comp, dict) else "NEUTRAL"

        LOGGER.info("[EVAL_PRE] %s STRUCT trend=%s bos=%s choch=%s cos=%s",
                    symbol, bias, struct.get("bos"), struct.get("choch"), struct.get("cos"))
        LOGGER.info("[EVAL_PRE] %s MOMENTUM=%s", symbol, mom)
        LOGGER.info("[EVAL_PRE] %s MOMENTUM_COMPOSITE score=%.2f label=%s", symbol, comp_score, comp_label)
        LOGGER.info("[EVAL_PRE] %s VOL_REGIME=%s EXTENSION=%s", symbol, vol_reg, ext_sig)

        # Structure gating / fallback bias
        if bias not in ("LONG", "SHORT"):
            if REQUIRE_STRUCTURE:
                LOGGER.info("[EVAL_REJECT] %s no_clear_trend_range", symbol)
                return {"valid": False, "reject_reason": "no_clear_trend_range", "structure": struct}

            # fallback desk direction from momentum/composite (direction-aware)
            fb = _composite_bias_fallback(comp_score, comp_label)
            if mom in ("BULLISH", "STRONG_BULLISH") or fb == "LONG":
                bias = "LONG"
                LOGGER.info("[EVAL_PRE] %s bias_fallback=LONG", symbol)
            elif mom in ("BEARISH", "STRONG_BEARISH") or fb == "SHORT":
                bias = "SHORT"
                LOGGER.info("[EVAL_PRE] %s bias_fallback=SHORT", symbol)
            else:
                LOGGER.info("[EVAL_REJECT] %s no_bias_fallback", symbol)
                return {"valid": False, "reject_reason": "no_clear_trend_range", "structure": struct}

        # HTF veto
        if REQUIRE_HTF_ALIGN and bias in ("LONG", "SHORT"):
            if not htf_trend_ok(df_h4, bias):
                LOGGER.info("[EVAL_REJECT] %s htf_veto", symbol)
                return {"valid": False, "reject_reason": "htf_veto", "structure": struct}

        # BOS / quality
        entry_mkt = _last_close(df_h1)
        bos_flag = bool(struct.get("bos", False))
        bos_dir = struct.get("bos_direction", None)
        bos_type = struct.get("bos_type", None)

        if (not DESK_EV_MODE) and (not bos_flag):
            LOGGER.info("[EVAL_REJECT] %s no_bos", symbol)
            return {"valid": False, "reject_reason": "no_bos", "structure": struct}

        oi_series = struct.get("oi_series", None)
        LOGGER.info("[EVAL_PRE] %s OI_STRUCT has_oi=%s", symbol, oi_series is not None)

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

        if REQUIRE_MOMENTUM:
            if bias == "LONG" and mom not in ("BULLISH", "STRONG_BULLISH"):
                LOGGER.info("[EVAL_REJECT] %s momentum_not_bullish", symbol)
                return {"valid": False, "reject_reason": "momentum_not_bullish", "structure": struct}
            if bias == "SHORT" and mom not in ("BEARISH", "STRONG_BEARISH"):
                LOGGER.info("[EVAL_REJECT] %s momentum_not_bearish", symbol)
                return {"valid": False, "reject_reason": "momentum_not_bearish", "structure": struct}

        premium, discount = compute_premium_discount(df_h1)
        LOGGER.info("[EVAL_PRE] %s PREMIUM=%s DISCOUNT=%s", symbol, premium, discount)

        # Liquidity sweep context
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

        # Entry pick (OTE/FVG/RAID)
        entry_pick = _pick_entry(df_h1, struct, bias)
        entry = float(entry_pick["entry_used"])
        entry_type = str(entry_pick["entry_type"])
        note = str(entry_pick.get("note"))
        atr14 = float(entry_pick.get("atr") or _atr(df_h1, 14))

        LOGGER.info("[EVAL_PRE] %s ENTRY_PICK type=%s entry_mkt=%s entry_used=%s in_zone=%s note=%s atr=%.6g",
                    symbol, entry_type, entry_pick.get("entry_mkt"), entry, entry_pick.get("in_zone"), note, atr14)

        # Extension hygiene (avoid chasing)
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

        # Premium/discount guard ONLY for market-chasing
        # In DESK_EV_MODE we can allow a momentum-breakout market entry if flow confirms (checked after inst fetch).
        if (not DESK_EV_MODE) and entry_type == "MARKET":
            if bias == "LONG" and premium:
                LOGGER.info("[EVAL_REJECT] %s long_in_premium_market", symbol)
                return {"valid": False, "reject_reason": "long_in_premium_market", "structure": struct, "entry_pick": entry_pick}
            if bias == "SHORT" and discount:
                LOGGER.info("[EVAL_REJECT] %s short_in_discount_market", symbol)
                return {"valid": False, "reject_reason": "short_in_discount_market", "structure": struct, "entry_pick": entry_pick}

        # Compute exits + RR first
        tick = estimate_tick_from_price(entry)
        exits = _compute_exits(df_h1, entry, bias, tick=tick)
        sl = float(exits["sl"])
        tp1 = float(exits["tp1"])
        rr = _safe_rr(entry, sl, tp1, bias)

        # Desk SL hygiene (push SL beyond sweep / nearest EQ pool)
        try:
            atr_last = _atr(df_h1, 14)
        except Exception:
            atr_last = 0.0
        buf = max((atr_last * 0.08) if atr_last > 0 else 0.0, float(tick) * 2.0, entry * 0.0004)

        # Sweep-based SL
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

        # EQ-level SL
        try:
            lv = detect_equal_levels(df_h1.tail(200), max_window=200, tol_mult_atr=0.10)
            eq_highs = lv.get("eq_highs", []) or []
            eq_lows = lv.get("eq_lows", []) or []

            if bias == "LONG" and eq_lows:
                lvl = min(eq_lows, key=lambda x: abs(entry - float(x)))
                lvl = float(lvl)
                if lvl > 0 and lvl < entry and sl > (lvl - buf):
                    sl = min(sl, lvl - buf)

            if bias == "SHORT" and eq_highs:
                lvl = min(eq_highs, key=lambda x: abs(entry - float(x)))
                lvl = float(lvl)
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
            return {"valid": False, "reject_reason": "rr_invalid", "structure": struct, "entry_pick": entry_pick}

        # Determine which setups are even possible BEFORE calling Binance
        can_bos = bool(bos_flag and ((not REQUIRE_BOS_QUALITY) or bos_quality_ok))
        can_raid = bool(isinstance(struct.get("raid_displacement"), dict) and struct["raid_displacement"].get("ok"))
        can_sweep = bool(isinstance(liq_sweep, dict) and liq_sweep.get("ok"))

        need_inst = False
        unfavorable_market = bool(entry_type == "MARKET" and ((bias == "LONG" and premium) or (bias == "SHORT" and discount)))

        if DESK_EV_MODE:
            need_inst = True
        if unfavorable_market:
            need_inst = True
        if (rr < RR_MIN_STRICT) and (rr >= RR_MIN_TOLERATED_WITH_INST):
            need_inst = True
        if can_sweep or can_raid:
            need_inst = True

        # Fetch institutional (best effort)
        inst: Dict[str, Any] = {
            "institutional_score": 0,
            "binance_symbol": None,
            "available": False,
            "warnings": ["not_fetched"],
        }

        if need_inst:
            try:
                inst = await compute_full_institutional_analysis(symbol, bias)
            except Exception as e:
                inst = {
                    "institutional_score": 0,
                    "binance_symbol": None,
                    "available": False,
                    "warnings": [f"inst_exception:{e}"],
                }

        inst_score = int(inst.get("institutional_score") or 0)
        available = bool(inst.get("available", False))
        binance_symbol = inst.get("binance_symbol")
        bypass_inst = (not available) or (binance_symbol is None)

        inst_score_eff, inst_override = _desk_inst_score_eff(inst, bias, inst_score)

        LOGGER.info("[INST_RAW] %s score=%s eff=%s override=%s available=%s binance_symbol=%s need_inst=%s",
                    symbol, inst_score, inst_score_eff, inst_override, available, binance_symbol, need_inst)

        if (not bypass_inst) and inst_override:
            LOGGER.warning("[INST_OVERRIDE] %s %s", symbol, inst_override)

        # -----------------------------------------------------------------
        # DESK "momentum breakout" permission for MARKET entries in bad location
        # -----------------------------------------------------------------
        if unfavorable_market and DESK_EV_MODE:
            breakout_ok = False
            if not bypass_inst:
                try:
                    flow = str(inst.get("flow_regime") or "").lower()
                    tape = float(inst.get("tape_delta") or 0.0)
                    ob = float(inst.get("orderbook_imbalance") or 0.0)

                    # Require structure context OR strong momentum to avoid random chases
                    ctx_ok = bool(bos_flag or (mom in ("STRONG_BULLISH", "STRONG_BEARISH")))
                    if ext_sig is None and ctx_ok and (inst_score_eff >= max(MIN_INST_SCORE, 2)):
                        if bias == "LONG" and flow in ("strong_buy", "buy") and tape >= 0.25 and ob >= 0.05:
                            breakout_ok = True
                        if bias == "SHORT" and flow in ("strong_sell", "sell") and tape <= -0.25 and ob <= -0.05:
                            breakout_ok = True
                except Exception:
                    breakout_ok = False

            if not breakout_ok:
                if bias == "LONG" and premium and entry_type == "MARKET":
                    LOGGER.info("[EVAL_REJECT] %s long_in_premium_market", symbol)
                    return {"valid": False, "reject_reason": "long_in_premium_market", "structure": struct, "entry_pick": entry_pick, "institutional": inst}
                if bias == "SHORT" and discount and entry_type == "MARKET":
                    LOGGER.info("[EVAL_REJECT] %s short_in_discount_market", symbol)
                    return {"valid": False, "reject_reason": "short_in_discount_market", "structure": struct, "entry_pick": entry_pick, "institutional": inst}
            else:
                LOGGER.info("[EVAL_PRE] %s MARKET_BREAKOUT_ALLOWED flow=%s tape=%.3f ob=%.3f inst_eff=%s",
                            symbol, str(inst.get("flow_regime")), float(inst.get("tape_delta") or 0.0),
                            float(inst.get("orderbook_imbalance") or 0.0), int(inst_score_eff))

        # =================================================================
        # SETUPS (desk order)
        # =================================================================

        # 1) BOS_STRICT (PRIMARY) — inst is NOT hard gate here
        bos_ok = False
        bos_variant = "NO"
        if can_bos:
            if rr >= RR_MIN_STRICT:
                bos_ok = True
                bos_variant = "RR_STRICT"
            elif (not bypass_inst) and (inst_score_eff >= MIN_INST_SCORE) and (rr >= RR_MIN_TOLERATED_WITH_INST):
                bos_ok = True
                bos_variant = "RR_RELAX_WITH_INST"
            elif bypass_inst and rr >= RR_MIN_STRICT:
                bos_ok = True
                bos_variant = "BYPASS_RR_STRICT"

        LOGGER.info("[EVAL_PRE] %s BOS_STRICT_CHECK can_bos=%s rr=%.3f ok=%s variant=%s",
                    symbol, can_bos, float(rr), bos_ok, bos_variant)

        if bos_ok:
            setup_type = "BOS_STRICT"
            LOGGER.info("[EVAL] %s VALID RR=%.3f SETUP=%s (%s) DESK_EV_MODE=%s",
                        symbol, float(rr), setup_type, bos_variant, DESK_EV_MODE)
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
                "setup_variant": bos_variant,
                "structure": struct,
                "bos_quality": bos_q,
                "institutional": inst,
                "inst_score_eff": int(inst_score_eff),
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

        # 2) RAID_DISPLACEMENT
        raid_ok = False
        raid_reason = ""
        if DESK_EV_MODE and can_raid:
            raid_ok = float(rr) >= float(RR_MIN_DESK_PRIORITY)

            if bias == "LONG":
                if "BEAR" in comp_label.upper() and comp_score >= 70.0:
                    raid_ok = False
                    raid_reason = "comp_opposes_long"
            else:
                if "BULL" in comp_label.upper() and comp_score <= 30.0:
                    raid_ok = False
                    raid_reason = "comp_opposes_short"

            if raid_ok and (not bypass_inst) and inst_score_eff < 1:
                raid_ok = False
                raid_reason = "inst_too_weak_for_raid"

        LOGGER.info("[EVAL_PRE] %s RAID_CHECK can_raid=%s ok=%s rr=%.3f reason=%s",
                    symbol, can_raid, raid_ok, float(rr), raid_reason)

        if DESK_EV_MODE and raid_ok:
            setup_type = "RAID_DISPLACEMENT"
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
                "setup_variant": str((struct.get("raid_displacement") or {}).get("note") or "raid_ok"),
                "structure": struct,
                "bos_quality": bos_q,
                "institutional": inst,
                "inst_score_eff": int(inst_score_eff),
                "inst_override": inst_override,
                "momentum": mom,
                "composite": comp,
                "composite_score": comp_score,
                "composite_label": comp_label,
                "premium": premium,
                "discount": discount,
                "entry_pick": entry_pick,
                "liquidity_sweep": liq_sweep,
            }

        # 3) LIQ_SWEEP
        liq_ok = False
        liq_reason = ""
        if DESK_EV_MODE and can_sweep:
            liq_ok = float(rr) >= float(RR_MIN_DESK_PRIORITY)

            if bias == "LONG" and ("BEAR" in comp_label.upper()) and comp_score >= 75.0:
                liq_ok = False
                liq_reason = "comp_opposes_long"
            if bias == "SHORT" and ("BULL" in comp_label.upper()) and comp_score <= 25.0:
                liq_ok = False
                liq_reason = "comp_opposes_short"

            if liq_ok and (not bypass_inst) and inst_score_eff < MIN_INST_SCORE:
                liq_ok = False
                liq_reason = "inst_low_for_sweep"

        LOGGER.info("[EVAL_PRE] %s LIQ_SWEEP_CHECK can_sweep=%s ok=%s rr=%.3f reason=%s",
                    symbol, can_sweep, liq_ok, float(rr), liq_reason)

        if DESK_EV_MODE and liq_ok:
            setup_type = "LIQ_SWEEP"
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
                "setup_variant": str(liq_sweep.get("kind") if isinstance(liq_sweep, dict) else "liq_ok"),
                "structure": struct,
                "bos_quality": bos_q,
                "institutional": inst,
                "inst_score_eff": int(inst_score_eff),
                "inst_override": inst_override,
                "momentum": mom,
                "composite": comp,
                "composite_score": comp_score,
                "composite_label": comp_label,
                "premium": premium,
                "discount": discount,
                "entry_pick": entry_pick,
                "liquidity_sweep": liq_sweep,
            }

        # 4) INST_CONTINUATION (fixed SHORT logic)
        inst_cont_ok = False
        inst_cont_reason = ""
        if DESK_EV_MODE and (not bypass_inst) and bias in ("LONG", "SHORT"):
            good_inst = inst_score_eff >= INST_SCORE_DESK_PRIORITY
            good_rr = float(rr) >= float(RR_MIN_DESK_PRIORITY)

            comp_thr = 65.0
            good_comp = _composite_bias_ok(bias, comp_score, comp_label, thr=comp_thr)

            inst_cont_ok = bool(good_inst and good_rr and good_comp)
            inst_cont_reason = (
                f"good_inst={good_inst}({inst_score_eff}>={INST_SCORE_DESK_PRIORITY}) "
                f"good_rr={good_rr}({float(rr):.3f}>={RR_MIN_DESK_PRIORITY}) "
                f"good_comp={good_comp}({comp_score:.1f},{comp_label},thr={comp_thr})"
            )

        LOGGER.info("[EVAL_PRE] %s INST_CONTINUATION_CHECK %s ok=%s",
                    symbol, inst_cont_reason, inst_cont_ok)

        if DESK_EV_MODE and inst_cont_ok:
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
                "setup_variant": "INST_CONTINUATION",
                "structure": struct,
                "bos_quality": bos_q,
                "institutional": inst,
                "inst_score_eff": int(inst_score_eff),
                "inst_override": inst_override,
                "momentum": mom,
                "composite": comp,
                "composite_score": comp_score,
                "composite_label": comp_label,
                "premium": premium,
                "discount": discount,
                "entry_pick": entry_pick,
                "sl_meta": exits.get("sl_meta"),
                "inst_continuation_reason": inst_cont_reason,
            }

        LOGGER.info("[EVAL_REJECT] %s no_setup_validated (DESK_EV_MODE=%s)", symbol, DESK_EV_MODE)
        return {
            "valid": False,
            "reject_reason": "no_setup_validated",
            "structure": struct,
            "institutional": inst,
            "bos_quality": bos_q,
            "entry_pick": entry_pick,
            "rr": float(rr),
            "inst_score_eff": int(inst_score_eff),
            "composite_score": comp_score,
            "composite_label": comp_label,
        }
