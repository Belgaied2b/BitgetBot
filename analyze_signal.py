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
# Composite momentum helpers (direction-aware)
#   - composite_score in [0..100] with 0=strong bearish, 100=strong bullish.
# =====================================================================

def _composite_bias_ok(bias: str, score: float, label: str, thr: float = 65.0) -> bool:
    try:
        b = (bias or "").upper()
        s = float(score)
        lab = str(label or "").upper()

        if b == "LONG":
            return (s >= thr) or ("BULL" in lab and s >= (thr - 5.0))
        if b == "SHORT":
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

        b = (bias or "").upper()
        if b == "LONG":
            risk = entry - sl
            reward = tp1 - entry
        elif b == "SHORT":
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
        b = (bias or "").upper()

        if b == "LONG":
            return (entry_mkt - entry) >= 0.25 * atr
        if b == "SHORT":
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
    b = (bias or "").upper()

    for z in zones:
        parsed = _parse_zone_bounds(z)
        if not parsed:
            continue
        zl, zh, zdir = parsed

        if b == "LONG" and zdir and "bear" in zdir:
            continue
        if b == "SHORT" and zdir and "bull" in zdir:
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

    try:
        if atr > 0 and best_dist > 0:
            if b == "LONG" and float(best_mid) > float(entry_mkt) + 0.1 * atr:
                return None, "fvg_not_better_side_long"
            if b == "SHORT" and float(best_mid) < float(entry_mkt) - 0.1 * atr:
                return None, "fvg_not_better_side_short"
    except Exception:
        pass

    if atr > 0 and best_dist > float(max_dist_atr) * atr:
        return None, f"fvg_too_far dist={best_dist:.6g} atr={atr:.6g} max_atr={max_dist_atr}"

    return float(best_mid), f"fvg_ok dist={best_dist:.6g} atr={atr:.6g} max_atr={max_dist_atr}"


def _pick_entry(df_h1: pd.DataFrame, struct: Dict[str, Any], bias: str) -> Dict[str, Any]:
    """
    Entry priority:
      1) RAID→DISPLACEMENT→FVG (si dispo)
      2) OTE (si in-zone)
      3) OTE pullback (même si pas in-zone)
      4) FVG general
      5) fallback = market close
    """
    entry_mkt = _last_close(df_h1)
    atr = _atr(df_h1, 14)

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

    # OTE
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

    # OTE pullback (même si pas in-zone)
    if ote_entry is not None and (not ote_in_zone):
        try:
            dist = abs(float(entry_mkt) - float(ote_entry))
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

    # FVG
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

    # fallback
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
# Institutional override helper (kept, but hard-gate uses eff score)
# =====================================================================

def _desk_inst_score_eff(inst: Dict[str, Any], bias: str, base_score: int) -> Tuple[int, Optional[str]]:
    """
    Desk override: évite inst_score=0 quand le flow est clairement fort.
    On ne force PAS si régime crowded.
    """
    try:
        if base_score >= MIN_INST_SCORE:
            return base_score, None

        b = (bias or "").upper()
        flow = str(inst.get("flow_regime") or "").lower()
        crowd = str(inst.get("crowding_regime") or "").lower()
        cvd = float(inst.get("cvd_slope") or 0.0)

        if "crowd" in crowd or "over" in crowd or "risky" in crowd:
            return base_score, None

        if b == "LONG" and flow == "strong_buy" and cvd >= 1.0:
            return MIN_INST_SCORE, "override_flow_cvd_strong_buy"
        if b == "SHORT" and flow == "strong_sell" and cvd <= -1.0:
            return MIN_INST_SCORE, "override_flow_cvd_strong_sell"

        return base_score, None
    except Exception:
        return base_score, None


def _reject_with_inst(reason: str, inst: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"valid": False, "reject_reason": reason, "institutional": inst}
    out.update(extra)
    return out


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

        # ---- cheap direction hints first (for inst) ----
        mom = institutional_momentum(df_h1)
        comp = composite_momentum(df_h1)
        vol_reg = volatility_regime(df_h1)
        ext_sig = extension_signal(df_h1)

        comp_score = float(comp.get("score", 50.0)) if isinstance(comp, dict) else 50.0
        comp_label = str(comp.get("label", "NEUTRAL")) if isinstance(comp, dict) else "NEUTRAL"

        # ---- structure (used AFTER inst hard-gate for confirmation + entries) ----
        struct = analyze_structure(df_h1)
        bias = str(struct.get("trend", "")).upper()

        # fallback bias if structure range (used to call inst correctly)
        if bias not in ("LONG", "SHORT"):
            fb = _composite_bias_fallback(comp_score, comp_label)
            if mom in ("BULLISH", "STRONG_BULLISH") or fb == "LONG":
                bias = "LONG"
            elif mom in ("BEARISH", "STRONG_BEARISH") or fb == "SHORT":
                bias = "SHORT"
            else:
                bias = "RANGE"

        LOGGER.info(
            "[EVAL_PRE] %s STRUCT trend=%s bos=%s choch=%s cos=%s",
            symbol, str(struct.get("trend", "")).upper(), struct.get("bos"), struct.get("choch"), struct.get("cos")
        )
        LOGGER.info("[EVAL_PRE] %s MOMENTUM=%s", symbol, mom)
        LOGGER.info("[EVAL_PRE] %s MOMENTUM_COMPOSITE score=%.2f label=%s", symbol, comp_score, comp_label)
        LOGGER.info("[EVAL_PRE] %s VOL_REGIME=%s EXTENSION=%s", symbol, vol_reg, ext_sig)

        # =================================================================
        # 1) INSTITUTIONAL — HARD GATE (priority #1)
        # =================================================================
        inst: Dict[str, Any]
        try:
            inst = await compute_full_institutional_analysis(symbol, "LONG" if bias == "RANGE" else bias)
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

        inst_score_eff, inst_override = _desk_inst_score_eff(inst, bias, inst_score)

        LOGGER.info(
            "[INST_RAW] %s score=%s eff=%s override=%s available=%s binance_symbol=%s bias=%s",
            symbol, inst_score, inst_score_eff, inst_override, available, binance_symbol, bias
        )

        if inst_override:
            LOGGER.warning("[INST_OVERRIDE] %s %s", symbol, inst_override)

        # HARD GATE: must be available + mapped + score >= MIN_INST_SCORE
        if (not available) or (binance_symbol is None):
            LOGGER.info("[EVAL_REJECT] %s inst_unavailable", symbol)
            return _reject_with_inst("inst_unavailable", inst, structure=struct, momentum=mom, composite=comp)

        if inst_score_eff < int(MIN_INST_SCORE):
            LOGGER.info("[EVAL_REJECT] %s inst_score_low eff=%s < MIN_INST_SCORE=%s", symbol, inst_score_eff, MIN_INST_SCORE)
            return _reject_with_inst("inst_score_low", inst, inst_score_eff=int(inst_score_eff), structure=struct, momentum=mom, composite=comp)

        # =================================================================
        # 2) STRUCTURE CONFIRMATION (after inst hard-gate)
        # =================================================================

        # Structure gating / fallback bias (now as confirmation)
        if bias not in ("LONG", "SHORT"):
            if REQUIRE_STRUCTURE:
                LOGGER.info("[EVAL_REJECT] %s no_clear_trend_range", symbol)
                return _reject_with_inst("no_clear_trend_range", inst, structure=struct, momentum=mom, composite=comp)

            fb = _composite_bias_fallback(comp_score, comp_label)
            if mom in ("BULLISH", "STRONG_BULLISH") or fb == "LONG":
                bias = "LONG"
                LOGGER.info("[EVAL_PRE] %s bias_fallback=LONG", symbol)
            elif mom in ("BEARISH", "STRONG_BEARISH") or fb == "SHORT":
                bias = "SHORT"
                LOGGER.info("[EVAL_PRE] %s bias_fallback=SHORT", symbol)
            else:
                LOGGER.info("[EVAL_REJECT] %s no_bias_fallback", symbol)
                return _reject_with_inst("no_clear_trend_range", inst, structure=struct, momentum=mom, composite=comp)

        # HTF veto
        if REQUIRE_HTF_ALIGN and bias in ("LONG", "SHORT"):
            if not htf_trend_ok(df_h4, bias):
                LOGGER.info("[EVAL_REJECT] %s htf_veto", symbol)
                return _reject_with_inst("htf_veto", inst, structure=struct)

        # BOS / quality
        entry_mkt = _last_close(df_h1)
        bos_flag = bool(struct.get("bos", False))
        bos_dir = struct.get("bos_direction", None)
        bos_type = struct.get("bos_type", None)

        if (not DESK_EV_MODE) and (not bos_flag):
            LOGGER.info("[EVAL_REJECT] %s no_bos", symbol)
            return _reject_with_inst("no_bos", inst, structure=struct)

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
        LOGGER.info(
            "[EVAL_PRE] %s BOS_QUALITY ok=%s score=%s reasons=%s bos_flag=%s bos_type=%s bos_dir=%s",
            symbol, bos_quality_ok, bos_q.get("score"), bos_q.get("reasons"),
            bos_flag, bos_type, bos_dir
        )

        if REQUIRE_MOMENTUM:
            if bias == "LONG" and mom not in ("BULLISH", "STRONG_BULLISH"):
                LOGGER.info("[EVAL_REJECT] %s momentum_not_bullish", symbol)
                return _reject_with_inst("momentum_not_bullish", inst, structure=struct)
            if bias == "SHORT" and mom not in ("BEARISH", "STRONG_BEARISH"):
                LOGGER.info("[EVAL_REJECT] %s momentum_not_bearish", symbol)
                return _reject_with_inst("momentum_not_bearish", inst, structure=struct)

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

        LOGGER.info(
            "[EVAL_PRE] %s ENTRY_PICK type=%s entry_mkt=%s entry_used=%s in_zone=%s note=%s atr=%.6g",
            symbol, entry_type, entry_pick.get("entry_mkt"), entry, entry_pick.get("in_zone"), note, atr14
        )

        # Extension hygiene (avoid chasing)
        if ext_sig == "OVEREXTENDED_LONG" and bias == "LONG":
            if entry_type == "MARKET":
                LOGGER.info("[EVAL_REJECT] %s overextended_long_market", symbol)
                return _reject_with_inst("overextended_long_market", inst, structure=struct, entry_pick=entry_pick)
            if not _entry_pullback_ok(entry, entry_mkt, bias, atr14):
                LOGGER.info("[EVAL_REJECT] %s overextended_long_no_pullback", symbol)
                return _reject_with_inst("overextended_long_no_pullback", inst, structure=struct, entry_pick=entry_pick)

        if ext_sig == "OVEREXTENDED_SHORT" and bias == "SHORT":
            if entry_type == "MARKET":
                LOGGER.info("[EVAL_REJECT] %s overextended_short_market", symbol)
                return _reject_with_inst("overextended_short_market", inst, structure=struct, entry_pick=entry_pick)
            if not _entry_pullback_ok(entry, entry_mkt, bias, atr14):
                LOGGER.info("[EVAL_REJECT] %s overextended_short_no_pullback", symbol)
                return _reject_with_inst("overextended_short_no_pullback", inst, structure=struct, entry_pick=entry_pick)

        unfavorable_market = bool(entry_type == "MARKET" and ((bias == "LONG" and premium) or (bias == "SHORT" and discount)))

        # Premium/discount guard ONLY for market-chasing
        # In DESK_EV_MODE we can allow a momentum-breakout market entry if flow confirms.
        if unfavorable_market and (not DESK_EV_MODE):
            if bias == "LONG" and premium and entry_type == "MARKET":
                LOGGER.info("[EVAL_REJECT] %s long_in_premium_market", symbol)
                return _reject_with_inst("long_in_premium_market", inst, structure=struct, entry_pick=entry_pick)
            if bias == "SHORT" and discount and entry_type == "MARKET":
                LOGGER.info("[EVAL_REJECT] %s short_in_discount_market", symbol)
                return _reject_with_inst("short_in_discount_market", inst, structure=struct, entry_pick=entry_pick)

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
            return _reject_with_inst("sl_invalid_after_liq_adj", inst)

        exits["sl"] = sl
        rr = _safe_rr(entry, sl, tp1, bias)

        LOGGER.info(
            "[EVAL_PRE] %s EXITS entry=%s sl=%s tp1=%s tick=%s RR=%s raw_rr=%s entry_type=%s",
            symbol, entry, sl, tp1, tick, rr, exits["rr_used"], entry_type
        )

        if rr is None or rr <= 0:
            LOGGER.info("[EVAL_REJECT] %s rr_invalid", symbol)
            return _reject_with_inst("rr_invalid", inst, structure=struct, entry_pick=entry_pick)

        # -----------------------------------------------------------------
        # DESK "momentum breakout" permission for MARKET entries in bad location
        # -----------------------------------------------------------------
        if unfavorable_market and DESK_EV_MODE:
            breakout_ok = False
            try:
                flow = str(inst.get("flow_regime") or "").lower()
                tape = float(inst.get("tape_delta") or 0.0)
                ob = float(inst.get("orderbook_imbalance") or 0.0)

                ctx_ok = bool(bos_flag or (mom in ("STRONG_BULLISH", "STRONG_BEARISH")))
                not_overext = str(ext_sig or "").upper() not in ("OVEREXTENDED_LONG", "OVEREXTENDED_SHORT")

                if not_overext and ctx_ok and (inst_score_eff >= max(int(MIN_INST_SCORE), 2)):
                    if bias == "LONG" and flow in ("strong_buy", "buy") and tape >= 0.25 and ob >= 0.05:
                        breakout_ok = True
                    if bias == "SHORT" and flow in ("strong_sell", "sell") and tape <= -0.25 and ob <= -0.05:
                        breakout_ok = True
            except Exception:
                breakout_ok = False

            if not breakout_ok:
                if bias == "LONG" and premium and entry_type == "MARKET":
                    LOGGER.info("[EVAL_REJECT] %s long_in_premium_market", symbol)
                    return _reject_with_inst("long_in_premium_market", inst, structure=struct, entry_pick=entry_pick)
                if bias == "SHORT" and discount and entry_type == "MARKET":
                    LOGGER.info("[EVAL_REJECT] %s short_in_discount_market", symbol)
                    return _reject_with_inst("short_in_discount_market", inst, structure=struct, entry_pick=entry_pick)
            else:
                LOGGER.info(
                    "[EVAL_PRE] %s MARKET_BREAKOUT_ALLOWED flow=%s tape=%.3f ob=%.3f inst_eff=%s",
                    symbol, str(inst.get("flow_regime")), float(inst.get("tape_delta") or 0.0),
                    float(inst.get("orderbook_imbalance") or 0.0), int(inst_score_eff)
                )

        # =================================================================
        # SETUPS (structure-confirmed, inst already hard-gated)
        # =================================================================

        can_bos = bool(bos_flag and ((not REQUIRE_BOS_QUALITY) or bos_quality_ok))
        can_raid = bool(isinstance(struct.get("raid_displacement"), dict) and struct["raid_displacement"].get("ok"))
        can_sweep = bool(isinstance(liq_sweep, dict) and liq_sweep.get("ok"))

        # 1) BOS_STRICT (primary)
        bos_ok = False
        bos_variant = "NO"
        if can_bos:
            if rr >= RR_MIN_STRICT:
                bos_ok = True
                bos_variant = "RR_STRICT"
            elif rr >= RR_MIN_TOLERATED_WITH_INST:
                bos_ok = True
                bos_variant = "RR_RELAX_WITH_INST"

        LOGGER.info(
            "[EVAL_PRE] %s BOS_STRICT_CHECK can_bos=%s rr=%.3f ok=%s variant=%s",
            symbol, can_bos, float(rr), bos_ok, bos_variant
        )

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

        # 2) RAID_DISPLACEMENT (desk)
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

            # inst already gated, but keep a minimal filter for raid quality
            if raid_ok and inst_score_eff < 1:
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

        # 3) LIQ_SWEEP (desk)
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

            if liq_ok and inst_score_eff < int(MIN_INST_SCORE):
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

        # 4) INST_CONTINUATION (desk)
        inst_cont_ok = False
        inst_cont_reason = ""
        if DESK_EV_MODE and bias in ("LONG", "SHORT"):
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
