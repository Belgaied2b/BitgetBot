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
# Zone entry settings (FVG / OTE)
# =====================================================================
USE_ZONE_ENTRY = True
ZONE_ENTRY_MODE = "LIMIT"   # "LIMIT" (propose une entry au milieu zone) ou "MARKET" (ignore) ou "REJECT" (refuse si pas dans zone)
ZONE_PRIORITY = "FVG"       # "FVG" ou "OTE"
ZONE_MAX_ATR_DIST = 1.5     # garde-fou: si zone trop loin (> 1.5 ATR), on fallback market (ou reject si ZONE_ENTRY_MODE="REJECT")


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


def estimate_tick_from_price(price: float) -> float:
    """
    Estimation du tick à partir du niveau de prix.
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
    return {"sl": sl, "tp1": tp1, "rr_used": rr_used, "sl_meta": meta}


def _atr14(df: pd.DataFrame, n: int = 14) -> float:
    if df is None or df.empty or len(df) < n + 2:
        try:
            return float((df["high"] - df["low"]).tail(20).mean())
        except Exception:
            return 0.0

    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean().iloc[-1]
    if pd.isna(atr):
        atr = tr.iloc[-1]
    return float(atr or 0.0)


def _parse_ote(df: pd.DataFrame, bias: str) -> Optional[dict]:
    """
    Supporte plusieurs formats selon ton compute_ote().
    Retour: {"low":..., "high":..., "mid":..., "in_zone": bool}
    """
    try:
        res = compute_ote(df, bias)
    except Exception:
        return None

    if res is None:
        return None

    last = float(df["close"].iloc[-1])

    if isinstance(res, dict):
        low = res.get("low") or res.get("discount_low") or res.get("ote_low")
        high = res.get("high") or res.get("discount_high") or res.get("ote_high")
        if low is None or high is None:
            return None
        low = float(low)
        high = float(high)
        lo, hi = min(low, high), max(low, high)
        mid = (lo + hi) / 2.0
        inz = res.get("in_zone")
        if inz is None:
            inz = (lo <= last <= hi)
        return {"low": lo, "high": hi, "mid": float(mid), "in_zone": bool(inz)}

    if isinstance(res, (tuple, list)):
        if len(res) == 3:
            inz, low, high = res
            low = float(low)
            high = float(high)
            lo, hi = min(low, high), max(low, high)
            mid = (lo + hi) / 2.0
            return {"low": lo, "high": hi, "mid": float(mid), "in_zone": bool(inz)}
        if len(res) == 2:
            low, high = res
            low = float(low)
            high = float(high)
            lo, hi = min(low, high), max(low, high)
            mid = (lo + hi) / 2.0
            inz = (lo <= last <= hi)
            return {"low": lo, "high": hi, "mid": float(mid), "in_zone": bool(inz)}

    return None


def _pick_fvg_zone(struct: dict, bias: str, last: float) -> Optional[dict]:
    """
    struct["fvg_zones"] doit être une liste.
    Retour: {"low":..., "high":..., "mid":...}
    """
    zones = struct.get("fvg_zones") or []
    if not isinstance(zones, list) or not zones:
        return None

    parsed = []
    for z in zones:
        if isinstance(z, dict):
            lo = z.get("low") or z.get("bottom") or z.get("l")
            hi = z.get("high") or z.get("top") or z.get("h")
            if lo is None or hi is None:
                continue
            lo = float(lo)
            hi = float(hi)
            lo2, hi2 = min(lo, hi), max(lo, hi)
            parsed.append({"low": lo2, "high": hi2, "mid": (lo2 + hi2) / 2.0})
        elif isinstance(z, (tuple, list)) and len(z) >= 2:
            lo = float(z[0])
            hi = float(z[1])
            lo2, hi2 = min(lo, hi), max(lo, hi)
            parsed.append({"low": lo2, "high": hi2, "mid": (lo2 + hi2) / 2.0})

    if not parsed:
        return None

    if bias == "LONG":
        under = [z for z in parsed if z["high"] <= last]
        cand = under if under else parsed
        return sorted(cand, key=lambda z: abs(last - z["mid"]))[0]

    if bias == "SHORT":
        over = [z for z in parsed if z["low"] >= last]
        cand = over if over else parsed
        return sorted(cand, key=lambda z: abs(last - z["mid"]))[0]

    return None


def _choose_zone_entry(df: pd.DataFrame, struct: dict, bias: str, entry_mkt: float) -> dict:
    """
    Retour:
      {"entry": float, "type": "MARKET|FVG|OTE", "note": str, "in_zone": bool}
    """
    last = float(entry_mkt)
    atr = _atr14(df)
    max_dist = ZONE_MAX_ATR_DIST * atr if atr and atr > 0 else None

    fvg = _pick_fvg_zone(struct, bias, last)
    ote = _parse_ote(df, bias)

    in_fvg = False
    if fvg:
        in_fvg = (fvg["low"] <= last <= fvg["high"])

    in_ote = bool(ote and ote.get("in_zone"))

    # déjà dans une zone -> entry au milieu zone (meilleur RR / logique setup)
    if ZONE_PRIORITY.upper() == "FVG":
        if in_fvg and fvg:
            return {"entry": float(fvg["mid"]), "type": "FVG", "note": f"in_fvg low={fvg['low']:.6g} high={fvg['high']:.6g}", "in_zone": True}
        if in_ote and ote:
            return {"entry": float(ote["mid"]), "type": "OTE", "note": f"in_ote low={ote['low']:.6g} high={ote['high']:.6g}", "in_zone": True}
    else:
        if in_ote and ote:
            return {"entry": float(ote["mid"]), "type": "OTE", "note": f"in_ote low={ote['low']:.6g} high={ote['high']:.6g}", "in_zone": True}
        if in_fvg and fvg:
            return {"entry": float(fvg["mid"]), "type": "FVG", "note": f"in_fvg low={fvg['low']:.6g} high={fvg['high']:.6g}", "in_zone": True}

    # pas dedans -> propose LIMIT vers la meilleure zone
    candidates = []
    if fvg:
        candidates.append(("FVG", float(fvg["mid"]), f"to_fvg mid={fvg['mid']:.6g}"))
    if ote:
        candidates.append(("OTE", float(ote["mid"]), f"to_ote mid={ote['mid']:.6g}"))

    if not candidates:
        return {"entry": last, "type": "MARKET", "note": "no_zone", "in_zone": False}

    # priorité
    if ZONE_PRIORITY.upper() == "FVG":
        candidates.sort(key=lambda x: (0 if x[0] == "FVG" else 1, abs(last - x[1])))
    else:
        candidates.sort(key=lambda x: (0 if x[0] == "OTE" else 1, abs(last - x[1])))

    ztype, zentry, note = candidates[0]
    dist = abs(last - zentry)

    if max_dist is not None and dist > max_dist:
        return {"entry": last, "type": "MARKET", "note": f"zone_too_far dist={dist:.6g} atr={atr:.6g}", "in_zone": False}

    return {"entry": zentry, "type": ztype, "note": note, "in_zone": False}


def _rej(reason: str, **debug) -> Dict[str, Any]:
    out = {"valid": False, "reject_reason": reason}
    if debug:
        out["debug"] = debug
    return out


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
         - BOS non requis
         - inst_score >= INST_SCORE_DESK_PRIORITY
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

        entry_mkt = float(df_h1["close"].iloc[-1])

        # ------------------------------------------------------------------
        # 1 — STRUCTURE H1
        # ------------------------------------------------------------------
        struct = analyze_structure(df_h1)
        bias = str(struct.get("trend", "")).upper()
        LOGGER.info(f"[EVAL_PRE] STRUCT={struct}")

        oi_series = struct.get("oi_series", None)
        has_oi_struct = oi_series is not None
        LOGGER.info(f"[EVAL_PRE] OI_STRUCT symbol={symbol} has_oi={has_oi_struct}")

        if REQUIRE_STRUCTURE:
            if bias not in ("LONG", "SHORT"):
                LOGGER.info("[EVAL_REJECT] No clear trend (RANGE) and REQUIRE_STRUCTURE=True")
                return _rej("no_clear_trend_range", bias=bias)
        else:
            if bias not in ("LONG", "SHORT"):
                LOGGER.info("[EVAL_PRE] Trend RANGE but REQUIRE_STRUCTURE=False")
                # on continue mais pas de setup clean
                return _rej("no_clear_trend_range", bias=bias)

        # ------------------------------------------------------------------
        # 1B — ENTRY PICK (FVG / OTE)
        # ------------------------------------------------------------------
        entry_type = "MARKET"
        entry = float(entry_mkt)

        if USE_ZONE_ENTRY and ZONE_ENTRY_MODE.upper() != "MARKET":
            pick = _choose_zone_entry(df_h1, struct, bias, entry_mkt)
            entry = float(pick["entry"])
            entry_type = str(pick["type"])
            LOGGER.info(
                f"[EVAL_PRE] ENTRY_PICK type={entry_type} entry_mkt={entry_mkt} entry_used={entry} "
                f"in_zone={pick.get('in_zone')} note={pick.get('note')}"
            )

            if ZONE_ENTRY_MODE.upper() == "REJECT" and not pick.get("in_zone", False) and entry_type != "MARKET":
                # Si tu veux strict: on refuse si pas déjà dans la zone
                return _rej("entry_not_in_zone", entry_type=entry_type, entry_mkt=entry_mkt, entry_used=entry)

        # ------------------------------------------------------------------
        # 2 — H4 ALIGNMENT (HTF veto)
        # ------------------------------------------------------------------
        if REQUIRE_HTF_ALIGN:
            if not htf_trend_ok(df_h4, bias):
                LOGGER.info("[EVAL_REJECT] HTF trend veto (REQUIRE_HTF_ALIGN=True)")
                return _rej("htf_veto", bias=bias)

        # ------------------------------------------------------------------
        # 3 — BOS / BOS_QUALITY
        # ------------------------------------------------------------------
        bos_flag = bool(struct.get("bos", False))
        bos_dir = struct.get("bos_direction", None)
        bos_type = struct.get("bos_type", None)

        bos_q = bos_quality_details(
            df_h1,
            oi_series=oi_series,
            df_liq=df_h1,
            price=entry_mkt,  # qualité BOS se base sur contexte actuel
            direction=bos_dir,
        )
        LOGGER.info(f"[EVAL_PRE] BOS_QUALITY={bos_q} bos_flag={bos_flag} bos_type={bos_type}")

        bos_quality_ok = bool(bos_q.get("ok", True))

        # ------------------------------------------------------------------
        # 4 — INSTITUTIONNEL (Binance)
        # ------------------------------------------------------------------
        inst = await compute_full_institutional_analysis(symbol, bias)
        inst_score = int(inst.get("institutional_score") or 0)
        LOGGER.info(f"[INST_RAW] score={inst_score} details={inst}")

        if inst_score < int(MIN_INST_SCORE):
            LOGGER.info(
                f"[EVAL_REJECT] Institutional score < MIN_INST_SCORE "
                f"({inst_score} < {MIN_INST_SCORE})"
            )
            return _rej("inst_score_low", inst_score=inst_score)

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
                LOGGER.info("[EVAL_REJECT] Momentum not bullish for LONG")
                return _rej("momentum_not_bullish", mom=mom)
            if bias == "SHORT" and mom not in ("BEARISH", "STRONG_BEARISH"):
                LOGGER.info("[EVAL_REJECT] Momentum not bearish for SHORT")
                return _rej("momentum_not_bearish", mom=mom)

        if ext_sig == "OVEREXTENDED_LONG" and bias == "LONG":
            LOGGER.info("[EVAL_REJECT] Extension signal OVEREXTENDED_LONG for LONG bias")
            return _rej("overextended_long")
        if ext_sig == "OVEREXTENDED_SHORT" and bias == "SHORT":
            LOGGER.info("[EVAL_REJECT] Extension signal OVEREXTENDED_SHORT for SHORT bias")
            return _rej("overextended_short")

        # ------------------------------------------------------------------
        # 6 — PREMIUM / DISCOUNT (info)
        # ------------------------------------------------------------------
        discount, premium = compute_premium_discount(df_h1)
        LOGGER.info(f"[EVAL_PRE] PREMIUM={premium} DISCOUNT={discount}")

        # ------------------------------------------------------------------
        # 7 — SL / TP1 / RR
        # ------------------------------------------------------------------
        tick = estimate_tick_from_price(entry)
        exits = _compute_exits(df_h1, entry, bias, tick=tick)
        sl = float(exits["sl"])
        tp1 = float(exits["tp1"])

        rr = _safe_rr(entry, sl, tp1, bias)
        LOGGER.info(
            f"[EVAL_PRE] EXITS entry={entry} sl={sl} tp1={tp1} tick={tick} "
            f"RR={rr} raw_rr={exits['rr_used']} entry_type={entry_type}"
        )

        if rr is None or rr <= 0:
            LOGGER.info("[EVAL_REJECT] Invalid RR (<=0)")
            return _rej("invalid_rr", rr=rr, entry=entry, sl=sl, tp1=tp1)

        # ------------------------------------------------------------------
        # 8 — PATH 1 : BOS_STRICT
        # ------------------------------------------------------------------
        bos_strict_ok = False

        if bos_flag:
            if (not REQUIRE_BOS_QUALITY) or bos_quality_ok:
                if rr >= float(RR_MIN_STRICT) and inst_score >= int(MIN_INST_SCORE):
                    bos_strict_ok = True

        LOGGER.info(
            "[EVAL_PRE] BOS_STRICT_CHECK "
            f"bos_flag={bos_flag}, bos_quality_ok={bos_quality_ok}, "
            f"rr={rr} vs RR_MIN_STRICT={RR_MIN_STRICT}, "
            f"inst_score={inst_score} vs MIN_INST_SCORE={MIN_INST_SCORE}, "
            f"bos_strict_ok={bos_strict_ok}"
        )

        if bos_strict_ok:
            LOGGER.info(f"[EVAL] VALID {symbol} RR={rr} SETUP=BOS_STRICT (DESK_EV_MODE={DESK_EV_MODE})")
            return {
                "valid": True,
                "symbol": symbol,
                "side": "BUY" if bias == "LONG" else "SELL",
                "bias": bias,
                "entry": float(entry),
                "entry_mkt": float(entry_mkt),
                "entry_type": entry_type,
                "sl": float(sl),
                "tp1": float(tp1),
                "tp2": None,
                "rr": float(rr),
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
            good_inst = inst_score >= int(INST_SCORE_DESK_PRIORITY)
            good_rr = rr >= float(RR_MIN_DESK_PRIORITY)

            if bias == "LONG":
                good_comp = comp_score >= 70 and comp_label in ("BULLISH", "SLIGHT_BULLISH")
            else:
                good_comp = comp_score >= 70 and comp_label in ("BEARISH", "SLIGHT_BEARISH")

            inst_continuation_ok = bool(good_inst and good_rr and good_comp)

            LOGGER.info(
                "[EVAL_PRE] INST_CONTINUATION_CHECK "
                f"inst_score={inst_score} vs INST_SCORE_DESK_PRIORITY={INST_SCORE_DESK_PRIORITY}, "
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
                "entry": float(entry),
                "entry_mkt": float(entry_mkt),
                "entry_type": entry_type,
                "sl": float(sl),
                "tp1": float(tp1),
                "tp2": None,
                "rr": float(rr),
                "qty": 1,
                "setup_type": "INST_CONTINUATION",
                "structure": struct,
                "bos_quality": bos_q,
                "institutional": inst,
                "momentum": mom,
                "premium": premium,
                "discount": discount,
            }

        LOGGER.info(
            "[EVAL_REJECT] No setup validated "
            f"(BOS_STRICT_OK={bos_strict_ok}, INST_CONTINUATION_OK={inst_continuation_ok}, DESK_EV_MODE={DESK_EV_MODE})"
        )
        return _rej("no_setup_validated", bos_strict_ok=bos_strict_ok, inst_continuation_ok=inst_continuation_ok)
