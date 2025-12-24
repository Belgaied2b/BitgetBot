# =====================================================================
# analyze_signal.py — Desk Institutional (BOS_STRICT needs OTE/FVG in_zone)
# Returns: valid/side/entry/sl/tp1/rr/setup_type/entry_type/in_zone/institutional
# =====================================================================

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd

logger = logging.getLogger(__name__)

# Desk params (adjust in code if you want)
DESK_EV_MODE = True

MIN_INST_SCORE = 2
RR_MIN_STRICT = 1.5

# “Continuation” is allowed with lower RR if flow is strong
INST_SCORE_DESK_PRIORITY = 2
RR_MIN_DESK_PRIORITY = 1.1

# Zone acceptance: how far current price can be from chosen OTE/FVG entry
ENTRY_ZONE_ATR_TOL = 0.8

# ---------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))

def _macd(close: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    sig = _ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist

def _estimate_tick(price: float) -> float:
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

# ---------------------------------------------------------------------
# Simple structure + zones
# ---------------------------------------------------------------------

def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}
    # try common variants
    for k in ("open", "high", "low", "close", "volume"):
        if k not in cols and k in df.columns:
            cols[k] = k
    # normalize
    for k in ("open", "high", "low", "close"):
        if k not in cols:
            raise ValueError(f"missing column {k}")
        df[k] = pd.to_numeric(df[cols[k]], errors="coerce")
    if "volume" in cols:
        df["volume"] = pd.to_numeric(df[cols["volume"]], errors="coerce").fillna(0.0)
    else:
        df["volume"] = 0.0
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return df

def _last_swings(df: pd.DataFrame, w: int = 4) -> Tuple[Optional[float], Optional[float]]:
    # crude swing: local extrema with window
    highs = df["high"]
    lows = df["low"]
    sh = highs[(highs.shift(w) < highs) & (highs.shift(-w) < highs)]
    sl = lows[(lows.shift(w) > lows) & (lows.shift(-w) > lows)]
    last_sh = float(sh.iloc[-1]) if len(sh) else None
    last_sl = float(sl.iloc[-1]) if len(sl) else None
    return last_sh, last_sl

def _find_fvg_zones(df: pd.DataFrame, direction: str) -> List[Tuple[float, float]]:
    # simple 3-candle imbalance
    zones: List[Tuple[float, float]] = []
    h = df["high"].values
    l = df["low"].values
    for i in range(2, len(df)):
        # bullish gap: high[i-2] < low[i]
        if h[i - 2] < l[i]:
            zones.append((float(h[i - 2]), float(l[i])))
        # bearish gap: low[i-2] > high[i]
        if l[i - 2] > h[i]:
            zones.append((float(h[i]), float(l[i - 2])))
    # pick last few
    zones = zones[-6:]
    if direction == "LONG":
        # bullish zones where low>high[i-2]
        return zones
    else:
        return zones

def _ote_zone(swing_high: float, swing_low: float, direction: str) -> Optional[Tuple[float, float, float]]:
    if swing_high is None or swing_low is None:
        return None
    if swing_high <= swing_low:
        return None
    # fib zone 0.62-0.79
    rng = swing_high - swing_low
    if direction == "LONG":
        z1 = swing_high - 0.79 * rng
        z2 = swing_high - 0.62 * rng
        entry = (z1 + z2) / 2.0
        return (min(z1, z2), max(z1, z2), entry)
    else:
        z1 = swing_low + 0.62 * rng
        z2 = swing_low + 0.79 * rng
        entry = (z1 + z2) / 2.0
        return (min(z1, z2), max(z1, z2), entry)

# ---------------------------------------------------------------------
# Institutional snapshot (best-effort, uses your existing module if present)
# ---------------------------------------------------------------------

def _funding_regime(fr: float) -> str:
    a = abs(fr)
    if a > 0.003:
        return "very_negative" if fr < 0 else "very_positive"
    if a > 0.0005:
        return "negative" if fr < 0 else "positive"
    return "neutral"

def _flow_regime(cvd_slope: float) -> str:
    if cvd_slope > 10:
        return "strong_buy"
    if cvd_slope > 0.5:
        return "buy"
    if cvd_slope < -10:
        return "strong_sell"
    if cvd_slope < -0.5:
        return "sell"
    return "balanced"

def _get_institutional(symbol: str) -> Dict[str, Any]:
    # Try your project module if it exists
    try:
        import institutional_data  # type: ignore
        # common patterns
        for fn in ("get_institutional_snapshot", "fetch_institutional_snapshot", "get_snapshot", "get_institutional_data"):
            if hasattr(institutional_data, fn):
                snap = getattr(institutional_data, fn)(symbol)
                if isinstance(snap, dict) and "institutional_score" in snap:
                    score = int(snap.get("institutional_score") or 0)
                    logger.info("[INST_RAW] score=%s details=%s", score, snap)
                    return snap
    except Exception:
        pass

    # fallback (keeps interface stable)
    snap = {
        "institutional_score": 0,
        "binance_symbol": str(symbol).upper(),
        "available": False,
        "oi": None,
        "oi_slope": 0.0,
        "cvd_slope": 0.0,
        "funding_rate": 0.0,
        "funding_regime": "neutral",
        "crowding_regime": "balanced",
        "flow_regime": "balanced",
        "warnings": ["institutional_unavailable"],
    }
    logger.info("[INST_RAW] score=%s details=%s", snap["institutional_score"], snap)
    return snap

# ---------------------------------------------------------------------
# Signal Analyzer
# ---------------------------------------------------------------------

class SignalAnalyzer:
    async def analyze(self, symbol: str, df_h1: pd.DataFrame, df_h4: pd.DataFrame, macro: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        sym = str(symbol).upper()
        logger.info("[EVAL] ▶ START %s", sym)

        try:
            df1 = _ensure_ohlcv(df_h1)
            df4 = _ensure_ohlcv(df_h4)
        except Exception as e:
            return {"valid": False, "reject_reason": f"bad_df:{e}"}

        # compute indicators
        atr1 = _atr(df1, 14)
        atr = float(atr1.iloc[-1]) if len(atr1) else 0.0
        atr = atr if atr and atr > 0 else float((df1["high"] - df1["low"]).rolling(14).mean().iloc[-1])

        ema20 = _ema(df1["close"], 20)
        ema50 = _ema(df1["close"], 50)
        ema200 = _ema(df1["close"], 200) if len(df1) >= 200 else _ema(df1["close"], 100)
        rsi = _rsi(df1["close"], 14)
        _, _, macd_hist = _macd(df1["close"])

        close = float(df1["close"].iloc[-1])

        # Trend regime (simple but stable)
        trend = "RANGE"
        if float(ema20.iloc[-1]) > float(ema50.iloc[-1]) > float(ema200.iloc[-1]):
            trend = "LONG"
        elif float(ema20.iloc[-1]) < float(ema50.iloc[-1]) < float(ema200.iloc[-1]):
            trend = "SHORT"

        # HTF veto: require H4 trend to not contradict
        ema20_4 = _ema(df4["close"], 20)
        ema50_4 = _ema(df4["close"], 50)
        trend4 = "RANGE"
        if float(ema20_4.iloc[-1]) > float(ema50_4.iloc[-1]):
            trend4 = "LONG"
        elif float(ema20_4.iloc[-1]) < float(ema50_4.iloc[-1]):
            trend4 = "SHORT"

        if trend == "RANGE":
            logger.info("[EVAL_REJECT] no_clear_trend_range")
            return {"valid": False, "reject_reason": "no_clear_trend_range"}

        if trend4 != "RANGE" and trend4 != trend:
            logger.info("[EVAL_REJECT] htf_veto h4=%s h1=%s", trend4, trend)
            return {"valid": False, "reject_reason": "htf_veto"}

        side = "BUY" if trend == "LONG" else "SELL"

        # Institutional snapshot
        inst = _get_institutional(sym)
        inst_score = int(inst.get("institutional_score") or 0)

        logger.info("[EVAL_PRE] OI_STRUCT symbol=%s has_oi=%s", sym, bool(inst.get("oi")))
        if inst.get("available") is False:
            logger.info("[EVAL_WARN] bypass MIN_INST_SCORE (binance unmapped) symbol=%s", sym)

        # Momentum label (timing, not dictator)
        rsi_v = float(rsi.iloc[-1]) if len(rsi) else 50.0
        mh = float(macd_hist.iloc[-1]) if len(macd_hist) else 0.0
        if trend == "LONG":
            mom = "STRONG_BULLISH" if (mh > 0 and rsi_v >= 55) else "BULLISH"
        else:
            mom = "STRONG_BEARISH" if (mh < 0 and rsi_v <= 45) else "BEARISH"
        logger.info("[EVAL_PRE] MOMENTUM=%s", mom)

        # Swings & BOS (simple break of last swing)
        sh, slw = _last_swings(df1, w=4)
        bos_flag = False
        if trend == "LONG" and sh is not None:
            bos_flag = close > (sh + 0.05 * atr)
        if trend == "SHORT" and slw is not None:
            bos_flag = close < (slw - 0.05 * atr)

        # Zones (FVG preferred, else OTE)
        fvg = _find_fvg_zones(df1, trend)
        ote = _ote_zone(sh, slw, trend) if (sh is not None and slw is not None) else None

        entry_type = "MARKET"
        in_zone = False
        entry_used = close
        zone_note = ""

        # choose entry for BOS_STRICT: OTE/FVG
        # pick latest zone closest to price (but must be within ATR tolerance)
        cand_entries: List[Tuple[str, float, float, float]] = []  # (type, z_low, z_high, entry)
        if ote:
            zlow, zhigh, ent = ote
            cand_entries.append(("OTE", float(zlow), float(zhigh), float(ent)))
        if fvg:
            # take last fvg zone
            zl, zh = fvg[-1]
            ent = (zl + zh) / 2.0
            cand_entries.append(("FVG", float(min(zl, zh)), float(max(zl, zh)), float(ent)))

        if cand_entries:
            # pick by smallest distance to current close
            cand_entries.sort(key=lambda x: abs(close - x[3]))
            et, zlow, zhigh, ent = cand_entries[0]
            dist = abs(close - ent)
            in_zone = (close >= zlow and close <= zhigh) or (dist <= ENTRY_ZONE_ATR_TOL * atr)
            entry_used = ent
            entry_type = et
            zone_note = "ok" if in_zone else "zone_too_far"
        else:
            zone_note = "no_zone"

        logger.info(
            "[EVAL_PRE] ENTRY_PICK type=%s entry_mkt=%s entry_used=%s in_zone=%s note=%s dist=%s atr=%s",
            entry_type, close, entry_used, in_zone, zone_note, abs(close - entry_used), atr
        )

        # SL: beyond last swing + small buffer
        if trend == "LONG":
            base = slw if slw is not None else float(df1["low"].tail(20).min())
            sl_price = float(base - 0.20 * atr)
        else:
            base = sh if sh is not None else float(df1["high"].tail(20).max())
            sl_price = float(base + 0.20 * atr)

        # risk and TP1 for desk RR targets
        risk = abs(sl_price - entry_used)
        if risk <= 0:
            return {"valid": False, "reject_reason": "bad_risk"}

        # Choose RR target by setup candidate
        rr_target_strict = RR_MIN_STRICT
        rr_target_cont = max(RR_MIN_DESK_PRIORITY, 1.35)

        # Default: build TP1 for strict RR first
        if trend == "LONG":
            tp1_price = entry_used + rr_target_strict * risk
        else:
            tp1_price = entry_used - rr_target_strict * risk

        rr = abs(tp1_price - entry_used) / max(1e-12, risk)
        tick = _estimate_tick(entry_used)

        logger.info(
            "[EVAL_PRE] EXITS entry=%s sl=%s tp1=%s tick=%s RR=%s raw_rr=%s entry_type=%s",
            entry_used, sl_price, tp1_price, tick, rr, rr, entry_type
        )

        # Setup validation
        # BOS_STRICT requires: BOS + RR>=1.5 + (inst_score>=2 unless unmapped) + in_zone
        min_inst_ok = True
        if inst.get("available") is True:
            min_inst_ok = (inst_score >= MIN_INST_SCORE)

        bos_strict_ok = bool(bos_flag and rr >= RR_MIN_STRICT and min_inst_ok and in_zone)
        logger.info(
            "[EVAL_PRE] BOS_STRICT_CHECK bos_flag=%s rr=%s vs RR_MIN_STRICT=%s inst_score=%s vs MIN_INST_SCORE=%s in_zone=%s bos_strict_ok=%s",
            bos_flag, rr, RR_MIN_STRICT, inst_score, MIN_INST_SCORE, in_zone, bos_strict_ok
        )

        # INST_CONTINUATION: strong flow + momentum aligned, RR >= RR_MIN_DESK_PRIORITY, can be MARKET
        flow = str(inst.get("flow_regime") or _flow_regime(float(inst.get("cvd_slope") or 0.0)))
        cont_mom_ok = (mom.startswith("STRONG") or mom in ("BULLISH", "BEARISH"))
        cont_flow_ok = (flow in ("strong_buy", "buy") and trend == "LONG") or (flow in ("strong_sell", "sell") and trend == "SHORT")
        inst_continuation_ok = (inst_score >= INST_SCORE_DESK_PRIORITY and rr >= RR_MIN_DESK_PRIORITY and cont_mom_ok and cont_flow_ok)

        logger.info(
            "[EVAL_PRE] INST_CONTINUATION_CHECK inst_score=%s vs INST_SCORE_DESK_PRIORITY=%s rr=%s vs RR_MIN_DESK_PRIORITY=%s flow=%s mom=%s ok=%s",
            inst_score, INST_SCORE_DESK_PRIORITY, rr, RR_MIN_DESK_PRIORITY, flow, mom, inst_continuation_ok
        )

        setup_type = None
        if DESK_EV_MODE:
            if bos_strict_ok:
                setup_type = "BOS_STRICT"
            elif inst_continuation_ok:
                setup_type = "INST_CONTINUATION"
                # continuation can be market; if zone too far, use market entry instead
                if not in_zone:
                    entry_type = "MARKET"
                    entry_used = close
                    # recompute TP1 for continuation RR target
                    if trend == "LONG":
                        tp1_price = entry_used + rr_target_cont * risk
                    else:
                        tp1_price = entry_used - rr_target_cont * risk
                    rr = abs(tp1_price - entry_used) / max(1e-12, risk)
            else:
                logger.info("[EVAL_REJECT] No setup validated (BOS_STRICT_OK=%s, INST_CONTINUATION_OK=%s, DESK_EV_MODE=%s)", bos_strict_ok, inst_continuation_ok, DESK_EV_MODE)
                return {"valid": False, "reject_reason": "no_setup_validated", "institutional": inst}
        else:
            # fallback: allow anything with RR ok + direction
            setup_type = "LEGACY"

        # Final strict desk gate: BOS_STRICT MUST be zone entry (OTE/FVG)
        if setup_type == "BOS_STRICT" and not in_zone:
            logger.info("[EVAL_REJECT] BOS_STRICT but entry not in zone (zone_too_far)")
            return {"valid": False, "reject_reason": "zone_too_far", "institutional": inst}

        logger.info("[EVAL] VALID %s RR=%s SETUP=%s (DESK_EV_MODE=%s)", sym, rr, setup_type, DESK_EV_MODE)

        return {
            "valid": True,
            "side": side,
            "setup_type": setup_type,
            "entry_type": entry_type,
            "in_zone": bool(in_zone),
            "entry": float(entry_used),
            "sl": float(sl_price),
            "tp1": float(tp1_price),
            "rr": float(rr),
            "institutional": inst,
        }
