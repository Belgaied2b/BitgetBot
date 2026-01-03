# =====================================================================
# analyze_signal.py — Desk institutional analyzer (TTL-compatible setup)
# - Integrates new institutional_data.py gate (raw_components_sum) + ok_count
# - 2-pass institutional policy (LIGHT then NORMAL/FULL)
# - Liquidations PASS2-only by default (anti-ban)
# - TTL-compatible setup_type suffixing for scanner.py
# =====================================================================

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple, List

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
    # NEW (grading / pass2 policy)
    PASS2_ONLY_FOR_PRIORITY,
    priority_at_least,
)

LOGGER = logging.getLogger(__name__)

REQUIRED_COLS = ("open", "high", "low", "close", "volume")

# Optional: allow technical fallback if Binance is banned/down (default False)
ALLOW_TECH_FALLBACK_WHEN_INST_DOWN = str(os.getenv("ALLOW_TECH_FALLBACK_WHEN_INST_DOWN", "0")).strip() == "1"

# =====================================================================
# ✅ scanner.py compatibility helper (TTL policy uses setup string)
# =====================================================================
def _setup_ttl_compatible(setup_type: str, entry_type: str) -> str:
    """
    scanner.py uses _entry_ttl_s(entry_type, setup) but looks for OTE/FVG/RAID in setup.
    We keep the canonical setup type, and append a suffix when needed so TTL works.

    Rules:
      - If entry_type contains RAID => suffix _RAID (do NOT add _FVG, or TTL would pick FVG before RAID)
      - Else if entry_type contains OTE => suffix _OTE
      - Else if entry_type contains FVG => suffix _FVG
    """
    base = str(setup_type or "").strip()
    if not base:
        base = "OTHER"
    s = base.upper()
    et = str(entry_type or "").upper()

    if "RAID" in et and ("RAID" not in s and "SWEEP" not in s):
        return f"{base}_RAID"
    if "OTE" in et and "OTE" not in s:
        return f"{base}_OTE"
    if "FVG" in et and "FVG" not in s:
        return f"{base}_FVG"
    return base


# =====================================================================
# ✅ Liquidations + 2-pass institutional policy
# =====================================================================
def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip() == "1"


def _env_str(name: str, default: str) -> str:
    return str(os.getenv(name, default)).strip()


# Liquidations enabled (global)
INST_ENABLE_LIQUIDATIONS = _env_flag("INST_ENABLE_LIQUIDATIONS", "1")  # default ON

# If 1: only call liquidations on PASS2 (recommended)
INST_LIQ_PASS2_ONLY = _env_flag("INST_LIQ_PASS2_ONLY", "1")  # default ON (anti-ban)

# Pass modes (override institutional_data INST_MODE)
INST_PASS1_MODE = _env_str("INST_PASS1_MODE", "LIGHT").upper()
INST_PASS2_MODE = _env_str("INST_PASS2_MODE", "NORMAL").upper()  # NORMAL by default (FULL if you want)

# Enable/disable pass2 completely
INST_PASS2_ENABLED = _env_flag("INST_PASS2_ENABLED", "1")

# Only do pass2 when gate >= this value (1 = “un peu intéressant”)
try:
    INST_PASS2_MIN_GATE = int(_env_str("INST_PASS2_MIN_GATE", "1"))
except Exception:
    INST_PASS2_MIN_GATE = 1

if INST_PASS1_MODE not in ("LIGHT", "NORMAL", "FULL"):
    INST_PASS1_MODE = "LIGHT"
if INST_PASS2_MODE not in ("LIGHT", "NORMAL", "FULL"):
    INST_PASS2_MODE = "NORMAL"


# =====================================================================
# Priority helpers (A→E)
# =====================================================================
_PRIORITY_ORDER = ("A", "B", "C", "D", "E")
_PRIORITY_IDX = {p: i for i, p in enumerate(_PRIORITY_ORDER)}  # A=0 ... E=4


def _prio_norm(p: str, default: str = "C") -> str:
    s = (p or "").strip().upper()
    return s if s in _PRIORITY_IDX else default


def _downgrade(p: str, steps: int = 1) -> str:
    p = _prio_norm(p)
    i = _PRIORITY_IDX[p]
    i2 = min(len(_PRIORITY_ORDER) - 1, i + max(0, int(steps)))
    return _PRIORITY_ORDER[i2]


def _upgrade(p: str, steps: int = 1) -> str:
    p = _prio_norm(p)
    i = _PRIORITY_IDX[p]
    i2 = max(0, i - max(0, int(steps)))
    return _PRIORITY_ORDER[i2]


def _pre_grade_candidate(
    *,
    bos_flag: bool,
    raid_ok: bool,
    sweep_ok: bool,
    mom: str,
    comp_score: float,
    used_bias_fallback: bool,
    ext_sig: str,
) -> Tuple[str, List[str]]:
    """
    PRE-grade "cheap" (avant PASS2) pour limiter les calls institutionnels lourds.
    """
    reasons: List[str] = []
    m = (mom or "").upper()
    ext = (ext_sig or "").upper()

    # Base bucket from structure triggers
    if bos_flag:
        p = "B"
        reasons.append("pre:bos_trigger")
    elif raid_ok or sweep_ok:
        p = "C"
        reasons.append("pre:raid_or_sweep")
    else:
        p = "D"
        reasons.append("pre:no_trigger")

    # Momentum & composite bump
    if m in ("STRONG_BULLISH", "STRONG_BEARISH"):
        p = _upgrade(p, 1)
        reasons.append("pre:strong_momentum")
    if float(comp_score) >= 67.0:
        p = _upgrade(p, 1)
        reasons.append("pre:high_composite")
    elif float(comp_score) <= 45.0:
        p = _downgrade(p, 1)
        reasons.append("pre:weak_composite")

    # Penalize uncertainty
    if used_bias_fallback:
        p = _downgrade(p, 1)
        reasons.append("pre:bias_fallback")

    # Penalize overextension (still can be valid, but less worth PASS2)
    if ext.startswith("OVEREXTENDED"):
        p = _downgrade(p, 1)
        reasons.append("pre:overextended")

    return _prio_norm(p), reasons


def _final_grade(
    *,
    setup_type: str,
    setup_variant: str,
    rr: float,
    gate: int,
    ok_count: int,
    inst_available: bool,
    bos_quality_ok: bool,
    used_bias_fallback: bool,
    entry_type: str,
    ext_sig: str,
    unfavorable_market: bool,
    mom: str,
    soft_vetoes: Optional[List[str]] = None,
) -> Tuple[str, List[str]]:
    """
    Grade final A→E + reasons. (A meilleur)
    """
    reasons: List[str] = []
    st = (setup_type or "").upper()
    sv = (setup_variant or "").upper()
    et = (entry_type or "").upper()
    ext = (ext_sig or "").upper()
    m = (mom or "").upper()
    svs = soft_vetoes or []

    # Base from setup type
    if st == "BOS_STRICT":
        if "RR_STRICT" in sv and int(gate) >= int(MIN_INST_SCORE) and bos_quality_ok:
            p = "A"
            reasons.append("setup:BOS_STRICT+RR_STRICT")
        else:
            p = "B"
            reasons.append("setup:BOS_STRICT")
    elif st in ("RAID_DISPLACEMENT", "LIQ_SWEEP"):
        if int(gate) >= int(INST_SCORE_DESK_PRIORITY):
            p = "B"
            reasons.append(f"setup:{st}+inst_ok")
        else:
            p = "C"
            reasons.append(f"setup:{st}")
    elif st == "INST_CONTINUATION":
        p = "C"
        reasons.append("setup:INST_CONTINUATION")
    else:
        p = "D"
        reasons.append("setup:OTHER")

    reasons.append(f"rr:{float(rr):.3f}")
    reasons.append(f"inst_gate:{int(gate)} ok_count:{int(ok_count)}")
    if bos_quality_ok:
        reasons.append("bos_quality:ok")

    if not inst_available:
        p = _downgrade(p, 1)
        reasons.append("downgrade:inst_unavailable")

    if used_bias_fallback:
        p = _downgrade(p, 1)
        reasons.append("downgrade:bias_fallback")

    if et == "MARKET":
        p = _downgrade(p, 1)
        reasons.append("downgrade:market_entry")

    if unfavorable_market:
        p = _downgrade(p, 1)
        reasons.append("downgrade:premium_discount_veto_path")

    if ext.startswith("OVEREXTENDED"):
        p = _downgrade(p, 1)
        reasons.append("downgrade:overextended")

    if svs:
        p = _downgrade(p, 1)
        reasons.append(f"downgrade:soft_vetoes({','.join(svs)})")

    if m in ("STRONG_BULLISH", "STRONG_BEARISH"):
        p = _upgrade(p, 1)
        reasons.append("upgrade:strong_momentum")

    return _prio_norm(p), reasons


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
# Tick helpers (for post-adjust rounding hygiene)
# =====================================================================
def _safe_tick(tick: float) -> float:
    try:
        t = float(tick)
        if not np.isfinite(t) or t <= 0:
            return 0.0
        return t
    except Exception:
        return 0.0


def _floor_to_tick(price: float, tick: float) -> float:
    t = _safe_tick(tick)
    if t <= 0:
        return float(price)
    p = float(price)
    return float(np.floor(p / t) * t)


def _ceil_to_tick(price: float, tick: float) -> float:
    t = _safe_tick(tick)
    if t <= 0:
        return float(price)
    p = float(price)
    return float(np.ceil(p / t) * t)


def _round_sl_for_side(sl: float, entry: float, bias: str, tick: float) -> float:
    """
    Align SL to tick grid safely:
      - LONG SL rounds DOWN
      - SHORT SL rounds UP
    Also re-enforces SL on correct side of entry if something broke after adjustments.
    """
    b = (bias or "").upper()
    t = _safe_tick(tick)
    sl_f = float(sl)
    entry_f = float(entry)

    if b == "LONG":
        sl_f = _floor_to_tick(sl_f, t)
        if sl_f >= entry_f:
            gap = max((2.0 * t if t > 0 else 0.0), abs(entry_f) * 0.001)
            sl_f = _floor_to_tick(entry_f - gap, t)
        return float(sl_f)

    if b == "SHORT":
        sl_f = _ceil_to_tick(sl_f, t)
        if sl_f <= entry_f:
            gap = max((2.0 * t if t > 0 else 0.0), abs(entry_f) * 0.001)
            sl_f = _ceil_to_tick(entry_f + gap, t)
        return float(sl_f)

    return float(sl_f)


# =====================================================================
# Composite helpers
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


def _composite_bias_fallback(score: float, label: str, mom: str) -> Optional[str]:
    """
    More permissive fallback to reduce `no_bias_fallback_for_inst`.
    """
    try:
        s = float(score)
        lab = str(label or "").upper()
        m = str(mom or "").upper()

        if m in ("STRONG_BULLISH", "BULLISH"):
            return "LONG"
        if m in ("STRONG_BEARISH", "BEARISH"):
            return "SHORT"

        if (s >= 58.0) or ("BULL" in lab and s >= 55.0):
            return "LONG"
        if (s <= 42.0) or ("BEAR" in lab and s <= 45.0):
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


def _compute_exits(
    df: pd.DataFrame,
    entry: float,
    bias: str,
    tick: float,
    *,
    setup: Optional[str] = None,
    entry_type: Optional[str] = None,
    htf_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Uses stops.py v3 policy-aware stops:
      protective_stop_*(..., setup=..., entry_type=..., htf_df=...)
    """
    if bias == "LONG":
        sl, meta = protective_stop_long(
            df,
            entry,
            tick,
            return_meta=True,
            setup=setup,
            entry_type=entry_type,
            htf_df=htf_df,
        )
    else:
        sl, meta = protective_stop_short(
            df,
            entry,
            tick,
            return_meta=True,
            setup=setup,
            entry_type=entry_type,
            htf_df=htf_df,
        )

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
      3) OTE pullback
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
# Institutional gate helpers (NEW score integration)
# =====================================================================
def _inst_ok_count(inst: Dict[str, Any]) -> int:
    comp = inst.get("score_components") or {}
    try:
        return int(sum(1 for v in comp.values() if int(v) > 0))
    except Exception:
        return 0


def _inst_gate_value(inst: Dict[str, Any]) -> Tuple[int, int]:
    """
    Returns:
      gate_score = integer used for gating (prefers raw sum of components)
      ok_count   = number of components that are >0 (debug only)
    """
    try:
        inst_score = int(inst.get("institutional_score") or 0)
        meta = inst.get("score_meta") or {}
        raw_sum = meta.get("raw_components_sum")
        gate = int(raw_sum) if raw_sum is not None else inst_score
        ok_count = int(meta.get("ok_count") or _inst_ok_count(inst))
        return gate, ok_count
    except Exception:
        return 0, _inst_ok_count(inst)


def _desk_inst_gate_override(inst: Dict[str, Any], bias: str, gate: int) -> Tuple[int, Optional[str]]:
    """
    If flow is clearly strong and crowding is not risky, allow bumping gate to MIN_INST_SCORE.
    """
    try:
        if gate >= int(MIN_INST_SCORE):
            return gate, None

        b = (bias or "").upper()
        flow = str(inst.get("flow_regime") or "").lower()
        crowd = str(inst.get("crowding_regime") or "").lower()
        cvd = inst.get("cvd_slope")
        cvd = float(cvd) if cvd is not None else 0.0

        if "risky" in crowd or "crowded" in crowd:
            return gate, None

        if b == "LONG" and flow == "strong_buy" and cvd >= 0.2:
            return int(MIN_INST_SCORE), "override_flow_strong_buy"
        if b == "SHORT" and flow == "strong_sell" and cvd <= -0.2:
            return int(MIN_INST_SCORE), "override_flow_strong_sell"
        return gate, None
    except Exception:
        return gate, None


def _reject(reason: str, **extra: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"valid": False, "reject_reason": reason}
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
            return _reject("bad_df_h1")

        if not _ensure_ohlcv(df_h4) or len(df_h4) < 80:
            LOGGER.info("[EVAL_REJECT] %s bad_df_h4", symbol)
            return _reject("bad_df_h4")

        # ---- cheap direction hints first ----
        mom = institutional_momentum(df_h1)
        comp = composite_momentum(df_h1)
        vol_reg = volatility_regime(df_h1)
        ext_sig = extension_signal(df_h1)

        comp_score = float(comp.get("score", 50.0)) if isinstance(comp, dict) else 50.0
        comp_label = str(comp.get("label", "NEUTRAL")) if isinstance(comp, dict) else "NEUTRAL"

        # ---- structure ----
        struct = analyze_structure(df_h1)
        struct_trend = str(struct.get("trend", "")).upper()
        bias = str(struct_trend).upper()

        # Fallback bias if range
        used_bias_fallback = False
        if bias not in ("LONG", "SHORT"):
            fb = _composite_bias_fallback(comp_score, comp_label, mom)
            if fb is not None:
                bias = fb
                used_bias_fallback = True
                LOGGER.info("[EVAL_PRE] %s bias_fallback_for_inst=%s", symbol, bias)
            else:
                LOGGER.info("[EVAL_REJECT] %s no_bias_fallback_for_inst", symbol)
                return _reject("no_bias_fallback_for_inst", structure=struct, momentum=mom, composite=comp)

        LOGGER.info(
            "[EVAL_PRE] %s STRUCT trend=%s bos=%s choch=%s cos=%s",
            symbol, struct_trend, struct.get("bos"), struct.get("choch"), struct.get("cos")
        )
        LOGGER.info("[EVAL_PRE] %s MOMENTUM=%s", symbol, mom)
        LOGGER.info("[EVAL_PRE] %s MOMENTUM_COMPOSITE score=%.2f label=%s", symbol, comp_score, comp_label)
        LOGGER.info("[EVAL_PRE] %s VOL_REGIME=%s EXTENSION=%s", symbol, vol_reg, ext_sig)

        # ---------------------------------------------------------------
        # PRE-FILTERS (cheap) BEFORE institutional call (anti-ban)
        # (In DESK_EV_MODE, HTF/MOMENTUM veto are treated as soft vetoes)
        # ---------------------------------------------------------------
        soft_vetoes: List[str] = []

        bos_flag = bool(struct.get("bos", False))
        raid_ok = bool(isinstance(struct.get("raid_displacement"), dict) and struct["raid_displacement"].get("ok"))
        liq_sweep_pre = liquidity_sweep_details(df_h1, bias, lookback=180)
        sweep_ok = bool(isinstance(liq_sweep_pre, dict) and liq_sweep_pre.get("ok"))

        # REQUIRE_STRUCTURE: keep strict only outside desk mode.
        if REQUIRE_STRUCTURE and used_bias_fallback and (not DESK_EV_MODE):
            LOGGER.info("[EVAL_REJECT] %s no_clear_trend_range", symbol)
            return _reject("no_clear_trend_range", structure=struct, momentum=mom, composite=comp)

        # HTF alignment gating
        if REQUIRE_HTF_ALIGN and bias in ("LONG", "SHORT"):
            if not htf_trend_ok(df_h4, bias):
                if DESK_EV_MODE:
                    soft_vetoes.append("htf_veto")
                    LOGGER.info("[EVAL_PRE] %s HTF_VETO (soft, desk_mode)", symbol)
                else:
                    LOGGER.info("[EVAL_REJECT] %s htf_veto", symbol)
                    return _reject("htf_veto", structure=struct)

        # In non-desk mode, require at least one structural trigger (BOS or RAID or SWEEP)
        if (not DESK_EV_MODE) and (not (bos_flag or raid_ok or sweep_ok)):
            LOGGER.info("[EVAL_REJECT] %s no_structure_trigger", symbol)
            return _reject("no_structure_trigger", structure=struct, sweep=liq_sweep_pre)

        # Momentum gating (cheap)
        if REQUIRE_MOMENTUM:
            if bias == "LONG" and mom not in ("BULLISH", "STRONG_BULLISH"):
                if DESK_EV_MODE:
                    soft_vetoes.append("momentum_not_bullish")
                    LOGGER.info("[EVAL_PRE] %s MOM_VETO (soft, desk_mode)", symbol)
                else:
                    LOGGER.info("[EVAL_REJECT] %s momentum_not_bullish", symbol)
                    return _reject("momentum_not_bullish", structure=struct)
            if bias == "SHORT" and mom not in ("BEARISH", "STRONG_BEARISH"):
                if DESK_EV_MODE:
                    soft_vetoes.append("momentum_not_bearish")
                    LOGGER.info("[EVAL_PRE] %s MOM_VETO (soft, desk_mode)", symbol)
                else:
                    LOGGER.info("[EVAL_REJECT] %s momentum_not_bearish", symbol)
                    return _reject("momentum_not_bearish", structure=struct)

        # ===============================================================
        # PRE-PRIORITY (cheap) → controls PASS2
        # ===============================================================
        pre_priority, pre_priority_reasons = _pre_grade_candidate(
            bos_flag=bos_flag,
            raid_ok=raid_ok,
            sweep_ok=sweep_ok,
            mom=mom,
            comp_score=comp_score,
            used_bias_fallback=used_bias_fallback,
            ext_sig=ext_sig,
        )

        pass2_allowed_by_priority = priority_at_least(pre_priority, PASS2_ONLY_FOR_PRIORITY)

        LOGGER.info(
            "[EVAL_PRE] %s PRE_PRIORITY=%s (min_for_pass2=%s) pass2_allowed=%s reasons=%s soft_vetoes=%s",
            symbol,
            pre_priority,
            PASS2_ONLY_FOR_PRIORITY,
            pass2_allowed_by_priority,
            pre_priority_reasons,
            soft_vetoes,
        )

        # ===============================================================
        # 1) INSTITUTIONAL — 2-PASS (anti-ban) + Liquidations
        # ===============================================================
        inst: Dict[str, Any]

        # Pass1: cheap, no liquidations by default
        pass1_liq = bool(INST_ENABLE_LIQUIDATIONS and (not INST_LIQ_PASS2_ONLY))
        try:
            inst = await compute_full_institutional_analysis(
                symbol,
                bias,
                mode=INST_PASS1_MODE,
                include_liquidations=pass1_liq,
            )
        except Exception as e:
            inst = {
                "institutional_score": 0,
                "binance_symbol": None,
                "available": False,
                "warnings": [f"inst_exception:{e}"],
                "score_components": {"flow": 0, "oi": 0, "crowding": 0, "orderbook": 0},
                "score_meta": {"raw_components_sum": 0, "ok_count": 0, "mode": INST_PASS1_MODE},
            }

        available = bool(inst.get("available", False))
        binance_symbol = inst.get("binance_symbol")

        gate, ok_count = _inst_gate_value(inst)
        gate, override = _desk_inst_gate_override(inst, bias, int(gate))

        mode_eff_1 = (inst.get("score_meta") or {}).get("mode") or INST_PASS1_MODE
        liq_eff_1 = inst.get("liquidation_intensity") is not None

        LOGGER.info(
            "[INST_RAW] %s pass=1 mode=%s liq_req=%s liq_ok=%s inst_score=%s ok_count=%s gate=%s override=%s available=%s binance_symbol=%s bias=%s comps=%s",
            symbol,
            mode_eff_1,
            pass1_liq,
            liq_eff_1,
            inst.get("institutional_score"),
            ok_count,
            gate,
            override,
            available,
            binance_symbol,
            bias,
            inst.get("available_components") or [],
        )

        # If inst not available: strict reject unless fallback enabled
        if (not available) or (binance_symbol is None):
            if ALLOW_TECH_FALLBACK_WHEN_INST_DOWN:
                LOGGER.warning("[INST_DOWN] %s tech_fallback_enabled", symbol)
            else:
                LOGGER.info("[EVAL_REJECT] %s inst_unavailable", symbol)
                return _reject(
                    "inst_unavailable",
                    institutional=inst,
                    structure=struct,
                    momentum=mom,
                    composite=comp,
                    pre_priority=pre_priority,
                    pre_priority_reasons=pre_priority_reasons,
                    soft_vetoes=soft_vetoes,
                )

        # Pass2: only for “candidates” + liquidations ON here
        do_pass2 = bool(
            INST_PASS2_ENABLED
            and pass2_allowed_by_priority
            and available
            and (binance_symbol is not None)
            and (int(gate) >= int(INST_PASS2_MIN_GATE) or bool(DESK_EV_MODE))
        )

        if do_pass2:
            pass2_liq = bool(INST_ENABLE_LIQUIDATIONS)  # liquidations ON here
            try:
                inst2 = await compute_full_institutional_analysis(
                    symbol,
                    bias,
                    mode=INST_PASS2_MODE,
                    include_liquidations=pass2_liq,
                )
            except Exception as e:
                inst2 = {
                    "institutional_score": 0,
                    "binance_symbol": binance_symbol,
                    "available": False,
                    "warnings": [f"inst_pass2_exception:{e}"],
                    "score_components": {"flow": 0, "oi": 0, "crowding": 0, "orderbook": 0},
                    "score_meta": {"raw_components_sum": 0, "ok_count": 0, "mode": INST_PASS2_MODE},
                }

            if isinstance(inst2, dict) and bool(inst2.get("available")) and (inst2.get("binance_symbol") is not None):
                inst = inst2

                available = bool(inst.get("available", False))
                binance_symbol = inst.get("binance_symbol")

                gate, ok_count = _inst_gate_value(inst)
                gate, override = _desk_inst_gate_override(inst, bias, int(gate))

                mode_eff_2 = (inst.get("score_meta") or {}).get("mode") or INST_PASS2_MODE
                liq_eff_2 = inst.get("liquidation_intensity") is not None

                LOGGER.info(
                    "[INST_RAW] %s pass=2 mode=%s liq_req=%s liq_ok=%s inst_score=%s ok_count=%s gate=%s override=%s available=%s binance_symbol=%s bias=%s comps=%s",
                    symbol,
                    mode_eff_2,
                    pass2_liq,
                    liq_eff_2,
                    inst.get("institutional_score"),
                    ok_count,
                    gate,
                    override,
                    available,
                    binance_symbol,
                    bias,
                    inst.get("available_components") or [],
                )
        else:
            LOGGER.info(
                "[INST_RAW] %s pass=2 SKIP (enabled=%s allowed_by_priority=%s pre_priority=%s min_for_pass2=%s gate=%s min_gate=%s desk=%s available=%s)",
                symbol,
                INST_PASS2_ENABLED,
                pass2_allowed_by_priority,
                pre_priority,
                PASS2_ONLY_FOR_PRIORITY,
                gate,
                INST_PASS2_MIN_GATE,
                DESK_EV_MODE,
                available,
            )

        # Hard gate (if inst up)
        if (not ALLOW_TECH_FALLBACK_WHEN_INST_DOWN) and int(gate) < int(MIN_INST_SCORE):
            LOGGER.info("[EVAL_REJECT] %s inst_gate_low gate=%s < %s", symbol, gate, MIN_INST_SCORE)
            return _reject(
                "inst_gate_low",
                institutional=inst,
                structure=struct,
                inst_gate=int(gate),
                ok_count=int(ok_count),
                pre_priority=pre_priority,
                pre_priority_reasons=pre_priority_reasons,
                soft_vetoes=soft_vetoes,
            )

        # ===============================================================
        # 2) STRUCTURE / QUALITY / ENTRY / EXITS
        # ===============================================================
        entry_mkt = _last_close(df_h1)

        bos_dir = struct.get("bos_direction", None)
        bos_q = bos_quality_details(
            df_h1,
            oi_series=struct.get("oi_series", None),
            df_liq=df_h1,
            price=entry_mkt,
            direction=bos_dir,
        )
        bos_quality_ok = bool(bos_q.get("ok", True))

        LOGGER.info(
            "[EVAL_PRE] %s BOS_QUALITY ok=%s score=%s reasons=%s bos_flag=%s bos_dir=%s",
            symbol, bos_quality_ok, bos_q.get("score"), bos_q.get("reasons"), bos_flag, bos_dir
        )

        if REQUIRE_BOS_QUALITY and not bos_quality_ok:
            LOGGER.info("[EVAL_REJECT] %s bos_quality_low", symbol)
            return _reject(
                "bos_quality_low",
                institutional=inst,
                structure=struct,
                bos_quality=bos_q,
                pre_priority=pre_priority,
                pre_priority_reasons=pre_priority_reasons,
                soft_vetoes=soft_vetoes,
            )

        premium, discount = compute_premium_discount(df_h1)
        LOGGER.info("[EVAL_PRE] %s PREMIUM=%s DISCOUNT=%s", symbol, premium, discount)

        # reuse precomputed sweep
        liq_sweep = liq_sweep_pre

        entry_pick = _pick_entry(df_h1, struct, bias)
        entry = float(entry_pick["entry_used"])
        entry_type = str(entry_pick["entry_type"])
        atr14 = float(entry_pick.get("atr") or _atr(df_h1, 14))

        LOGGER.info(
            "[EVAL_PRE] %s ENTRY_PICK type=%s entry_mkt=%s entry_used=%s in_zone=%s note=%s atr=%.6g",
            symbol,
            entry_type,
            entry_pick.get("entry_mkt"),
            entry_pick.get("entry_used"),
            entry_pick.get("in_zone"),
            entry_pick.get("note"),
            atr14,
        )

        # Extension hygiene
        if ext_sig == "OVEREXTENDED_LONG" and bias == "LONG":
            if entry_type == "MARKET":
                LOGGER.info("[EVAL_REJECT] %s overextended_long_market", symbol)
                return _reject(
                    "overextended_long_market",
                    institutional=inst,
                    structure=struct,
                    entry_pick=entry_pick,
                    pre_priority=pre_priority,
                    pre_priority_reasons=pre_priority_reasons,
                    soft_vetoes=soft_vetoes,
                )
            if not _entry_pullback_ok(entry, entry_mkt, bias, atr14):
                LOGGER.info("[EVAL_REJECT] %s overextended_long_no_pullback", symbol)
                return _reject(
                    "overextended_long_no_pullback",
                    institutional=inst,
                    structure=struct,
                    entry_pick=entry_pick,
                    pre_priority=pre_priority,
                    pre_priority_reasons=pre_priority_reasons,
                    soft_vetoes=soft_vetoes,
                )

        if ext_sig == "OVEREXTENDED_SHORT" and bias == "SHORT":
            if entry_type == "MARKET":
                LOGGER.info("[EVAL_REJECT] %s overextended_short_market", symbol)
                return _reject(
                    "overextended_short_market",
                    institutional=inst,
                    structure=struct,
                    entry_pick=entry_pick,
                    pre_priority=pre_priority,
                    pre_priority_reasons=pre_priority_reasons,
                    soft_vetoes=soft_vetoes,
                )
            if not _entry_pullback_ok(entry, entry_mkt, bias, atr14):
                LOGGER.info("[EVAL_REJECT] %s overextended_short_no_pullback", symbol)
                return _reject(
                    "overextended_short_no_pullback",
                    institutional=inst,
                    structure=struct,
                    entry_pick=entry_pick,
                    pre_priority=pre_priority,
                    pre_priority_reasons=pre_priority_reasons,
                    soft_vetoes=soft_vetoes,
                )

        unfavorable_market = bool(entry_type == "MARKET" and ((bias == "LONG" and premium) or (bias == "SHORT" and discount)))
        if unfavorable_market and (not DESK_EV_MODE):
            if bias == "LONG" and premium:
                LOGGER.info("[EVAL_REJECT] %s long_in_premium_market", symbol)
                return _reject(
                    "long_in_premium_market",
                    institutional=inst,
                    structure=struct,
                    entry_pick=entry_pick,
                    pre_priority=pre_priority,
                    pre_priority_reasons=pre_priority_reasons,
                    soft_vetoes=soft_vetoes,
                )
            if bias == "SHORT" and discount:
                LOGGER.info("[EVAL_REJECT] %s short_in_discount_market", symbol)
                return _reject(
                    "short_in_discount_market",
                    institutional=inst,
                    structure=struct,
                    entry_pick=entry_pick,
                    pre_priority=pre_priority,
                    pre_priority_reasons=pre_priority_reasons,
                    soft_vetoes=soft_vetoes,
                )

        # --------------------------
        # Exits (policy-aware SL v3)
        # --------------------------
        setup_hint: Optional[str] = None
        if bos_flag:
            setup_hint = "BOS_STRICT"
        elif DESK_EV_MODE and raid_ok:
            setup_hint = "RAID_DISPLACEMENT"
        elif DESK_EV_MODE and sweep_ok:
            setup_hint = "LIQ_SWEEP"
        elif DESK_EV_MODE:
            setup_hint = "INST_CONTINUATION"

        tick = estimate_tick_from_price(entry)
        exits = _compute_exits(
            df_h1,
            entry,
            bias,
            tick=tick,
            setup=setup_hint,
            entry_type=entry_type,
            htf_df=df_h4,
        )

        sl = float(exits["sl"])
        tp1 = float(exits["tp1"])
        rr = _safe_rr(entry, sl, tp1, bias)

        # SL hygiene (push beyond sweep / eq levels) + keep tick rounding consistent
        atr_last = _atr(df_h1, 14)
        buf = max((atr_last * 0.08) if atr_last > 0 else 0.0, float(tick) * 2.0, entry * 0.0004)

        sl_pre = float(sl)
        sl_adj: Dict[str, Any] = {
            "setup_hint": setup_hint,
            "entry_type": entry_type,
            "sl_pre": sl_pre,
            "buf": float(buf),
            "did_sweep": False,
            "did_eq": False,
            "sweep_extreme": None,
            "eq_level": None,
            "sl_post_round": None,
        }

        try:
            if isinstance(liq_sweep, dict) and liq_sweep.get("ok"):
                ext = float(liq_sweep.get("sweep_extreme") or 0.0)
                if ext > 0 and buf > 0:
                    if bias == "LONG":
                        new_sl = min(sl, ext - buf)
                    else:
                        new_sl = max(sl, ext + buf)
                    if np.isfinite(new_sl) and float(new_sl) != float(sl):
                        sl = float(new_sl)
                        sl_adj["did_sweep"] = True
                        sl_adj["sweep_extreme"] = float(ext)
        except Exception:
            pass

        try:
            lv = detect_equal_levels(df_h1.tail(200), max_window=200, tol_mult_atr=0.10)
            eq_highs = lv.get("eq_highs", []) or []
            eq_lows = lv.get("eq_lows", []) or []

            if bias == "LONG" and eq_lows:
                lvl = min(eq_lows, key=lambda x: abs(entry - float(x)))
                lvl = float(lvl)
                if lvl > 0 and lvl < entry and sl > (lvl - buf):
                    new_sl = min(sl, lvl - buf)
                    if np.isfinite(new_sl) and float(new_sl) != float(sl):
                        sl = float(new_sl)
                        sl_adj["did_eq"] = True
                        sl_adj["eq_level"] = float(lvl)

            if bias == "SHORT" and eq_highs:
                lvl = min(eq_highs, key=lambda x: abs(entry - float(x)))
                lvl = float(lvl)
                if lvl > 0 and lvl > entry and sl < (lvl + buf):
                    new_sl = max(sl, lvl + buf)
                    if np.isfinite(new_sl) and float(new_sl) != float(sl):
                        sl = float(new_sl)
                        sl_adj["did_eq"] = True
                        sl_adj["eq_level"] = float(lvl)
        except Exception:
            pass

        # tick-grid rounding AFTER adjustments
        sl = _round_sl_for_side(sl, entry, bias, tick)
        sl_adj["sl_post_round"] = float(sl)

        if sl <= 0:
            LOGGER.info("[EVAL_REJECT] %s sl_invalid_after_liq_adj sl=%s", symbol, sl)
            return _reject(
                "sl_invalid_after_liq_adj",
                institutional=inst,
                structure=struct,
                pre_priority=pre_priority,
                pre_priority_reasons=pre_priority_reasons,
                soft_vetoes=soft_vetoes,
            )

        # recalc TP1 after SL adjustments
        try:
            tp1, _rr_used2 = compute_tp1(entry, sl, bias, df=df_h1, tick=tick)
            tp1 = float(tp1)
        except Exception:
            tp1 = float(tp1)

        rr = _safe_rr(entry, sl, tp1, bias)

        # merge SL meta (original + post adjustments)
        base_sl_meta = exits.get("sl_meta") if isinstance(exits.get("sl_meta"), dict) else {}
        sl_meta_out = dict(base_sl_meta)
        sl_meta_out["post_adjust"] = sl_adj

        LOGGER.info(
            "[EVAL_PRE] %s EXITS entry=%s sl=%s tp1=%s tick=%s RR=%s raw_rr=%s entry_type=%s setup_hint=%s",
            symbol, entry, sl, tp1, tick, rr, exits.get("rr_used"), entry_type, setup_hint
        )

        if rr is None or rr <= 0:
            LOGGER.info("[EVAL_REJECT] %s rr_invalid", symbol)
            return _reject(
                "rr_invalid",
                institutional=inst,
                structure=struct,
                entry_pick=entry_pick,
                pre_priority=pre_priority,
                pre_priority_reasons=pre_priority_reasons,
                soft_vetoes=soft_vetoes,
            )

        # ===============================================================
        # SETUPS
        # ===============================================================
        can_bos = bool(bos_flag and ((not REQUIRE_BOS_QUALITY) or bos_quality_ok))
        can_raid = bool(isinstance(struct.get("raid_displacement"), dict) and struct["raid_displacement"].get("ok"))
        can_sweep = bool(isinstance(liq_sweep, dict) and liq_sweep.get("ok"))

        # 1) BOS_STRICT
        bos_ok = False
        bos_variant = "NO"
        if can_bos:
            if rr >= RR_MIN_STRICT:
                bos_ok = True
                bos_variant = "RR_STRICT"
            elif rr >= RR_MIN_TOLERATED_WITH_INST:
                bos_ok = True
                bos_variant = "RR_RELAX_WITH_INST"

        if bos_ok:
            priority, priority_reasons = _final_grade(
                setup_type="BOS_STRICT",
                setup_variant=bos_variant,
                rr=float(rr),
                gate=int(gate),
                ok_count=int(ok_count),
                inst_available=bool(available),
                bos_quality_ok=bool(bos_quality_ok),
                used_bias_fallback=bool(used_bias_fallback),
                entry_type=str(entry_type),
                ext_sig=str(ext_sig),
                unfavorable_market=bool(unfavorable_market),
                mom=str(mom),
                soft_vetoes=soft_vetoes,
            )

            setup_core = "BOS_STRICT"
            setup_ttl = _setup_ttl_compatible(setup_core, entry_type)

            return {
                "valid": True,
                "symbol": symbol,
                "side": "BUY" if bias == "LONG" else "SELL",
                "bias": bias,
                "entry": entry,
                "entry_type": entry_type,
                "sl": float(sl),
                "tp1": float(tp1),
                "tp2": None,
                "rr": float(rr),
                "qty": 1,
                "setup_type": setup_ttl,          # TTL-compatible for scanner.py
                "setup_type_core": setup_core,    # canonical for debug
                "setup_variant": bos_variant,
                "priority": priority,
                "priority_reasons": priority_reasons,
                "pre_priority": pre_priority,
                "pre_priority_reasons": pre_priority_reasons,
                "pass2_done": bool(do_pass2),
                "soft_vetoes": soft_vetoes,
                "structure": struct,
                "bos_quality": bos_q,
                "institutional": inst,
                "inst_gate": int(gate),
                "inst_ok_count": int(ok_count),
                "inst_score_eff": int(gate),  # gate used for acceptance
                "momentum": mom,
                "composite": comp,
                "composite_score": comp_score,
                "composite_label": comp_label,
                "premium": premium,
                "discount": discount,
                "entry_pick": entry_pick,
                "sl_meta": sl_meta_out,
            }

        # 2) RAID_DISPLACEMENT (desk)
        if DESK_EV_MODE and can_raid and float(rr) >= float(RR_MIN_DESK_PRIORITY):
            variant = str((struct.get("raid_displacement") or {}).get("note") or "raid_ok")
            priority, priority_reasons = _final_grade(
                setup_type="RAID_DISPLACEMENT",
                setup_variant=variant,
                rr=float(rr),
                gate=int(gate),
                ok_count=int(ok_count),
                inst_available=bool(available),
                bos_quality_ok=bool(bos_quality_ok),
                used_bias_fallback=bool(used_bias_fallback),
                entry_type=str(entry_type),
                ext_sig=str(ext_sig),
                unfavorable_market=bool(unfavorable_market),
                mom=str(mom),
                soft_vetoes=soft_vetoes,
            )

            setup_core = "RAID_DISPLACEMENT"
            setup_ttl = _setup_ttl_compatible(setup_core, entry_type)

            return {
                "valid": True,
                "symbol": symbol,
                "side": "BUY" if bias == "LONG" else "SELL",
                "bias": bias,
                "entry": entry,
                "entry_type": entry_type,
                "sl": float(sl),
                "tp1": float(tp1),
                "tp2": None,
                "rr": float(rr),
                "qty": 1,
                "setup_type": setup_ttl,          # TTL-compatible
                "setup_type_core": setup_core,
                "setup_variant": variant,
                "priority": priority,
                "priority_reasons": priority_reasons,
                "pre_priority": pre_priority,
                "pre_priority_reasons": pre_priority_reasons,
                "pass2_done": bool(do_pass2),
                "soft_vetoes": soft_vetoes,
                "structure": struct,
                "bos_quality": bos_q,
                "institutional": inst,
                "inst_gate": int(gate),
                "inst_ok_count": int(ok_count),
                "inst_score_eff": int(gate),
                "momentum": mom,
                "composite": comp,
                "composite_score": comp_score,
                "composite_label": comp_label,
                "premium": premium,
                "discount": discount,
                "entry_pick": entry_pick,
                "liquidity_sweep": liq_sweep,
                "sl_meta": sl_meta_out,
            }

        # 3) LIQ_SWEEP (desk)
        if DESK_EV_MODE and can_sweep and float(rr) >= float(RR_MIN_DESK_PRIORITY):
            variant = str(liq_sweep.get("kind") if isinstance(liq_sweep, dict) else "liq_ok")
            priority, priority_reasons = _final_grade(
                setup_type="LIQ_SWEEP",
                setup_variant=variant,
                rr=float(rr),
                gate=int(gate),
                ok_count=int(ok_count),
                inst_available=bool(available),
                bos_quality_ok=bool(bos_quality_ok),
                used_bias_fallback=bool(used_bias_fallback),
                entry_type=str(entry_type),
                ext_sig=str(ext_sig),
                unfavorable_market=bool(unfavorable_market),
                mom=str(mom),
                soft_vetoes=soft_vetoes,
            )

            setup_core = "LIQ_SWEEP"
            setup_ttl = _setup_ttl_compatible(setup_core, entry_type)

            return {
                "valid": True,
                "symbol": symbol,
                "side": "BUY" if bias == "LONG" else "SELL",
                "bias": bias,
                "entry": entry,
                "entry_type": entry_type,
                "sl": float(sl),
                "tp1": float(tp1),
                "tp2": None,
                "rr": float(rr),
                "qty": 1,
                "setup_type": setup_ttl,          # TTL-compatible
                "setup_type_core": setup_core,
                "setup_variant": variant,
                "priority": priority,
                "priority_reasons": priority_reasons,
                "pre_priority": pre_priority,
                "pre_priority_reasons": pre_priority_reasons,
                "pass2_done": bool(do_pass2),
                "soft_vetoes": soft_vetoes,
                "structure": struct,
                "bos_quality": bos_q,
                "institutional": inst,
                "inst_gate": int(gate),
                "inst_ok_count": int(ok_count),
                "inst_score_eff": int(gate),
                "momentum": mom,
                "composite": comp,
                "composite_score": comp_score,
                "composite_label": comp_label,
                "premium": premium,
                "discount": discount,
                "entry_pick": entry_pick,
                "liquidity_sweep": liq_sweep,
                "sl_meta": sl_meta_out,
            }

        # 4) INST_CONTINUATION (desk)
        if DESK_EV_MODE and bias in ("LONG", "SHORT"):
            good_inst = int(gate) >= int(INST_SCORE_DESK_PRIORITY)
            good_rr = float(rr) >= float(RR_MIN_DESK_PRIORITY)
            good_comp = _composite_bias_ok(bias, comp_score, comp_label, thr=65.0)

            if good_inst and good_rr and good_comp:
                priority, priority_reasons = _final_grade(
                    setup_type="INST_CONTINUATION",
                    setup_variant="INST_CONTINUATION",
                    rr=float(rr),
                    gate=int(gate),
                    ok_count=int(ok_count),
                    inst_available=bool(available),
                    bos_quality_ok=bool(bos_quality_ok),
                    used_bias_fallback=bool(used_bias_fallback),
                    entry_type=str(entry_type),
                    ext_sig=str(ext_sig),
                    unfavorable_market=bool(unfavorable_market),
                    mom=str(mom),
                    soft_vetoes=soft_vetoes,
                )

                setup_core = "INST_CONTINUATION"
                setup_ttl = _setup_ttl_compatible(setup_core, entry_type)

                return {
                    "valid": True,
                    "symbol": symbol,
                    "side": "BUY" if bias == "LONG" else "SELL",
                    "bias": bias,
                    "entry": entry,
                    "entry_type": entry_type,
                    "sl": float(sl),
                    "tp1": float(tp1),
                    "tp2": None,
                    "rr": float(rr),
                    "qty": 1,
                    "setup_type": setup_ttl,          # TTL-compatible
                    "setup_type_core": setup_core,
                    "setup_variant": "INST_CONTINUATION",
                    "priority": priority,
                    "priority_reasons": priority_reasons,
                    "pre_priority": pre_priority,
                    "pre_priority_reasons": pre_priority_reasons,
                    "pass2_done": bool(do_pass2),
                    "soft_vetoes": soft_vetoes,
                    "structure": struct,
                    "bos_quality": bos_q,
                    "institutional": inst,
                    "inst_gate": int(gate),
                    "inst_ok_count": int(ok_count),
                    "inst_score_eff": int(gate),
                    "momentum": mom,
                    "composite": comp,
                    "composite_score": comp_score,
                    "composite_label": comp_label,
                    "premium": premium,
                    "discount": discount,
                    "entry_pick": entry_pick,
                    "sl_meta": sl_meta_out,
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
            "inst_gate": int(gate),
            "inst_ok_count": int(ok_count),
            "inst_score_eff": int(gate),
            "composite_score": comp_score,
            "composite_label": comp_label,
            "sl_meta": sl_meta_out,
            "pre_priority": pre_priority,
            "pre_priority_reasons": pre_priority_reasons,
            "soft_vetoes": soft_vetoes,
        }
