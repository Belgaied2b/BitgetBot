# =====================================================================
# ai_model.py — Expert Desk Model (Pre-calibrated, production-ready)
# =====================================================================
# Goal:
#   Provide a "pretrained-like" expert model you can deploy immediately:
#   - No training required to start (weights embedded).
#   - Robust to missing fields (safe parsing).
#   - Outputs: ai_score (0..100), ai_prob (0..1), risk_factor (>0), veto + reason.
#
# IMPORTANT NOTE (honesty):
#   This is not a proprietary institutional model trained on private order-flow datasets.
#   It is a strong expert-calibrated logistic model + desk gating rules (institutional priors).
#   You can later retrain offline on YOUR execution history and load weights via AI_MODEL_PATH.
# =====================================================================

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


MODEL_NAME = "UltraDesk-AI-Expert"
MODEL_VERSION = "v1.2.0-frozen-expert-priors"
DEFAULT_MODEL_PATH = os.getenv("AI_MODEL_PATH", "data/ai/weights.json")

# -----------------------------
# Utilities
# -----------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default

def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(float(x))
    except Exception:
        return default

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _sigmoid(z: float) -> float:
    # stable sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)

def _logit(p: float) -> float:
    p = _clamp(p, 1e-6, 1.0 - 1e-6)
    return math.log(p / (1.0 - p))

def _get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def _bps_from_prices(best_bid: float, best_ask: float) -> float:
    if best_bid <= 0 or best_ask <= 0:
        return 9999.0
    mid = (best_bid + best_ask) / 2.0
    if mid <= 0:
        return 9999.0
    return abs(best_ask - best_bid) / mid * 10000.0

def _norm01(x: float, lo: float, hi: float) -> float:
    """Normalize x into [0,1] with clamp."""
    if hi <= lo:
        return 0.0
    return _clamp((x - lo) / (hi - lo), 0.0, 1.0)

def _tanh(x: float) -> float:
    # stable tanh
    if x > 20:
        return 1.0
    if x < -20:
        return -1.0
    e2x = math.exp(2.0 * x)
    return (e2x - 1.0) / (e2x + 1.0)


# -----------------------------
# Feature engineering
# -----------------------------

FEATURE_ORDER: List[str] = [
    # Institutional gate quality
    "inst_score_norm",           # 0..1
    "spread_bps_neg",            # negative (tight spread => positive)
    "depth_usd_log",             # log depth
    "depth_imbalance",           # [-1,1], bid-ask imbalance
    "tape_delta_5m_tanh",        # [-1,1]
    "funding_rate_tanh",         # [-1,1]
    "oi_change_1h_tanh",         # [-1,1]
    "atr_pct_neg",               # negative (too volatile => lower)
    "momentum_composite",        # [-1,1]
    "htf_align",                 # 0/1
    "bos_trigger",               # 0/1
    "choch_penalty",             # 0/1
    "range_penalty",             # 0/1
    "session_liquidity",         # 0..1
    "vol_regime_low",            # 0/1
    "vol_regime_med",            # 0/1
    "vol_regime_high",           # 0/1
]

@dataclass
class AIResult:
    ok: bool
    model: str
    version: str
    ts_ms: int

    ai_prob: float
    ai_score: int
    risk_factor: float

    veto: bool
    veto_reason: str

    contributions: Dict[str, float]


def build_features(payload: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert your candidate/signal/institutional snapshot into numeric features.
    The payload is expected to be a merged dict, for example:
      {
        "direction": "LONG"/"SHORT",
        "inst_score": 0..5,
        "best_bid": ..., "best_ask": ...,
        "bid_depth_usd": ..., "ask_depth_usd": ...,
        "spread": ... (optional),
        "tape_delta_5m": ...,
        "funding_rate": ...,
        "oi_change_1h": ...,
        "atr_pct": ...,
        "momentum_label": ..., "momentum_score": ...,
        "momentum_composite_score": ...,
        "htf_ok": True/False,
        "bos": True/False,
        "choch": True/False,
        "trend": "LONG"/"SHORT"/"RANGE",
        "vol_regime": "LOW"/"MEDIUM"/"HIGH"/"UNKNOWN",
        "session": {"in_london":..., "in_ny":...} or "session_liquidity"
      }
    """
    inst_score = _safe_int(_get(payload, "inst_score", "inst_score_eff", default=0), 0)
    # normalize inst score to [0,1]; assume typical 0..4/5
    inst_score_norm = _clamp(inst_score / 4.0, 0.0, 1.0)

    best_bid = _safe_float(_get(payload, "best_bid", default=0.0), 0.0)
    best_ask = _safe_float(_get(payload, "best_ask", default=0.0), 0.0)

    spread_bps = _safe_float(_get(payload, "spread_bps", default=0.0), 0.0)
    if spread_bps <= 0:
        # if spread provided as absolute, try compute bps from bid/ask
        spread_abs = _safe_float(_get(payload, "spread", default=0.0), 0.0)
        if spread_abs > 0 and (best_bid > 0 and best_ask > 0):
            mid = (best_bid + best_ask) / 2.0
            spread_bps = (spread_abs / max(mid, 1e-12)) * 10000.0
        else:
            spread_bps = _bps_from_prices(best_bid, best_ask)

    # Tight spread is good => negative feature for spread
    spread_bps_neg = -_norm01(spread_bps, lo=1.0, hi=30.0)

    bid_depth = _safe_float(_get(payload, "bid_depth_usd", "bid_depth", default=0.0), 0.0)
    ask_depth = _safe_float(_get(payload, "ask_depth_usd", "ask_depth", default=0.0), 0.0)
    depth_sum = max(bid_depth + ask_depth, 0.0)

    # log depth: stable, treats 0.. as low liquidity
    depth_usd_log = math.log10(max(depth_sum, 1.0)) / 6.0  # ~ 1e6 => 1.0
    depth_usd_log = _clamp(depth_usd_log, 0.0, 1.0)

    # imbalance [-1,1]
    depth_imbalance = 0.0
    if depth_sum > 0:
        depth_imbalance = (bid_depth - ask_depth) / depth_sum
        depth_imbalance = _clamp(depth_imbalance, -1.0, 1.0)

    tape_delta_5m = _safe_float(_get(payload, "tape_delta_5m", default=0.0), 0.0)
    tape_delta_5m_tanh = _tanh(tape_delta_5m / 100000.0)  # scale for USD delta

    funding_rate = _safe_float(_get(payload, "funding_rate", default=0.0), 0.0)
    funding_rate_tanh = _tanh(funding_rate * 20.0)  # typical funding small, scale it

    oi_change_1h = _safe_float(_get(payload, "oi_change_1h", "open_interest_change_1h", default=0.0), 0.0)
    oi_change_1h_tanh = _tanh(oi_change_1h / 5.0)  # in % maybe

    # atr_pct: high ATR% => riskier, penalize
    atr_pct = _safe_float(_get(payload, "atr_pct", default=0.0), 0.0)
    atr_pct_neg = -_norm01(atr_pct, lo=0.008, hi=0.06)  # 0.8%..6%

    # momentum composite: map from label/score if available
    mom = _safe_float(_get(payload, "momentum_composite", "momentum_composite_score", default=0.0), 0.0)
    # If mom seems like 0..20 scale, normalize to [-1,1] via tanh
    if abs(mom) > 2.0:
        momentum_composite = _tanh(mom / 10.0)
    else:
        momentum_composite = _clamp(mom, -1.0, 1.0)

    # HTF alignment (bool)
    htf_ok = bool(_get(payload, "htf_ok", "htf_align", default=False))
    htf_align = 1.0 if htf_ok else 0.0

    bos = bool(_get(payload, "bos", "bos_trigger", default=False))
    bos_trigger = 1.0 if bos else 0.0

    choch = bool(_get(payload, "choch", default=False))
    choch_penalty = 1.0 if choch else 0.0

    trend = str(_get(payload, "trend", default="")).upper()
    range_penalty = 1.0 if trend == "RANGE" else 0.0

    # Session liquidity proxy: NY/London higher, Asia medium
    session_liquidity = _safe_float(_get(payload, "session_liquidity", default=0.0), 0.0)
    if session_liquidity <= 0:
        sess = _get(payload, "session", default=None)
        if isinstance(sess, dict):
            in_london = bool(sess.get("in_london", False))
            in_ny = bool(sess.get("in_ny", False))
            # modest proxy
            session_liquidity = 1.0 if (in_london or in_ny) else 0.6
        else:
            session_liquidity = 0.7
    session_liquidity = _clamp(session_liquidity, 0.0, 1.0)

    vol_regime = str(_get(payload, "vol_regime", "volatility_regime", default="UNKNOWN")).upper()
    vol_regime_low = 1.0 if vol_regime == "LOW" else 0.0
    vol_regime_med = 1.0 if vol_regime == "MEDIUM" else 0.0
    vol_regime_high = 1.0 if vol_regime == "HIGH" else 0.0

    feats = {
        "inst_score_norm": float(inst_score_norm),
        "spread_bps_neg": float(spread_bps_neg),
        "depth_usd_log": float(depth_usd_log),
        "depth_imbalance": float(depth_imbalance),
        "tape_delta_5m_tanh": float(tape_delta_5m_tanh),
        "funding_rate_tanh": float(funding_rate_tanh),
        "oi_change_1h_tanh": float(oi_change_1h_tanh),
        "atr_pct_neg": float(atr_pct_neg),
        "momentum_composite": float(momentum_composite),
        "htf_align": float(htf_align),
        "bos_trigger": float(bos_trigger),
        "choch_penalty": float(choch_penalty),
        "range_penalty": float(range_penalty),
        "session_liquidity": float(session_liquidity),
        "vol_regime_low": float(vol_regime_low),
        "vol_regime_med": float(vol_regime_med),
        "vol_regime_high": float(vol_regime_high),
    }
    return feats


# -----------------------------
# Frozen expert weights (pre-calibrated priors)
# -----------------------------
# These weights reflect desk-style priorities:
# - institutional quality and liquidity heavily weighted
# - spread and depth strongly matter
# - avoid chaos: choch/range penalized
# - ATR% too high penalized
# - require HTF alignment / BOS adds edge
# - funding/OI/tape act as context tilt

DEFAULT_BIAS = _logit(0.53)  # base win-prob prior ~53%

DEFAULT_WEIGHTS: Dict[str, float] = {
    "inst_score_norm": 1.35,
    "spread_bps_neg": 0.95,
    "depth_usd_log": 1.15,
    "depth_imbalance": 0.35,
    "tape_delta_5m_tanh": 0.40,
    "funding_rate_tanh": 0.18,
    "oi_change_1h_tanh": 0.30,
    "atr_pct_neg": 0.55,
    "momentum_composite": 0.75,
    "htf_align": 0.70,
    "bos_trigger": 0.40,
    "choch_penalty": -0.85,
    "range_penalty": -0.60,
    "session_liquidity": 0.25,
    "vol_regime_low": 0.18,
    "vol_regime_med": 0.08,
    "vol_regime_high": -0.25,
}

# Desk gating defaults
AI_ENABLE = str(os.getenv("AI_MODEL_ENABLE", "1")).strip() == "1"
AI_SCORE_MIN = _safe_int(os.getenv("AI_SCORE_MIN", "58"), 58)          # veto if below (unless overridden)
AI_PROB_MIN = _safe_float(os.getenv("AI_PROB_MIN", "0.55"), 0.55)      # veto if below
AI_REQUIRE_INST_SCORE = _safe_int(os.getenv("AI_REQUIRE_INST_SCORE", "2"), 2)

# Size modulation
AI_RISK_FLOOR = _safe_float(os.getenv("AI_RISK_FLOOR", "0.65"), 0.65)  # min multiplier
AI_RISK_CAP = _safe_float(os.getenv("AI_RISK_CAP", "1.35"), 1.35)      # max multiplier
AI_RISK_SLOPE = _safe_float(os.getenv("AI_RISK_SLOPE", "0.55"), 0.55)  # how strongly prob affects size


# -----------------------------
# Model class
# -----------------------------

class ExpertDeskModel:
    def __init__(self) -> None:
        self.model = MODEL_NAME
        self.version = MODEL_VERSION
        self.bias = float(DEFAULT_BIAS)
        self.weights: Dict[str, float] = dict(DEFAULT_WEIGHTS)
        self.loaded_from: str = "embedded"
        self.loaded_ts_ms: int = _now_ms()

    def load(self, path: str = DEFAULT_MODEL_PATH) -> bool:
        """Load weights from JSON if present (optional)."""
        try:
            if not path or not os.path.exists(path):
                return False
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if not isinstance(obj, dict):
                return False

            bias = obj.get("bias", None)
            if bias is not None:
                self.bias = float(bias)

            w = obj.get("weights", None)
            if isinstance(w, dict):
                # only accept known features
                for k in FEATURE_ORDER:
                    if k in w:
                        self.weights[k] = float(w[k])

            self.loaded_from = str(path)
            self.loaded_ts_ms = _now_ms()
            return True
        except Exception:
            return False

    def score(self, feats: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Return (probability, per-feature contributions)."""
        z = float(self.bias)
        contrib: Dict[str, float] = {}
        for k in FEATURE_ORDER:
            x = _safe_float(feats.get(k, 0.0), 0.0)
            w = _safe_float(self.weights.get(k, 0.0), 0.0)
            c = w * x
            contrib[k] = c
            z += c
        p = _sigmoid(z)
        return p, contrib

    def compute_risk_factor(self, p: float, feats: Dict[str, float]) -> float:
        """
        Convert probability into size multiplier.
        - Higher p => larger size, within [floor, cap]
        - Liquidity gate also affects size (depth, spread).
        """
        p = _clamp(p, 0.0, 1.0)
        # base from probability around 0.5
        adj = (p - 0.5) * (2.0 * AI_RISK_SLOPE)  # roughly [-slope, +slope]
        base = 1.0 + adj

        # liquidity modifier: depth increases, spread decreases
        depth = _safe_float(feats.get("depth_usd_log", 0.0), 0.0)
        spr = _safe_float(feats.get("spread_bps_neg", 0.0), 0.0)  # negative => tight
        liq_bonus = 0.85 + 0.35 * depth + 0.20 * (-spr)  # convert negative->positive
        liq_bonus = _clamp(liq_bonus, 0.70, 1.35)

        rf = base * liq_bonus
        return _clamp(rf, AI_RISK_FLOOR, AI_RISK_CAP)

    def veto_reason(self, payload: Dict[str, Any], feats: Dict[str, float], p: float, ai_score: int) -> Tuple[bool, str]:
        """
        Desk-like gating rules to avoid junk signals.
        Keeps false positives low; you can tune via env.
        """
        inst_score = _safe_int(_get(payload, "inst_score", "inst_score_eff", default=0), 0)

        if inst_score < AI_REQUIRE_INST_SCORE:
            return True, f"inst_score<{AI_REQUIRE_INST_SCORE}"

        # Hard liquidity gate: ultra-wide spread and low depth
        # (using engineered features)
        depth = _safe_float(feats.get("depth_usd_log", 0.0), 0.0)
        spr_neg = _safe_float(feats.get("spread_bps_neg", 0.0), 0.0)
        spread_is_bad = (spr_neg < -0.85)  # very wide spread => spread_bps_neg near -1
        depth_is_bad = (depth < 0.25)      # low depth

        if spread_is_bad and depth_is_bad:
            return True, "liq_gate:wide_spread_low_depth"

        # Chaos regime: CHoCH + High vol => veto more often
        choch = bool(_get(payload, "choch", default=False))
        vol_regime = str(_get(payload, "vol_regime", "volatility_regime", default="UNKNOWN")).upper()
        if choch and vol_regime == "HIGH" and p < 0.62:
            return True, "chaos_gate:choch_high_vol"

        # Low confidence
        if ai_score < int(AI_SCORE_MIN):
            return True, f"ai_score<{AI_SCORE_MIN}"

        if p < float(AI_PROB_MIN):
            return True, f"ai_prob<{AI_PROB_MIN:.2f}"

        return False, ""

# Global singleton
_MODEL = ExpertDeskModel()
_MODEL.load(DEFAULT_MODEL_PATH)


# -----------------------------
# Public API (what you import)
# -----------------------------

def score_signal(payload: Dict[str, Any]) -> AIResult:
    """
    Main entrypoint.
    payload: merged dict (candidate + institutional snapshot + analysis context)
    Returns AIResult with ai_score, prob, risk_factor, veto.
    """
    ts = _now_ms()
    if not AI_ENABLE:
        return AIResult(
            ok=True,
            model=_MODEL.model,
            version=_MODEL.version,
            ts_ms=ts,
            ai_prob=0.50,
            ai_score=50,
            risk_factor=1.0,
            veto=False,
            veto_reason="",
            contributions={},
        )

    feats = build_features(payload)
    p, contrib = _MODEL.score(feats)

    # Convert probability to 0..100 score (desk-friendly)
    # Nonlinear mapping: favors higher certainty
    ai_score = int(round(_clamp((p ** 0.85) * 100.0, 0.0, 100.0)))

    rf = _MODEL.compute_risk_factor(p, feats)
    veto, reason = _MODEL.veto_reason(payload, feats, p, ai_score)

    return AIResult(
        ok=True,
        model=_MODEL.model,
        version=_MODEL.version,
        ts_ms=ts,
        ai_prob=float(p),
        ai_score=int(ai_score),
        risk_factor=float(rf),
        veto=bool(veto),
        veto_reason=str(reason or ""),
        contributions=contrib,
    )


def predict_vol_liquidity(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lightweight helper:
    - predicts volatility regime and liquidity tier, using current features.
    Use this to modulate TP clamp, position sizing, or skip illiquid symbols.
    """
    feats = build_features(payload)

    # Liquidity tier from depth + spread
    depth = _safe_float(feats.get("depth_usd_log", 0.0), 0.0)     # 0..1
    spr_neg = _safe_float(feats.get("spread_bps_neg", 0.0), 0.0)  # [-1..0]
    # Convert to intuitive signals
    spread_quality = _clamp(-spr_neg, 0.0, 1.0)  # tight spread => closer to 1
    liq_score = 0.55 * depth + 0.45 * spread_quality

    if liq_score >= 0.72:
        liq_tier = "A"
    elif liq_score >= 0.55:
        liq_tier = "B"
    elif liq_score >= 0.40:
        liq_tier = "C"
    else:
        liq_tier = "D"

    # Volatility regime from atr_pct if present, else infer from flags
    atr_pct = _safe_float(_get(payload, "atr_pct", default=0.0), 0.0)
    if atr_pct > 0:
        if atr_pct < 0.015:
            vol = "LOW"
        elif atr_pct < 0.035:
            vol = "MEDIUM"
        else:
            vol = "HIGH"
    else:
        # fallback from one-hot
        if feats.get("vol_regime_high", 0.0) > 0.5:
            vol = "HIGH"
        elif feats.get("vol_regime_med", 0.0) > 0.5:
            vol = "MEDIUM"
        elif feats.get("vol_regime_low", 0.0) > 0.5:
            vol = "LOW"
        else:
            vol = "UNKNOWN"

    return {
        "ok": True,
        "model": _MODEL.model,
        "version": _MODEL.version,
        "ts_ms": _now_ms(),
        "liq_tier": liq_tier,
        "liq_score": float(liq_score),
        "vol_regime": str(vol),
        "features": {
            "depth_usd_log": float(depth),
            "spread_quality": float(spread_quality),
        },
    }


def explain_score(payload: Dict[str, Any], top_k: int = 8) -> Dict[str, Any]:
    """
    Debug helper: see which features push score up/down.
    """
    r = score_signal(payload)
    items = sorted(r.contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top = items[: max(1, int(top_k))]
    return {
        "ok": True,
        "model": r.model,
        "version": r.version,
        "ts_ms": r.ts_ms,
        "ai_prob": r.ai_prob,
        "ai_score": r.ai_score,
        "risk_factor": r.risk_factor,
        "veto": r.veto,
        "veto_reason": r.veto_reason,
        "top_contrib": [{"feat": k, "contrib": float(v)} for k, v in top],
    }


# -----------------------------
# Optional: dumping current weights (for reproducibility)
# -----------------------------

def dump_weights(path: str = DEFAULT_MODEL_PATH) -> bool:
    """
    Write current weights to JSON (for versioning).
    Does not train; just exports configuration.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        obj = {
            "model": _MODEL.model,
            "version": _MODEL.version,
            "ts_ms": _now_ms(),
            "bias": float(_MODEL.bias),
            "weights": {k: float(_MODEL.weights.get(k, 0.0)) for k in FEATURE_ORDER},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


# -----------------------------
# Quick self-test (optional)
# -----------------------------
if __name__ == "__main__":
    sample = {
        "inst_score": 3,
        "best_bid": 1.0000,
        "best_ask": 1.0005,
        "bid_depth_usd": 420000,
        "ask_depth_usd": 380000,
        "tape_delta_5m": 65000,
        "funding_rate": 0.0003,
        "oi_change_1h": 1.8,
        "atr_pct": 0.022,
        "momentum_composite_score": 9.5,
        "htf_ok": True,
        "bos": True,
        "choch": False,
        "trend": "SHORT",
        "vol_regime": "MEDIUM",
        "session": {"in_london": False, "in_ny": True},
    }
    print(explain_score(sample, top_k=10))
    print(predict_vol_liquidity(sample))
