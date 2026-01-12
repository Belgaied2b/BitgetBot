from __future__ import annotations

import asyncio
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from indicators import (
    atr,
    ema,
    rsi,
    stoch_rsi,
    vwma,
)
from institutional_data import compute_full_institutional_analysis
from logger import get_logger
from macro_data import fetch_macro_snapshot
from options_data import fetch_options_snapshot
from risk_manager import RiskManager
from smt_utils import compute_smt_veto
from stops import compute_stops
from structure_utils import compute_structure
from tp_clamp import clamp_tp
from trend_filters import (
    compute_htf_bias,
    compute_momentum_label,
    compute_momentum_score,
    compute_vol_regime,
)

logger = get_logger("analyze_signal")


# -----------------------------
# Helpers / small utils
# -----------------------------
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _now_ms() -> int:
    return int(time.time() * 1000)


def _pct(a: float, b: float, default: float = 0.0) -> float:
    if b == 0:
        return default
    return (a / b) * 100.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _median(xs: List[float], default: float = 0.0) -> float:
    try:
        if not xs:
            return default
        return statistics.median(xs)
    except Exception:
        return default


def _mean(xs: List[float], default: float = 0.0) -> float:
    try:
        if not xs:
            return default
        return statistics.mean(xs)
    except Exception:
        return default


def _std(xs: List[float], default: float = 0.0) -> float:
    try:
        if len(xs) < 2:
            return default
        return statistics.pstdev(xs)
    except Exception:
        return default


def _zscore(x: float, mean_: float, std_: float, default: float = 0.0) -> float:
    if std_ == 0:
        return default
    return (x - mean_) / std_


def _isfinite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _coalesce(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


# -----------------------------
# Config / constants
# -----------------------------
DEFAULT_TF = "15m"
DEFAULT_HTF = "4h"

# internal scoring weights (keep as-is unless you know the strategy)
W_STRUCT = 0.35
W_MOM = 0.35
W_INST = 0.30

# institutional passes (LIGHT first, then NORMAL)
INST_PASS1_MODE = "LIGHT"
INST_PASS2_MODE = "NORMAL"

# -----------------------------
# Data containers
# -----------------------------
@dataclass
class EvalContext:
    symbol: str
    timeframe: str
    htf: str
    ts_ms: int = field(default_factory=_now_ms)
    macro: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None
    ws_hub: Any = None  # institutional_ws_hub instance


@dataclass
class PreEval:
    trend: str
    bos: bool
    choch: bool
    cos: bool
    momentum_label: str
    momentum_score: float
    vol_regime: str
    extension: Optional[str] = None
    bias_fallback_for_inst: Optional[str] = None


@dataclass
class PrePriority:
    priority: str
    pass2_allowed: bool
    reasons: List[str]
    soft_vetoes: List[str]
    smt_veto: bool
    options_filter: Dict[str, Any]
    session: Dict[str, Any]
    trad: Dict[str, Any]
    iv: Optional[Dict[str, Any]] = None


@dataclass
class InstSummary:
    ok: bool
    available: bool
    bitget_symbol: Optional[str]
    bias: Optional[str]
    mode: str
    pass_n: int
    liq_req: bool
    inst_score: float
    inst_score_norm: float
    ok_count: int
    gate: int
    comps: List[str]
    override: Optional[str] = None


# -----------------------------
# Core analyzer
# -----------------------------
class SignalAnalyzer:
    def __init__(self, client, settings: Dict[str, Any]):
        self.client = client
        self.settings = settings
        self.risk = RiskManager(settings)

    async def analyze_symbol(
        self,
        symbol: str,
        timeframe: str = DEFAULT_TF,
        htf: str = DEFAULT_HTF,
        ws_hub=None,
        macro: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Main entry used by scanner.

        Returns a dict with:
        - ok: bool
        - reason: str (if not ok)
        - fields: various analysis components (structure/momentum/institutional/risk)
        """
        logger.info("[EVAL] ▶ START %s", symbol)
        t0 = time.time()

        ctx = EvalContext(symbol=symbol, timeframe=timeframe, htf=htf, macro=macro, options=options, ws_hub=ws_hub)

        # -----------------------------
        # 1) Fetch candles (Bitget client)
        # -----------------------------
        try:
            candles = await self.client.fetch_klines(symbol, timeframe=timeframe, limit=self.settings.get("KLINES_LIMIT", 240))
        except Exception as e:
            return self._reject(symbol, f"fetch_klines_error:{e}")

        if not candles or len(candles) < 80:
            return self._reject(symbol, "not_enough_klines")

        # candles expected: list of dicts or tuples; normalize
        o, h, l, c, v = self._split_ohlcv(candles)
        if len(c) < 80:
            return self._reject(symbol, "bad_klines_format")

        # -----------------------------
        # 2) Structure / momentum / regime
        # -----------------------------
        struct = compute_structure(o, h, l, c, self.settings)
        trend = struct.get("trend", "RANGE")
        bos = bool(struct.get("bos", False))
        choch = bool(struct.get("choch", False))
        cos = bool(struct.get("cos", False))

        logger.info("[EVAL_PRE] %s STRUCT trend=%s bos=%s choch=%s cos=%s", symbol, trend, bos, choch, cos)

        momentum_label = compute_momentum_label(o, h, l, c, self.settings)
        logger.info("[EVAL_PRE] %s MOMENTUM=%s", symbol, momentum_label)

        mom_score = compute_momentum_score(o, h, l, c, self.settings)
        mom_label2 = "NEUTRAL"
        if mom_score >= 80:
            mom_label2 = "STRONG_BULLISH"
        elif mom_score >= 60:
            mom_label2 = "BULLISH"
        elif mom_score <= 20:
            mom_label2 = "STRONG_BEARISH"
        elif mom_score <= 40:
            mom_label2 = "BEARISH"
        logger.info("[EVAL_PRE] %s MOMENTUM_COMPOSITE score=%.2f label=%s", symbol, mom_score, mom_label2)

        vol_regime, extension = compute_vol_regime(o, h, l, c, self.settings)
        logger.info("[EVAL_PRE] %s VOL_REGIME=%s EXTENSION=%s", symbol, vol_regime, extension)

        pre = PreEval(
            trend=trend,
            bos=bos,
            choch=choch,
            cos=cos,
            momentum_label=momentum_label,
            momentum_score=mom_score,
            vol_regime=vol_regime,
            extension=extension,
        )

        # -----------------------------
        # 3) HTF bias check
        # -----------------------------
        try:
            htf_bias = await compute_htf_bias(self.client, symbol, htf, self.settings)
        except Exception as e:
            return self._reject(symbol, f"htf_bias_error:{e}")

        bias = self._bias_from_trend(pre.trend, pre.momentum_label, pre.momentum_score, htf_bias)
        if bias is None:
            # fallback for inst computations (keeps old behavior)
            pre.bias_fallback_for_inst = "LONG" if pre.momentum_score >= 50 else "SHORT"
            logger.info("[EVAL_PRE] %s bias_fallback_for_inst=%s", symbol, pre.bias_fallback_for_inst)
        else:
            pre.bias_fallback_for_inst = bias

        # If bias mismatch with HTF, reject early
        if bias is not None and htf_bias is not None and bias != htf_bias:
            logger.info("[EVAL_REJECT] %s htf_bias_mismatch bias=%s htf_bias=%s", symbol, bias, htf_bias)
            return self._reject(symbol, "htf_bias_mismatch")

        # -----------------------------
        # 4) Macro / options filters (already provided by scanner, but safe fallback)
        # -----------------------------
        if ctx.macro is None:
            try:
                ctx.macro = await fetch_macro_snapshot()
            except Exception:
                ctx.macro = {"ok": False}

        if ctx.options is None:
            try:
                ctx.options = await fetch_options_snapshot()
            except Exception:
                ctx.options = {"ok": False}

        options_filter = self._options_filter(ctx.options, pre.vol_regime)
        session_info = self._session_filter()
        trad = await self._tradability_filter(symbol, o, h, l, c, v)

        smt_veto = False
        if self.settings.get("SMT_ENABLED", True):
            try:
                smt_veto = bool(await compute_smt_veto(self.client, symbol, timeframe, self.settings))
            except Exception:
                smt_veto = False

        # -----------------------------
        # 5) Pre priority decision
        # -----------------------------
        pre_priority = self._compute_pre_priority(symbol, pre, options_filter, session_info, trad, smt_veto)
        logger.info(
            "[EVAL_PRE] %s PRE_PRIORITY=%s pass2_allowed=%s reasons=%s soft_vetoes=%s smt_veto=%s options_filter=%s session=%s trad=%s iv=%s",
            symbol,
            pre_priority.priority,
            pre_priority.pass2_allowed,
            pre_priority.reasons,
            pre_priority.soft_vetoes,
            pre_priority.smt_veto,
            pre_priority.options_filter,
            pre_priority.session,
            pre_priority.trad,
            pre_priority.iv,
        )

        # If pre stage blocks, reject
        if pre_priority.priority in ("E", "F"):
            return self._reject(symbol, "pre_filtered")

        # -----------------------------
        # 6) Institutional analysis (Bitget WS+REST)
        # -----------------------------
        inst = None
        inst_summary = None
        try:
            # Pass 1: LIGHT
            inst = await compute_full_institutional_analysis(
                symbol,
                bias=(bias or pre.bias_fallback_for_inst or "LONG"),
                ws_hub=ws_hub,
                mode=INST_PASS1_MODE,
                require_liquidations=False,
                use_tape=False,
            )
        except Exception as e:
            logger.exception("[INST] error pass1 %s: %s", symbol, e)
            inst = {"ok": False, "error": str(e), "available": False, "bitget_symbol": symbol, "bias": bias, "mode": INST_PASS1_MODE}

        # Normalize symbol field for this file (Bitget-only naming)
        bitget_symbol = (inst.get("bitget_symbol") or inst.get("symbol") or symbol)

        # If no institutional availability, skip gracefully (still allows signal, depending on settings)
        if not inst.get("available") or not bitget_symbol:
            inst = inst or {}
            inst_summary = InstSummary(
                ok=bool(inst.get("ok", False)),
                available=bool(inst.get("available", False)),
                bitget_symbol=bitget_symbol,
                bias=(inst.get("bias") if isinstance(inst, dict) else None),
                mode=INST_PASS1_MODE,
                pass_n=1,
                liq_req=False,
                inst_score=_safe_float(inst.get("inst_score", 0.0)),
                inst_score_norm=_safe_float(inst.get("inst_score_norm", 0.0)),
                ok_count=int(inst.get("ok_count", 0) or 0),
                gate=int(inst.get("gate", 0) or 0),
                comps=list(inst.get("comps", []) or []),
                override=inst.get("override"),
            )
        else:
            inst_score = _safe_float(inst.get("inst_score", 0.0))
            ok_count = int(inst.get("ok_count", 0) or 0)
            gate = int(inst.get("gate", 0) or 0)
            inst_score_norm = _safe_float(inst.get("inst_score_norm", 0.0))

            logger.info(
                "[INST_RAW] %s pass=%s mode=%s liq_req=%s inst_score=%s ok_count=%s gate=%s inst_score_norm=%s override=%s available=%s bitget_symbol=%s bias=%s comps=%s",
                symbol,
                1,
                INST_PASS1_MODE,
                False,
                inst_score,
                ok_count,
                gate,
                inst_score_norm,
                inst.get("override"),
                inst.get("available"),
                bitget_symbol,
                inst.get("bias"),
                inst.get("comps"),
            )

            do_pass2 = bool(inst.get("available")) and bool(bitget_symbol) and bool(pre_priority.pass2_allowed)

            if do_pass2:
                try:
                    inst2 = await compute_full_institutional_analysis(
                        symbol,
                        bias=(bias or pre.bias_fallback_for_inst or "LONG"),
                        ws_hub=ws_hub,
                        mode=INST_PASS2_MODE,
                        require_liquidations=True,
                        use_tape=True,
                    )
                    if isinstance(inst2, dict) and bool(inst2.get("available")):
                        inst = inst2
                        bitget_symbol = (inst.get("bitget_symbol") or inst.get("symbol") or symbol)
                except Exception as e:
                    logger.exception("[INST] error pass2 %s: %s", symbol, e)

            inst_score = _safe_float(inst.get("inst_score", 0.0))
            ok_count = int(inst.get("ok_count", 0) or 0)
            gate = int(inst.get("gate", 0) or 0)
            inst_score_norm = _safe_float(inst.get("inst_score_norm", 0.0))
            override = inst.get("override")

            logger.info(
                "[INST_RAW] %s pass=%s mode=%s liq_req=%s inst_score=%s ok_count=%s gate=%s inst_score_norm=%s override=%s available=%s bitget_symbol=%s bias=%s comps=%s",
                symbol,
                2 if do_pass2 else 1,
                INST_PASS2_MODE if do_pass2 else INST_PASS1_MODE,
                True if do_pass2 else False,
                inst_score,
                ok_count,
                gate,
                inst_score_norm,
                override,
                inst.get("available"),
                bitget_symbol,
                inst.get("bias"),
                inst.get("comps"),
            )

            inst_summary = InstSummary(
                ok=bool(inst.get("ok", False)),
                available=bool(inst.get("available", False)),
                bitget_symbol=bitget_symbol,
                bias=inst.get("bias"),
                mode=INST_PASS2_MODE if do_pass2 else INST_PASS1_MODE,
                pass_n=2 if do_pass2 else 1,
                liq_req=True if do_pass2 else False,
                inst_score=inst_score,
                inst_score_norm=inst_score_norm,
                ok_count=ok_count,
                gate=gate,
                comps=list(inst.get("comps", []) or []),
                override=override,
            )

        # -----------------------------
        # 7) Gate checks / validation
        # -----------------------------
        decision = self._validate_setup(symbol, pre, pre_priority, inst, inst_summary, bias, htf_bias)
        if not decision["ok"]:
            return decision

        # -----------------------------
        # 8) Compute risk / stops / TP
        # -----------------------------
        stops = compute_stops(symbol, o, h, l, c, bias or pre.bias_fallback_for_inst or "LONG", self.settings)
        risk = self.risk.compute(symbol, bias=(bias or pre.bias_fallback_for_inst or "LONG"), stops=stops, trad=trad)

        tp = clamp_tp(symbol, stops=stops, risk=risk, settings=self.settings)

        elapsed = int((time.time() - t0) * 1000)

        # -----------------------------
        # 9) Return full payload
        # -----------------------------
        return {
            "ok": True,
            "symbol": symbol,
            "bias": bias or pre.bias_fallback_for_inst,
            "timeframe": timeframe,
            "htf": htf,
            "htf_bias": htf_bias,
            "elapsed_ms": elapsed,
            "pre": {
                "trend": pre.trend,
                "bos": pre.bos,
                "choch": pre.choch,
                "cos": pre.cos,
                "momentum": pre.momentum_label,
                "momentum_score": pre.momentum_score,
                "momentum_composite_label": mom_label2,
                "vol_regime": pre.vol_regime,
                "extension": pre.extension,
                "bias_fallback_for_inst": pre.bias_fallback_for_inst,
            },
            "pre_priority": {
                "priority": pre_priority.priority,
                "pass2_allowed": pre_priority.pass2_allowed,
                "reasons": pre_priority.reasons,
                "soft_vetoes": pre_priority.soft_vetoes,
                "smt_veto": pre_priority.smt_veto,
                "options_filter": pre_priority.options_filter,
                "session": pre_priority.session,
                "trad": pre_priority.trad,
                "iv": pre_priority.iv,
            },
            "institutional": inst if isinstance(inst, dict) else {},
            "institutional_summary": inst_summary.__dict__ if inst_summary else {},
            "stops": stops,
            "risk": risk,
            "tp": tp,
        }

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _split_ohlcv(self, candles):
        """
        Supports:
        - list of dicts: {"open":..,"high":..,"low":..,"close":..,"volume":..}
        - list of lists/tuples: [ts, open, high, low, close, volume]
        """
        o, h, l, c, v = [], [], [], [], []
        for x in candles:
            if isinstance(x, dict):
                o.append(_safe_float(x.get("open")))
                h.append(_safe_float(x.get("high")))
                l.append(_safe_float(x.get("low")))
                c.append(_safe_float(x.get("close")))
                v.append(_safe_float(x.get("volume")))
            else:
                # assume tuple-like
                if len(x) >= 6:
                    o.append(_safe_float(x[1]))
                    h.append(_safe_float(x[2]))
                    l.append(_safe_float(x[3]))
                    c.append(_safe_float(x[4]))
                    v.append(_safe_float(x[5]))
        return o, h, l, c, v

    def _bias_from_trend(self, trend: str, momentum_label: str, mom_score: float, htf_bias: Optional[str]) -> Optional[str]:
        """
        Conservative bias selection.
        """
        if trend == "LONG":
            return "LONG"
        if trend == "SHORT":
            return "SHORT"
        # trend range: infer from momentum and HTF if strong enough
        if mom_score >= 85:
            return "LONG"
        if mom_score <= 15:
            return "SHORT"
        # else defer to HTF
        return htf_bias

    def _options_filter(self, options: Optional[Dict[str, Any]], vol_regime: str) -> Dict[str, Any]:
        if not options or not options.get("ok"):
            return {"ok": True, "score": 0, "regime": vol_regime, "reason": "options_unavailable"}
        # keep current behavior
        return {
            "ok": True,
            "score": int(options.get("score", 0) or 0),
            "regime": str(options.get("regime", vol_regime)),
            "reason": str(options.get("reason", "")),
            "avg_dvol": _safe_float(options.get("avg", 0.0)),
            "dvol_change_24h_pct": _safe_float(options.get("chg24h_pct", 0.0)),
            "spike": bool(options.get("spike", False)),
            "risk_factor": _safe_float(options.get("risk_factor", 1.0), 1.0),
            "position_mode": str(options.get("position_mode", "neutral")),
            "setup_type": str(options.get("setup_type", "OTHER")),
        }

    def _session_filter(self) -> Dict[str, Any]:
        """
        Session gating: uses UTC windows from settings.
        """
        enabled = bool(self.settings.get("SESSION_FILTER_ENABLED", True))
        tz = self.settings.get("SESSION_TZ", "UTC")
        london = self.settings.get("SESSION_LONDON", "07:00-10:00")
        ny = self.settings.get("SESSION_NY", "13:00-16:00")
        # The existing project computes actual in-session elsewhere; keep simple
        now = time.time()
        # provide same fields as logs
        return {
            "enabled": enabled,
            "tz": tz,
            "now": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now)),
            "in_london": False,
            "in_ny": False,
            "london": london,
            "ny": ny,
        }

    async def _tradability_filter(self, symbol: str, o, h, l, c, v) -> Dict[str, Any]:
        """
        Approx tradability from recent candles and volume.
        """
        enabled = bool(self.settings.get("TRADABILITY_ENABLED", True))
        if not enabled:
            return {"enabled": False, "ok": True, "reason": "disabled"}

        # proxy dollar volume: close * volume (20 bars)
        n = int(self.settings.get("TRADABILITY_LOOKBACK", 20))
        closes = c[-n:]
        vols = v[-n:]
        dollar_vol_20 = sum([closes[i] * vols[i] for i in range(min(len(closes), len(vols)))]) if closes and vols else 0.0

        # spread proxy using candle wicks / body ratio
        wickiness_20 = self._wickiness(o[-n:], h[-n:], l[-n:], c[-n:])
        atr14 = atr(h, l, c, 14) if len(c) >= 15 else 0.0
        last_range = (h[-1] - l[-1]) if h and l else 0.0
        range_atr_mult = (last_range / atr14) if atr14 else 0.0

        min_dvol = _safe_float(self.settings.get("TRADABILITY_MIN_DOLLAR_VOL", 250_000.0), 250_000.0)

        ok = dollar_vol_20 >= min_dvol
        reason = "ok" if ok else "low_dollar_volume"

        return {
            "enabled": True,
            "dollar_vol_20": dollar_vol_20,
            "spread_proxy_20": self._spread_proxy(o[-n:], h[-n:], l[-n:], c[-n:]),
            "wickiness_20": wickiness_20,
            "atr14": atr14,
            "last_range": last_range,
            "range_atr_mult": range_atr_mult,
            "ok": ok,
            "reason": reason,
        }

    def _wickiness(self, o, h, l, c) -> float:
        if not o or not h or not l or not c:
            return 0.0
        vals = []
        for i in range(len(c)):
            body = abs(c[i] - o[i])
            wick = (h[i] - max(o[i], c[i])) + (min(o[i], c[i]) - l[i])
            denom = body + wick
            vals.append((wick / denom) if denom else 0.0)
        return _mean(vals, 0.0)

    def _spread_proxy(self, o, h, l, c) -> float:
        if not h or not l or not c:
            return 0.0
        vals = []
        for i in range(len(c)):
            mid = c[i] if c[i] else 1.0
            vals.append((h[i] - l[i]) / mid if mid else 0.0)
        return _mean(vals, 0.0)

    def _compute_pre_priority(self, symbol: str, pre: PreEval, options_filter: Dict[str, Any], session: Dict[str, Any], trad: Dict[str, Any], smt_veto: bool) -> PrePriority:
        reasons = []
        soft_vetoes = []

        # triggers
        trigger = False
        if pre.bos or pre.choch or pre.cos:
            trigger = True

        # momentum gates (keep current style)
        strong_mom = pre.momentum_score >= 85 or pre.momentum_score <= 15
        weak_mom = 40 <= pre.momentum_score <= 60

        # composite labels
        high_composite = pre.momentum_score >= 80
        weak_composite = pre.momentum_score <= 20

        # session soft veto
        if session.get("enabled") and not (session.get("in_london") or session.get("in_ny")):
            soft_vetoes.append("outside_killzones")

        # tradability soft veto
        if trad.get("enabled") and not trad.get("ok"):
            soft_vetoes.append(f"tradability:{trad.get('reason')}")

        # SMT veto
        if smt_veto:
            reasons.append("pre:smt_veto")

        # Determine priority
        priority = "D"
        pass2_allowed = False

        if not trigger:
            reasons.append("pre:no_trigger")
            priority = "D"
        else:
            # With trigger, go higher
            priority = "B"
            pass2_allowed = True

        if strong_mom and trigger:
            reasons.append("pre:strong_momentum")
        if weak_mom:
            reasons.append("pre:weak_momentum")

        if high_composite:
            reasons.append("pre:high_composite")
        if weak_composite:
            reasons.append("pre:weak_composite")

        if pre.bias_fallback_for_inst:
            reasons.append("pre:bias_fallback")

        if pre.bos:
            reasons.append("pre:bos_trigger")
        if pre.choch:
            reasons.append("pre:choch_trigger")
        if pre.cos:
            reasons.append("pre:cos_trigger")

        # downgrade if SMT veto hard
        if smt_veto:
            priority = "E"
            pass2_allowed = False

        return PrePriority(
            priority=priority,
            pass2_allowed=pass2_allowed,
            reasons=reasons,
            soft_vetoes=soft_vetoes,
            smt_veto=smt_veto,
            options_filter=options_filter,
            session=session,
            trad=trad,
            iv=None,
        )

    def _validate_setup(self, symbol: str, pre: PreEval, pre_priority: PrePriority, inst: Dict[str, Any], inst_summary: Optional[InstSummary], bias: Optional[str], htf_bias: Optional[str]) -> Dict[str, Any]:
        """
        Final "go/no-go" decision combining structure/momentum/institutional and config rules.
        """
        # If requested in settings: require a validated setup trigger
        if self.settings.get("REQUIRE_TRIGGER", True):
            if not (pre.bos or pre.choch or pre.cos):
                logger.info("[EVAL_REJECT] %s no_setup_validated (DESK_EV_MODE=%s)", symbol, self.settings.get("DESK_EV_MODE", False))
                return self._reject(symbol, "no_setup_validated")

        # If institutional gate enabled
        if self.settings.get("INSTITUTIONAL_ENABLED", True):
            if not inst_summary or not inst_summary.available:
                if self.settings.get("INSTITUTIONAL_REQUIRED", False):
                    return self._reject(symbol, "institutional_unavailable")
            else:
                # gate rule: minimum inst score and minimum gate
                min_gate = int(self.settings.get("INSTITUTIONAL_MIN_GATE", 1))
                min_norm = float(self.settings.get("INSTITUTIONAL_MIN_NORM", 0.0))
                if inst_summary.gate < min_gate and inst_summary.override is None:
                    return self._reject(symbol, "institutional_gate_fail")
                if inst_summary.inst_score_norm < min_norm and inst_summary.override is None:
                    return self._reject(symbol, "institutional_norm_fail")

        # If tradability is hard-required
        if self.settings.get("TRADABILITY_HARD_REQUIRE", False):
            if pre_priority.trad.get("enabled") and not pre_priority.trad.get("ok"):
                return self._reject(symbol, "tradability_fail")

        # Soft veto handling
        if self.settings.get("SOFT_VETO_HARD", False) and pre_priority.soft_vetoes:
            return self._reject(symbol, "soft_veto")

        return {"ok": True}

    def _reject(self, symbol: str, reason: str, **kwargs) -> Dict[str, Any]:
        payload = {"ok": False, "symbol": symbol, "reason": reason}
        if kwargs:
            payload.update(kwargs)
        return payload
