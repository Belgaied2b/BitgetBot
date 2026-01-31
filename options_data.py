from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import aiohttp

LOGGER = logging.getLogger(__name__)

DERIBIT_BASE = os.getenv("DERIBIT_BASE", "https://www.deribit.com/api/v2")
OPTIONS_TTL_S = float(os.getenv("OPTIONS_TTL_S", "120"))
HTTP_TIMEOUT_S = float(os.getenv("OPTIONS_HTTP_TIMEOUT_S", "10"))

# Window / resolution
DVOL_WINDOW_H = float(os.getenv("DVOL_WINDOW_H", "24"))            # 24h
DVOL_RESOLUTION_S = int(os.getenv("DVOL_RESOLUTION_S", "3600"))    # 1h candles

# Regime thresholds (Deribit DVOL ~ index points)
DVOL_LOW_TH = float(os.getenv("DVOL_LOW_TH", "45"))
DVOL_HIGH_TH = float(os.getenv("DVOL_HIGH_TH", "70"))

# Spike detection: if 24h change >= this => "vol_spike"
DVOL_SPIKE_PCT = float(os.getenv("DVOL_SPIKE_PCT", "12"))  # +12% in 24h

# If Deribit is flaky, keep last good snapshot instead of "unknown" hard-drop
KEEP_LAST_GOOD_ON_ERROR = str(os.getenv("OPTIONS_KEEP_LAST_GOOD", "1")).strip() == "1"


@dataclass
class OptionsSnapshot:
    ts: float
    dvol_btc: Optional[float]
    dvol_eth: Optional[float]

    # extra stats (NEW)
    avg_dvol: Optional[float]
    dvol_change_24h_pct: Optional[float]  # computed from BTC if possible, else avg
    spike: bool

    regime: str
    raw: Dict[str, Any]


class OptionsCache:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._snap: Optional[OptionsSnapshot] = None

    async def get(self, force: bool = False) -> OptionsSnapshot:
        now = time.time()
        async with self._lock:
            if (not force) and self._snap and (now - self._snap.ts) < OPTIONS_TTL_S:
                return self._snap

        snap = await _fetch_options_snapshot(last_good=self._snap)
        async with self._lock:
            self._snap = snap
        return snap


# ---------------------------------------------------------------------
# Deribit DVOL fetch
# ---------------------------------------------------------------------

async def _fetch_dvol_with_session(
    session: aiohttp.ClientSession,
    currency: str,
    resolution: int = DVOL_RESOLUTION_S,
) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    """
    Returns:
      - last_close (DVOL)
      - change_24h_pct (approx from first/last close in window), or None
      - raw json
    """
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - int(DVOL_WINDOW_H * 3600 * 1000)
    params = {
        "currency": currency,
        "start_timestamp": str(start_ts),
        "end_timestamp": str(end_ts),
        "resolution": str(int(resolution)),
    }
    url = f"{DERIBIT_BASE}/public/get_volatility_index_data"

    raw: Dict[str, Any] = {}
    try:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                return None, None, {}
            raw = await resp.json()
    except Exception as e:
        LOGGER.debug("[OPTIONS] dvol fetch %s failed: %s", currency, e)
        return None, None, {}

    close_v: Optional[float] = None
    chg_pct: Optional[float] = None

    # Expect: {"jsonrpc":"2.0","result":{"data":[[ts,o,h,l,c],...], ...}}
    try:
        res = raw.get("result") if isinstance(raw, dict) else None
        data = (res or {}).get("data") if isinstance(res, dict) else None
        if isinstance(data, list) and len(data) >= 2:
            first = data[0]
            last = data[-1]
            if isinstance(last, (list, tuple)) and len(last) >= 5:
                close_v = float(last[4])
            if isinstance(first, (list, tuple)) and len(first) >= 5 and close_v is not None:
                first_close = float(first[4])
                if first_close > 0:
                    chg_pct = float((close_v - first_close) / first_close * 100.0)
    except Exception:
        close_v = None
        chg_pct = None

    return close_v, chg_pct, (raw if isinstance(raw, dict) else {})


def _regime_from_avg(avg: Optional[float], spike: bool) -> str:
    if avg is None:
        return "unknown"
    try:
        a = float(avg)
    except Exception:
        return "unknown"

    if spike:
        return "vol_spike"

    if a >= DVOL_HIGH_TH:
        return "high_vol"
    if a <= DVOL_LOW_TH:
        return "low_vol"
    return "mid_vol"


async def _fetch_options_snapshot(last_good: Optional[OptionsSnapshot] = None) -> OptionsSnapshot:
    t0 = time.time()

    timeout = aiohttp.ClientTimeout(total=float(HTTP_TIMEOUT_S))
    btc_v: Optional[float] = None
    eth_v: Optional[float] = None
    btc_chg: Optional[float] = None
    eth_chg: Optional[float] = None
    raw_btc: Dict[str, Any] = {}
    raw_eth: Dict[str, Any] = {}

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [
                _fetch_dvol_with_session(session, "BTC", resolution=int(DVOL_RESOLUTION_S)),
                _fetch_dvol_with_session(session, "ETH", resolution=int(DVOL_RESOLUTION_S)),
            ]
            res = await asyncio.gather(*tasks, return_exceptions=True)

        # unpack (tolerate exceptions)
        if len(res) >= 2:
            if not isinstance(res[0], Exception):
                btc_v, btc_chg, raw_btc = res[0]  # type: ignore[misc]
            if not isinstance(res[1], Exception):
                eth_v, eth_chg, raw_eth = res[1]  # type: ignore[misc]
    except Exception as e:
        LOGGER.debug("[OPTIONS] snapshot fetch failed: %s", e)

    # compute avg
    vals = [v for v in (btc_v, eth_v) if isinstance(v, (int, float))]
    avg_dvol: Optional[float] = None
    if vals:
        try:
            avg_dvol = float(sum(vals) / len(vals))
        except Exception:
            avg_dvol = None

    # compute change pct (prefer BTC, else ETH, else None)
    dvol_change_24h_pct: Optional[float] = None
    if isinstance(btc_chg, (int, float)):
        dvol_change_24h_pct = float(btc_chg)
    elif isinstance(eth_chg, (int, float)):
        dvol_change_24h_pct = float(eth_chg)

    spike = False
    try:
        if dvol_change_24h_pct is not None and float(dvol_change_24h_pct) >= float(DVOL_SPIKE_PCT):
            spike = True
    except Exception:
        spike = False

    regime = _regime_from_avg(avg_dvol, spike)

    dt_ms = int((time.time() - t0) * 1000)
    LOGGER.info(
        "[OPTIONS] ms=%d dvol_btc=%s dvol_eth=%s avg=%s chg24h_pct=%s spike=%s regime=%s",
        dt_ms,
        f"{btc_v:.3g}" if isinstance(btc_v, (int, float)) else None,
        f"{eth_v:.3g}" if isinstance(eth_v, (int, float)) else None,
        f"{avg_dvol:.3g}" if isinstance(avg_dvol, (int, float)) else None,
        f"{dvol_change_24h_pct:.2f}" if isinstance(dvol_change_24h_pct, (int, float)) else None,
        spike,
        regime,
    )

    snap = OptionsSnapshot(
        ts=time.time(),
        dvol_btc=btc_v,
        dvol_eth=eth_v,
        avg_dvol=avg_dvol,
        dvol_change_24h_pct=dvol_change_24h_pct,
        spike=bool(spike),
        regime=regime,
        raw={"btc": raw_btc, "eth": raw_eth},
    )

    # If everything failed, optionally keep last good (NEW)
    if KEEP_LAST_GOOD_ON_ERROR:
        if (snap.avg_dvol is None) and last_good is not None and isinstance(last_good.avg_dvol, (int, float)):
            # keep regime context rather than "unknown"
            return OptionsSnapshot(
                ts=time.time(),
                dvol_btc=last_good.dvol_btc,
                dvol_eth=last_good.dvol_eth,
                avg_dvol=last_good.avg_dvol,
                dvol_change_24h_pct=last_good.dvol_change_24h_pct,
                spike=bool(last_good.spike),
                regime=str(last_good.regime or "unknown"),
                raw={"btc": raw_btc, "eth": raw_eth, "fallback_used": True},
            )

    return snap


# ---------------------------------------------------------------------
# Scoring / Position sizing adjustments
# ---------------------------------------------------------------------

def _is_trend_setup(setup_type: Optional[str]) -> bool:
    """
    Heuristic: trend/breakout setups benefit from mid/high vol,
    but extreme spike -> reduce size.
    """
    st = str(setup_type or "").upper()
    if st in ("BOS_STRICT", "RAID_DISPLACEMENT", "LIQ_SWEEP", "INST_CONTINUATION"):
        return True
    # allow TTL suffix variants e.g. BOS_STRICT_OTE / _FVG
    if st.startswith("BOS_STRICT") or st.startswith("RAID_DISPLACEMENT") or st.startswith("LIQ_SWEEP") or st.startswith("INST_CONTINUATION"):
        return True
    return False


def _risk_factor_from_regime(regime: str, *, setup_type: Optional[str] = None) -> Tuple[float, str]:
    """
    Returns:
      risk_factor (multiplier on risk or notional),
      position_mode: "reduce" | "neutral" | "increase"

    Non-blocking by design.
    """
    reg = str(regime or "unknown").lower()
    trend = _is_trend_setup(setup_type)

    # Defaults
    rf = 1.0
    mode = "neutral"

    if reg == "unknown":
        return 1.0, "neutral"

    # Spike: usually spreads/liquidations risk up -> reduce
    if reg == "vol_spike":
        return 0.75, "reduce"

    if reg == "high_vol":
        # high vol supports breakouts, but still riskier -> slight reduce
        return (0.85, "reduce") if trend else (0.90, "reduce")

    if reg == "mid_vol":
        # best regime for trend follow-through
        return (1.05, "increase") if trend else (1.0, "neutral")

    if reg == "low_vol":
        # low vol: fewer follow-through breakouts -> reduce for trend setups
        return (0.90, "reduce") if trend else (1.0, "neutral")

    return rf, mode


def score_options_context(opt: Any, bias: str, *, setup_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Non-blocking options vol context.

    Backward compatible keys:
      - ok, score, regime, reason

    NEW keys:
      - avg_dvol, dvol_change_24h_pct, spike
      - risk_factor, position_mode, setup_type

    score semantics (simple):
      +1 if vol is mid/high, else 0.
    """
    b = (bias or "").upper()
    _ = b  # bias kept for potential future use

    if not isinstance(opt, dict) and not isinstance(opt, OptionsSnapshot):
        return {
            "ok": True,
            "score": 0,
            "regime": "unknown",
            "reason": "no_options",
            "avg_dvol": None,
            "dvol_change_24h_pct": None,
            "spike": False,
            "risk_factor": 1.0,
            "position_mode": "neutral",
            "setup_type": setup_type,
        }

    regime = opt.regime if isinstance(opt, OptionsSnapshot) else str(opt.get("regime") or "unknown")
    avg_dvol = opt.avg_dvol if isinstance(opt, OptionsSnapshot) else opt.get("avg_dvol")
    chg = opt.dvol_change_24h_pct if isinstance(opt, OptionsSnapshot) else opt.get("dvol_change_24h_pct")
    spike = bool(opt.spike) if isinstance(opt, OptionsSnapshot) else bool(opt.get("spike", False))

    score = 0
    reason = "neutral"

    if regime in ("mid_vol", "high_vol"):
        score = 1
        reason = "vol_supports_trend"
    elif regime == "vol_spike":
        score = 1
        reason = "vol_spike_caution"
    elif regime == "low_vol":
        score = 0
        reason = "low_vol_neutral"
    else:
        score = 0
        reason = "unknown"

    rf, mode = _risk_factor_from_regime(regime, setup_type=setup_type)

    return {
        "ok": True,
        "score": int(score),
        "regime": str(regime),
        "reason": reason,
        "avg_dvol": float(avg_dvol) if isinstance(avg_dvol, (int, float)) else None,
        "dvol_change_24h_pct": float(chg) if isinstance(chg, (int, float)) else None,
        "spike": bool(spike),
        "risk_factor": float(rf),
        "position_mode": str(mode),
        "setup_type": setup_type,
    }


def apply_options_adjustment(
    signal: Dict[str, Any],
    opt: Any,
    *,
    setup_type: Optional[str] = None,
    field: str = "risk_factor",
) -> Dict[str, Any]:
    """
    Helper: attaches options context to a signal dict and returns a patched copy.

    Typical usage in analyze_signal or scanner:
      opt_ctx = score_options_context(opt_snap, bias, setup_type=setup_core)
      sig2 = apply_options_adjustment(sig, opt_ctx, setup_type=setup_core)

    If you want to actually scale order sizing:
      - multiply your notional/risk by sig2["options"]["risk_factor"]
      - keep it non-blocking (do NOT reject only because options unknown)
    """
    out = dict(signal or {})
    ctx = score_options_context(opt, out.get("bias") or out.get("side"), setup_type=setup_type)

    # attach options block
    out["options"] = ctx

    # convenience: expose risk_factor at top-level if you want
    try:
        rf = float(ctx.get("risk_factor", 1.0))
    except Exception:
        rf = 1.0

    out[field] = rf
    return out
