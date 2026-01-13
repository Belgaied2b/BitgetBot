from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import aiohttp

LOGGER = logging.getLogger(__name__)

# =====================================================================
# options_data.py — Deribit DVOL context (robuste + stable API)
# =====================================================================
# ✅ Single shared session (optional) + connector (keeps sockets healthy)
# ✅ TTL cache + "in-flight" dedupe (no stampede on refresh)
# ✅ Rate-limit friendly concurrency (BTC/ETH) + per-call timeout
# ✅ Better parsing tolerance (data shape variations)
# ✅ Keep-last-good logic improved: keeps entire snapshot consistently
# ✅ score_options_context accepts OptionsSnapshot OR dict OR raw snapshot dict
# ✅ risk_factor clamped (0.25..1.25 by default) + env overrides
# ✅ Always returns stable keys (so scanner never KeyError)
# =====================================================================

DERIBIT_BASE = os.getenv("DERIBIT_BASE", "https://www.deribit.com/api/v2").rstrip("/")
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

# Risk factor clamp (avoid crazy sizing)
RF_MIN = float(os.getenv("OPTIONS_RF_MIN", "0.25"))
RF_MAX = float(os.getenv("OPTIONS_RF_MAX", "1.25"))

# HTTP pool / reuse (optional)
_OPTIONS_HTTP_POOL = str(os.getenv("OPTIONS_HTTP_POOL", "1")).strip() == "1"
_TCP_LIMIT = int(os.getenv("OPTIONS_HTTP_TCP_LIMIT", "20"))
_DNS_TTL = int(os.getenv("OPTIONS_HTTP_DNS_TTL", "300"))


def _now_s() -> float:
    return time.time()


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:  # NaN
            return None
        return v
    except Exception:
        return None


def _clamp(v: float, lo: float, hi: float) -> float:
    try:
        x = float(v)
    except Exception:
        return 1.0
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@dataclass
class OptionsSnapshot:
    ts: float
    dvol_btc: Optional[float]
    dvol_eth: Optional[float]

    # extra stats
    avg_dvol: Optional[float]
    dvol_change_24h_pct: Optional[float]  # computed from BTC if possible, else ETH, else None
    spike: bool

    regime: str
    raw: Dict[str, Any]


# =====================================================================
# Shared session (optional)
# =====================================================================

_SESSION: Optional[aiohttp.ClientSession] = None
_SESSION_LOCK = asyncio.Lock()


async def _get_session() -> aiohttp.ClientSession:
    global _SESSION
    if not _OPTIONS_HTTP_POOL:
        timeout = aiohttp.ClientTimeout(total=float(HTTP_TIMEOUT_S))
        return aiohttp.ClientSession(timeout=timeout)

    if _SESSION is not None and not _SESSION.closed:
        return _SESSION

    async with _SESSION_LOCK:
        if _SESSION is not None and not _SESSION.closed:
            return _SESSION
        timeout = aiohttp.ClientTimeout(total=float(HTTP_TIMEOUT_S))
        connector = aiohttp.TCPConnector(limit=_TCP_LIMIT, ttl_dns_cache=_DNS_TTL, enable_cleanup_closed=True)
        _SESSION = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return _SESSION


async def close_options_session() -> None:
    """Optional: call on shutdown."""
    global _SESSION
    async with _SESSION_LOCK:
        if _SESSION is not None and not _SESSION.closed:
            try:
                await _SESSION.close()
            except Exception:
                pass
        _SESSION = None


# =====================================================================
# Cache + in-flight dedupe
# =====================================================================

class OptionsCache:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._snap: Optional[OptionsSnapshot] = None
        self._inflight: Optional[asyncio.Task] = None

    async def get(self, force: bool = False) -> OptionsSnapshot:
        now = _now_s()
        async with self._lock:
            if (not force) and self._snap and (now - float(self._snap.ts)) < float(OPTIONS_TTL_S):
                return self._snap

            # stampede protection: if refresh already running, await it
            if self._inflight and not self._inflight.done():
                task = self._inflight
            else:
                task = asyncio.create_task(_fetch_options_snapshot(last_good=self._snap))
                self._inflight = task

        try:
            snap = await task
        except Exception as e:
            LOGGER.debug("[OPTIONS] inflight fetch exception: %s", e)
            # fallback: last good if any, else minimal snapshot
            if self._snap is not None and KEEP_LAST_GOOD_ON_ERROR:
                return self._snap
            return OptionsSnapshot(
                ts=_now_s(),
                dvol_btc=None,
                dvol_eth=None,
                avg_dvol=None,
                dvol_change_24h_pct=None,
                spike=False,
                regime="unknown",
                raw={"error": "fetch_exception"},
            )

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
    end_ts = int(_now_s() * 1000)
    start_ts = end_ts - int(float(DVOL_WINDOW_H) * 3600 * 1000)

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
                return None, None, {"http_status": resp.status}
            raw = await resp.json()
    except Exception as e:
        LOGGER.debug("[OPTIONS] dvol fetch %s failed: %s", currency, e)
        return None, None, {"error": str(e)}

    close_v: Optional[float] = None
    chg_pct: Optional[float] = None

    # Expect: {"jsonrpc":"2.0","result":{"data":[[ts,o,h,l,c],...], ...}}
    try:
        res = raw.get("result") if isinstance(raw, dict) else None
        data = (res or {}).get("data") if isinstance(res, dict) else None

        # tolerate if data nested differently
        if isinstance(data, dict) and "data" in data:
            data = data.get("data")

        if isinstance(data, list) and len(data) >= 2:
            first = data[0]
            last = data[-1]

            if isinstance(last, (list, tuple)) and len(last) >= 5:
                close_v = _safe_float(last[4])

            if isinstance(first, (list, tuple)) and len(first) >= 5 and close_v is not None:
                first_close = _safe_float(first[4])
                if first_close and first_close > 0:
                    chg_pct = float((float(close_v) - float(first_close)) / float(first_close) * 100.0)
    except Exception:
        close_v = None
        chg_pct = None

    return close_v, chg_pct, (raw if isinstance(raw, dict) else {"raw_type": str(type(raw))})


def _regime_from_avg(avg: Optional[float], spike: bool) -> str:
    if avg is None:
        return "unknown"
    a = _safe_float(avg)
    if a is None:
        return "unknown"
    if spike:
        return "vol_spike"
    if a >= float(DVOL_HIGH_TH):
        return "high_vol"
    if a <= float(DVOL_LOW_TH):
        return "low_vol"
    return "mid_vol"


async def _fetch_options_snapshot(last_good: Optional[OptionsSnapshot] = None) -> OptionsSnapshot:
    t0 = _now_s()

    btc_v: Optional[float] = None
    eth_v: Optional[float] = None
    btc_chg: Optional[float] = None
    eth_chg: Optional[float] = None
    raw_btc: Dict[str, Any] = {}
    raw_eth: Dict[str, Any] = {}

    session: Optional[aiohttp.ClientSession] = None
    own_session = False

    try:
        session = await _get_session()
        own_session = not _OPTIONS_HTTP_POOL  # if pooling disabled, _get_session created a new one

        tasks = [
            _fetch_dvol_with_session(session, "BTC", resolution=int(DVOL_RESOLUTION_S)),
            _fetch_dvol_with_session(session, "ETH", resolution=int(DVOL_RESOLUTION_S)),
        ]
        res = await asyncio.gather(*tasks, return_exceptions=True)

        if len(res) >= 2:
            if not isinstance(res[0], Exception):
                btc_v, btc_chg, raw_btc = res[0]  # type: ignore[misc]
            else:
                raw_btc = {"error": "btc_exception", "detail": str(res[0])}
            if not isinstance(res[1], Exception):
                eth_v, eth_chg, raw_eth = res[1]  # type: ignore[misc]
            else:
                raw_eth = {"error": "eth_exception", "detail": str(res[1])}

    except Exception as e:
        LOGGER.debug("[OPTIONS] snapshot fetch failed: %s", e)
        raw_btc = {"error": str(e)}
        raw_eth = {"error": str(e)}
    finally:
        if own_session and session is not None:
            try:
                await session.close()
            except Exception:
                pass

    # avg
    vals = [v for v in (btc_v, eth_v) if isinstance(v, (int, float))]
    avg_dvol: Optional[float] = None
    if vals:
        try:
            avg_dvol = float(sum(vals) / len(vals))
        except Exception:
            avg_dvol = None

    # change pct (prefer BTC, else ETH)
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

    dt_ms = int((_now_s() - t0) * 1000)
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
        ts=_now_s(),
        dvol_btc=btc_v,
        dvol_eth=eth_v,
        avg_dvol=avg_dvol,
        dvol_change_24h_pct=dvol_change_24h_pct,
        spike=bool(spike),
        regime=regime,
        raw={"btc": raw_btc, "eth": raw_eth},
    )

    # Keep-last-good logic: only if this snapshot is effectively empty
    if KEEP_LAST_GOOD_ON_ERROR and last_good is not None:
        if (snap.avg_dvol is None) and (last_good.avg_dvol is not None):
            # preserve previous fields, but keep new timestamp + indicate fallback
            return OptionsSnapshot(
                ts=_now_s(),
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
    st = str(setup_type or "").upper()
    if st in ("BOS_STRICT", "RAID_DISPLACEMENT", "LIQ_SWEEP", "INST_CONTINUATION"):
        return True
    if st.startswith("BOS_STRICT") or st.startswith("RAID_DISPLACEMENT") or st.startswith("LIQ_SWEEP") or st.startswith("INST_CONTINUATION"):
        return True
    return False


def _risk_factor_from_regime(regime: str, *, setup_type: Optional[str] = None) -> Tuple[float, str]:
    """
    Returns:
      (risk_factor, position_mode)

    NOTE: risk_factor is clamped by OPTIONS_RF_MIN/OPTIONS_RF_MAX.
    """
    reg = str(regime or "unknown").lower()
    trend = _is_trend_setup(setup_type)

    if reg == "unknown":
        return 1.0, "neutral"

    if reg == "vol_spike":
        return _clamp(0.75, RF_MIN, RF_MAX), "reduce"

    if reg == "high_vol":
        rf = 0.85 if trend else 0.90
        return _clamp(rf, RF_MIN, RF_MAX), "reduce"

    if reg == "mid_vol":
        rf = 1.05 if trend else 1.0
        return _clamp(rf, RF_MIN, RF_MAX), ("increase" if trend else "neutral")

    if reg == "low_vol":
        rf = 0.90 if trend else 1.0
        return _clamp(rf, RF_MIN, RF_MAX), ("reduce" if trend else "neutral")

    return 1.0, "neutral"


def _coerce_snapshot(opt: Any) -> Optional[OptionsSnapshot]:
    """
    Accept:
      - OptionsSnapshot
      - dict with regime/avg_dvol/...
      - dict containing {"options": {...}} style
    """
    if isinstance(opt, OptionsSnapshot):
        return opt

    if isinstance(opt, dict):
        # unwrap common nesting patterns
        if "options" in opt and isinstance(opt.get("options"), dict):
            opt = opt["options"]

        regime = str(opt.get("regime") or "unknown")
        avg_dvol = _safe_float(opt.get("avg_dvol"))
        chg = _safe_float(opt.get("dvol_change_24h_pct"))
        spike = bool(opt.get("spike", False))

        # allow raw snapshot pass-through
        return OptionsSnapshot(
            ts=_now_s(),
            dvol_btc=_safe_float(opt.get("dvol_btc")),
            dvol_eth=_safe_float(opt.get("dvol_eth")),
            avg_dvol=avg_dvol,
            dvol_change_24h_pct=chg,
            spike=spike,
            regime=regime,
            raw=dict(opt.get("raw") or {}),
        )

    return None


def score_options_context(opt: Any, bias: str, *, setup_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Non-blocking options vol context.

    Stable keys:
      ok, score, regime, reason,
      avg_dvol, dvol_change_24h_pct, spike,
      risk_factor, position_mode, setup_type
    """
    _ = (bias or "").upper()  # reserved for future use

    snap = _coerce_snapshot(opt)
    if snap is None:
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

    regime = str(snap.regime or "unknown")
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
        "regime": regime,
        "reason": reason,
        "avg_dvol": float(snap.avg_dvol) if isinstance(snap.avg_dvol, (int, float)) else None,
        "dvol_change_24h_pct": float(snap.dvol_change_24h_pct) if isinstance(snap.dvol_change_24h_pct, (int, float)) else None,
        "spike": bool(snap.spike),
        "risk_factor": float(_clamp(rf, RF_MIN, RF_MAX)),
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
    Attaches options context to a signal dict and returns a patched copy.

    Usage:
      opt_ctx = score_options_context(opt_snap, bias, setup_type=setup_core)
      sig2 = apply_options_adjustment(sig, opt_ctx, setup_type=setup_core)

    If you want to actually scale order sizing:
      multiply your notional/risk by sig2["options"]["risk_factor"] (or sig2[field]).
    """
    out = dict(signal or {})
    ctx = score_options_context(opt, out.get("bias") or out.get("side"), setup_type=setup_type)

    out["options"] = ctx

    try:
        rf = float(ctx.get("risk_factor", 1.0))
    except Exception:
        rf = 1.0
    out[field] = float(_clamp(rf, RF_MIN, RF_MAX))

    return out
