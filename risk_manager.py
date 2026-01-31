# =====================================================================
# risk_manager.py — Desk Supreme Risk Engine (Desk-lead Hardened)
# =====================================================================
# ✅ Stable daily state reset
# ✅ Thread-safe (asyncio.Lock) for scanner concurrency
# ✅ Proper "reserve then confirm" flow (avoid double-counting on order fail)
# ✅ Exposure caps (gross + per symbol) using settings.py values
# ✅ Optional RR / inst_score gating (lightweight, not blocking by default)
# ✅ Tilt cooldown + daily loss hard stop
# ✅ Snapshot for logs/monitoring
#
# NEW (non-breaking):
# ✅ Reservation TTL cleanup (stale reservations won't block exposure forever)
# ✅ Partial fills support (confirm_open filled_notional/filled_risk + apply_fill)
# ✅ Time-stop helpers (scanner can query due positions)
# ✅ Optional volatility-targeted risk sizing (if you pass volatility_atr_pct)
# ✅ Net RR helper (compute_rr_net) + rr_net support in gating when provided
# ✅ Safety fix: forbid opening a new position on a symbol that already has one
#
# UPGRADE #2 (Portfolio / Institutional):
# ✅ Cluster exposure cap for high-BTC-correlation symbols:
#    - if abs(corr_btc) >= CORR_BTC_THRESHOLD -> cluster = BTC_BETA_HIGH
#    - enforce cap: cluster_notional_after <= CORR_GROUP_CAP * equity
#    - includes open + reservations (concurrency-safe)
#    - stored in Reservation/Position for monitoring
#
# FIXES (this version):
# ✅ can_trade() now lock-protected (prevents race with concurrent reserves)
# ✅ reserve_trade prevents double-reservation on same symbol (safety)
# ✅ TTL purge returns metrics + avoids silent stuck states
# ✅ caps sanitized (no negative / NaN)
# =====================================================================

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional, List

from settings import (
    RISK_USDT,
    ACCOUNT_EQUITY_USDT,
    MAX_GROSS_EXPOSURE,
    MAX_SYMBOL_EXPOSURE,
    RR_MIN_STRICT,
    RR_MIN_TOLERATED_WITH_INST,
    MIN_INST_SCORE,
    DESK_EV_MODE,
    MAX_DAILY_LOSS,
    MAX_TRADES_PER_DAY,
    MAX_OPEN_POSITIONS,
    MAX_LONG_POSITIONS,
    MAX_SHORT_POSITIONS,
    MAX_CONSECUTIVE_LOSSES,
    TILT_COOLDOWN_SECONDS,
    DRAWDOWN_RISK_FACTOR,
    SYMBOL_MAX_DAILY_LOSS,
    SYMBOL_MAX_TRADES_PER_DAY,
    CORR_GROUP_CAP,
    CORR_BTC_THRESHOLD,
)

LOGGER = logging.getLogger(__name__)


def _sf(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return float(default)
        return v
    except Exception:
        return float(default)


def _si(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        v = float(x)
    except Exception:
        return float(lo)
    if v < lo:
        return float(lo)
    if v > hi:
        return float(hi)
    return float(v)


# =====================================================================
# CONFIG
# =====================================================================

@dataclass
class RiskConfig:
    risk_per_trade: float = float(RISK_USDT)
    max_daily_loss: float = float(MAX_DAILY_LOSS)
    max_trades_per_day: int = int(MAX_TRADES_PER_DAY)

    max_open_positions: int = int(MAX_OPEN_POSITIONS)
    max_long_positions: int = int(MAX_LONG_POSITIONS)
    max_short_positions: int = int(MAX_SHORT_POSITIONS)

    max_consecutive_losses: int = int(MAX_CONSECUTIVE_LOSSES)
    tilt_cooldown_seconds: int = int(TILT_COOLDOWN_SECONDS)

    drawdown_risk_factor: float = float(DRAWDOWN_RISK_FACTOR)

    account_equity_usdt: float = float(ACCOUNT_EQUITY_USDT)
    max_gross_exposure: float = float(MAX_GROSS_EXPOSURE)
    max_symbol_exposure: float = float(MAX_SYMBOL_EXPOSURE)

    corr_btc_threshold: float = float(CORR_BTC_THRESHOLD)
    corr_group_cap: float = float(CORR_GROUP_CAP)

    reservation_ttl_seconds: int = 6 * 60
    max_position_age_seconds: int = 12 * 60 * 60

    use_volatility_targeting: bool = False
    target_atr_pct: float = 0.010
    vol_risk_min_factor: float = 0.35
    vol_risk_max_factor: float = 1.25

    symbol_max_daily_loss: float = float(SYMBOL_MAX_DAILY_LOSS)
    symbol_max_trades_per_day: int = int(SYMBOL_MAX_TRADES_PER_DAY)

    day_reset_use_utc: bool = False

    def sanitize(self) -> None:
        self.risk_per_trade = max(0.0, _sf(self.risk_per_trade, 0.0))
        self.max_daily_loss = max(0.0, _sf(self.max_daily_loss, 0.0))
        self.max_trades_per_day = max(0, _si(self.max_trades_per_day, 0))

        self.max_open_positions = max(0, _si(self.max_open_positions, 0))
        self.max_long_positions = max(0, _si(self.max_long_positions, 0))
        self.max_short_positions = max(0, _si(self.max_short_positions, 0))

        self.max_consecutive_losses = max(0, _si(self.max_consecutive_losses, 0))
        self.tilt_cooldown_seconds = max(0, _si(self.tilt_cooldown_seconds, 0))

        self.drawdown_risk_factor = _clamp(_sf(self.drawdown_risk_factor, 1.0), 0.0, 1.0)

        self.account_equity_usdt = max(0.0, _sf(self.account_equity_usdt, 0.0))
        self.max_gross_exposure = max(0.0, _sf(self.max_gross_exposure, 0.0))
        self.max_symbol_exposure = max(0.0, _sf(self.max_symbol_exposure, 0.0))

        self.corr_btc_threshold = _clamp(_sf(self.corr_btc_threshold, 0.0), 0.0, 1.0)
        self.corr_group_cap = max(0.0, _sf(self.corr_group_cap, 0.0))

        self.reservation_ttl_seconds = max(30, _si(self.reservation_ttl_seconds, 360))
        self.max_position_age_seconds = max(60, _si(self.max_position_age_seconds, 3600))

        self.target_atr_pct = max(1e-6, _sf(self.target_atr_pct, 0.01))
        self.vol_risk_min_factor = _clamp(_sf(self.vol_risk_min_factor, 0.35), 0.05, 2.0)
        self.vol_risk_max_factor = _clamp(_sf(self.vol_risk_max_factor, 1.25), self.vol_risk_min_factor, 3.0)

        self.symbol_max_daily_loss = max(0.0, _sf(self.symbol_max_daily_loss, 0.0))
        self.symbol_max_trades_per_day = max(0, _si(self.symbol_max_trades_per_day, 0))


# =====================================================================
# INTERNAL STATE
# =====================================================================

@dataclass
class DailyState:
    date_key: str
    trades_opened: int = 0
    pnl: float = 0.0
    losses_count: int = 0
    symbol_pnl: Dict[str, float] = field(default_factory=dict)
    symbol_trades: Dict[str, int] = field(default_factory=dict)


@dataclass
class PositionState:
    symbol: str
    side: str
    notional: float
    risk: float
    opened_at: float = field(default_factory=lambda: time.time())

    filled_notional: float = 0.0
    avg_entry: Optional[float] = None

    cluster: Optional[str] = None
    corr_btc: Optional[float] = None


@dataclass
class Reservation:
    rid: str
    symbol: str
    side: str
    notional: float
    risk: float
    created_at: float = field(default_factory=lambda: time.time())

    cluster: Optional[str] = None
    corr_btc: Optional[float] = None


# =====================================================================
# Risk Manager
# =====================================================================

class RiskManager:
    def __init__(self, config: Optional[RiskConfig] = None):
        self.config: RiskConfig = config or RiskConfig()
        self.config.sanitize()

        self._daily: Optional[DailyState] = None
        self.open_positions: Dict[str, PositionState] = {}
        self._reservations: Dict[str, Reservation] = {}
        self.direction_counts = {"LONG": 0, "SHORT": 0}

        self._tilt_active: bool = False
        self._tilt_activated_at: float = 0.0

        self._lock = asyncio.Lock()

        # metrics
        self._last_ttl_purge_ts: float = 0.0
        self._ttl_purged_total: int = 0

    # ------------------------------------------------------------------
    # daily helpers
    # ------------------------------------------------------------------

    def _current_date_key(self) -> str:
        if bool(self.config.day_reset_use_utc):
            return time.strftime("%Y-%m-%d", time.gmtime())
        return time.strftime("%Y-%m-%d", time.localtime())

    def _purge_expired_reservations_nolock(self) -> int:
        """
        Prevent stale reservations from blocking exposure caps forever.
        Returns number purged.
        """
        purged = 0
        try:
            ttl = int(max(30, self.config.reservation_ttl_seconds))
            now = time.time()
            expired = [rid for rid, r in self._reservations.items() if (now - float(r.created_at)) >= ttl]
            for rid in expired:
                self._reservations.pop(rid, None)
                purged += 1
        except Exception:
            return purged

        if purged:
            self._ttl_purged_total += int(purged)
            self._last_ttl_purge_ts = time.time()
        return purged

    def _ensure_daily_state_nolock(self) -> None:
        today = self._current_date_key()
        if self._daily is None or self._daily.date_key != today:
            self._daily = DailyState(date_key=today)
            self._tilt_active = False
            self._tilt_activated_at = 0.0
            self._reservations.clear()
        else:
            if self._daily.symbol_pnl is None:
                self._daily.symbol_pnl = {}
            if self._daily.symbol_trades is None:
                self._daily.symbol_trades = {}
        self._purge_expired_reservations_nolock()

    def _daily_loss_nolock(self) -> float:
        self._ensure_daily_state_nolock()
        return float(self._daily.pnl if self._daily else 0.0)

    def _daily_trades_nolock(self) -> int:
        self._ensure_daily_state_nolock()
        return int(self._daily.trades_opened if self._daily else 0)

    def _daily_losses_nolock(self) -> int:
        self._ensure_daily_state_nolock()
        return int(self._daily.losses_count if self._daily else 0)

    def _is_tilt_active_nolock(self) -> bool:
        if not self._tilt_active:
            return False
        elapsed = time.time() - self._tilt_activated_at
        if elapsed >= self.config.tilt_cooldown_seconds:
            self._tilt_active = False
            self._tilt_activated_at = 0.0
            return False
        return True

    # ------------------------------------------------------------------
    # normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _norm_side(side: str) -> str:
        s = (side or "").upper()
        if s == "BUY":
            return "LONG"
        if s == "SELL":
            return "SHORT"
        if s in ("LONG", "SHORT"):
            return s
        return "LONG"

    @staticmethod
    def _norm_symbol(symbol: str) -> str:
        return (symbol or "UNKNOWN").upper().strip()

    @staticmethod
    def _norm_cluster(cluster: Optional[str]) -> Optional[str]:
        if cluster is None:
            return None
        c = str(cluster).strip()
        return c if c else None

    def _derive_cluster(self, corr_btc: Optional[float], cluster: Optional[str]) -> Optional[str]:
        c_in = self._norm_cluster(cluster)
        if c_in:
            return c_in

        v = _safe_corr(corr_btc)
        if v is None:
            return None

        if abs(v) >= float(self.config.corr_btc_threshold):
            return "BTC_BETA_HIGH"
        return None

    # ------------------------------------------------------------------
    # exposure computations
    # ------------------------------------------------------------------

    def _gross_open_notional_nolock(self) -> float:
        return float(sum(_sf(p.notional, 0.0) for p in self.open_positions.values()))

    def _gross_reserved_notional_nolock(self) -> float:
        return float(sum(_sf(r.notional, 0.0) for r in self._reservations.values()))

    def _symbol_open_notional_nolock(self, symbol: str) -> float:
        sym = self._norm_symbol(symbol)
        p = self.open_positions.get(sym)
        return float(_sf(p.notional, 0.0)) if p else 0.0

    def _symbol_reserved_notional_nolock(self, symbol: str) -> float:
        sym = self._norm_symbol(symbol)
        return float(sum(_sf(r.notional, 0.0) for r in self._reservations.values() if r.symbol == sym))

    def _cluster_open_notional_nolock(self, cluster: Optional[str]) -> float:
        c = self._norm_cluster(cluster)
        if not c:
            return 0.0
        return float(sum(_sf(p.notional, 0.0) for p in self.open_positions.values() if (p.cluster or None) == c))

    def _cluster_reserved_notional_nolock(self, cluster: Optional[str]) -> float:
        c = self._norm_cluster(cluster)
        if not c:
            return 0.0
        return float(sum(_sf(r.notional, 0.0) for r in self._reservations.values() if (r.cluster or None) == c))

    def _symbol_has_reservation_nolock(self, symbol: str) -> bool:
        sym = self._norm_symbol(symbol)
        for r in self._reservations.values():
            if r.symbol == sym:
                return True
        return False

    # ------------------------------------------------------------------
    # Level 1 gate: can_open (nolock version)
    # ------------------------------------------------------------------

    def _can_open_nolock(self, symbol: str, side: str) -> Tuple[bool, str]:
        self._ensure_daily_state_nolock()
        sym = self._norm_symbol(symbol)
        s = self._norm_side(side)

        if self._is_tilt_active_nolock():
            return False, "tilt_cooldown"

        if self._daily_trades_nolock() >= int(self.config.max_trades_per_day):
            return False, "max_trades_per_day_reached"

        if self._daily_loss_nolock() <= -abs(float(self.config.max_daily_loss)):
            return False, "max_daily_loss_reached"

        if len(self.open_positions) >= int(self.config.max_open_positions):
            return False, "max_open_positions_reached"

        if s == "LONG" and self.direction_counts.get("LONG", 0) >= int(self.config.max_long_positions):
            return False, "max_long_exposure"

        if s == "SHORT" and self.direction_counts.get("SHORT", 0) >= int(self.config.max_short_positions):
            return False, "max_short_exposure"

        # SAFETY: already open on symbol
        if sym in self.open_positions:
            return False, "position_already_open_symbol"

        # SAFETY: already reserved on symbol (prevents double reservation in same scan)
        if self._symbol_has_reservation_nolock(sym):
            return False, "symbol_already_reserved"

        return True, "OK"

    def can_open(self, symbol: str, side: str) -> Tuple[bool, str]:
        """
        Sync helper (no lock). Prefer reserve_trade() in scanner.
        """
        # keep backward behavior; core concurrency-safe checks happen under reserve_trade lock
        self._ensure_daily_state_nolock()
        return self._can_open_nolock(symbol, side)

    # ------------------------------------------------------------------
    # Level 2 gate: can_trade (legacy sync) — NOW LOCKED
    # ------------------------------------------------------------------

    def can_trade(self, *args: Any, **kwargs: Any) -> Tuple[bool, str]:
        """
        Legacy API: returns (allowed, reason). DOES NOT reserve.
        Made lock-protected to avoid races with concurrent reserve_trade().
        """
        extracted = self._extract_args(args, kwargs)

        async def _run() -> Tuple[bool, str]:
            async with self._lock:
                self._ensure_daily_state_nolock()
                allowed, reason = self._can_trade_core_nolock(**extracted)
                return (bool(allowed), str(reason))

        # if called from async context, user should call reserve_trade anyway.
        # but keep safe: run loop if possible, else fallback to nolock (best-effort).
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # cannot block; best effort
                allowed, reason = self._can_trade_core_nolock(**extracted)
                return (bool(allowed), str(reason) if allowed else str(reason))
        except Exception:
            pass

        try:
            return asyncio.run(_run())  # type: ignore[arg-type]
        except Exception:
            allowed, reason = self._can_trade_core_nolock(**extracted)
            return (bool(allowed), str(reason))

    # ------------------------------------------------------------------
    # Recommended async API: reserve/confirm/cancel
    # ------------------------------------------------------------------

    async def reserve_trade(
        self,
        *,
        symbol: str,
        side: str,
        notional: float,
        rr: Optional[float] = None,
        rr_net: Optional[float] = None,
        inst_score: Optional[int] = None,
        commitment: Optional[float] = None,
        volatility_atr_pct: Optional[float] = None,
        corr_btc: Optional[float] = None,
        cluster: Optional[str] = None,
        **_extra: Any,
    ) -> Tuple[bool, str, Optional[str]]:
        async with self._lock:
            self._ensure_daily_state_nolock()

            sym = self._norm_symbol(symbol)
            s = self._norm_side(side)

            derived_cluster = self._derive_cluster(corr_btc, cluster)

            allowed, reason = self._can_trade_core_nolock(
                symbol=sym,
                side=s,
                notional=notional,
                rr=rr,
                rr_net=rr_net,
                inst_score=inst_score,
                commitment=commitment,
                volatility_atr_pct=volatility_atr_pct,
                corr_btc=corr_btc,
                cluster=derived_cluster,
            )
            if not allowed:
                return False, str(reason), None

            risk_used = self.risk_for_this_trade(volatility_atr_pct=volatility_atr_pct)

            rid = uuid.uuid4().hex[:12]
            self._reservations[rid] = Reservation(
                rid=rid,
                symbol=sym,
                side=s,
                notional=float(max(0.0, _sf(notional, 0.0))),
                risk=float(max(0.0, _sf(risk_used, 0.0))),
                cluster=derived_cluster,
                corr_btc=_safe_corr(corr_btc),
            )
            return True, "OK", rid

    async def cancel_reservation(self, rid: Optional[str]) -> None:
        if not rid:
            return
        async with self._lock:
            self._reservations.pop(str(rid), None)

    async def confirm_open(
        self,
        rid: str,
        *,
        filled_notional: Optional[float] = None,
        filled_risk: Optional[float] = None,
        avg_entry: Optional[float] = None,
        **_extra: Any,
    ) -> None:
        async with self._lock:
            self._ensure_daily_state_nolock()
            r = self._reservations.pop(str(rid), None)
            if not r:
                return

            notional = float(_sf(filled_notional, r.notional)) if filled_notional is not None else float(r.notional)
            risk = float(_sf(filled_risk, r.risk)) if filled_risk is not None else float(r.risk)

            if notional <= 0:
                return

            # safety: prevent duplicate open (shouldn’t happen but keep hard)
            if r.symbol in self.open_positions:
                return

            self._register_open_nolock(
                r.symbol,
                r.side,
                notional,
                risk,
                filled_notional=notional,
                avg_entry=avg_entry,
                cluster=r.cluster,
                corr_btc=r.corr_btc,
            )

    async def apply_fill(
        self,
        symbol: str,
        side: str,
        *,
        fill_notional_delta: float,
        risk_delta: float = 0.0,
        avg_entry: Optional[float] = None,
    ) -> None:
        async with self._lock:
            self._ensure_daily_state_nolock()
            sym = self._norm_symbol(symbol)
            s = self._norm_side(side)

            p = self.open_positions.get(sym)
            if not p or p.side != s:
                return

            dn = float(_sf(fill_notional_delta, 0.0))
            dr = float(_sf(risk_delta, 0.0))

            p.notional = float(max(0.0, _sf(p.notional, 0.0) + dn))
            p.risk = float(max(0.0, _sf(p.risk, 0.0) + dr))
            p.filled_notional = float(max(0.0, _sf(p.filled_notional, 0.0) + dn))

            if avg_entry is not None:
                p.avg_entry = float(avg_entry)

            if p.notional <= 0.0:
                self.open_positions.pop(sym, None)
                self.direction_counts[s] = max(0, self.direction_counts.get(s, 0) - 1)

    # ------------------------------------------------------------------
    # Risk sizing
    # ------------------------------------------------------------------

    def risk_for_this_trade(self, *, volatility_atr_pct: Optional[float] = None) -> float:
        self.config.sanitize()
        self._ensure_daily_state_nolock()

        base = float(max(0.0, self.config.risk_per_trade))

        dloss = float(self._daily_loss_nolock())
        if dloss < -2.0 * base and base > 0:
            base = float(base * float(self.config.drawdown_risk_factor))

        if not bool(self.config.use_volatility_targeting):
            return float(base)

        v = None
        try:
            if volatility_atr_pct is not None:
                v = float(volatility_atr_pct)
        except Exception:
            v = None

        if v is None or v <= 0:
            return float(base)

        tgt = float(max(1e-9, self.config.target_atr_pct))
        factor = float(tgt / v)
        factor = float(max(self.config.vol_risk_min_factor, min(self.config.vol_risk_max_factor, factor)))
        return float(base * factor)

    # ------------------------------------------------------------------
    # Register open/close
    # ------------------------------------------------------------------

    def register_open(self, symbol: str, side: str, notional: float, risk: float) -> None:
        self._ensure_daily_state_nolock()
        sym = self._norm_symbol(symbol)
        s = self._norm_side(side)

        if sym in self.open_positions:
            return
        if self._symbol_has_reservation_nolock(sym):
            return

        self._register_open_nolock(sym, s, float(_sf(notional, 0.0)), float(_sf(risk, 0.0)), filled_notional=float(_sf(notional, 0.0)))

    def _register_open_nolock(
        self,
        sym: str,
        side: str,
        notional: float,
        risk: float,
        *,
        filled_notional: float = 0.0,
        avg_entry: Optional[float] = None,
        cluster: Optional[str] = None,
        corr_btc: Optional[float] = None,
    ) -> None:
        self.open_positions[sym] = PositionState(
            symbol=sym,
            side=side,
            notional=float(max(0.0, _sf(notional, 0.0))),
            risk=float(max(0.0, _sf(risk, 0.0))),
            filled_notional=float(max(0.0, _sf(filled_notional, 0.0))),
            avg_entry=float(avg_entry) if avg_entry is not None else None,
            cluster=self._norm_cluster(cluster),
            corr_btc=_safe_corr(corr_btc),
        )
        self.direction_counts[side] = self.direction_counts.get(side, 0) + 1
        if self._daily:
            self._daily.trades_opened += 1
            self._daily.symbol_trades[sym] = int(self._daily.symbol_trades.get(sym, 0)) + 1

    def register_closed(self, symbol: str, side: str, pnl: float) -> None:
        self._ensure_daily_state_nolock()
        sym = self._norm_symbol(symbol)
        s = self._norm_side(side)

        if self._daily:
            self._daily.pnl += float(_sf(pnl, 0.0))
            self._daily.symbol_pnl[sym] = float(_sf(self._daily.symbol_pnl.get(sym, 0.0), 0.0) + float(_sf(pnl, 0.0)))

            if float(_sf(pnl, 0.0)) < 0:
                self._daily.losses_count += 1
            else:
                self._daily.losses_count = 0

            if self._daily.losses_count >= int(self.config.max_consecutive_losses) and int(self.config.max_consecutive_losses) > 0:
                self._tilt_active = True
                self._tilt_activated_at = time.time()

        pos = self.open_positions.pop(sym, None)
        if pos is not None:
            self.direction_counts[pos.side] = max(0, self.direction_counts.get(pos.side, 0) - 1)
        else:
            self.direction_counts[s] = max(0, self.direction_counts.get(s, 0) - 1)

    # ------------------------------------------------------------------
    # Time-stop helpers
    # ------------------------------------------------------------------

    def position_age_seconds(self, symbol: str) -> Optional[float]:
        sym = self._norm_symbol(symbol)
        p = self.open_positions.get(sym)
        if not p:
            return None
        return float(max(0.0, time.time() - float(p.opened_at)))

    def time_stop_due(self, symbol: str, *, max_age_seconds: Optional[int] = None) -> bool:
        p_age = self.position_age_seconds(symbol)
        if p_age is None:
            return False
        limit = int(max_age_seconds) if max_age_seconds is not None else int(self.config.max_position_age_seconds)
        return bool(p_age >= float(max(60, limit)))

    def positions_due_time_stop(self, *, max_age_seconds: Optional[int] = None) -> List[Dict[str, Any]]:
        limit = int(max_age_seconds) if max_age_seconds is not None else int(self.config.max_position_age_seconds)
        limit = int(max(60, limit))
        out: List[Dict[str, Any]] = []
        for sym, p in self.open_positions.items():
            age = float(max(0.0, time.time() - float(p.opened_at)))
            if age >= limit:
                out.append({
                    "symbol": sym,
                    "side": p.side,
                    "age_s": age,
                    "notional": float(p.notional),
                    "risk": float(p.risk),
                    "cluster": p.cluster,
                    "corr_btc": p.corr_btc,
                })
        return out

    # ------------------------------------------------------------------
    # Core gating (nolock, called under reserve_trade lock)
    # ------------------------------------------------------------------

    def _can_trade_core_nolock(
        self,
        *,
        symbol: str,
        side: str,
        notional: float,
        rr: Optional[float],
        rr_net: Optional[float],
        inst_score: Optional[int],
        commitment: Optional[float],
        volatility_atr_pct: Optional[float],
        corr_btc: Optional[float] = None,
        cluster: Optional[str] = None,
    ) -> Tuple[bool, str]:
        self.config.sanitize()
        self._ensure_daily_state_nolock()

        sym = self._norm_symbol(symbol)
        s = self._norm_side(side)

        notional_f = float(_sf(notional, 0.0))
        if notional_f <= 0:
            return False, "notional_invalid"

        # per-symbol daily caps
        if self._daily:
            if float(self.config.symbol_max_daily_loss) > 0:
                sym_pnl = float(_sf(self._daily.symbol_pnl.get(sym, 0.0), 0.0))
                if sym_pnl <= -abs(float(self.config.symbol_max_daily_loss)):
                    return False, "symbol_daily_loss_cap"
            if int(self.config.symbol_max_trades_per_day) > 0:
                sym_trades = int(_si(self._daily.symbol_trades.get(sym, 0), 0))
                if sym_trades >= int(self.config.symbol_max_trades_per_day):
                    return False, "symbol_trades_cap"

        allowed, reason = self._can_open_nolock(sym, s)
        if not allowed:
            return False, reason

        eq = float(self.config.account_equity_usdt)
        gross_cap = float(self.config.max_gross_exposure) * eq
        sym_cap = float(self.config.max_symbol_exposure) * eq

        gross_after = self._gross_open_notional_nolock() + self._gross_reserved_notional_nolock() + notional_f
        if gross_cap > 0 and gross_after > gross_cap:
            return False, "gross_exposure_cap"

        sym_after = self._symbol_open_notional_nolock(sym) + self._symbol_reserved_notional_nolock(sym) + notional_f
        if sym_cap > 0 and sym_after > sym_cap:
            return False, "symbol_exposure_cap"

        derived_cluster = self._derive_cluster(corr_btc, cluster)
        if derived_cluster == "BTC_BETA_HIGH":
            cluster_cap = float(self.config.corr_group_cap) * eq
            cluster_after = self._cluster_open_notional_nolock(derived_cluster) + self._cluster_reserved_notional_nolock(derived_cluster) + notional_f
            if cluster_cap > 0 and cluster_after > cluster_cap:
                return False, "cluster_exposure_cap_btc_beta_high"

        # RR gating (optional)
        rr_used = None
        if rr_net is not None:
            try:
                rr_used = float(rr_net)
            except Exception:
                rr_used = None
        if rr_used is None and rr is not None:
            try:
                rr_used = float(rr)
            except Exception:
                rr_used = None

        if rr_used is not None and rr_used > 0:
            iscore = None
            try:
                if inst_score is not None:
                    iscore = int(inst_score)
            except Exception:
                iscore = None

            if (iscore is None or iscore < int(MIN_INST_SCORE)) and rr_used < float(RR_MIN_STRICT):
                return False, "rr_below_strict_no_inst"

            if iscore is not None and iscore >= int(MIN_INST_SCORE):
                if rr_used < float(RR_MIN_TOLERATED_WITH_INST):
                    return False, "rr_below_tolerated_even_with_inst"

        if DESK_EV_MODE and commitment is not None:
            pass

        if self.config.use_volatility_targeting and volatility_atr_pct is not None:
            try:
                v = float(volatility_atr_pct)
                if v > 0.0 and v >= 4.5 * float(self.config.target_atr_pct):
                    return False, "volatility_too_high"
            except Exception:
                pass

        return True, "OK"

    # ------------------------------------------------------------------
    # Args extraction (compat)
    # ------------------------------------------------------------------

    def _extract_args(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        symbol = kwargs.get("symbol")
        side = kwargs.get("side")
        notional = kwargs.get("notional") or kwargs.get("notional_usdt") or kwargs.get("size_notional")
        rr = kwargs.get("rr") or kwargs.get("rr_actual")
        rr_net = kwargs.get("rr_net") or kwargs.get("rr_after_fees")
        inst_score = kwargs.get("inst_score") or kwargs.get("institutional_score")
        commitment = kwargs.get("commitment")

        volatility_atr_pct = (
            kwargs.get("volatility_atr_pct")
            or kwargs.get("atr_pct")
            or kwargs.get("vol_atr_pct")
            or kwargs.get("volatility")
        )

        corr_btc = (
            kwargs.get("corr_btc")
            or kwargs.get("btc_corr")
            or kwargs.get("corr_to_btc")
            or kwargs.get("btc_correlation")
        )

        cluster = kwargs.get("cluster") or kwargs.get("corr_group")

        str_args = [a for a in args if isinstance(a, str)]
        num_args = [a for a in args if isinstance(a, (int, float))]

        if symbol is None and str_args:
            symbol = str_args[0]
        if side is None and len(str_args) >= 2:
            side = str_args[1]

        if notional is None and num_args:
            notional = float(max(float(x) for x in num_args))

        if rr is None and num_args:
            cands = [float(x) for x in num_args if 0 < float(x) < 10]
            if cands:
                rr = min(cands)

        if symbol is None:
            symbol = "UNKNOWN"
        if side is None:
            side = "LONG"
        if notional is None:
            notional = float(self.config.risk_per_trade * 10.0)

        iscore = None
        try:
            if inst_score is not None:
                iscore = int(inst_score)
        except Exception:
            iscore = None

        com = None
        try:
            if commitment is not None:
                com = float(commitment)
        except Exception:
            com = None

        rr_f = None
        rr_net_f = None
        try:
            if rr is not None:
                rr_f = float(rr)
        except Exception:
            rr_f = None
        try:
            if rr_net is not None:
                rr_net_f = float(rr_net)
        except Exception:
            rr_net_f = None

        vol_f = None
        try:
            if volatility_atr_pct is not None:
                vol_f = float(volatility_atr_pct)
        except Exception:
            vol_f = None

        corr_f = _safe_corr(corr_btc)

        return {
            "symbol": str(symbol),
            "side": str(side),
            "notional": float(_sf(notional, 0.0)),
            "rr": rr_f,
            "rr_net": rr_net_f,
            "inst_score": iscore,
            "commitment": com,
            "volatility_atr_pct": vol_f,
            "corr_btc": corr_f,
            "cluster": self._norm_cluster(cluster),
        }

    # ------------------------------------------------------------------
    # Net RR helper
    # ------------------------------------------------------------------

    @staticmethod
    def compute_rr_net(
        *,
        entry: float,
        stop: float,
        tp: float,
        side: str,
        fee_rate_roundtrip: float = 0.0,
        funding_rate_per_hour: float = 0.0,
        funding_hours: float = 0.0,
        slippage_bps_roundtrip: float = 0.0,
    ) -> Optional[float]:
        try:
            e = float(entry)
            st = float(stop)
            t = float(tp)
            if e <= 0 or st <= 0 or t <= 0:
                return None

            s = (side or "").upper()
            if s == "BUY":
                s = "LONG"
            if s == "SELL":
                s = "SHORT"
            if s not in ("LONG", "SHORT"):
                s = "LONG"

            risk = abs(e - st)
            reward = abs(t - e)
            if risk <= 0:
                return None

            rr_gross = reward / risk

            fee = float(max(0.0, fee_rate_roundtrip))
            slip = float(max(0.0, slippage_bps_roundtrip)) / 10000.0
            funding = float(max(0.0, funding_rate_per_hour)) * float(max(0.0, funding_hours))

            total_rate = fee + slip + funding
            cost_move = e * total_rate
            rr_net = rr_gross - (cost_move / risk)

            return float(max(0.0, rr_net))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot_state(self) -> Dict[str, Any]:
        self._ensure_daily_state_nolock()

        # aggregation helpers (for monitoring)
        reserved_by_symbol: Dict[str, float] = {}
        reserved_by_cluster: Dict[str, float] = {}
        for r in self._reservations.values():
            reserved_by_symbol[r.symbol] = float(reserved_by_symbol.get(r.symbol, 0.0) + _sf(r.notional, 0.0))
            if r.cluster:
                reserved_by_cluster[r.cluster] = float(reserved_by_cluster.get(r.cluster, 0.0) + _sf(r.notional, 0.0))

        return {
            "date": self._daily.date_key if self._daily else None,
            "daily_pnl": float(self._daily.pnl if self._daily else 0.0),
            "daily_trades": int(self._daily.trades_opened if self._daily else 0),
            "daily_losses": int(self._daily.losses_count if self._daily else 0),
            "symbol_pnl": dict(self._daily.symbol_pnl if self._daily else {}),
            "symbol_trades": dict(self._daily.symbol_trades if self._daily else {}),
            "tilt_active": bool(self._is_tilt_active_nolock()),

            "open_positions": {
                sym: {
                    "side": pos.side,
                    "notional": float(pos.notional),
                    "risk": float(pos.risk),
                    "opened_at": float(pos.opened_at),
                    "age_s": float(max(0.0, time.time() - float(pos.opened_at))),
                    "filled_notional": float(pos.filled_notional),
                    "avg_entry": float(pos.avg_entry) if pos.avg_entry is not None else None,
                    "cluster": pos.cluster,
                    "corr_btc": pos.corr_btc,
                }
                for sym, pos in self.open_positions.items()
            },

            "reservations": {
                rid: {
                    "symbol": r.symbol,
                    "side": r.side,
                    "notional": float(r.notional),
                    "risk": float(r.risk),
                    "created_at": float(r.created_at),
                    "age_s": float(max(0.0, time.time() - float(r.created_at))),
                    "cluster": r.cluster,
                    "corr_btc": r.corr_btc,
                }
                for rid, r in self._reservations.items()
            },

            "reserved_by_symbol": reserved_by_symbol,
            "reserved_by_cluster": reserved_by_cluster,

            "direction_counts": dict(self.direction_counts),
            "gross_open_notional": float(sum(_sf(p.notional, 0.0) for p in self.open_positions.values())),
            "gross_reserved_notional": float(sum(_sf(r.notional, 0.0) for r in self._reservations.values())),

            "time_stop_due": self.positions_due_time_stop(),

            "ttl_purged_total": int(self._ttl_purged_total),
            "last_ttl_purge_ts": float(self._last_ttl_purge_ts) if self._last_ttl_purge_ts else 0.0,

            "config": {
                "risk_per_trade": float(self.config.risk_per_trade),
                "max_daily_loss": float(self.config.max_daily_loss),
                "max_open_positions": int(self.config.max_open_positions),
                "max_gross_exposure": float(self.config.max_gross_exposure),
                "max_symbol_exposure": float(self.config.max_symbol_exposure),
                "corr_btc_threshold": float(self.config.corr_btc_threshold),
                "corr_group_cap": float(self.config.corr_group_cap),
                "symbol_max_daily_loss": float(self.config.symbol_max_daily_loss),
                "symbol_max_trades_per_day": int(self.config.symbol_max_trades_per_day),
                "reservation_ttl_seconds": int(self.config.reservation_ttl_seconds),
                "max_position_age_seconds": int(self.config.max_position_age_seconds),
                "use_volatility_targeting": bool(self.config.use_volatility_targeting),
                "target_atr_pct": float(self.config.target_atr_pct),
            },
        }


def _safe_corr(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:
            return None
        if v < -1.0:
            v = -1.0
        if v > 1.0:
            v = 1.0
        return float(v)
    except Exception:
        return None
