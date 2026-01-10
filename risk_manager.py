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
)

LOGGER = logging.getLogger(__name__)


# =====================================================================
# CONFIG
# =====================================================================

@dataclass
class RiskConfig:
    # risk per trade (USDT) — keep aligned with settings.RISK_USDT
    risk_per_trade: float = float(RISK_USDT)

    # daily loss hard stop
    max_daily_loss: float = float(MAX_DAILY_LOSS)

    # daily trades cap
    max_trades_per_day: int = int(MAX_TRADES_PER_DAY)

    # max concurrently open positions
    max_open_positions: int = int(MAX_OPEN_POSITIONS)

    # directional caps
    max_long_positions: int = int(MAX_LONG_POSITIONS)
    max_short_positions: int = int(MAX_SHORT_POSITIONS)

    # tilt: consecutive losses => cooldown
    max_consecutive_losses: int = int(MAX_CONSECUTIVE_LOSSES)
    tilt_cooldown_seconds: int = int(TILT_COOLDOWN_SECONDS)

    # drawdown risk reducer
    drawdown_risk_factor: float = float(DRAWDOWN_RISK_FACTOR)

    # exposure caps (gross + per symbol) relative to equity
    account_equity_usdt: float = float(ACCOUNT_EQUITY_USDT)
    max_gross_exposure: float = float(MAX_GROSS_EXPOSURE)     # e.g. 2.0 => 2x equity notional
    max_symbol_exposure: float = float(MAX_SYMBOL_EXPOSURE)   # e.g. 0.25 => 25% equity per symbol

    # reservations: prevent stuck exposure if order flow never confirms
    reservation_ttl_seconds: int = 6 * 60  # 6 minutes

    # time-stop helper (scanner decides how to close)
    max_position_age_seconds: int = 12 * 60 * 60  # 12h

    # optional volatility targeting:
    # if you pass volatility_atr_pct (ATR/price), risk can be scaled.
    # Example: volatility_atr_pct=0.01 means 1% ATR.
    use_volatility_targeting: bool = False
    target_atr_pct: float = 0.010  # 1%
    vol_risk_min_factor: float = 0.35
    vol_risk_max_factor: float = 1.25

    # per-symbol daily caps (0 disables)
    symbol_max_daily_loss: float = float(SYMBOL_MAX_DAILY_LOSS)
    symbol_max_trades_per_day: int = int(SYMBOL_MAX_TRADES_PER_DAY)

    # daily reset mode (keeps current default behavior: localtime)
    day_reset_use_utc: bool = False


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
    side: str  # "LONG"/"SHORT"
    notional: float
    risk: float
    opened_at: float = field(default_factory=lambda: time.time())
    # for partial fill tracking
    filled_notional: float = 0.0
    avg_entry: Optional[float] = None  # optional, if you want to store it later


@dataclass
class Reservation:
    rid: str
    symbol: str
    side: str
    notional: float
    risk: float
    created_at: float = field(default_factory=lambda: time.time())


# =====================================================================
# Risk Manager
# =====================================================================

class RiskManager:
    """
    Desk risk engine.

    Recommended flow:
      allowed, reason, rid = await rm.reserve_trade(...)
      if not allowed: return
      send order...
      if fail: await rm.cancel_reservation(rid)
      if ok:   await rm.confirm_open(rid, filled_notional=..., filled_risk=...)

    Backward compatible:
      allowed, reason = rm.can_trade(...)
      rm.register_open(...)  # old style (less accurate under concurrency)
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config: RiskConfig = config or RiskConfig()

        self._daily: Optional[DailyState] = None

        # open positions: symbol -> PositionState
        self.open_positions: Dict[str, PositionState] = {}

        # reservations (pre-open)
        self._reservations: Dict[str, Reservation] = {}

        # direction counts
        self.direction_counts = {"LONG": 0, "SHORT": 0}

        # tilt / cooldown
        self._tilt_active: bool = False
        self._tilt_activated_at: float = 0.0

        # lock for concurrent scanner tasks
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # daily helpers
    # ------------------------------------------------------------------

    def _current_date_key(self) -> str:
        if bool(self.config.day_reset_use_utc):
            return time.strftime("%Y-%m-%d", time.gmtime())
        return time.strftime("%Y-%m-%d", time.localtime())

    def _purge_expired_reservations_nolock(self) -> None:
        """
        Prevent stale reservations from blocking exposure caps forever.
        """
        try:
            ttl = int(max(30, self.config.reservation_ttl_seconds))
            now = time.time()
            expired = [rid for rid, r in self._reservations.items() if (now - float(r.created_at)) >= ttl]
            for rid in expired:
                self._reservations.pop(rid, None)
        except Exception:
            # never break risk flow
            return

    def _ensure_daily_state(self) -> None:
        today = self._current_date_key()
        if self._daily is None or self._daily.date_key != today:
            self._daily = DailyState(date_key=today)
            self._tilt_active = False
            self._tilt_activated_at = 0.0
            # do not auto-close open positions; clear reservations to avoid stuck state
            self._reservations.clear()
        else:
            if self._daily.symbol_pnl is None:
                self._daily.symbol_pnl = {}
            if self._daily.symbol_trades is None:
                self._daily.symbol_trades = {}
        # also keep the reservation map clean in runtime
        self._purge_expired_reservations_nolock()

    def _daily_loss(self) -> float:
        self._ensure_daily_state()
        return float(self._daily.pnl if self._daily else 0.0)

    def _daily_trades(self) -> int:
        self._ensure_daily_state()
        return int(self._daily.trades_opened if self._daily else 0)

    def _daily_losses(self) -> int:
        self._ensure_daily_state()
        return int(self._daily.losses_count if self._daily else 0)

    def _is_tilt_active(self) -> bool:
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

    # ------------------------------------------------------------------
    # exposure computations
    # ------------------------------------------------------------------

    def _gross_open_notional(self) -> float:
        return float(sum(p.notional for p in self.open_positions.values()))

    def _gross_reserved_notional(self) -> float:
        return float(sum(r.notional for r in self._reservations.values()))

    def _symbol_open_notional(self, symbol: str) -> float:
        sym = (symbol or "").upper()
        p = self.open_positions.get(sym)
        return float(p.notional) if p else 0.0

    def _symbol_reserved_notional(self, symbol: str) -> float:
        sym = (symbol or "").upper()
        return float(sum(r.notional for r in self._reservations.values() if r.symbol == sym))

    # ------------------------------------------------------------------
    # Level 1 gate: can_open
    # ------------------------------------------------------------------

    def can_open(self, symbol: str, side: str) -> Tuple[bool, str]:
        self._ensure_daily_state()
        sym = (symbol or "UNKNOWN").upper()
        s = self._norm_side(side)

        if self._is_tilt_active():
            return False, "tilt_cooldown"

        if self._daily_trades() >= self.config.max_trades_per_day:
            return False, "max_trades_per_day_reached"

        if self._daily_loss() <= -abs(self.config.max_daily_loss):
            return False, "max_daily_loss_reached"

        if len(self.open_positions) >= self.config.max_open_positions:
            return False, "max_open_positions_reached"

        if s == "LONG" and self.direction_counts.get("LONG", 0) >= self.config.max_long_positions:
            return False, "max_long_exposure"

        if s == "SHORT" and self.direction_counts.get("SHORT", 0) >= self.config.max_short_positions:
            return False, "max_short_exposure"

        # SAFETY FIX:
        # with open_positions keyed by symbol, allowing opposite-side opens would overwrite state.
        if sym in self.open_positions:
            return False, "position_already_open_symbol"

        return True, "OK"

    # ------------------------------------------------------------------
    # Level 2 gate: can_trade (sync legacy)
    # ------------------------------------------------------------------

    def can_trade(self, *args: Any, **kwargs: Any) -> Tuple[bool, str]:
        """
        Legacy API: returns (allowed, reason).
        This DOES NOT reserve. Prefer reserve_trade() in scanner for accuracy.
        """
        extracted = self._extract_args(args, kwargs)
        allowed, reason = self._can_trade_core(**extracted)
        if not allowed:
            return False, reason
        return True, "OK"

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
        **_extra: Any,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Returns (allowed, reason, reservation_id).
        If allowed, reserves exposure immediately (prevents overfill in concurrent scan tasks).

        Pass volatility_atr_pct if you want volatility targeting.
        Pass rr_net if you computed net RR (fees/funding/slippage).
        """
        async with self._lock:
            self._ensure_daily_state()

            allowed, reason = self._can_trade_core(
                symbol=symbol,
                side=side,
                notional=notional,
                rr=rr,
                rr_net=rr_net,
                inst_score=inst_score,
                commitment=commitment,
                volatility_atr_pct=volatility_atr_pct,
            )
            if not allowed:
                return False, reason, None

            risk_used = self.risk_for_this_trade(volatility_atr_pct=volatility_atr_pct)

            rid = uuid.uuid4().hex[:12]
            sym = (symbol or "UNKNOWN").upper()
            s = self._norm_side(side)

            self._reservations[rid] = Reservation(
                rid=rid,
                symbol=sym,
                side=s,
                notional=float(notional),
                risk=float(risk_used),
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
        """
        Turns reservation into an open position and increments daily trades.
        Supports partial fills:
          - filled_notional: if None, uses reserved notional
          - filled_risk: if None, uses reserved risk
        """
        async with self._lock:
            self._ensure_daily_state()
            r = self._reservations.pop(str(rid), None)
            if not r:
                return

            notional = float(filled_notional) if (filled_notional is not None) else float(r.notional)
            risk = float(filled_risk) if (filled_risk is not None) else float(r.risk)

            # If filled is 0 => treat as no position opened (avoid polluting state)
            if notional <= 0:
                return

            self._register_open_nolock(r.symbol, r.side, notional, risk, filled_notional=notional, avg_entry=avg_entry)

    async def apply_fill(
        self,
        symbol: str,
        side: str,
        *,
        fill_notional_delta: float,
        risk_delta: float = 0.0,
        avg_entry: Optional[float] = None,
    ) -> None:
        """
        Apply incremental fills to an already-open position (partial fills / adds).
        This does NOT bypass exposure gates; scanner should reserve/confirm for adds.

        - fill_notional_delta can be positive (more filled) or negative (reduce)
        """
        async with self._lock:
            self._ensure_daily_state()
            sym = (symbol or "UNKNOWN").upper()
            s = self._norm_side(side)
            p = self.open_positions.get(sym)
            if not p:
                return
            if p.side != s:
                return

            p.notional = float(max(0.0, p.notional + float(fill_notional_delta)))
            p.risk = float(max(0.0, p.risk + float(risk_delta)))
            p.filled_notional = float(max(0.0, p.filled_notional + float(fill_notional_delta)))
            if avg_entry is not None:
                p.avg_entry = float(avg_entry)

            # If position is effectively gone
            if p.notional <= 0.0:
                self.open_positions.pop(sym, None)
                self.direction_counts[s] = max(0, self.direction_counts.get(s, 0) - 1)

    # ------------------------------------------------------------------
    # Risk sizing
    # ------------------------------------------------------------------

    def risk_for_this_trade(self, *, volatility_atr_pct: Optional[float] = None) -> float:
        """
        Base risk sizing with optional drawdown reduction + optional volatility targeting.

        Volatility targeting requires passing volatility_atr_pct (ATR/price).
        If not passed, this returns baseline risk (with drawdown reducer only).
        """
        self._ensure_daily_state()
        base = float(self.config.risk_per_trade)

        # drawdown reducer
        dloss = self._daily_loss()
        if dloss < -2.0 * base:
            base = float(base * self.config.drawdown_risk_factor)

        # optional volatility targeting
        if not self.config.use_volatility_targeting:
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
        # scale risk inversely to volatility
        factor = tgt / v
        factor = float(max(self.config.vol_risk_min_factor, min(self.config.vol_risk_max_factor, factor)))
        return float(base * factor)

    # ------------------------------------------------------------------
    # Register open/close (legacy + internal)
    # ------------------------------------------------------------------

    def register_open(self, symbol: str, side: str, notional: float, risk: float) -> None:
        """
        Legacy immediate register (non-reserved). Prefer reserve_trade/confirm_open.
        """
        self._ensure_daily_state()
        s = self._norm_side(side)
        sym = (symbol or "UNKNOWN").upper()

        # keep same safety rule as can_open
        if sym in self.open_positions:
            return

        self._register_open_nolock(sym, s, float(notional), float(risk), filled_notional=float(notional))

    def _register_open_nolock(
        self,
        sym: str,
        side: str,
        notional: float,
        risk: float,
        *,
        filled_notional: float = 0.0,
        avg_entry: Optional[float] = None,
    ) -> None:
        self.open_positions[sym] = PositionState(
            symbol=sym,
            side=side,
            notional=float(notional),
            risk=float(risk),
            filled_notional=float(filled_notional),
            avg_entry=float(avg_entry) if avg_entry is not None else None,
        )
        self.direction_counts[side] = self.direction_counts.get(side, 0) + 1
        if self._daily:
            self._daily.trades_opened += 1
            self._daily.symbol_trades[sym] = int(self._daily.symbol_trades.get(sym, 0)) + 1

    def register_closed(self, symbol: str, side: str, pnl: float) -> None:
        self._ensure_daily_state()

        s = self._norm_side(side)
        sym = (symbol or "UNKNOWN").upper()

        if self._daily:
            self._daily.pnl += float(pnl)
            self._daily.symbol_pnl[sym] = float(self._daily.symbol_pnl.get(sym, 0.0)) + float(pnl)

            if pnl < 0:
                self._daily.losses_count += 1
            else:
                self._daily.losses_count = 0

            if self._daily.losses_count >= self.config.max_consecutive_losses:
                self._tilt_active = True
                self._tilt_activated_at = time.time()

        pos = self.open_positions.pop(sym, None)
        if pos is not None:
            self.direction_counts[pos.side] = max(0, self.direction_counts.get(pos.side, 0) - 1)
        else:
            self.direction_counts[s] = max(0, self.direction_counts.get(s, 0) - 1)

    # ------------------------------------------------------------------
    # Time-stop helper
    # ------------------------------------------------------------------

    def position_age_seconds(self, symbol: str) -> Optional[float]:
        sym = (symbol or "").upper()
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
                out.append({"symbol": sym, "side": p.side, "age_s": age, "notional": float(p.notional), "risk": float(p.risk)})
        return out

    # ------------------------------------------------------------------
    # Core gating logic
    # ------------------------------------------------------------------

    def _can_trade_core(
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
    ) -> Tuple[bool, str]:
        self._ensure_daily_state()
        sym = (symbol or "UNKNOWN").upper()
        s = self._norm_side(side)

        try:
            notional = float(notional or 0.0)
        except Exception:
            notional = 0.0

        if notional <= 0:
            return False, "notional_invalid"

        # Per-symbol daily caps (optional)
        if self._daily:
            if self.config.symbol_max_daily_loss > 0:
                sym_pnl = float(self._daily.symbol_pnl.get(sym, 0.0))
                if sym_pnl <= -float(self.config.symbol_max_daily_loss):
                    return False, "symbol_daily_loss_cap"
            if self.config.symbol_max_trades_per_day > 0:
                sym_trades = int(self._daily.symbol_trades.get(sym, 0))
                if sym_trades >= int(self.config.symbol_max_trades_per_day):
                    return False, "symbol_trades_cap"

        allowed, reason = self.can_open(sym, s)
        if not allowed:
            return False, reason

        # Exposure caps (gross + per symbol), include reservations too
        eq = float(self.config.account_equity_usdt)
        gross_cap = float(self.config.max_gross_exposure) * eq
        sym_cap = float(self.config.max_symbol_exposure) * eq

        gross_after = self._gross_open_notional() + self._gross_reserved_notional() + notional
        if gross_after > gross_cap:
            return False, "gross_exposure_cap"

        sym_after = self._symbol_open_notional(sym) + self._symbol_reserved_notional(sym) + notional
        if sym_after > sym_cap:
            return False, "symbol_exposure_cap"

        # Optional EV sanity gates.
        # Prefer rr_net if provided, else rr.
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
            # If no/low institutional info and rr below strict => veto
            if (inst_score is None or int(inst_score) < int(MIN_INST_SCORE)) and rr_used < float(RR_MIN_STRICT):
                return False, "rr_below_strict_no_inst"

            # If inst strong, allow down to tolerated threshold
            if inst_score is not None and int(inst_score) >= int(MIN_INST_SCORE):
                if rr_used < float(RR_MIN_TOLERATED_WITH_INST):
                    return False, "rr_below_tolerated_even_with_inst"

        # Commitment (optional): left non-blocking unless DESK_EV_MODE later evolves
        if DESK_EV_MODE and commitment is not None:
            pass

        # Optional volatility targeting check (non-blocking): if targeting is enabled
        # and volatility is extremely high, you might want a desk veto.
        if self.config.use_volatility_targeting and volatility_atr_pct is not None:
            try:
                v = float(volatility_atr_pct)
                if v > 0.0 and v >= 4.5 * float(self.config.target_atr_pct):
                    # very volatile regime -> veto to avoid nasty spreads/liquidations
                    return False, "volatility_too_high"
            except Exception:
                pass

        return True, "OK"

    # ------------------------------------------------------------------
    # Flexible args extraction (compat with your scanner)
    # ------------------------------------------------------------------

    def _extract_args(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a dict matching _can_trade_core signature.

        Supported kw aliases:
          - notional_usdt / size_notional
          - rr_actual (gross) / rr_net (preferred)
          - inst_score / institutional_score
          - volatility_atr_pct / atr_pct / vol_atr_pct
        """
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

        str_args = [a for a in args if isinstance(a, str)]
        num_args = [a for a in args if isinstance(a, (int, float))]

        if symbol is None and str_args:
            symbol = str_args[0]
        if side is None and len(str_args) >= 2:
            side = str_args[1]

        # pick biggest numeric as notional (common pattern)
        if notional is None and num_args:
            notional = float(max(num_args))

        # rr: smallest positive < 10 if not provided
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

        # normalize inst_score
        if inst_score is not None:
            try:
                inst_score = int(inst_score)
            except Exception:
                inst_score = None

        # normalize commitment
        if commitment is not None:
            try:
                commitment = float(commitment)
            except Exception:
                commitment = None

        # normalize rr/rr_net
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

        # normalize volatility_atr_pct
        vol_f = None
        try:
            if volatility_atr_pct is not None:
                vol_f = float(volatility_atr_pct)
        except Exception:
            vol_f = None

        return {
            "symbol": str(symbol),
            "side": str(side),
            "notional": float(notional),
            "rr": rr_f,
            "rr_net": rr_net_f,
            "inst_score": inst_score,
            "commitment": commitment,
            "volatility_atr_pct": vol_f,
        }

    # ------------------------------------------------------------------
    # Net RR helper (explicit inputs, no guessing)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_rr_net(
        *,
        entry: float,
        stop: float,
        tp: float,
        side: str,
        fee_rate_roundtrip: float = 0.0,     # e.g. 0.001 means 0.10% total entry+exit
        funding_rate_per_hour: float = 0.0,  # e.g. 0.00001 means 0.001%/hour
        funding_hours: float = 0.0,
        slippage_bps_roundtrip: float = 0.0, # e.g. 5 => 5 bps total
    ) -> Optional[float]:
        """
        Computes a net R:R using explicit parameters only.
        Returns None if inputs invalid.

        - fee_rate_roundtrip is applied on notional, approximated as price cost in move terms.
        - slippage_bps_roundtrip is converted to a rate (bps -> fraction).
        - funding cost is applied as rate = funding_rate_per_hour * funding_hours.

        This is a simplification; it avoids hidden assumptions.
        """
        try:
            e = float(entry)
            st = float(stop)
            t = float(tp)
            if e <= 0 or st <= 0 or t <= 0:
                return None

            s = (side or "").upper()
            if s not in ("LONG", "SHORT", "BUY", "SELL"):
                s = "LONG"
            if s == "BUY":
                s = "LONG"
            if s == "SELL":
                s = "SHORT"

            risk = abs(e - st)
            reward = abs(t - e)
            if risk <= 0:
                return None

            rr_gross = reward / risk

            fee = float(max(0.0, fee_rate_roundtrip))
            slip = float(max(0.0, slippage_bps_roundtrip)) / 10000.0
            funding = float(max(0.0, funding_rate_per_hour)) * float(max(0.0, funding_hours))

            # Convert rates into an approximate "price move" cost:
            # cost_move ≈ entry * total_rate; then convert to R units via / risk
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
        self._ensure_daily_state()
        return {
            "date": self._daily.date_key if self._daily else None,
            "daily_pnl": float(self._daily.pnl if self._daily else 0.0),
            "daily_trades": int(self._daily.trades_opened if self._daily else 0),
            "daily_losses": int(self._daily.losses_count if self._daily else 0),
            "symbol_pnl": dict(self._daily.symbol_pnl if self._daily else {}),
            "symbol_trades": dict(self._daily.symbol_trades if self._daily else {}),
            "tilt_active": bool(self._is_tilt_active()),
            "open_positions": {
                sym: {
                    "side": pos.side,
                    "notional": float(pos.notional),
                    "risk": float(pos.risk),
                    "opened_at": float(pos.opened_at),
                    "age_s": float(max(0.0, time.time() - float(pos.opened_at))),
                    "filled_notional": float(pos.filled_notional),
                    "avg_entry": float(pos.avg_entry) if pos.avg_entry is not None else None,
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
                }
                for rid, r in self._reservations.items()
            },
            "direction_counts": dict(self.direction_counts),
            "gross_open_notional": float(self._gross_open_notional()),
            "gross_reserved_notional": float(self._gross_reserved_notional()),
            "time_stop_due": self.positions_due_time_stop(),
            "config": {
                "risk_per_trade": float(self.config.risk_per_trade),
                "max_daily_loss": float(self.config.max_daily_loss),
                "max_open_positions": int(self.config.max_open_positions),
                "max_gross_exposure": float(self.config.max_gross_exposure),
                "max_symbol_exposure": float(self.config.max_symbol_exposure),
                "symbol_max_daily_loss": float(self.config.symbol_max_daily_loss),
                "symbol_max_trades_per_day": int(self.config.symbol_max_trades_per_day),
                "reservation_ttl_seconds": int(self.config.reservation_ttl_seconds),
                "max_position_age_seconds": int(self.config.max_position_age_seconds),
                "use_volatility_targeting": bool(self.config.use_volatility_targeting),
                "target_atr_pct": float(self.config.target_atr_pct),
            },
        }
