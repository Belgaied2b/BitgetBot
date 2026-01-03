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
# Important integration change (scanner recommended):
#   - Call can_trade(...) -> if allowed, it returns a reservation_id
#   - If order fails => cancel_reservation(reservation_id)
#   - If order sent OK => confirm_open(reservation_id, symbol, side, notional, risk)
#   - When position closes => register_closed(...)
#
# Backward compatible:
#   - can_trade(...) still returns (allowed, reason) if you don't use reservation
#     but you should upgrade scanner to use reservation for accuracy.
# =====================================================================

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional

from settings import (
    RISK_USDT,
    ACCOUNT_EQUITY_USDT,
    MAX_GROSS_EXPOSURE,
    MAX_SYMBOL_EXPOSURE,
    RR_MIN_STRICT,
    RR_MIN_TOLERATED_WITH_INST,
    MIN_INST_SCORE,
    DESK_EV_MODE,
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
    max_daily_loss: float = 60.0

    # daily trades cap
    max_trades_per_day: int = 500

    # max concurrently open positions
    max_open_positions: int = 20

    # directional caps
    max_long_positions: int = 15
    max_short_positions: int = 15

    # tilt: consecutive losses => cooldown
    max_consecutive_losses: int = 5
    tilt_cooldown_seconds: int = 60 * 60

    # drawdown risk reducer
    drawdown_risk_factor: float = 0.5

    # exposure caps (gross + per symbol) relative to equity
    account_equity_usdt: float = float(ACCOUNT_EQUITY_USDT)
    max_gross_exposure: float = float(MAX_GROSS_EXPOSURE)     # e.g. 2.0 => 2x equity notional
    max_symbol_exposure: float = float(MAX_SYMBOL_EXPOSURE)   # e.g. 0.25 => 25% equity per symbol


# =====================================================================
# INTERNAL STATE
# =====================================================================

@dataclass
class DailyState:
    date_key: str
    trades_opened: int = 0
    pnl: float = 0.0
    losses_count: int = 0


@dataclass
class PositionState:
    symbol: str
    side: str  # "LONG"/"SHORT"
    notional: float
    risk: float
    opened_at: float = field(default_factory=lambda: time.time())


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
      if ok:   await rm.confirm_open(rid)

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
        return time.strftime("%Y-%m-%d", time.localtime())

    def _ensure_daily_state(self) -> None:
        today = self._current_date_key()
        if self._daily is None or self._daily.date_key != today:
            self._daily = DailyState(date_key=today)
            self._tilt_active = False
            self._tilt_activated_at = 0.0

            # daily reset does NOT auto-close open positions (runtime memory);
            # but it clears reservations to avoid stuck state after restarts.
            self._reservations.clear()

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

        if sym in self.open_positions and self.open_positions[sym].side == s:
            return False, "position_already_open_same_side"

        return True, "OK"

    # ------------------------------------------------------------------
    # Level 2 gate: can_trade (sync legacy)
    # ------------------------------------------------------------------

    def can_trade(self, *args: Any, **kwargs: Any) -> Tuple[bool, str]:
        """
        Legacy API: returns (allowed, reason).
        This DOES NOT reserve. Prefer reserve_trade() in scanner for accuracy.
        """
        symbol, side, notional, rr, inst_score, commitment = self._extract_args(args, kwargs)
        allowed, reason = self._can_trade_core(symbol, side, notional, rr, inst_score, commitment)
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
        inst_score: Optional[int] = None,
        commitment: Optional[float] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Returns (allowed, reason, reservation_id).
        If allowed, reserves exposure immediately (prevents overfill in concurrent scan tasks).
        """
        async with self._lock:
            self._ensure_daily_state()

            allowed, reason = self._can_trade_core(symbol, side, notional, rr, inst_score, commitment)
            if not allowed:
                return False, reason, None

            risk_used = self.risk_for_this_trade()

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

    async def confirm_open(self, rid: str) -> None:
        """
        Turns reservation into an open position and increments daily trades.
        """
        async with self._lock:
            self._ensure_daily_state()
            r = self._reservations.pop(str(rid), None)
            if not r:
                return
            self._register_open_nolock(r.symbol, r.side, r.notional, r.risk)

    # ------------------------------------------------------------------
    # Risk sizing
    # ------------------------------------------------------------------

    def risk_for_this_trade(self) -> float:
        self._ensure_daily_state()
        base = float(self.config.risk_per_trade)
        dloss = self._daily_loss()
        if dloss < -2.0 * base:
            return float(base * self.config.drawdown_risk_factor)
        return base

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
        self._register_open_nolock(sym, s, float(notional), float(risk))

    def _register_open_nolock(self, sym: str, side: str, notional: float, risk: float) -> None:
        self.open_positions[sym] = PositionState(symbol=sym, side=side, notional=notional, risk=risk)
        self.direction_counts[side] = self.direction_counts.get(side, 0) + 1
        if self._daily:
            self._daily.trades_opened += 1

    def register_closed(self, symbol: str, side: str, pnl: float) -> None:
        self._ensure_daily_state()

        s = self._norm_side(side)
        sym = (symbol or "UNKNOWN").upper()

        if self._daily:
            self._daily.pnl += float(pnl)

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
    # Core gating logic
    # ------------------------------------------------------------------

    def _can_trade_core(
        self,
        symbol: str,
        side: str,
        notional: float,
        rr: Optional[float],
        inst_score: Optional[int],
        commitment: Optional[float],
    ) -> Tuple[bool, str]:
        sym = (symbol or "UNKNOWN").upper()
        s = self._norm_side(side)
        notional = float(notional or 0.0)
        if notional <= 0:
            return False, "notional_invalid"

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

        # Optional EV sanity gates (light + non-destructive)
        # Keep analysis logic in analyze_signal; this is just a final desk veto.
        if rr is not None:
            try:
                rr_f = float(rr)
            except Exception:
                rr_f = None

            if rr_f is not None and rr_f > 0:
                # If no/low institutional info and rr below strict => veto
                if (inst_score is None or int(inst_score) < int(MIN_INST_SCORE)) and rr_f < float(RR_MIN_STRICT):
                    return False, "rr_below_strict_no_inst"

                # If inst strong, allow down to tolerated threshold
                if inst_score is not None and int(inst_score) >= int(MIN_INST_SCORE):
                    if rr_f < float(RR_MIN_TOLERATED_WITH_INST):
                        return False, "rr_below_tolerated_even_with_inst"

        # Commitment (optional): never hard-veto by default
        # If you want later: add gating under DESK_EV_MODE.
        if DESK_EV_MODE and commitment is not None:
            pass

        return True, "OK"

    # ------------------------------------------------------------------
    # Flexible args extraction (compat with your scanner)
    # ------------------------------------------------------------------

    def _extract_args(
        self,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Tuple[str, str, float, Optional[float], Optional[int], Optional[float]]:
        symbol = kwargs.get("symbol")
        side = kwargs.get("side")
        notional = kwargs.get("notional") or kwargs.get("notional_usdt") or kwargs.get("size_notional")
        rr = kwargs.get("rr") or kwargs.get("rr_actual")
        inst_score = kwargs.get("inst_score") or kwargs.get("institutional_score")
        commitment = kwargs.get("commitment")

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

        return str(symbol), str(side), float(notional), (float(rr) if rr is not None else None), inst_score, commitment

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
            "tilt_active": bool(self._is_tilt_active()),
            "open_positions": {
                sym: {
                    "side": pos.side,
                    "notional": float(pos.notional),
                    "risk": float(pos.risk),
                    "opened_at": float(pos.opened_at),
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
                }
                for rid, r in self._reservations.items()
            },
            "direction_counts": dict(self.direction_counts),
            "gross_open_notional": float(self._gross_open_notional()),
            "gross_reserved_notional": float(self._gross_reserved_notional()),
        }
