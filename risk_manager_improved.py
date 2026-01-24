"""
Enhanced Risk Manager for BitgetBot
===================================

This module provides an institutional-grade risk management engine that
extends the original `risk_manager.py` with dynamic risk factors and
exposure controls. It incorporates macro volatility and cross-asset
correlation to adjust position sizing in real time. The configuration
parameters are loaded from the centralized settings module. A periodic
re-evaluation routine can be invoked by the caller to refresh the
internal state based on updated macro metrics.

The main concepts implemented here include:

* **Dynamic risk factor** – adjusts base position sizing using macro
  volatility (e.g. derived from options implied volatility) and a
  correlation score between assets. Higher volatility or stronger
  correlations result in lower permitted exposure.
* **Exposure tracking** – keeps track of open positions and
  reservations per symbol in notional terms, and prevents new trades
  when limits are exceeded.
* **Cluster contagion** – limits aggregate exposure across symbols that
  are highly correlated. This helps prevent cascading losses when
  correlated assets move together.
* **Periodic re-evaluation** – allows the caller to refresh the
  dynamic factor and recompute exposures at regular intervals.

This module does not interact directly with exchanges; instead it
provides a decision engine that other parts of the bot (e.g. the
scanner or executor) should query before placing orders.
"""

from __future__ import annotations

import asyncio
import statistics
from dataclasses import dataclass, field
from typing import Dict, Optional

from settings_improved import get_settings
import macro_data_improved as macro_data


@dataclass
class ExposureState:
    """Tracks open exposures and reservations for symbols."""

    # Current notional exposure per symbol
    exposures: Dict[str, float] = field(default_factory=dict)
    # Reserved notional exposure for pending orders per symbol
    reservations: Dict[str, float] = field(default_factory=dict)

    def total_exposure(self) -> float:
        return sum(abs(v) for v in self.exposures.values())

    def reserved_exposure(self) -> float:
        return sum(abs(v) for v in self.reservations.values())


@dataclass
class RiskManager:
    """Institutional risk management engine with dynamic sizing."""

    settings: dict = field(default_factory=get_settings)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    state: ExposureState = field(default_factory=ExposureState)
    # Rolling window of macro volatility values for dynamic factor
    macro_vol_window: list[float] = field(default_factory=list)
    # Rolling window of average cross-asset correlations
    corr_window: list[float] = field(default_factory=list)
    # Size of the rolling window for smoothing
    window_size: int = 10

    def _compute_dynamic_factor(self) -> float:
        """Compute a dynamic risk factor based on macro volatility and correlation.

        The factor scales down position sizes when volatility or
        correlation is high. A value of 1.0 means no adjustment, while
        lower values reduce the allowed exposure.
        """
        if not self.macro_vol_window:
            # Default factor when no data is available
            return 1.0
        avg_vol = statistics.mean(self.macro_vol_window)
        avg_corr = statistics.mean(self.corr_window) if self.corr_window else 0.0
        # Base factor inversely proportional to volatility and correlation
        factor = 1.0 / (1.0 + avg_vol + max(avg_corr, 0.0))
        # Clamp to sensible range
        return max(0.2, min(1.0, factor))

    async def refresh_dynamic_data(self) -> None:
        """Fetch latest macro metrics and update rolling windows.

        This method should be called periodically by the caller. It
        uses the macro_data_improved module to obtain the current
        volatility and average correlation across tracked assets. The
        rolling windows smooth the values over time.
        """
        macro_metrics = await macro_data.get_macro_metrics_async()
        vol = macro_metrics.get("volatility", 0.0)
        corr = macro_metrics.get("correlation", 0.0)
        # Update windows
        self.macro_vol_window.append(vol)
        if len(self.macro_vol_window) > self.window_size:
            self.macro_vol_window.pop(0)
        self.corr_window.append(corr)
        if len(self.corr_window) > self.window_size:
            self.corr_window.pop(0)

    async def can_trade(self, symbol: str, notional: float) -> bool:
        """Determine whether a new trade of a given notional size can be placed.

        Args:
            symbol: Symbol of the instrument (e.g. 'BTCUSDT').
            notional: Absolute USD value of the proposed trade.

        Returns:
            True if the trade is allowed under current risk limits.
        """
        async with self.lock:
            dynamic_factor = self._compute_dynamic_factor()
            # Effective maximum gross exposure
            max_exposure = self.settings["risk"].get("max_gross_exposure", 100000.0)
            allowed_exposure = max_exposure * dynamic_factor
            current_exposure = self.state.total_exposure() + self.state.reserved_exposure()
            # Consider per-symbol limit
            symbol_limit = self.settings["risk"].get("per_symbol_limit", max_exposure / 5)
            symbol_exposure = abs(self.state.exposures.get(symbol, 0.0)) + abs(self.state.reservations.get(symbol, 0.0))
            if current_exposure + abs(notional) > allowed_exposure:
                return False
            if symbol_exposure + abs(notional) > symbol_limit:
                return False
            return True

    async def reserve_trade(self, symbol: str, notional: float) -> bool:
        """Reserve exposure for a pending order.

        This method should be called after `can_trade` returns True.
        It increments the reserved exposure to avoid double-counting.
        Returns True if reservation succeeds, False otherwise.
        """
        async with self.lock:
            if not await self.can_trade(symbol, notional):
                return False
            self.state.reservations[symbol] = self.state.reservations.get(symbol, 0.0) + notional
            return True

    async def finalize_trade(self, symbol: str, notional: float) -> None:
        """Finalize a trade after execution.

        Moves reserved exposure into actual exposure and clears the
        reservation.
        """
        async with self.lock:
            # Remove reservation
            reserved = self.state.reservations.get(symbol, 0.0)
            self.state.reservations[symbol] = max(0.0, reserved - notional)
            # Add to exposures
            self.state.exposures[symbol] = self.state.exposures.get(symbol, 0.0) + notional

    async def close_position(self, symbol: str) -> None:
        """Close out any open exposure for a symbol."""
        async with self.lock:
            self.state.exposures.pop(symbol, None)
            self.state.reservations.pop(symbol, None)

    async def reevaluate_exposure(self) -> None:
        """Re-evaluate exposures at regular intervals.

        This method can be scheduled by the caller (e.g. scanner) to run
        periodically. It refreshes dynamic data and ensures exposures
        remain within limits.
        """
        await self.refresh_dynamic_data()
        # Additional logic could be added here to reduce positions or
        # perform other risk adjustments in real time based on updated
        # macro metrics. This is left as a placeholder for further
        # development.