"""
Improved Signal Analyzer
=======================

This module provides a modular and asynchronous framework for signal
analysis in the BitgetBot. It splits the analysis pipeline into
independent components (structure, momentum, options, macro and risk)
that can be evaluated concurrently. Thresholds and coefficients are
externally configurable via the central settings.

The `SignalAnalyzer` class exposes a single method `analyze` which
takes a symbol and a minimal set of market data (entry price, ATR,
momentum, liquidity, order book depth). It returns either a
dictionary describing a suggested trade (including position size and
take-profit targets) or `None` if the signal does not meet risk and
confidence requirements.

This implementation is intentionally simplified relative to the
production-grade analyzer. It demonstrates how to integrate the
improved risk manager, take‑profit clamp and macro/options data
modules while allowing external calibration of thresholds.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional

from settings_improved import get_settings
from risk_manager_improved import RiskManager
from tp_clamp_improved import TPClamp
from macro_data_improved import get_macro_metrics_async
from options_data_improved import get_options_metrics_async


@dataclass
class SignalAnalyzer:
    settings: dict
    risk_manager: RiskManager
    tp_clamp: TPClamp
    confidence_threshold: float

    def __init__(self, settings: dict | None = None) -> None:
        self.settings = settings or get_settings()
        self.risk_manager = RiskManager(self.settings)
        self.tp_clamp = TPClamp(self.settings)
        # Configure threshold from settings
        analyzer_cfg = self.settings.get("analyzer", {})
        self.confidence_threshold = analyzer_cfg.get("confidence_threshold", 0.5)

    async def _analyze_structure(self, market_data: Dict[str, float]) -> float:
        """Compute a structural score based on trend and liquidity patterns.

        In a real system this would consider market structure, order book
        imbalance and higher time-frame patterns. Here we use a simple
        heuristic: higher liquidity yields better structure scores.
        """
        liquidity = market_data.get("liquidity", 0.0)
        # Normalize liquidity into [0, 1] using a logistic function
        import math
        return 1.0 / (1.0 + math.exp(-liquidity / 1e6))

    async def _analyze_momentum(self, market_data: Dict[str, float]) -> float:
        """Compute a momentum score based on a momentum indicator.

        A real implementation would use indicators like RSI or MACD. We
        scale the provided momentum value into [0, 1].
        """
        momentum = market_data.get("momentum", 0.0)
        # Assume momentum is already in a reasonable range [-1, 1]
        return max(0.0, min(1.0, (momentum + 1.0) / 2.0))

    async def _analyze_options(self) -> float:
        """Compute a risk factor based on options volatility regimes."""
        options_metrics = await get_options_metrics_async()
        btc_regime = options_metrics["BTC"]["regime"]
        # Map regimes to numerical factors (lower is more favourable)
        regime_map = {"low": 0.1, "mid": 0.2, "high": 0.3, "extreme": 0.4}
        return regime_map.get(btc_regime, 0.25)

    async def _analyze_macro(self) -> float:
        """Compute a macro risk factor based on macro volatility and correlation."""
        macro_metrics = await get_macro_metrics_async()
        # Higher volatility or correlation yields a higher risk factor
        vol = macro_metrics.get("volatility", 0.0)
        corr = macro_metrics.get("correlation", 0.0)
        return vol + corr

    async def analyze(self, symbol: str, market_data: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Analyze a market signal asynchronously.

        Args:
            symbol: Trading pair (e.g. 'BTCUSDT').
            market_data: A dictionary containing at least the following
                fields: entry_price, atr, momentum, liquidity, order_book_depth.

        Returns:
            A suggestion dictionary with keys 'symbol', 'entry_price',
            'size', 'tp1', 'tp2' if a trade is recommended; otherwise
            None.
        """
        # Run independent analyses concurrently
        structure_score, momentum_score, options_factor, macro_factor = await asyncio.gather(
            self._analyze_structure(market_data),
            self._analyze_momentum(market_data),
            self._analyze_options(),
            self._analyze_macro(),
        )
        # Aggregate into a confidence score (higher is better)
        confidence = (structure_score + momentum_score) / 2.0
        # Aggregate risk factors (lower is better)
        risk_factor = options_factor + macro_factor
        # Decide whether to proceed based on threshold
        if confidence - risk_factor < self.confidence_threshold:
            return None
        entry_price = market_data["entry_price"]
        atr = market_data["atr"]
        momentum = market_data["momentum"]
        liquidity = market_data["liquidity"]
        order_book_depth = market_data["order_book_depth"]
        # Determine trade direction from momentum (simplified rule)
        direction = 1 if momentum >= 0 else -1
        # Compute notional size based on base risk and dynamic factor
        base_risk = self.settings.get("risk", {}).get("base_notional", 1000.0)
        # Refresh risk manager dynamic data
        await self.risk_manager.refresh_dynamic_data()
        dynamic_factor = self.risk_manager._compute_dynamic_factor()
        size = base_risk * dynamic_factor * confidence
        # Check risk limits via risk manager
        can_trade = await self.risk_manager.can_trade(symbol, size)
        if not can_trade:
            return None
        # Reserve exposure
        reserved = await self.risk_manager.reserve_trade(symbol, size)
        if not reserved:
            return None
        # Compute take profits
        tp1, tp2 = self.tp_clamp.compute_targets(
            entry_price=entry_price,
            direction=direction,
            atr=atr,
            momentum=momentum,
            liquidity=liquidity,
            order_book_depth=order_book_depth,
        )
        # Build suggestion
        suggestion = {
            "symbol": symbol,
            "entry_price": entry_price,
            "size": size,
            "direction": direction,
            "tp1": tp1,
            "tp2": tp2,
            "confidence": confidence,
            "risk_factor": risk_factor,
        }
        return suggestion