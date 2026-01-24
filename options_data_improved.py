"""
Improved Options Data Module
============================

This module enriches options market context for trading strategies. It
extends the original `options_data.py` by smoothing implied volatility
data over a rolling window and computing risk factors based on
volatility regimes. The design is symmetrical to
`macro_data_improved.py` to allow consistent integration with risk and
signal analysis.

For demonstration purposes, the current implementation uses randomly
generated volatility values. In a production environment, you would
replace the `_fetch_current_vols` method with actual API calls (e.g.
Deribit or Centralized Exchange vol surfaces).
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Dict, List

from settings_improved import get_settings


@dataclass
class OptionsDataManager:
    """Manages implied volatility data and derives regimes."""

    settings: dict = field(default_factory=get_settings)
    window_size: int = field(default_factory=lambda: get_settings().get("options", {}).get("window_size", 14))
    btc_vol_window: List[float] = field(default_factory=list)
    eth_vol_window: List[float] = field(default_factory=list)

    async def _fetch_current_vols(self) -> Dict[str, float]:
        """Simulate fetching current implied volatility for BTC and ETH options.

        Replace this implementation with API calls to a real options
        data provider. Values are annualized volatilities in decimal form.
        """
        btc_vol = random.uniform(0.4, 1.2)  # 40% – 120% annualized
        eth_vol = random.uniform(0.5, 1.5)  # 50% – 150% annualized
        return {"BTC": btc_vol, "ETH": eth_vol}

    async def update(self) -> None:
        """Update the rolling windows with the latest implied volatilities."""
        vols = await self._fetch_current_vols()
        btc_vol = vols["BTC"]
        eth_vol = vols["ETH"]
        self.btc_vol_window.append(btc_vol)
        if len(self.btc_vol_window) > self.window_size:
            self.btc_vol_window.pop(0)
        self.eth_vol_window.append(eth_vol)
        if len(self.eth_vol_window) > self.window_size:
            self.eth_vol_window.pop(0)

    def _compute_regime(self, vol: float) -> str:
        """Classify volatility into discrete regimes."""
        # Thresholds could be configured via settings
        low = 0.5  # 50%
        mid = 0.8  # 80%
        high = 1.1  # 110%
        if vol < low:
            return "low"
        if vol < mid:
            return "mid"
        if vol < high:
            return "high"
        return "extreme"

    def get_smoothed_metrics(self) -> Dict[str, Dict[str, float | str]]:
        """Return smoothed vol and regime for BTC and ETH."""
        if not self.btc_vol_window:
            return {"BTC": {"vol": 0.0, "regime": "unknown"}, "ETH": {"vol": 0.0, "regime": "unknown"}}
        btc_avg = sum(self.btc_vol_window) / len(self.btc_vol_window)
        eth_avg = sum(self.eth_vol_window) / len(self.eth_vol_window)
        return {
            "BTC": {"vol": btc_avg, "regime": self._compute_regime(btc_avg)},
            "ETH": {"vol": eth_avg, "regime": self._compute_regime(eth_avg)},
        }


# Global instance for convenience
_GLOBAL_MANAGER: OptionsDataManager | None = None

def _get_manager() -> OptionsDataManager:
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        _GLOBAL_MANAGER = OptionsDataManager()
    return _GLOBAL_MANAGER


async def get_options_metrics_async() -> Dict[str, Dict[str, float | str]]:
    """Fetch smoothed implied volatility metrics for BTC and ETH.

    Updates the internal windows and returns average volatility and
    regime classification for each asset.
    """
    manager = _get_manager()
    await manager.update()
    return manager.get_smoothed_metrics()