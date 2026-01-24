"""
Improved Macro Data Module
==========================

This module enriches macroeconomic inputs used by trading strategies. It
extends the original `macro_data.py` by including additional metrics
such as interest rates, inflation, stablecoin volume and social
sentiment. It also applies smoothing to volatility measurements over
sliding windows to avoid reacting to short-lived spikes.

The primary interface is the asynchronous function
`get_macro_metrics_async` which returns a dictionary containing
standardized macro metrics. It can be integrated with the improved
risk manager and signal analyzer.

Note: Real implementations should query external APIs (e.g. FRED,
CoinGecko, Santiment) to retrieve current data. Here we provide
placeholder implementations to illustrate the architecture. Users are
encouraged to replace the placeholder data with actual API calls.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List

from settings_improved import get_settings

import random


@dataclass
class MacroDataManager:
    """Manages macro data and applies smoothing to volatility measures."""

    settings: dict = field(default_factory=get_settings)
    window_size: int = field(default_factory=lambda: get_settings().get("macro", {}).get("window_size", 14))
    vol_window: List[float] = field(default_factory=list)
    corr_window: List[float] = field(default_factory=list)

    async def _fetch_current_data(self) -> Dict[str, float]:
        """Fetch current macroeconomic metrics.

        In a production setting, this method would call external APIs to
        retrieve interest rates, inflation, stablecoin volume and social
        sentiment. Here we simulate these values using random numbers
        drawn from reasonable ranges.
        """
        # Simulated macro data (replace with real API calls)
        interest_rate = random.uniform(0.02, 0.07)  # 2% – 7%
        inflation = random.uniform(0.01, 0.05)      # 1% – 5%
        stablecoin_volume = random.uniform(5e9, 2e10)  # USD volume
        social_sentiment = random.uniform(-1.0, 1.0)   # -1 = very bearish, +1 = very bullish
        # Simulated realized volatility (e.g. 30-day)
        volatility = random.uniform(0.3, 1.0)  # 30% – 100% annualized
        return {
            "interest_rate": interest_rate,
            "inflation": inflation,
            "stablecoin_volume": stablecoin_volume,
            "social_sentiment": social_sentiment,
            "volatility": volatility,
        }

    async def update(self) -> None:
        """Update internal rolling windows with the latest metrics."""
        data = await self._fetch_current_data()
        vol = data["volatility"]
        # For correlation, we use difference between interest rate and inflation as a proxy
        corr_proxy = abs(data["interest_rate"] - data["inflation"])
        self.vol_window.append(vol)
        if len(self.vol_window) > self.window_size:
            self.vol_window.pop(0)
        self.corr_window.append(corr_proxy)
        if len(self.corr_window) > self.window_size:
            self.corr_window.pop(0)

    def get_smoothed_metrics(self) -> Dict[str, float]:
        """Return smoothed volatility and correlation proxies."""
        if not self.vol_window:
            return {"volatility": 0.0, "correlation": 0.0}
        avg_vol = sum(self.vol_window) / len(self.vol_window)
        avg_corr = sum(self.corr_window) / len(self.corr_window) if self.corr_window else 0.0
        return {"volatility": avg_vol, "correlation": avg_corr}


# Create a global manager instance for convenience
_GLOBAL_MANAGER: MacroDataManager | None = None

def _get_manager() -> MacroDataManager:
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        _GLOBAL_MANAGER = MacroDataManager()
    return _GLOBAL_MANAGER


async def get_macro_metrics_async() -> Dict[str, float]:
    """Public API to obtain smoothed macro volatility and correlation.

    This function updates the macro data manager with the latest
    measurements and then returns the smoothed metrics. It is designed
    to be awaited by other modules (e.g. risk manager) when they need
    fresh macro inputs.
    """
    manager = _get_manager()
    await manager.update()
    return manager.get_smoothed_metrics()