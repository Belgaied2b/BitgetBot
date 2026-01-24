"""
Improved TP Clamp Engine
========================

This module provides an enhanced take-profit calculation engine for
institutional trading strategies. It parameterizes the coefficients
used to determine the distance of take-profit targets based on
volatility, momentum and liquidity, and integrates order book depth to
ensure that targets are placed at realistic levels where sufficient
liquidity exists.

The primary interface is the `TPClamp` class with a method
`compute_targets` that returns two take-profit levels (TP1 and TP2)
relative to an entry price. The calculation uses coefficients defined
in the central configuration loaded via `settings_improved`.

Formula Overview
----------------

For a trade direction (long = +1, short = -1), the first take-profit
is computed as:

```
TP1 = entry + direction * (c_atr * ATR + c_mom * momentum + c_liq / max(depth, 1e-6))
```

Where:

* `ATR` is the average true range on the relevant timeframe
* `momentum` is a momentum metric (e.g. RSI deviation or rate of change)
* `liquidity` is an estimate of available volume at the top of book
* `depth` is the order book depth at the best bid/ask for the symbol
* `c_atr`, `c_mom` and `c_liq` are coefficients configurable via settings

The second take-profit (TP2) is placed further out based on a risk
reward multiplier `c_rr`:

```
TP2 = entry + direction * c_rr * (TP1 - entry)
```

Both levels are clamped to respect minimum distances from the entry to
avoid unrealistic targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from settings_improved import get_settings


@dataclass
class TPClamp:
    settings: dict

    def __init__(self, settings: dict | None = None) -> None:
        self.settings = settings or get_settings()
        # Read coefficients from settings
        clamp_settings = self.settings.get("tp_clamp", {})
        # Note: coefficient names correspond to their effects.
        self.c_atr = clamp_settings.get("atr_coeff", 1.0)
        self.c_mom = clamp_settings.get("momentum_coeff", 1.0)
        self.c_liq = clamp_settings.get("liquidity_coeff", 1.0)
        self.c_rr = clamp_settings.get("rr_multiplier", 2.0)
        self.min_distance = clamp_settings.get("min_distance_ticks", 1.0)

    def compute_targets(
        self,
        entry_price: float,
        direction: int,
        atr: float,
        momentum: float,
        liquidity: float,
        order_book_depth: float,
    ) -> Tuple[float, float]:
        """Compute the first and second take-profit levels.

        Args:
            entry_price: The price at which the trade is entered.
            direction: +1 for long trades, -1 for short trades.
            atr: Average true range (volatility measure).
            momentum: Momentum indicator value (positive for upward momentum).
            liquidity: Estimated liquidity factor (higher = more liquid).
            order_book_depth: Available depth at top of book (USD size).

        Returns:
            A tuple (tp1, tp2) with the computed take-profit prices.
        """
        # Compute raw distance for TP1
        depth_factor = 1.0 / (order_book_depth + 1e-6)
        raw_distance = (
            self.c_atr * atr + self.c_mom * momentum + self.c_liq * depth_factor
        )
        # Ensure a minimum distance
        distance = max(self.min_distance, raw_distance)
        tp1 = entry_price + direction * distance
        # Second target uses risk reward multiplier
        tp2_distance = self.c_rr * distance
        tp2 = entry_price + direction * tp2_distance
        return tp1, tp2