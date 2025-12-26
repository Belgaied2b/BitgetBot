# =====================================================================
# sizing.py — position sizing institutionnel Bitget (USDT risk fixed)
# =====================================================================
import math


def compute_position_size(
    entry: float,
    stop: float,
    risk_usdt: float,
    lot_multiplier: float,
    lot_size: float,
) -> float:
    """
    Calcule la taille de position basée sur :
        - distance SL
        - risque USDT fixe
        - multiplier Bitget
        - lot size minimum

    entry: prix entrée
    stop: prix stop
    risk_usdt: montant de risque USDT
    lot_multiplier: quantité sous-jacente par lot (ex: 0.001 BTC)
    lot_size: lot minimal imposé par Bitget
    """

    dist = abs(entry - stop)
    if dist <= 0:
        return 0.0

    # Perte par lot = dist * lot_multiplier
    loss_per_lot = dist * float(lot_multiplier)
    if loss_per_lot <= 0:
        return lot_size

    est = risk_usdt / loss_per_lot

    # Arrondi au lot minimal
    lots = math.floor(est / lot_size) * lot_size
    return max(lot_size, lots)
