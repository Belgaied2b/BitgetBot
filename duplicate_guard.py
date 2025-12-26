# =====================================================================
# duplicate_guard.py — Anti-doublons signaux / ordres
# =====================================================================
# Objectif:
# - Empêcher d'envoyer / trader le même signal plusieurs fois dans une fenêtre TTL
# - Fonctionne avec un "fingerprint" (string) stable
#
# API:
# - is_duplicate(fingerprint) -> bool
# - mark(fingerprint) -> None
# - seen(fingerprint) -> bool   (retro-compat: check + mark)
# - fingerprint(symbol, side, entry, sl, tp1=None, extra=None) -> str
# =====================================================================

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DuplicateGuard:
    """
    Garde-fou anti doublons via fingerprints.

    ttl_seconds: durée de vie d'un fingerprint (ex: 3600 = 1h)
    """

    ttl_seconds: int = 3600

    def __post_init__(self) -> None:
        # cache: fingerprint -> timestamp
        self._cache: Dict[str, float] = {}

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------

    def _now(self) -> float:
        return time.time()

    def _cleanup(self) -> None:
        """Nettoie le cache des fingerprints expirés."""
        now = self._now()
        expired = [k for k, ts in self._cache.items() if (now - ts) > self.ttl_seconds]
        for k in expired:
            del self._cache[k]

    # -----------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------

    def is_duplicate(self, fingerprint: str) -> bool:
        """
        True si fingerprint déjà vu dans la fenêtre TTL.
        """
        if not fingerprint:
            return False
        self._cleanup()
        return fingerprint in self._cache

    def mark(self, fingerprint: str) -> None:
        """
        Enregistre fingerprint "vu maintenant".
        """
        if not fingerprint:
            return
        self._cleanup()
        self._cache[fingerprint] = self._now()

    def seen(self, fingerprint: str) -> bool:
        """
        Retro-compat: check + mark en une seule opération.
        Retourne True si déjà vu, sinon False et enregistre.
        """
        if self.is_duplicate(fingerprint):
            return True
        self.mark(fingerprint)
        return False

    def purge(self) -> None:
        """Force un nettoyage complet."""
        self._cache.clear()

    def size(self) -> int:
        """Nombre de fingerprints actuellement en mémoire (après cleanup)."""
        self._cleanup()
        return len(self._cache)


# ---------------------------------------------------------------------
# Helper: générer un fingerprint stable
# ---------------------------------------------------------------------

def fingerprint(
    symbol: str,
    side: str,
    entry: float,
    sl: float,
    tp1: Optional[float] = None,
    extra: Optional[Any] = None,
    precision: int = 6,
) -> str:
    """
    Construit un fingerprint stable.

    - precision: arrondi (ex: 6) pour éviter des micro variations float
    - extra: peut être un setup_type, timeframe, mode, etc.
    """
    sym = (symbol or "").strip().upper()
    sd = (side or "").strip().upper()

    def r(x: Optional[float]) -> str:
        if x is None:
            return "NA"
        try:
            return str(round(float(x), precision))
        except Exception:
            return "NA"

    parts = [sym, sd, r(entry), r(sl), r(tp1)]
    if extra is not None:
        parts.append(str(extra))

    return "|".join(parts)
