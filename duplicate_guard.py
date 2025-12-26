# =====================================================================
# duplicate_guard.py — Anti-doublons signaux / ordres (Desk-lead FINAL)
# =====================================================================
# ✅ O(1) avg cleanup using expiry queue (no full dict scan every call)
# ✅ Thread-safe (Lock) for asyncio/to_thread safety
# ✅ Optional persistence (load/save) to survive restarts (Railway)
#
# API kept:
# - is_duplicate(fingerprint) -> bool
# - mark(fingerprint) -> None
# - seen(fingerprint) -> bool
# - purge() / size()
# - fingerprint(symbol, side, entry, sl, tp1=None, extra=None, precision=6) -> str
# =====================================================================

from __future__ import annotations

import json
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple


def _now() -> float:
    return time.time()


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _fmt_num(x: Optional[float], precision: int) -> str:
    if x is None:
        return "NA"
    try:
        v = float(x)
        # avoid "-0.0"
        if abs(v) < 10 ** (-(precision + 2)):
            v = 0.0
        # stable representation
        s = f"{v:.{int(precision)}f}"
        # strip trailing zeros/dot for stability
        s = s.rstrip("0").rstrip(".")
        return s if s else "0"
    except Exception:
        return "NA"


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

    - precision: arrondi/format stable (évite micro variations float)
    - extra: setup_type / entry_type / mode / htf, etc.
    """
    sym = (symbol or "").strip().upper()
    sd = (side or "").strip().upper()

    e = _safe_float(entry)
    s = _safe_float(sl)
    t = _safe_float(tp1)

    parts = [sym, sd, _fmt_num(e, precision), _fmt_num(s, precision), _fmt_num(t, precision)]
    if extra is not None:
        parts.append(str(extra).strip())

    return "|".join(parts)


# ---------------------------------------------------------------------
# Duplicate Guard
# ---------------------------------------------------------------------
@dataclass
class DuplicateGuard:
    """
    Garde-fou anti doublons via fingerprints.

    ttl_seconds: durée de vie d'un fingerprint (ex: 3600 = 1h)
    max_items: optionnel, pour éviter une dérive mémoire si jamais ça spam.
    """

    ttl_seconds: int = 3600
    max_items: int = 50_000

    def __post_init__(self) -> None:
        # fp -> expiry_ts
        self._cache: Dict[str, float] = {}
        # queue of (expiry_ts, fp) for efficient cleanup
        self._q: Deque[Tuple[float, str]] = deque()
        self._lock = threading.Lock()

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------
    def _cleanup(self, now: Optional[float] = None) -> None:
        now = _now() if now is None else float(now)

        # Pop expired from queue; only delete if expiry matches current cache expiry
        while self._q:
            exp, fp = self._q[0]
            if exp > now:
                break
            self._q.popleft()
            cur = self._cache.get(fp)
            if cur is not None and cur <= now and abs(cur - exp) < 1e-9:
                self._cache.pop(fp, None)

        # Hard cap memory
        if self.max_items > 0 and len(self._cache) > self.max_items:
            # Evict oldest by draining queue until under cap
            target = int(self.max_items * 0.95)
            while self._q and len(self._cache) > target:
                exp, fp = self._q.popleft()
                cur = self._cache.get(fp)
                # Remove even if not expired (we're enforcing a cap)
                if cur is not None and abs(cur - exp) < 1e-9:
                    self._cache.pop(fp, None)

    # -----------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------
    def is_duplicate(self, fp: str) -> bool:
        """
        True si fingerprint déjà vu dans la fenêtre TTL.
        """
        if not fp:
            return False
        now = _now()
        with self._lock:
            self._cleanup(now)
            exp = self._cache.get(fp)
            return (exp is not None) and (exp > now)

    def mark(self, fp: str) -> None:
        """
        Enregistre fingerprint "vu maintenant".
        """
        if not fp:
            return
        now = _now()
        exp = now + float(self.ttl_seconds)
        with self._lock:
            self._cleanup(now)
            self._cache[fp] = exp
            self._q.append((exp, fp))

    def seen(self, fp: str) -> bool:
        """
        Retro-compat: check + mark en une seule opération.
        Retourne True si déjà vu, sinon False et enregistre.
        """
        if not fp:
            return False
        now = _now()
        with self._lock:
            self._cleanup(now)
            exp = self._cache.get(fp)
            if exp is not None and exp > now:
                return True
            exp2 = now + float(self.ttl_seconds)
            self._cache[fp] = exp2
            self._q.append((exp2, fp))
            return False

    def purge(self) -> None:
        """Force un nettoyage complet."""
        with self._lock:
            self._cache.clear()
            self._q.clear()

    def size(self) -> int:
        """Nombre de fingerprints actuellement en mémoire (après cleanup)."""
        now = _now()
        with self._lock:
            self._cleanup(now)
            return len(self._cache)

    # -----------------------------------------------------------------
    # Optional persistence (restart-safe)
    # -----------------------------------------------------------------
    def save(self, path: str) -> None:
        """
        Sauvegarde les fingerprints non expirés.
        """
        if not path:
            return
        now = _now()
        with self._lock:
            self._cleanup(now)
            data = {"ttl_seconds": self.ttl_seconds, "items": self._cache}
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        import os
        os.replace(tmp, path)

    def load(self, path: str) -> int:
        """
        Recharge les fingerprints non expirés.
        Retourne le nombre d'items chargés.
        """
        if not path:
            return 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return 0

        now = _now()
        items = (data or {}).get("items")
        if not isinstance(items, dict):
            return 0

        kept = 0
        with self._lock:
            self._cache.clear()
            self._q.clear()
            for fp, exp in items.items():
                try:
                    expf = float(exp)
                except Exception:
                    continue
                if expf > now:
                    self._cache[str(fp)] = expf
                    self._q.append((expf, str(fp)))
                    kept += 1
            self._cleanup(now)
        return kept
