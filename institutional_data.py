# institutional_data.py
# -*- coding: utf-8 -*-

"""
Institutional data layer (Binance USD-M focused) with:
- LIGHT mode: read-only from a WebSocket hub cache (no per-symbol REST spam).
- FULL mode: optional REST supplements (e.g., Open Interest) with hard cooldown/backoff.
- Stable output keys to prevent KeyError and "missing field" issues.
- Component availability + gating helpers (available_components_count).

This module is designed to be safe under multi-symbol scans:
- If REST starts failing (429 / 418 / -1003), we cool down globally and per symbol.
- analyze_signal / scanner should prefer LIGHT for pass-1, FULL only for shortlist.

Expected (optional) hub module:
- institutional_ws_hub.py exposing a singleton or function returning a hub object with:
    hub.get(symbol) -> dict snapshot (fresh timestamps)
You can plug any hub; if missing, LIGHT returns mostly None and availability shows low.
"""

from __future__ import annotations

import os
import time
import math
import json
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# Optional dependency (most projects have it)
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore


# ----------------------------
# Optional project settings
# ----------------------------
def _load_settings_value(name: str, default: Any) -> Any:
    # 1) env
    if name in os.environ:
        return os.environ[name]
    # 2) settings.py if present
    try:
        import settings  # type: ignore

        if hasattr(settings, name):
            return getattr(settings, name)
    except Exception:
        pass
    return default


# REST base (Binance USD-M Futures)
BINANCE_FAPI_REST_BASE = _load_settings_value("BINANCE_FAPI_REST_BASE", "https://fapi.binance.com")

# Defaults
INST_DEFAULT_MODE = str(_load_settings_value("INST_DEFAULT_MODE", "LIGHT")).upper().strip()  # LIGHT / FULL
INST_LIGHT_TTL_MS = int(_load_settings_value("INST_LIGHT_TTL_MS", 7_000))  # hub freshness window
INST_FULL_TTL_MS = int(_load_settings_value("INST_FULL_TTL_MS", 15_000))

# Cooldowns/backoff
INST_SYMBOL_COOLDOWN_SEC = float(_load_settings_value("INST_SYMBOL_COOLDOWN_SEC", 8.0))
INST_GLOBAL_COOLDOWN_SEC = float(_load_settings_value("INST_GLOBAL_COOLDOWN_SEC", 3.0))
INST_BAN_COOLDOWN_SEC = float(_load_settings_value("INST_BAN_COOLDOWN_SEC", 90.0))  # when we detect ban-ish errors

# REST timeouts
INST_REST_TIMEOUT = float(_load_settings_value("INST_REST_TIMEOUT", 4.0))

# Scoring thresholds (directional)
# You can move these to settings.py if you want finer control.
INST_FUNDING_ABS_OK = float(_load_settings_value("INST_FUNDING_ABS_OK", 0.0005))  # 0.05%
INST_DELTA_USD_OK = float(_load_settings_value("INST_DELTA_USD_OK", 50_000.0))
INST_LIQ_USD_OK = float(_load_settings_value("INST_LIQ_USD_OK", 200_000.0))
INST_OI_CHANGE_OK = float(_load_settings_value("INST_OI_CHANGE_OK", 0.01))  # +1% in window

# If you want strict gating when components missing:
INST_REQUIRE_MIN_AVAILABLE = int(_load_settings_value("INST_REQUIRE_MIN_AVAILABLE", 2))


# ----------------------------
# Utilities
# ----------------------------
def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return float(x)
        if isinstance(x, (int, float)):
            if math.isfinite(float(x)):
                return float(x)
            return None
        s = str(x).strip()
        if s == "":
            return None
        v = float(s)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            if math.isfinite(x):
                return int(x)
            return None
        s = str(x).strip()
        if s == "":
            return None
        return int(float(s))
    except Exception:
        return None


def normalize_binance_symbol(symbol: str) -> str:
    """
    Tries to map KuCoin-ish or project symbols to Binance USD-M symbol format:
    - 'BTCUSDTM' -> 'BTCUSDT'
    - 'BTC-USDT' -> 'BTCUSDT'
    - 'BTC/USDT' -> 'BTCUSDT'
    - 'BTCUSDT_PERP' -> 'BTCUSDT'
    """
    if not symbol:
        return symbol
    s = symbol.upper().replace("-", "").replace("/", "").replace(":", "")
    s = s.replace("_PERP", "").replace("PERP", "")
    if s.endswith("USDTM"):
        s = s[:-1]  # drop trailing 'M' -> USDT
    return s


# ----------------------------
# Backoff / cooldown manager
# ----------------------------
@dataclass
class BackoffState:
    next_allowed_ts: float = 0.0
    ban_until_ts: float = 0.0
    consecutive_errors: int = 0

    def is_banned(self) -> bool:
        return time.time() < self.ban_until_ts

    def is_in_cooldown(self) -> bool:
        return time.time() < self.next_allowed_ts

    def mark_success(self) -> None:
        self.consecutive_errors = 0
        self.next_allowed_ts = time.time() + INST_GLOBAL_COOLDOWN_SEC

    def mark_error(self, severe: bool = False) -> None:
        self.consecutive_errors += 1
        base = INST_GLOBAL_COOLDOWN_SEC
        # exponential-ish
        cooldown = min(60.0, base * (1.6 ** min(self.consecutive_errors, 8)))
        self.next_allowed_ts = time.time() + cooldown
        if severe:
            self.ban_until_ts = max(self.ban_until_ts, time.time() + INST_BAN_COOLDOWN_SEC)


@dataclass
class SymbolBackoffState:
    next_allowed_ts: float = 0.0
    ban_until_ts: float = 0.0
    consecutive_errors: int = 0

    def is_banned(self) -> bool:
        return time.time() < self.ban_until_ts

    def is_in_cooldown(self) -> bool:
        return time.time() < self.next_allowed_ts

    def mark_success(self) -> None:
        self.consecutive_errors = 0
        self.next_allowed_ts = time.time() + INST_SYMBOL_COOLDOWN_SEC

    def mark_error(self, severe: bool = False) -> None:
        self.consecutive_errors += 1
        base = INST_SYMBOL_COOLDOWN_SEC
        cooldown = min(120.0, base * (1.7 ** min(self.consecutive_errors, 8)))
        self.next_allowed_ts = time.time() + cooldown
        if severe:
            self.ban_until_ts = max(self.ban_until_ts, time.time() + INST_BAN_COOLDOWN_SEC)


# ----------------------------
# REST client (optional supplements)
# ----------------------------
class BinanceFapiRestClient:
    def __init__(self, base_url: str = BINANCE_FAPI_REST_BASE, timeout: float = INST_REST_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Returns: (json_dict_or_none, meta)
        meta includes:
          - ok (bool)
          - http_status (int|None)
          - error_code (int|None)
          - error_msg (str|None)
        """
        meta = {"ok": False, "http_status": None, "error_code": None, "error_msg": None}

        if requests is None:
            meta["error_msg"] = "requests_not_available"
            return None, meta

        url = f"{self.base_url}{path}"
        try:
            r = requests.get(url, params=params or {}, timeout=self.timeout)
            meta["http_status"] = r.status_code
            # Binance may return JSON error payload with code/msg
            try:
                data = r.json()
            except Exception:
                data = None

            if r.status_code >= 200 and r.status_code < 300 and isinstance(data, dict):
                meta["ok"] = True
                return data, meta

            # errors
            if isinstance(data, dict):
                meta["error_code"] = _safe_int(data.get("code"))
                meta["error_msg"] = str(data.get("msg")) if data.get("msg") is not None else None
            else:
                meta["error_msg"] = f"non_json_error_http_{r.status_code}"
            return None, meta

        except Exception as e:
            meta["error_msg"] = f"exception:{type(e).__name__}"
            return None, meta

    def fetch_open_interest(self, symbol: str) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        USD-M endpoint (documented by Binance):
          GET /fapi/v1/openInterest?symbol=BTCUSDT
        Response has "openInterest" (string number).
        """
        data, meta = self._request_json("/fapi/v1/openInterest", params={"symbol": symbol})
        if not meta.get("ok") or not isinstance(data, dict):
            return None, meta
        return _safe_float(data.get("openInterest")), meta


# ----------------------------
# Hub adapter (optional)
# ----------------------------
class HubAdapter:
    """
    Minimal adapter contract:
      get(symbol) -> dict with latest fields and a timestamp (ms)
    """

    def __init__(self) -> None:
        self._hub = None
        self._init_attempted = False

    def _lazy_init(self) -> None:
        if self._init_attempted:
            return
        self._init_attempted = True
        try:
            # expected file you will add later
            import institutional_ws_hub  # type: ignore

            # common patterns:
            # - institutional_ws_hub.HUB
            # - institutional_ws_hub.get_hub()
            if hasattr(institutional_ws_hub, "HUB"):
                self._hub = getattr(institutional_ws_hub, "HUB")
            elif hasattr(institutional_ws_hub, "get_hub"):
                self._hub = institutional_ws_hub.get_hub()  # type: ignore
            else:
                self._hub = None
        except Exception:
            self._hub = None

    def available(self) -> bool:
        self._lazy_init()
        return self._hub is not None

    def get(self, symbol: str) -> Optional[Dict[str, Any]]:
        self._lazy_init()
        if self._hub is None:
            return None
        try:
            if hasattr(self._hub, "get"):
                return self._hub.get(symbol)  # type: ignore
            return None
        except Exception:
            return None


# ----------------------------
# Institutional service
# ----------------------------
class InstitutionalService:
    def __init__(self) -> None:
        self.rest = BinanceFapiRestClient()
        self.hub = HubAdapter()

        self._lock = threading.Lock()
        self._global = BackoffState()
        self._per_symbol: Dict[str, SymbolBackoffState] = {}

        # cache: symbol -> {"ts": ms, "data": {...}}
        self._cache_light: Dict[str, Dict[str, Any]] = {}
        self._cache_full: Dict[str, Dict[str, Any]] = {}

        # for OI delta
        self._last_oi: Dict[str, Tuple[int, float]] = {}

    def _sym_state(self, symbol: str) -> SymbolBackoffState:
        with self._lock:
            if symbol not in self._per_symbol:
                self._per_symbol[symbol] = SymbolBackoffState()
            return self._per_symbol[symbol]

    def _mark_error(self, symbol: str, meta: Dict[str, Any]) -> None:
        http_status = meta.get("http_status")
        err_code = meta.get("error_code")
        severe = False

        # severe ban-ish signals:
        # - HTTP 418 (IP ban) / 429 (rate limit) are common in practice
        # - API error code -1003 indicates too many requests
        if http_status in (418, 429):
            severe = True
        if err_code == -1003:
            severe = True

        with self._lock:
            self._global.mark_error(severe=severe)
            st = self._sym_state(symbol)
            st.mark_error(severe=severe)

    def _mark_success(self, symbol: str) -> None:
        with self._lock:
            self._global.mark_success()
            st = self._sym_state(symbol)
            st.mark_success()

    def global_ban_until(self) -> float:
        with self._lock:
            return self._global.ban_until_ts

    def symbol_ban_until(self, symbol: str) -> float:
        with self._lock:
            st = self._sym_state(symbol)
            return st.ban_until_ts

    def _cooldown_blocked(self, symbol: str) -> bool:
        with self._lock:
            if self._global.is_banned() or self._global.is_in_cooldown():
                return True
            st = self._sym_state(symbol)
            if st.is_banned() or st.is_in_cooldown():
                return True
        return False

    @staticmethod
    def _fresh(ts_ms: Optional[int], ttl_ms: int) -> bool:
        if ts_ms is None:
            return False
        return (_now_ms() - ts_ms) <= ttl_ms

    def _base_output(self) -> Dict[str, Any]:
        # Stable keys: keep these always present to avoid KeyError.
        return {
            "symbol": None,
            "ts": None,
            "mode": None,
            "source": None,  # "hub" | "rest" | "mixed" | "none"
            "available": False,
            "available_components_count": 0,
            # components (raw)
            "fundingRate": None,
            "openInterest": None,
            "openInterestChange": None,  # fractional change in a window
            "markPrice": None,
            "bookImbalance": None,  # (bid-ask)/(bid+ask)
            "deltaUsd_1m": None,
            "cvdUsd_5m": None,
            "liquidationsUsd_5m": None,
            # component availability flags
            "components": {
                "funding": False,
                "oi": False,
                "delta": False,
                "liq": False,
                "book": False,
            },
            # debug/meta
            "meta": {
                "global_ban_until": None,
                "symbol_ban_until": None,
                "rest": {},
                "hub": {},
            },
        }

    def _read_from_hub(self, symbol: str) -> Dict[str, Any]:
        out = self._base_output()
        out["symbol"] = symbol

        snap = self.hub.get(symbol) if self.hub.available() else None
        if not isinstance(snap, dict):
            out["source"] = "none"
            return out

        # timestamp in ms (common fields in caches)
        ts = _safe_int(snap.get("ts")) or _safe_int(snap.get("timestamp")) or _safe_int(snap.get("T"))
        out["ts"] = ts
        out["source"] = "hub"
        out["meta"]["hub"] = {"has_snapshot": True}

        # funding (from markPrice stream cache or derived)
        fr = _safe_float(snap.get("fundingRate") if "fundingRate" in snap else snap.get("funding_rate"))
        mp = _safe_float(snap.get("markPrice") if "markPrice" in snap else snap.get("mark_price"))

        # orderbook best bid/ask or imbalance
        bid = _safe_float(snap.get("bid") if "bid" in snap else snap.get("bestBid"))
        ask = _safe_float(snap.get("ask") if "ask" in snap else snap.get("bestAsk"))
        imb = _safe_float(snap.get("bookImbalance") if "bookImbalance" in snap else snap.get("book_imbalance"))
        if imb is None and bid is not None and ask is not None and (bid + ask) > 0:
            imb = (bid - ask) / (bid + ask)

        # delta & cvd in USD (if hub computed it)
        d1m = _safe_float(snap.get("deltaUsd_1m") if "deltaUsd_1m" in snap else snap.get("delta_usd_1m"))
        cvd5 = _safe_float(snap.get("cvdUsd_5m") if "cvdUsd_5m" in snap else snap.get("cvd_usd_5m"))

        # liquidations (if hub computed it)
        liq5 = _safe_float(
            snap.get("liquidationsUsd_5m") if "liquidationsUsd_5m" in snap else snap.get("liq_usd_5m")
        )

        # write stable keys
        out["fundingRate"] = fr
        out["markPrice"] = mp
        out["bookImbalance"] = imb
        out["deltaUsd_1m"] = d1m
        out["cvdUsd_5m"] = cvd5
        out["liquidationsUsd_5m"] = liq5

        # availability flags
        out["components"]["funding"] = fr is not None
        out["components"]["book"] = imb is not None
        out["components"]["delta"] = (d1m is not None) or (cvd5 is not None)
        out["components"]["liq"] = liq5 is not None

        return out

    def _supplement_open_interest(self, symbol: str, out: Dict[str, Any]) -> Dict[str, Any]:
        # REST supplement, guarded by cooldown/backoff
        out["meta"]["rest"]["attempted"] = True

        if self._cooldown_blocked(symbol):
            out["meta"]["rest"]["blocked"] = True
            return out

        oi, meta = self.rest.fetch_open_interest(symbol)
        out["meta"]["rest"]["openInterest"] = {"ok": meta.get("ok"), "http": meta.get("http_status"), "code": meta.get("error_code")}
        if oi is None:
            self._mark_error(symbol, meta)
            return out

        self._mark_success(symbol)
        out["openInterest"] = oi
        out["components"]["oi"] = True

        # compute change vs last seen
        now = _now_ms()
        with self._lock:
            prev = self._last_oi.get(symbol)
            self._last_oi[symbol] = (now, oi)

        if prev:
            prev_ts, prev_oi = prev
            # only compute if previous is recent-ish (15 min)
            if prev_oi > 0 and (now - prev_ts) <= 15 * 60 * 1000:
                out["openInterestChange"] = (oi - prev_oi) / prev_oi

        return out

    def get_institutional(self, raw_symbol: str, mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Main fetch:
        - LIGHT: hub only (no REST)
        - FULL : hub + optional REST supplements (currently Open Interest)
        """
        mode_use = (mode or INST_DEFAULT_MODE).upper().strip()
        symbol = normalize_binance_symbol(raw_symbol)

        # Light cache
        now = _now_ms()
        ttl = INST_LIGHT_TTL_MS if mode_use == "LIGHT" else INST_FULL_TTL_MS

        # Return cached if fresh
        cache = self._cache_light if mode_use == "LIGHT" else self._cache_full
        cached = cache.get(symbol)
        if isinstance(cached, dict) and self._fresh(_safe_int(cached.get("ts")), ttl):
            out = dict(cached)
            out["mode"] = mode_use
            return out

        out = self._read_from_hub(symbol)
        out["mode"] = mode_use
        out["meta"]["global_ban_until"] = self.global_ban_until()
        out["meta"]["symbol_ban_until"] = self.symbol_ban_until(symbol)

        # Determine freshness/availability
        out["available"] = self._fresh(_safe_int(out.get("ts")), ttl)

        # FULL mode: supplement with REST (guarded)
        if mode_use == "FULL":
            out["source"] = out["source"] if out["source"] != "none" else "rest"
            out = self._supplement_open_interest(symbol, out)
            if out["openInterest"] is not None and out["source"] == "hub":
                out["source"] = "mixed"

        # recompute available_components_count
        comp = out.get("components", {}) or {}
        out["available_components_count"] = sum(1 for k in ("funding", "oi", "delta", "liq", "book") if comp.get(k) is True)

        # cache store
        cache[symbol] = dict(out)
        return out


# Singleton service (simple drop-in)
_INST = InstitutionalService()


# ----------------------------
# Scoring / gating
# ----------------------------
@dataclass
class InstScoreConfig:
    funding_abs_ok: float = INST_FUNDING_ABS_OK
    delta_usd_ok: float = INST_DELTA_USD_OK
    liq_usd_ok: float = INST_LIQ_USD_OK
    oi_change_ok: float = INST_OI_CHANGE_OK
    min_available: int = INST_REQUIRE_MIN_AVAILABLE


def compute_institutional_score(
    inst: Dict[str, Any],
    direction: str,
    cfg: Optional[InstScoreConfig] = None,
) -> Dict[str, Any]:
    """
    Returns a stable, explainable score bundle:
      {
        "inst_score": int,
        "available_components_count": int,
        "required_min_available": int,
        "ok_components": [...],
        "warn_components": [...],
        "fail_components": [...],
        "details": {...}
      }
    Direction: "long"/"short" (anything starting with 'l' treated as long).
    """
    cfg = cfg or InstScoreConfig()
    is_long = str(direction).lower().startswith("l")

    comp_flags = (inst.get("components") or {}).copy()
    available = int(inst.get("available_components_count") or 0)

    out = {
        "inst_score": 0,
        "available_components_count": available,
        "required_min_available": int(cfg.min_available),
        "ok_components": [],
        "warn_components": [],
        "fail_components": [],
        "details": {},
    }

    # If too few components, we don't fabricate signals.
    if available <= 0:
        out["fail_components"].append("no_components")
        return out

    # Funding: typically directional interpretation depends on your strategy.
    # Here we do a conservative mapping:
    # - Funding near 0 is "good"; extreme funding is "warn" (crowded).
    fr = _safe_float(inst.get("fundingRate"))
    if fr is not None:
        out["details"]["fundingRate"] = fr
        if abs(fr) <= cfg.funding_abs_ok:
            out["inst_score"] += 1
            out["ok_components"].append("funding")
        else:
            out["warn_components"].append("funding")

    # Orderbook imbalance: positive favors long, negative favors short (very rough proxy)
    imb = _safe_float(inst.get("bookImbalance"))
    if imb is not None:
        out["details"]["bookImbalance"] = imb
        if (is_long and imb > 0) or ((not is_long) and imb < 0):
            out["inst_score"] += 1
            out["ok_components"].append("book")
        else:
            out["warn_components"].append("book")

    # Delta/CVD: if hub provides one of them
    d1m = _safe_float(inst.get("deltaUsd_1m"))
    cvd5 = _safe_float(inst.get("cvdUsd_5m"))
    best_delta = d1m if d1m is not None else cvd5
    if best_delta is not None:
        out["details"]["deltaProxyUsd"] = best_delta
        if (is_long and best_delta >= cfg.delta_usd_ok) or ((not is_long) and best_delta <= -cfg.delta_usd_ok):
            out["inst_score"] += 1
            out["ok_components"].append("delta")
        else:
            out["warn_components"].append("delta")

    # Liquidations: high liquidation flow can indicate forced moves / squeezes.
    # We only treat "high liq" as confirmation of activity, not direction.
    liq5 = _safe_float(inst.get("liquidationsUsd_5m"))
    if liq5 is not None:
        out["details"]["liquidationsUsd_5m"] = liq5
        if liq5 >= cfg.liq_usd_ok:
            out["inst_score"] += 1
            out["ok_components"].append("liq")
        else:
            out["warn_components"].append("liq")

    # Open interest change: only if we have a computed change
    oic = _safe_float(inst.get("openInterestChange"))
    if oic is not None:
        out["details"]["openInterestChange"] = oic
        # We only score OI change if it’s positive and meaningful; direction logic can be refined later.
        if oic >= cfg.oi_change_ok:
            out["inst_score"] += 1
            out["ok_components"].append("oi")
        else:
            out["warn_components"].append("oi")

    # Minimum availability gating (so you can implement A/B/C tiers cleanly)
    if available < cfg.min_available:
        out["fail_components"].append("available_below_min")

    return out


def institutional_gating_ok(score_bundle: Dict[str, Any]) -> bool:
    """
    Simple gate: enough components AND score >= 2.
    (Adjust in settings or in analyze_signal tiers.)
    """
    available = int(score_bundle.get("available_components_count") or 0)
    min_avail = int(score_bundle.get("required_min_available") or 0)
    score = int(score_bundle.get("inst_score") or 0)
    if available < min_avail:
        return False
    return score >= 2


# ----------------------------
# Public API (drop-in helpers)
# ----------------------------
def get_institutional_data(symbol: str, mode: str = "LIGHT") -> Dict[str, Any]:
    """
    Stable raw snapshot. Keys like fundingRate/openInterest always exist (may be None).
    """
    return _INST.get_institutional(symbol, mode=mode)


def get_institutional_score(
    symbol: str,
    direction: str,
    mode: str = "LIGHT",
    cfg: Optional[InstScoreConfig] = None,
) -> Dict[str, Any]:
    """
    Convenience: returns {"inst": <raw>, "score": <bundle>, "gating_ok": bool}
    """
    inst = _INST.get_institutional(symbol, mode=mode)
    score = compute_institutional_score(inst, direction=direction, cfg=cfg)
    return {"inst": inst, "score": score, "gating_ok": institutional_gating_ok(score)}


def get_institutional_availability(symbol: str, mode: str = "LIGHT") -> Dict[str, Any]:
    """
    Useful for scanner to decide tier C fallback:
    - available_components_count
    - ban timers
    """
    inst = _INST.get_institutional(symbol, mode=mode)
    return {
        "symbol": inst.get("symbol"),
        "available": bool(inst.get("available")),
        "available_components_count": int(inst.get("available_components_count") or 0),
        "global_ban_until": inst.get("meta", {}).get("global_ban_until"),
        "symbol_ban_until": inst.get("meta", {}).get("symbol_ban_until"),
        "source": inst.get("source"),
    }


# Optional: tiny debug helper
def dump_institutional(symbol: str, direction: str = "long", mode: str = "LIGHT") -> str:
    payload = get_institutional_score(symbol, direction=direction, mode=mode)
    return json.dumps(payload, indent=2, sort_keys=True, default=str)
