# =====================================================================
# bitget_client.py — INSTITUTIONAL MARKET CLIENT (Desk Lead Hardened)
# =====================================================================
# ✅ Robust HTTP/JSON handling + latency + ok flag
# ✅ Retry intelligent: network/5xx/429 + jitter backoff
# ✅ Handles Bitget v3 candles param mismatch (interval vs granularity)
# ✅ Supports limit > 100 via pagination using endTime
# ✅ Fix interval case: minutes must be "1m" not "1M"
# ✅ Stable query ordering (signature-safe)
# ✅ Contracts cache + normalize symbols
# =====================================================================

from __future__ import annotations

import aiohttp
import asyncio
import base64
import hashlib
import hmac
import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

import pandas as pd

from retry_utils import retry_async
from settings import PRODUCT_TYPE, BITGET_HTTP_CONCURRENCY, BITGET_MIN_INTERVAL_SEC, BITGET_HTTP_TIMEOUT_S, BITGET_HTTP_RETRIES

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Symbol normalization
# ---------------------------------------------------------------------

def normalize_symbol(sym: str) -> str:
    """
    Standardise :
      BTC-USDT, BTCUSDTM, BTCUSDT → BTCUSDT
      XBTUSDT → BTCUSDT
    """
    if not sym:
        return ""
    s = str(sym).upper().replace("-", "")
    s = s.replace("USDTM", "USDT").replace("USDTSWAP", "USDT")
    if s.startswith("XBT"):
        s = s.replace("XBT", "BTC", 1)
    return s


# ---------------------------------------------------------------------
# Internal retryable exception
# ---------------------------------------------------------------------

class _Retryable(Exception):
    pass


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _normalize_interval(tf: str) -> Optional[str]:
    """
    Bitget v3 candles expects:
      minutes => "1m","3m","5m","15m","30m"
      hours   => "1H","4H","6H","12H"
      day     => "1D"
    Your bot tends to pass "1H"/"4H" already.
    """
    if not tf:
        return None
    t = str(tf).strip()

    # common aliases
    t = t.replace(" ", "")
    t = t.replace("h", "H").replace("d", "D")

    # minutes must be lowercase m
    # ex: "1M" -> "1m"
    if t.endswith("M") and t[:-1].isdigit():
        t = t[:-1] + "m"
    if t.endswith("m") and t[:-1].isdigit():
        # ok already
        pass

    valid = {"1m", "3m", "5m", "15m", "30m", "1H", "4H", "6H", "12H", "1D"}
    return t if t in valid else None


def _parse_candles_rows(data: Any) -> pd.DataFrame:
    """
    Expected format (strings):
      [ [ts, open, high, low, close, volume, turnover], ... ]
    """
    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    row0 = data[0] if isinstance(data[0], list) else None
    if not row0 or len(row0) < 6:
        return pd.DataFrame()

    cols = ["time", "open", "high", "low", "close", "volume", "turnover"]
    df = pd.DataFrame(data, columns=cols[: len(row0)])

    # coerce numeric
    for c in ["time", "open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["time", "open", "high", "low", "close"])

    # ensure ms int
    df["time"] = df["time"].astype("int64")

    df = df.sort_values("time").reset_index(drop=True)
    df = df[["time", "open", "high", "low", "close", "volume"]]
    return df


# ---------------------------------------------------------------------
# Bitget Client
# ---------------------------------------------------------------------

class BitgetClient:
    BASE = "https://api.bitget.com"

    def __init__(self, api_key: str, api_secret: str, passphrase: str):
        self.api_key = api_key
        self.api_secret = (api_secret or "").encode()
        self.api_passphrase = passphrase or ""

        self.session: Optional[aiohttp.ClientSession] = None

        # contracts cache
        self._contracts_cache: Optional[List[str]] = None
        self._contracts_ts: float = 0.0
        self._contracts_lock = asyncio.Lock()
        self._contracts_info_cache: Optional[List[Dict[str, Any]]] = None
        self._contracts_info_ts: float = 0.0

        # HTTP pacing / concurrency control (institutional hygiene)
        # NOTE: uses settings BITGET_HTTP_CONCURRENCY + BITGET_MIN_INTERVAL_SEC
        self._req_sem = asyncio.Semaphore(int(max(1, BITGET_HTTP_CONCURRENCY)))
        self._pace_lock = asyncio.Lock()
        self._min_interval = float(max(0.0, BITGET_MIN_INTERVAL_SEC))
        self._last_req_ts = 0.0

    async def close(self) -> None:
        if self.session and not self.session.closed:
            await self.session.close()

    async def _ensure_session(self) -> None:
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=25)
            connector = aiohttp.TCPConnector(limit=80, ttl_dns_cache=300, enable_cleanup_closed=True)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)

    async def _pace(self) -> None:
        """Global pacing across concurrent tasks (best-effort)."""
        try:
            mi = float(self._min_interval)
        except Exception:
            mi = 0.0
        if mi <= 0:
            return

        async with self._pace_lock:
            now = time.time()
            dt = now - float(self._last_req_ts)
            wait_s = float(mi) - float(dt)
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            self._last_req_ts = time.time()

    def _sign(self, ts: str, method: str, path: str, query: str, body: str) -> str:
        msg = f"{ts}{method}{path}{query}{body}"
        mac = hmac.new(self.api_secret, msg.encode(), hashlib.sha256).digest()
        return base64.b64encode(mac).decode()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        auth: bool = True,
        timeout_s: int = int(BITGET_HTTP_TIMEOUT_S),
        retries: int = int(BITGET_HTTP_RETRIES),
        base_delay: float = 0.35,
    ) -> Dict[str, Any]:
        """
        Robust wrapper:
          - stable query ordering
          - retries on: network, timeout, 5xx, HTTP429, and "too frequent" messages
          - returns dict with: ok, code, msg, _http_status, _latency_ms
        """
        await self._ensure_session()

        params = params or {}
        data = data or {}

        # stable query order (signature-safe)
        query = ""
        if params:
            items = sorted(params.items(), key=lambda kv: kv[0])
            query = "?" + "&".join(f"{k}={v}" for k, v in items)

        url = self.BASE + path + query
        body = json.dumps(data, separators=(",", ":")) if data else ""

        async def _do() -> Dict[str, Any]:
            ts = str(_now_ms())
            headers: Dict[str, str] = {
                "Content-Type": "application/json",
                "User-Agent": "desk-bot/bitget-client",
            }

            if auth:
                sig = self._sign(ts, method.upper(), path, query, body)
                headers.update(
                    {
                        "ACCESS-KEY": self.api_key,
                        "ACCESS-SIGN": sig,
                        "ACCESS-TIMESTAMP": ts,
                        "ACCESS-PASSPHRASE": self.api_passphrase,
                    }
                )

            # Concurrency + pacing gate
            async with self._req_sem:
                await self._pace()

                t0 = time.time()
                try:
                    async with self.session.request(
                        method.upper(),
                        url,
                        headers=headers,
                        data=body if data else None,
                        timeout=aiohttp.ClientTimeout(total=timeout_s),
                    ) as resp:
                        txt = await resp.text()
                        status = resp.status
                        latency_ms = int((time.time() - t0) * 1000)

                        # HTTP 429 / 5xx => retry
                        if status == 429:
                            LOGGER.warning("HTTP 429 %s %s query=%s", method, path, query)
                            raise _Retryable("HTTP 429 Too Many Requests")
                        if 500 <= status <= 599:
                            LOGGER.warning("HTTP %s %s %s query=%s raw=%s", status, method, path, query, txt[:300])
                            raise _Retryable(f"HTTP {status}")

                        # 4xx (except 429) => no retry (hard error)
                        if status >= 400:
                            return {
                                "ok": False,
                                "code": str(status),
                                "msg": "http_error",
                                "raw": txt,
                                "_http_status": status,
                                "_latency_ms": latency_ms,
                                "_path": path,
                            }

                        # parse JSON
                        try:
                            js = json.loads(txt) if txt else {}
                        except Exception:
                            return {
                                "ok": False,
                                "code": "NONJSON",
                                "msg": "json_decode_error",
                                "raw": txt,
                                "_http_status": status,
                                "_latency_ms": latency_ms,
                                "_path": path,
                            }

                        code = str(js.get("code", ""))
                        msg = str(js.get("msg", ""))

                        js["ok"] = (code == "00000")
                        js["_http_status"] = status
                        js["_latency_ms"] = latency_ms
                        js["_path"] = path

                        # Bitget sometimes rate-limits via code/msg even with HTTP 200
                        if (not js["ok"]) and ("too many" in msg.lower() or "frequency" in msg.lower() or code in {"429"}):
                            raise _Retryable(f"API rate limited code={code} msg={msg}")

                        return js

                except _Retryable:
                    raise
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    raise _Retryable(str(e))

        # retry with jitter via retry_async (raises if final fail)
        async def _wrapped():
            try:
                return await _do()
            except _Retryable as e:
                # small jitter to de-sync bursts
                await asyncio.sleep(random.random() * 0.08)
                raise e

        try:
            return await retry_async(_wrapped, retries=retries, base_delay=base_delay)
        except Exception as e:
            return {
                "ok": False,
                "code": "EXC",
                "msg": str(e),
                "_http_status": 0,
                "_latency_ms": 0,
                "_path": path,
            }

    # =================================================================
    # CONTRACT LIST (v2)
    # =================================================================

    async def get_contracts_list(self) -> List[str]:
        """
        Returns symbols USDT-FUTURES: BTCUSDT, ETHUSDT...
        Cached for 5 minutes.
        """
        now = time.time()
        if self._contracts_cache and (now - self._contracts_ts) < 300:
            return self._contracts_cache

        async with self._contracts_lock:
            now = time.time()
            if self._contracts_cache and (now - self._contracts_ts) < 300:
                return self._contracts_cache
            info = await self.get_contracts_info()
            symbols = [str(x.get("symbol") or "") for x in (info or []) if isinstance(x, dict)]
            symbols = [normalize_symbol(s) for s in symbols if s]
            symbols = sorted(set(symbols))

            LOGGER.info("📈 Loaded %d symbols from Bitget Futures", len(symbols))
            self._contracts_cache = symbols
            self._contracts_ts = time.time()
            return symbols

    async def get_contracts_info(self) -> List[Dict[str, Any]]:
        """
        Returns contracts list with best-effort metrics (volume/turnover/oi when available).
        Cached for 5 minutes.
        """
        now = time.time()
        if self._contracts_info_cache and (now - self._contracts_info_ts) < 300:
            return self._contracts_info_cache

        async with self._contracts_lock:
            now = time.time()
            if self._contracts_info_cache and (now - self._contracts_info_ts) < 300:
                return self._contracts_info_cache

            params = {"productType": str(PRODUCT_TYPE or "USDT-FUTURES")}

            js = await self._request(
                "GET",
                "/api/v2/mix/market/contracts",
                params=params,
                auth=False,
            )

            if not isinstance(js, dict) or "data" not in js:
                LOGGER.error("❌ CONTRACT ERROR: %s", js)
                return []

            def _pick_float(d: Dict[str, Any], keys: List[str]) -> Optional[float]:
                for k in keys:
                    if k not in d:
                        continue
                    v = _safe_float(d.get(k))
                    if v is not None and v != 0.0:
                        return float(v)
                return None

            info: List[Dict[str, Any]] = []
            for c in js.get("data", []):
                if not isinstance(c, dict):
                    continue
                sym = c.get("symbol")
                if not sym:
                    continue
                info.append(
                    {
                        "symbol": normalize_symbol(sym),
                        "volume_24h": _pick_float(c, ["volume", "volume24h", "baseVolume", "vol24h"]),
                        "turnover_24h": _pick_float(c, ["turnover", "turnover24h", "quoteVolume", "usdtVolume", "amount24h"]),
                        "open_interest": _pick_float(c, ["openInterest", "holdingAmount", "holding"]),
                        "minTradeNum": _pick_float(c, ["minTradeNum", "minTradeAmount"]),
                        "contractSize": _pick_float(c, ["size", "contractSize"]),
                    }
                )

            self._contracts_info_cache = info
            self._contracts_info_ts = time.time()
            return info

    # =================================================================
    # CANDLES (v3) — supports pagination via endTime
    # =================================================================

    async def _candles_page(
        self,
        *,
        symbol: str,
        interval: str,
        limit: int,
        end_time: Optional[int] = None,
        candle_type: str = "market",
        try_granularity_fallback: bool = True,
    ) -> Dict[str, Any]:
        """
        One page. Bitget docs have a mismatch in example ("granularity") vs params ("interval").
        We try interval first; if it fails (param error), retry once with granularity.
        """
        sym = normalize_symbol(symbol)

        params: Dict[str, Any] = {
            "category": str(PRODUCT_TYPE or "USDT-FUTURES"),
            "symbol": sym,
            "interval": interval,          # doc param
            "type": candle_type,
            "limit": str(int(limit)),
        }
        if end_time is not None:
            params["endTime"] = str(int(end_time))

        js = await self._request("GET", "/api/v3/market/candles", params=params, auth=False)

        # fallback: replace interval->granularity if API complains
        if try_granularity_fallback and (not js.get("ok")):
            msg = str(js.get("msg") or "")
            code = str(js.get("code") or "")
            # generic parameter error heuristics
            if ("parameter" in msg.lower()) or ("param" in msg.lower()) or code in {"400", "40000", "60006"}:
                params2 = dict(params)
                params2.pop("interval", None)
                params2["granularity"] = interval  # example param
                js2 = await self._request("GET", "/api/v3/market/candles", params=params2, auth=False)
                # if better, use it
                if js2.get("ok") or js2.get("data"):
                    js = js2

        return js

    async def get_klines_df(
        self,
        symbol: str,
        tf: str = "1H",
        limit: int = 200,
    ) -> pd.DataFrame:
        """
        Candles v3:
          - can fetch up to 1000 (per docs; limit behavior can vary)
          - supports endTime for pagination
        """
        interval = _normalize_interval(tf)
        if not interval:
            LOGGER.error("❌ INVALID INTERVAL %s (symbol=%s)", tf, symbol)
            return pd.DataFrame()

        want = max(10, int(limit))
        want = min(want, 1200)  # hard cap to protect memory

        # try single shot first (best for latency)
        # (docs say up to 1000; but some deployments cap lower; we'll paginate if needed)
        per_page = min(want, 1000)

        out: List[List[Any]] = []
        seen_ts = set()

        end_time: Optional[int] = None
        safety_loops = 0

        while len(out) < want and safety_loops < 10:
            safety_loops += 1

            js = await self._candles_page(
                symbol=symbol,
                interval=interval,
                limit=per_page,
                end_time=end_time,
                candle_type="market",
                try_granularity_fallback=True,
            )

            if not isinstance(js, dict):
                LOGGER.error("❌ NON-DICT RESPONSE candles %s(%s) → %s", symbol, interval, js)
                break

            if str(js.get("code")) != "00000" or not js.get("data"):
                # don't spam logs for normal "no data"
                LOGGER.warning("⚠️ KLINES ERROR/EMPTY %s(%s) code=%s msg=%s", symbol, interval, js.get("code"), js.get("msg"))
                break

            data = js.get("data") or []
            if not isinstance(data, list) or not data:
                break

            # collect unique candles
            added = 0
            for row in data:
                if not isinstance(row, list) or len(row) < 6:
                    continue
                try:
                    ts = int(float(row[0]))
                except Exception:
                    continue
                if ts in seen_ts:
                    continue
                seen_ts.add(ts)
                out.append(row)
                added += 1

            if added == 0:
                break

            # paginate older: endTime = earliest_ts - 1
            try:
                earliest = min(int(float(r[0])) for r in data if isinstance(r, list) and r)
                end_time = int(earliest) - 1
            except Exception:
                break

            # if the API gives you less than per_page, likely no more history
            if len(data) < max(10, min(per_page, 100)):
                # soft break: still parse what we have
                break

        # parse to DF
        try:
            df = _parse_candles_rows(out)
            if df.empty:
                return df

            # keep last 'want'
            if len(df) > want:
                df = df.tail(want).reset_index(drop=True)

            return df

        except Exception as exc:
            LOGGER.exception("❌ PARSE ERROR candles %s(%s): %s", symbol, interval, exc)
            return pd.DataFrame()


# ---------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------

_client_instance: Optional[BitgetClient] = None


async def get_client(api_key: str, api_secret: str, passphrase: str) -> BitgetClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = BitgetClient(api_key, api_secret, passphrase)
    return _client_instance
