# =====================================================================
# bitget_client.py — INSTITUTIONAL MARKET CLIENT (2025)
# =====================================================================
# - Contrats : /api/v2/mix/market/contracts  (USDT-FUTURES)
# - Candles  : /api/v3/market/candles        (USDT-FUTURES)
# - Symbol   : BTCUSDT, ETHUSDT, etc. (SANS suffixe)
# - Logs détaillés sur erreurs (429, 4xx, code != 00000)
# =====================================================================

from __future__ import annotations

import aiohttp
import asyncio
import time
import hmac
import base64
import hashlib
import json
import logging
from typing import Any, Dict, Optional, List

import pandas as pd

LOGGER = logging.getLogger(__name__)


# =====================================================================
# RETRY ENGINE
# =====================================================================

async def _async_retry(fn, retries: int = 3, base_delay: float = 0.3):
    """
    Petit retry générique sur exceptions réseau/JSON.
    NE GÈRE PAS les codes 429 dans le JSON, c'est fait dans _request.
    """
    for attempt in range(retries + 1):
        try:
            return await fn()
        except Exception:
            if attempt >= retries:
                raise
            await asyncio.sleep(base_delay * (2 ** attempt))


# =====================================================================
# SYMBOL NORMALISATION
# =====================================================================

def normalize_symbol(sym: str) -> str:
    """
    Standardise :
      BTC-USDT, BTCUSDTM, BTCUSDT → BTCUSDT
      XBTUSDT → BTCUSDT
    """
    if not sym:
        return ""

    s = sym.upper().replace("-", "")
    s = s.replace("USDTM", "USDT").replace("USDTSWAP", "USDT")

    if s.startswith("XBT"):
        s = s.replace("XBT", "BTC")

    return s


# =====================================================================
# BITGET CLIENT
# =====================================================================

class BitgetClient:
    BASE = "https://api.bitget.com"

    def __init__(self, api_key: str, api_secret: str, passphrase: str):
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.api_passphrase = passphrase

        self.session: Optional[aiohttp.ClientSession] = None

        # cache contrats
        self._contracts_cache: Optional[List[str]] = None
        self._contracts_ts: float = 0.0

    async def close(self) -> None:
        if self.session and not self.session.closed:
            await self.session.close()

    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=25)
            self.session = aiohttp.ClientSession(timeout=timeout)

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
    ) -> Dict[str, Any]:
        """
        Wrapper générique Bitget avec logging et retry.
        """
        await self._ensure_session()

        params = params or {}
        data = data or {}

        query = ""
        if params:
            # ordre stable pour éviter des signatures bizarres
            items = sorted(params.items(), key=lambda kv: kv[0])
            query = "?" + "&".join(f"{k}={v}" for k, v in items)

        url = self.BASE + path + query
        body = json.dumps(data, separators=(",", ":")) if data else ""

        async def _do():
            ts = str(int(time.time() * 1000))
            headers: Dict[str, str] = {}

            if auth:
                sig = self._sign(ts, method.upper(), path, query, body)
                headers = {
                    "ACCESS-KEY": self.api_key,
                    "ACCESS-SIGN": sig,
                    "ACCESS-TIMESTAMP": ts,
                    "ACCESS-PASSPHRASE": self.api_passphrase,
                    "Content-Type": "application/json",
                }

            async with self.session.request(
                method.upper(),
                url,
                headers=headers,
                data=body if data else None,
            ) as resp:

                txt = await resp.text()
                status = resp.status

                if status == 429:
                    LOGGER.error(
                        "HTTP 429 %s %s params=%s body=%s raw=%s",
                        method, path, params, body, txt,
                    )
                    raise RuntimeError("HTTP 429 Too Many Requests")

                if status >= 400:
                    LOGGER.error(
                        "HTTP %s %s %s params=%s body=%s raw=%s",
                        status, method, path, params, body, txt,
                    )
                    raise RuntimeError(f"HTTP {status}")

                try:
                    js = json.loads(txt)
                except Exception:
                    LOGGER.error("❌ JSON ERROR %s %s → %s", method, path, txt)
                    raise

                return js

        return await _async_retry(_do)

    # =================================================================
    # CONTRACT LIST (v2)
    # =================================================================

    async def get_contracts_list(self) -> List[str]:
        """
        Renvoie la liste des symboles USDT-FUTURES : BTCUSDT, ETHUSDT, etc.
        """
        now = time.time()
        if self._contracts_cache and now - self._contracts_ts < 300:
            return self._contracts_cache

        params = {"productType": "USDT-FUTURES"}

        js = await self._request(
            "GET",
            "/api/v2/mix/market/contracts",
            params=params,
            auth=False,
        )

        if not isinstance(js, dict) or "data" not in js:
            LOGGER.error("❌ CONTRACT ERROR: %s", js)
            return []

        symbols: List[str] = []
        for c in js["data"]:
            sym = c.get("symbol")
            if not sym:
                continue
            symbols.append(normalize_symbol(sym))

        LOGGER.info("📈 Loaded %d symbols from Bitget Futures", len(symbols))

        self._contracts_cache = symbols
        self._contracts_ts = now
        return symbols

    # =================================================================
    # CANDLES (v3)
    # =================================================================

    async def get_klines_df(
        self,
        symbol: str,
        tf: str = "1H",
        limit: int = 200,
    ) -> pd.DataFrame:
        """
        Candles v3 : max 100 par page
        """
        interval = tf.upper()
        valid_intervals = {
            "1M", "3M", "5M", "15M", "30M",
            "1H", "4H", "6H", "12H", "1D",
        }
        if interval not in valid_intervals:
            LOGGER.error("❌ INVALID INTERVAL %s (symbol=%s)", tf, symbol)
            return pd.DataFrame()

        limit_int = max(10, min(int(limit), 100))

        params = {
            "category": "USDT-FUTURES",
            "symbol": normalize_symbol(symbol),
            "interval": interval,
            "type": "market",
            "limit": str(limit_int),
        }

        try:
            js = await self._request(
                "GET",
                "/api/v3/market/candles",
                params=params,
                auth=False,
            )
        except Exception as exc:
            LOGGER.error(
                "❌ REQUEST ERROR candles %s(%s) params=%s exc=%s",
                symbol, interval, params, exc,
            )
            return pd.DataFrame()

        if not isinstance(js, dict):
            LOGGER.error("❌ NON-DICT RESPONSE candles %s(%s) → %s", symbol, interval, js)
            return pd.DataFrame()

        code = js.get("code")
        data = js.get("data")

        if code != "00000" or not data:
            LOGGER.warning(
                "⚠️ EMPTY/ERROR KLINES for %s (%s) → RAW=%s",
                symbol, interval, js,
            )
            return pd.DataFrame()

        try:
            cols = ["time", "open", "high", "low", "close", "volume", "turnover"]
            df = pd.DataFrame(data, columns=cols[: len(data[0])])

            for c in ["time", "open", "high", "low", "close", "volume"]:
                df[c] = df[c].astype(float)

            df.rename(columns={"time": "ts"}, inplace=True)
            df.sort_values("ts", inplace=True)
            df.rename(columns={"ts": "time"}, inplace=True)

            df = df[["time", "open", "high", "low", "close", "volume"]]
            return df.reset_index(drop=True)

        except Exception as exc:
            LOGGER.exception("❌ PARSE ERROR candles %s(%s): %s", symbol, interval, exc)
            return pd.DataFrame()


# =====================================================================
# SINGLETON
# =====================================================================

_client_instance: Optional[BitgetClient] = None


async def get_client(api_key: str, api_secret: str, passphrase: str) -> BitgetClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = BitgetClient(api_key, api_secret, passphrase)
    return _client_instance
