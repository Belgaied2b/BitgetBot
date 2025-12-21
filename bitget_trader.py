# =====================================================================
# bitget_trader.py — Execution layer (Bitget) — FIX session + marginMode + 40774 fallback
# =====================================================================

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, Optional, Tuple, Union

from bitget_client import BitgetClient, normalize_symbol
from settings import BITGET_BASE_URL

logger = logging.getLogger(__name__)


# =====================================================================
# Small utils (Decimal-safe)
# =====================================================================

def _d(x: Union[str, float, int, Decimal]) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(str(x))


def _quantize_down(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    n = (value / step).to_integral_value(rounding=ROUND_DOWN)
    return n * step


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _is_ok(resp: Any) -> bool:
    return isinstance(resp, dict) and (resp.get("ok") is True or str(resp.get("code", "")) == "00000")


# =====================================================================
# Meta cache (tick sizes / size steps)
# =====================================================================

@dataclass
class SymbolMeta:
    symbol: str
    price_tick: Decimal
    size_step: Decimal
    min_size: Decimal

    def fmt_price(self, px: Decimal) -> str:
        # format with tick precision (no scientific)
        q = _quantize_down(px, self.price_tick)
        s = format(q, "f")
        return s

    def fmt_size(self, sz: Decimal) -> str:
        q = _quantize_down(sz, self.size_step)
        s = format(q, "f")
        return s


class SymbolMetaCache:
    """
    Pull meta from /api/v2/mix/market/contracts (via BitgetClient.get_contracts_list(raw=True))
    Tolerant parsing, because Bitget fields differ depending on product type / version.
    """

    def __init__(self, client: BitgetClient, product_type: str = "USDT-FUTURES", margin_coin: str = "USDT"):
        self.client = client
        self.product_type = product_type
        self.margin_coin = margin_coin
        self._cache: Dict[str, SymbolMeta] = {}
        self._last_refresh = 0.0

    @staticmethod
    def _parse_tick(contract: Dict[str, Any]) -> Decimal:
        # best effort
        if contract.get("priceTick") is not None:
            t = _safe_float(contract.get("priceTick"), 0.0)
            return _d(t) if t > 0 else _d("0.00000001")

        price_place = contract.get("pricePlace")
        price_end_step = contract.get("priceEndStep")
        if price_place is not None and price_end_step is not None:
            try:
                pp = int(price_place)
                pes = int(price_end_step)
                tick = Decimal(pes) * (Decimal(10) ** Decimal(-pp))
                return tick if tick > 0 else _d("0.00000001")
            except Exception:
                pass

        return _d("0.00000001")

    @staticmethod
    def _parse_size_step(contract: Dict[str, Any]) -> Decimal:
        if contract.get("sizeMultiplier") is not None:
            s = _safe_float(contract.get("sizeMultiplier"), 0.0)
            return _d(s) if s > 0 else _d("0.000001")

        volume_place = contract.get("volumePlace")
        if volume_place is not None:
            try:
                vp = int(volume_place)
                step = Decimal(10) ** Decimal(-vp)
                return step if step > 0 else _d("0.000001")
            except Exception:
                pass

        return _d("0.000001")

    @staticmethod
    def _parse_min_size(contract: Dict[str, Any]) -> Decimal:
        for k in ("minTradeNum", "minTradeAmount", "minOrderQty", "minSize"):
            if contract.get(k) is not None:
                v = _safe_float(contract.get(k), 0.0)
                if v > 0:
                    return _d(v)
        return _d("0.0")

    async def refresh(self, ttl_s: int = 900) -> None:
        now = time.time()
        if self._cache and (now - self._last_refresh) < ttl_s:
            return

        raw = await self.client.get_contracts_list(
            product_type=self.product_type,
            margin_coin=self.margin_coin,
            raw=True,
        )
        if not isinstance(raw, list):
            return

        new_cache: Dict[str, SymbolMeta] = {}
        for c in raw:
            if not isinstance(c, dict):
                continue
            sym_raw = c.get("symbol", "")
            sym = normalize_symbol(sym_raw)
            if not sym:
                continue

            price_tick = self._parse_tick(c)
            size_step = self._parse_size_step(c)
            min_size = self._parse_min_size(c)

            new_cache[sym] = SymbolMeta(
                symbol=sym,
                price_tick=price_tick,
                size_step=size_step,
                min_size=min_size,
            )

        if new_cache:
            self._cache = new_cache
            self._last_refresh = now
            logger.info("[bitget_trader] refreshed meta cache (%d symbols)", len(self._cache))

    async def get(self, symbol: str) -> Optional[SymbolMeta]:
        await self.refresh()
        return self._cache.get(symbol)


# =====================================================================
# Trader
# =====================================================================

class BitgetTrader:
    """
    Compatible avec tes deux styles :
      - BitgetTrader(API_KEY, API_SECRET, API_PASSPHRASE)
      - BitgetTrader(client, margin_mode="isolated", ...)
    """

    def __init__(
        self,
        client_or_key: Union[BitgetClient, str],
        api_secret: Optional[str] = None,
        api_passphrase: Optional[str] = None,
        *,
        product_type: str = "USDT-FUTURES",
        margin_coin: str = "USDT",
        margin_mode: str = "isolated",
    ):
        if isinstance(client_or_key, BitgetClient):
            self.client = client_or_key
        else:
            api_key = str(client_or_key or "")
            self.client = BitgetClient(
                api_key=api_key,
                api_secret=str(api_secret or ""),
                api_passphrase=str(api_passphrase or ""),
                base_url=BITGET_BASE_URL,
            )

        self.product_type = product_type
        self.margin_coin = margin_coin
        self.margin_mode = margin_mode  # "isolated" / "crossed"

        self._meta = SymbolMetaCache(self.client, product_type=self.product_type, margin_coin=self.margin_coin)

    # -----------------------------------------------------------------
    # Low-level request that DOES NOT raise on HTTP != 2xx
    # -----------------------------------------------------------------
    async def _request_any_status(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        auth: bool = True,
    ) -> Dict[str, Any]:
        await self.client._ensure_session()

        session = getattr(self.client, "session", None)
        if session is None:
            return {"ok": False, "code": "NO_SESSION", "msg": "aiohttp session is not initialized"}

        params = params or {}
        body = body or {}
        query = ""
        if params:
            from urllib.parse import urlencode
            query = "?" + urlencode({k: v for k, v in params.items() if v is not None})

        body_str = json.dumps(body, separators=(",", ":"))
        ts = str(int(time.time() * 1000))

        headers = {
            "Content-Type": "application/json",
            "locale": "en-US",
        }

        if auth:
            sign = self.client._sign(ts, method.upper(), path, query, body_str)
            headers.update(
                {
                    "ACCESS-KEY": self.client.api_key,
                    "ACCESS-PASSPHRASE": self.client.api_passphrase,
                    "ACCESS-TIMESTAMP": ts,
                    "ACCESS-SIGN": sign,
                }
            )

        url = f"{self.client.BASE}{path}{query}"

        try:
            async with session.request(method.upper(), url, data=body_str, headers=headers) as resp:
                text = await resp.text()
                try:
                    data = json.loads(text) if text else {}
                except Exception:
                    data = {"raw": text}

                # normalize response
                if isinstance(data, dict):
                    data.setdefault("http_status", resp.status)
                    data["ok"] = (resp.status < 400 and str(data.get("code", "")) == "00000")
                    return data

                return {"ok": False, "http_status": resp.status, "raw": text}

        except Exception as e:
            return {"ok": False, "code": "HTTP_ERROR", "msg": str(e)}

    # -----------------------------------------------------------------
    # Symbol conversion for API v1 (symbolId requires *_UMCBL in docs)
    # -----------------------------------------------------------------
    @staticmethod
    def _to_symbol_id_v1(symbol: str) -> str:
        # Bitget docs / FAQ: REST expects BTCUSDT_UMCBL (symbolId). :contentReference[oaicite:3]{index=3}
        if "_" in symbol:
            return symbol
        return f"{symbol}_UMCBL"

    # -----------------------------------------------------------------
    # ENTRY (LIMIT)
    # -----------------------------------------------------------------
    async def place_limit(self, symbol: str, side: str, price: float, qty: float, client_oid: Optional[str] = None) -> Dict[str, Any]:
        """
        Place une entrée LIMIT.
        - Essaie d'abord v2: /api/v2/mix/order/place-order (avec marginMode + tradeSide)
        - Si Bitget renvoie 40774 => fallback v1 avec side buy_single/sell_single (single_hold) :contentReference[oaicite:4]{index=4}
        """
        meta = await self._meta.get(symbol)

        px = _d(price)
        sz = _d(qty)

        if px <= 0 or sz <= 0:
            return {"ok": False, "code": "BAD_INPUT", "msg": f"price/qty invalid (price={price}, qty={qty})"}

        if meta:
            px_q = _quantize_down(px, meta.price_tick)
            sz_q = _quantize_down(sz, meta.size_step)
            if meta.min_size > 0 and sz_q < meta.min_size:
                sz_q = meta.min_size
            price_str = meta.fmt_price(px_q)
            size_str = meta.fmt_size(sz_q)
        else:
            # fallback, still avoid scientific
            price_str = format(px, "f")
            size_str = format(sz, "f")

        side_l = str(side).lower()
        if side_l in ("long", "buy"):
            side_l = "buy"
        elif side_l in ("short", "sell"):
            side_l = "sell"

        if side_l not in ("buy", "sell"):
            return {"ok": False, "code": "BAD_SIDE", "msg": f"side invalid: {side}"}

        if client_oid is None:
            client_oid = f"entry-{symbol}-{int(time.time() * 1000)}"

        # ---- V2 attempt (recommended)
        body_v2 = {
            "symbol": symbol,
            "productType": self.product_type,
            "marginCoin": self.margin_coin,
            "marginMode": self.margin_mode,
            "size": size_str,
            "price": price_str,
            "side": side_l,
            "tradeSide": "open",  # in hedge-mode required, in one-way ignored per doc snippet
            "orderType": "limit",
            "timeInForceValue": "normal",
            "clientOid": str(client_oid),
            "reduceOnly": "NO",
        }

        logger.info("[EXEC] place LIMIT %s %s price=%s size=%s", symbol, side_l, price_str, size_str)
        resp = await self._request_any_status("POST", "/api/v2/mix/order/place-order", body=body_v2, auth=True)
        if _is_ok(resp):
            return resp

        code = str(resp.get("code", ""))
        msg = str(resp.get("msg", ""))

        # ---- Fallback on unilateral/holdMode mismatch errors
        if code in ("40774", "45021") or "unilateral" in msg.lower():
            logger.warning("[EXEC] v2 rejected (%s: %s) -> fallback v1 single_hold side", code, msg)

            symbol_id = self._to_symbol_id_v1(symbol)
            side_v1 = "buy_single" if side_l == "buy" else "sell_single"  # single_hold doc :contentReference[oaicite:5]{index=5}

            body_v1 = {
                "symbol": symbol_id,
                "marginCoin": self.margin_coin,
                "size": size_str,
                "price": price_str,
                "side": side_v1,
                "orderType": "limit",
                "timeInForceValue": "normal",
                "clientOid": str(client_oid),
            }

            resp1 = await self._request_any_status("POST", "/api/mix/v1/order/placeOrder", body=body_v1, auth=True)
            if _is_ok(resp1):
                return resp1

            # second fallback: hedge-style open_long/open_short
            logger.warning("[EXEC] v1 single_hold rejected -> fallback v1 hedge side")
            side_v1b = "open_long" if side_l == "buy" else "open_short"

            body_v1b = dict(body_v1)
            body_v1b["side"] = side_v1b

            resp2 = await self._request_any_status("POST", "/api/mix/v1/order/placeOrder", body=body_v1b, auth=True)
            return resp2

        return resp

    # -----------------------------------------------------------------
    # STOP LOSS (plan order v2)
    # -----------------------------------------------------------------
    async def place_stop_loss(self, symbol: str, open_side: str, sl_price: float, qty: float, client_oid: Optional[str] = None) -> Dict[str, Any]:
        """
        Place un STOP LOSS via /api/v2/mix/order/place-plan-order.
        Structure alignée sur les exemples v2 (planType, marginMode, tradeSide...). :contentReference[oaicite:6]{index=6}
        """
        meta = await self._meta.get(symbol)

        px = _d(sl_price)
        sz = _d(qty)
        if px <= 0 or sz <= 0:
            return {"ok": False, "code": "BAD_INPUT", "msg": f"sl/qty invalid (sl={sl_price}, qty={qty})"}

        if meta:
            px_q = _quantize_down(px, meta.price_tick)
            sz_q = _quantize_down(sz, meta.size_step)
            if meta.min_size > 0 and sz_q < meta.min_size:
                sz_q = meta.min_size
            price_str = meta.fmt_price(px_q)
            size_str = meta.fmt_size(sz_q)
        else:
            price_str = format(px, "f")
            size_str = format(sz, "f")

        open_side_u = str(open_side).upper()
        # close side opposite of open
        close_side = "sell" if open_side_u in ("BUY", "LONG") else "buy"

        if client_oid is None:
            client_oid = f"sl-{symbol}-{int(time.time() * 1000)}"

        body = {
            "planType": "normal_plan",
            "symbol": symbol,
            "productType": self.product_type,
            "marginMode": self.margin_mode,
            "marginCoin": self.margin_coin,
            "size": size_str,
            "price": price_str,               # execute price for limit-plan
            "triggerPrice": price_str,
            "triggerType": "mark_price",
            "side": close_side,
            "tradeSide": "close",
            "orderType": "limit",
            "timeInForceValue": "normal",
            "clientOid": str(client_oid),
            "reduceOnly": "YES",
        }

        logger.info("[EXEC] place SL %s close_side=%s trigger=%s size=%s", symbol, close_side, price_str, size_str)
        return await self._request_any_status("POST", "/api/v2/mix/order/place-plan-order", body=body, auth=True)

    # -----------------------------------------------------------------
    # TAKE PROFIT (plan order v2)
    # -----------------------------------------------------------------
    async def place_take_profit(self, symbol: str, open_side: str, tp_price: float, qty: float, client_oid: Optional[str] = None) -> Dict[str, Any]:
        """
        Place un TAKE PROFIT via /api/v2/mix/order/place-plan-order.
        """
        meta = await self._meta.get(symbol)

        px = _d(tp_price)
        sz = _d(qty)
        if px <= 0 or sz <= 0:
            return {"ok": False, "code": "BAD_INPUT", "msg": f"tp/qty invalid (tp={tp_price}, qty={qty})"}

        if meta:
            px_q = _quantize_down(px, meta.price_tick)
            sz_q = _quantize_down(sz, meta.size_step)
            if meta.min_size > 0 and sz_q < meta.min_size:
                sz_q = meta.min_size
            price_str = meta.fmt_price(px_q)
            size_str = meta.fmt_size(sz_q)
        else:
            price_str = format(px, "f")
            size_str = format(sz, "f")

        open_side_u = str(open_side).upper()
        close_side = "sell" if open_side_u in ("BUY", "LONG") else "buy"

        if client_oid is None:
            client_oid = f"tp-{symbol}-{int(time.time() * 1000)}"

        body = {
            "planType": "normal_plan",
            "symbol": symbol,
            "productType": self.product_type,
            "marginMode": self.margin_mode,
            "marginCoin": self.margin_coin,
            "size": size_str,
            "price": price_str,
            "triggerPrice": price_str,
            "triggerType": "mark_price",
            "side": close_side,
            "tradeSide": "close",
            "orderType": "limit",
            "timeInForceValue": "normal",
            "clientOid": str(client_oid),
            "reduceOnly": "YES",
        }

        logger.info("[EXEC] place TP %s close_side=%s trigger=%s size=%s", symbol, close_side, price_str, size_str)
        return await self._request_any_status("POST", "/api/v2/mix/order/place-plan-order", body=body, auth=True)

    async def close(self) -> None:
        await self.client.close()
