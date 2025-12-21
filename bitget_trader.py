# =====================================================================
# bitget_trader.py — Trader Bitget FUTURES (compatible bitget_client.py)
# =====================================================================

from __future__ import annotations

import json
import logging
import math
import time
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, Optional

from bitget_client import BitgetClient, normalize_symbol

logger = logging.getLogger(__name__)


def _d(x: Any) -> Decimal:
    try:
        return Decimal(str(x))
    except Exception:
        return Decimal("0")


def _floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    n = (value / step).to_integral_value(rounding=ROUND_DOWN)
    return n * step


def _fmt_decimal(x: Decimal) -> str:
    # string sans scientific notation
    s = format(x, "f")
    # trim léger (optionnel)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def _is_ok(resp: Any) -> bool:
    return isinstance(resp, dict) and (resp.get("ok") is True or str(resp.get("code", "")) == "00000")


class ContractMetaCache:
    """
    Cache des metas depuis:
      GET /api/v2/mix/market/contracts?productType=USDT-FUTURES

    On parse:
      - pricePlace + priceEndStep -> tick
      - sizeMultiplier -> size_step
      - minTradeNum -> min_size
    """

    def __init__(self, client: BitgetClient, product_type: str = "USDT-FUTURES", ttl_s: int = 600):
        self.client = client
        self.product_type = product_type
        self.ttl_s = int(ttl_s)
        self._ts = 0.0
        self._meta: Dict[str, Dict[str, Any]] = {}

    async def refresh(self) -> None:
        now = time.time()
        if self._meta and (now - self._ts) < self.ttl_s:
            return

        # IMPORTANT: on n'appelle PAS client.get_contracts_list(product_type=...) (ça n'existe pas)
        await self.client._ensure_session()
        js = await self.client._request(
            "GET",
            "/api/v2/mix/market/contracts",
            params={"productType": self.product_type},
            auth=False,
        )

        data = js.get("data") if isinstance(js, dict) else None
        if not isinstance(data, list):
            logger.error("[META] unexpected contracts response: %s", js)
            return

        meta: Dict[str, Dict[str, Any]] = {}
        for c in data:
            try:
                sym = normalize_symbol(c.get("symbol", ""))
                if not sym:
                    continue

                price_place = int(c.get("pricePlace", 0) or 0)
                price_end_step = Decimal(str(c.get("priceEndStep", 1) or 1))
                size_multiplier = Decimal(str(c.get("sizeMultiplier", 1) or 1))
                min_trade_num = Decimal(str(c.get("minTradeNum", 0) or 0))

                tick = price_end_step * (Decimal(10) ** Decimal(-price_place))
                if tick <= 0:
                    tick = Decimal("0.00000001")
                if size_multiplier <= 0:
                    size_multiplier = Decimal("0.000001")

                meta[sym] = {
                    "tick": tick,
                    "size_step": size_multiplier,
                    "min_size": min_trade_num if min_trade_num > 0 else Decimal("0"),
                }
            except Exception:
                continue

        if meta:
            self._meta = meta
            self._ts = now
            logger.info("[META] refreshed contracts cache (%d symbols)", len(meta))

    async def get(self, symbol: str) -> Optional[Dict[str, Any]]:
        await self.refresh()
        return self._meta.get(normalize_symbol(symbol))


class BitgetTrader:
    """
    Trader qui accepte:
      BitgetTrader(client, margin_usdt=..., leverage=..., margin_mode="isolated")
    et aussi:
      BitgetTrader(api_key, api_secret, api_passphrase, ...)
    """

    def __init__(
        self,
        client_or_key,
        api_secret: Optional[str] = None,
        api_passphrase: Optional[str] = None,
        *,
        margin_usdt: float = 20.0,
        leverage: float = 10.0,
        margin_mode: str = "isolated",
        product_type: str = "USDT-FUTURES",
        margin_coin: str = "USDT",
        # compat alias
        target_margin_usdt: Optional[float] = None,
        **_ignored,
    ):
        if isinstance(client_or_key, BitgetClient):
            self.client = client_or_key
        else:
            self.client = BitgetClient(
                api_key=str(client_or_key or ""),
                api_secret=str(api_secret or ""),
                passphrase=str(api_passphrase or ""),
            )

        if target_margin_usdt is not None:
            margin_usdt = float(target_margin_usdt)

        self.margin_usdt = float(margin_usdt)
        self.leverage = float(leverage)

        mm = (margin_mode or "isolated").lower().strip()
        if mm not in ("isolated", "crossed"):
            mm = "isolated"
        self.margin_mode = mm

        self.product_type = product_type
        self.margin_coin = margin_coin

        self._meta = ContractMetaCache(self.client, product_type=self.product_type, ttl_s=600)

    async def _request_any_status(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        auth: bool = True,
    ) -> Dict[str, Any]:
        """
        Même signature que client._request, mais:
        - ne raise pas sur HTTP 4xx
        - retourne le JSON Bitget (code/msg/data)
        """
        await self.client._ensure_session()
        session = self.client.session
        if session is None:
            return {"ok": False, "code": "NO_SESSION", "msg": "session not initialized"}

        params = params or {}
        data = data or {}

        query = ""
        if params:
            query = "?" + "&".join(f"{k}={v}" for k, v in params.items())

        url = self.client.BASE + path + query
        body = json.dumps(data, separators=(",", ":")) if data else ""

        ts = str(int(time.time() * 1000))
        headers: Dict[str, str] = {"Content-Type": "application/json"}

        if auth:
            sig = self.client._sign(ts, method.upper(), path, query, body)
            headers.update(
                {
                    "ACCESS-KEY": self.client.api_key,
                    "ACCESS-SIGN": sig,
                    "ACCESS-TIMESTAMP": ts,
                    "ACCESS-PASSPHRASE": self.client.api_passphrase,
                }
            )

        try:
            async with session.request(method.upper(), url, headers=headers, data=body if data else None) as resp:
                txt = await resp.text()
                try:
                    js = json.loads(txt) if txt else {}
                except Exception:
                    js = {"raw": txt}

                if isinstance(js, dict):
                    js["http_status"] = resp.status
                    js["ok"] = (resp.status < 400 and str(js.get("code", "")) == "00000")
                return js if isinstance(js, dict) else {"ok": False, "raw": txt, "http_status": resp.status}
        except Exception as e:
            return {"ok": False, "code": "HTTP_ERROR", "msg": str(e)}

    @staticmethod
    def _to_side(side: str) -> str:
        s = (side or "").upper()
        return "buy" if s == "BUY" else "sell"

    @staticmethod
    def _symbol_id_v1(symbol: str) -> str:
        sym = normalize_symbol(symbol)
        return f"{sym}_UMCBL"

    async def place_limit(
        self,
        *,
        symbol: str,
        side: str,              # "BUY"/"SELL"
        price: float,
        size: Optional[float] = None,        # si None => calc auto via margin_usdt*leverage
        qty: Optional[float] = None,         # compat
        client_oid: Optional[str] = None,
        preset_sl: Optional[float] = None,   # acceptés mais non utilisés ici (stabilité)
        preset_tp: Optional[float] = None,   # acceptés mais non utilisés ici (stabilité)
        **_ignored,
    ) -> Dict[str, Any]:
        sym = normalize_symbol(symbol)
        sd = self._to_side(side)

        px = _d(price)
        if px <= 0:
            return {"ok": False, "code": "BAD_PRICE", "msg": f"price<=0 ({price})"}

        # qty/size resolution
        if qty is not None and size is None:
            size = qty

        if size is None:
            notional = Decimal(str(max(0.0, self.margin_usdt))) * Decimal(str(max(0.0, self.leverage)))
            raw_size = notional / px
        else:
            raw_size = _d(size)

        if raw_size <= 0:
            return {"ok": False, "code": "BAD_SIZE", "msg": f"size<=0 ({size})"}

        meta = await self._meta.get(sym)
        tick = meta["tick"] if meta else Decimal("0.00000001")
        step = meta["size_step"] if meta else Decimal("0.000001")
        min_size = meta["min_size"] if meta else Decimal("0")

        px_q = _floor_to_step(px, tick)
        sz_q = _floor_to_step(raw_size, step)

        if min_size > 0 and sz_q < min_size:
            sz_q = min_size

        price_s = _fmt_decimal(px_q)
        size_s = _fmt_decimal(sz_q)

        if client_oid is None:
            client_oid = f"entry-{sym}-{int(time.time() * 1000)}"

        logger.info(
            "[EXEC] place LIMIT %s %s price=%s size=%s notional≈%.2f (marginMode=%s lev=%.1fx)",
            sym, sd, price_s, size_s, float(px_q * sz_q), self.margin_mode, self.leverage
        )

        # ---- v2 attempt 1 (tradeSide open)
        payload_v2 = {
            "symbol": sym,
            "productType": self.product_type,
            "marginCoin": self.margin_coin,
            "marginMode": self.margin_mode,
            "size": size_s,
            "price": price_s,
            "side": sd,
            "tradeSide": "open",
            "orderType": "limit",
            "timeInForceValue": "normal",
            "clientOid": str(client_oid),
        }

        resp = await self._request_any_status("POST", "/api/v2/mix/order/place-order", data=payload_v2, auth=True)
        if _is_ok(resp):
            return resp

        code = str(resp.get("code", ""))
        msg = str(resp.get("msg", ""))

        # ---- v2 attempt 2 (no tradeSide) si 40774
        if code == "40774":
            payload_v2b = dict(payload_v2)
            payload_v2b.pop("tradeSide", None)
            resp2 = await self._request_any_status("POST", "/api/v2/mix/order/place-order", data=payload_v2b, auth=True)
            if _is_ok(resp2):
                return resp2
            code = str(resp2.get("code", ""))
            msg = str(resp2.get("msg", ""))

        # ---- fallback v1 si toujours "unilateral" / 40774
        if code == "40774" or "unilateral" in msg.lower():
            symbol_id = self._symbol_id_v1(sym)

            # single-hold
            side_v1 = "buy_single" if sd == "buy" else "sell_single"
            payload_v1 = {
                "symbol": symbol_id,
                "marginCoin": self.margin_coin,
                "size": size_s,
                "price": price_s,
                "side": side_v1,
                "orderType": "limit",
                "timeInForceValue": "normal",
                "clientOid": str(client_oid),
            }
            resp3 = await self._request_any_status("POST", "/api/mix/v1/order/placeOrder", data=payload_v1, auth=True)
            if _is_ok(resp3):
                return resp3

            # hedge fallback
            side_v1b = "open_long" if sd == "buy" else "open_short"
            payload_v1b = dict(payload_v1)
            payload_v1b["side"] = side_v1b
            resp4 = await self._request_any_status("POST", "/api/mix/v1/order/placeOrder", data=payload_v1b, auth=True)
            return resp4

        # presets (acceptés mais ignorés ici)
        _ = preset_sl, preset_tp
        return resp
