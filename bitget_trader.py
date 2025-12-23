# =====================================================================
# bitget_trader.py — Bitget Execution (Entry + TP1/TP2 + SL->BE)
# FIXES:
# - strict tick quantization using Decimal (prevents 40020 price errors)
# - direction-aware rounding
# - quantize triggerPrice for plan SL
# =====================================================================

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING, getcontext
from typing import Any, Dict, Optional, Tuple

from settings import PRODUCT_TYPE, MARGIN_COIN

logger = logging.getLogger(__name__)
getcontext().prec = 28


# =====================================================================
# Helpers
# =====================================================================

def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _is_ok(resp: Any) -> bool:
    if not isinstance(resp, dict):
        return False
    if resp.get("ok") is True:
        return True
    return str(resp.get("code", "")) == "00000"


def _d(x: Any) -> Decimal:
    return Decimal(str(x))


def _fmt_decimal(val: Decimal, places: int) -> str:
    if places < 0:
        return format(val, "f")
    q = Decimal(1).scaleb(-places)  # 10^-places
    return format(val.quantize(q), "f")


def _quant_price(price: float, tick: float, *, mode: str) -> Decimal:
    """
    mode: 'floor' or 'ceil'
    """
    if tick <= 0:
        return _d(price)

    p = _d(price)
    t = _d(tick)
    steps = (p / t)

    if mode == "ceil":
        k = steps.to_integral_value(rounding=ROUND_CEILING)
    else:
        k = steps.to_integral_value(rounding=ROUND_FLOOR)

    return k * t


def _quant_qty(qty: float, step: float) -> Decimal:
    if step <= 0:
        return _d(qty)
    q = _d(qty)
    s = _d(step)
    k = (q / s).to_integral_value(rounding=ROUND_FLOOR)
    return k * s


# =====================================================================
# Contract meta cache (tick/lot)
# =====================================================================

@dataclass
class ContractMeta:
    symbol: str
    price_place: int
    price_tick: float
    qty_place: int
    qty_step: float
    min_qty: float


class ContractMetaCache:
    def __init__(self, client, ttl_s: int = 600):
        self.client = client
        self.ttl_s = ttl_s
        self._ts = 0.0
        self._by_symbol: Dict[str, ContractMeta] = {}

    async def refresh(self) -> None:
        now = time.time()
        if self._by_symbol and (now - self._ts) < self.ttl_s:
            return

        js = await self.client._request(
            "GET",
            "/api/v2/mix/market/contracts",
            params={"productType": PRODUCT_TYPE},
            auth=False,
        )

        if not isinstance(js, dict) or "data" not in js:
            logger.error("[META] bad contracts response: %s", js)
            return

        by_symbol: Dict[str, ContractMeta] = {}

        for c in js.get("data", []):
            sym = (c.get("symbol") or "").upper()
            if not sym:
                continue

            price_place = int(_safe_float(c.get("pricePlace"), 6))
            qty_place = int(_safe_float(c.get("volumePlace"), 4))

            price_tick = _safe_float(c.get("priceEndStep"), 0.0) or (10 ** (-price_place))
            qty_step = (
                _safe_float(c.get("sizeMultiplier"), 0.0)
                or _safe_float(c.get("minTradeNum"), 0.0)
                or (10 ** (-qty_place))
            )

            min_qty = _safe_float(c.get("minTradeNum"), 0.0) or qty_step

            by_symbol[sym] = ContractMeta(
                symbol=sym,
                price_place=price_place,
                price_tick=float(price_tick),
                qty_place=qty_place,
                qty_step=float(qty_step),
                min_qty=float(min_qty),
            )

        self._by_symbol = by_symbol
        self._ts = now
        logger.info("[META] refreshed contracts cache (%d symbols)", len(by_symbol))

    async def get(self, symbol: str) -> Optional[ContractMeta]:
        sym = (symbol or "").upper()
        await self.refresh()
        return self._by_symbol.get(sym)


# =====================================================================
# Trader
# =====================================================================

class BitgetTrader:
    BASE = "https://api.bitget.com"

    def __init__(
        self,
        client,
        margin_usdt: float = 20.0,
        leverage: float = 10.0,
        margin_mode: str = "isolated",
        product_type: str = PRODUCT_TYPE,
        margin_coin: str = MARGIN_COIN,
        target_margin_usdt: Optional[float] = None,
    ):
        self.client = client
        self.margin_usdt = float(target_margin_usdt if target_margin_usdt is not None else margin_usdt)
        self.leverage = float(leverage)
        self.margin_mode = (margin_mode or "isolated").lower()
        self.product_type = (product_type or PRODUCT_TYPE)
        self.margin_coin = (margin_coin or MARGIN_COIN)

        self._meta = ContractMetaCache(client)

    async def _request_any_status(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        timeout: int = 12,
        auth: bool = True,
    ) -> Dict[str, Any]:
        await self.client._ensure_session()

        params = params or {}
        data = data or {}

        query = ""
        if params:
            query = "?" + "&".join(f"{k}={v}" for k, v in params.items())

        url = self.BASE + path + query
        body = json.dumps(data, separators=(",", ":")) if data else ""

        ts = str(_now_ms())
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
            async with self.client.session.request(
                method.upper(), url, headers=headers, data=body if data else None, timeout=timeout
            ) as resp:
                txt = await resp.text()
                status = resp.status
                try:
                    js = json.loads(txt) if txt else {}
                except Exception:
                    return {"ok": False, "status": status, "raw": txt}

                code = str(js.get("code", ""))
                ok = (status < 400) and (code == "00000")
                js["ok"] = ok
                js["_http_status"] = status
                return js

        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def _get_meta(self, symbol: str) -> Optional[ContractMeta]:
        return await self._meta.get(symbol)

    async def get_tick(self, symbol: str) -> float:
        meta = await self._get_meta(symbol)
        if not meta:
            return 1e-6
        return float(meta.price_tick or (10 ** (-meta.price_place)))

    async def _quantize_for_limit(
        self,
        symbol: str,
        *,
        price: float,
        qty: float,
        side: str,
        trade_side: str,
    ) -> Tuple[str, str, float, float]:
        meta = await self._get_meta(symbol)
        if not meta:
            return str(price), str(qty), float(price), float(qty)

        tick = float(meta.price_tick or (10 ** (-meta.price_place)))
        step = float(meta.qty_step or (10 ** (-meta.qty_place)))

        s = (side or "").lower()
        ts = (trade_side or "open").lower()

        # open: BUY floor, SELL ceil
        # close: SELL floor, BUY ceil
        if ts == "open":
            mode = "floor" if s == "buy" else "ceil"
        else:
            mode = "floor" if s == "sell" else "ceil"

        qp = _quant_price(price, tick, mode=mode)
        qq = _quant_qty(qty, step)

        if float(qq) < float(meta.min_qty):
            return "0", "0", 0.0, 0.0

        p_str = _fmt_decimal(qp, meta.price_place)
        q_str = _fmt_decimal(qq, meta.qty_place)
        return p_str, q_str, float(qp), float(qq)

    async def _quantize_for_plan_sl(
        self,
        symbol: str,
        *,
        trigger_price: float,
        qty: float,
        close_side: str,
    ) -> Tuple[str, str, float, float]:
        meta = await self._get_meta(symbol)
        if not meta:
            return str(trigger_price), str(qty), float(trigger_price), float(qty)

        tick = float(meta.price_tick or (10 ** (-meta.price_place)))
        step = float(meta.qty_step or (10 ** (-meta.qty_place)))

        cs = (close_side or "").lower()
        # close long => SELL stop => FLOOR trigger
        # close short => BUY stop => CEIL trigger
        mode = "floor" if cs == "sell" else "ceil"

        qp = _quant_price(trigger_price, tick, mode=mode)
        qq = _quant_qty(qty, step)

        if float(qq) < float(meta.min_qty):
            return "0", "0", 0.0, 0.0

        p_str = _fmt_decimal(qp, meta.price_place)
        q_str = _fmt_decimal(qq, meta.qty_place)
        return p_str, q_str, float(qp), float(qq)

    # ---------------------------------------------------------------
    # Orders
    # ---------------------------------------------------------------

    async def place_limit(
        self,
        symbol: str,
        side: str,
        price: float,
        size: Optional[float] = None,
        client_oid: Optional[str] = None,
        trade_side: str = "open",
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        sym = (symbol or "").upper()
        s = (side or "").lower()  # buy/sell

        if size is None:
            notional = self.margin_usdt * self.leverage
            raw_qty = notional / max(1e-12, float(price))
        else:
            raw_qty = float(size)

        p_str, q_str, p_f, q_f = await self._quantize_for_limit(
            sym, price=float(price), qty=float(raw_qty), side=s, trade_side=trade_side
        )
        if q_f <= 0:
            return {"ok": False, "code": "QTY0", "msg": "quantized qty is 0"}

        payload: Dict[str, Any] = {
            "symbol": sym,
            "productType": self.product_type,
            "marginCoin": self.margin_coin,
            "marginMode": self.margin_mode,
            "size": q_str,
            "price": p_str,
            "side": s,
            "orderType": "limit",
            "timeInForceValue": "normal",
            "clientOid": client_oid or f"oid-{sym}-{_now_ms()}",
            "tradeSide": (trade_side or "open").lower(),
        }

        if reduce_only:
            payload["reduceOnly"] = "YES"

        resp = await self._request_any_status("POST", "/api/v2/mix/order/place-order", data=payload, auth=True)
        if resp.get("ok") is True:
            resp["qty"] = q_f
            resp["price"] = p_f
        return resp

    async def place_reduce_limit_tp(
        self,
        symbol: str,
        close_side: str,
        price: float,
        qty: float,
        client_oid: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self.place_limit(
            symbol=symbol,
            side=close_side,
            price=price,
            size=qty,
            client_oid=client_oid,
            trade_side="close",
            reduce_only=True,
        )

    async def place_stop_market_sl(
        self,
        symbol: str,
        close_side: str,
        trigger_price: float,
        qty: float,
        *,
        client_oid: Optional[str] = None,
        trigger_type: str = "mark_price",
    ) -> Dict[str, Any]:
        sym = (symbol or "").upper()
        close_s = (close_side or "").lower()

        tp_str, q_str, tp_f, q_f = await self._quantize_for_plan_sl(
            sym, trigger_price=float(trigger_price), qty=float(qty), close_side=close_s
        )
        if q_f <= 0:
            return {"ok": False, "code": "QTY0", "msg": "quantized qty is 0"}

        payload = {
            "symbol": sym,
            "productType": self.product_type,
            "marginCoin": self.margin_coin,
            "marginMode": self.margin_mode,
            "size": q_str,
            "side": close_s,
            "orderType": "market",
            "triggerPrice": tp_str,
            "triggerType": trigger_type,
            "planType": "normal_plan",
            "tradeSide": "close",
            "reduceOnly": "YES",
            "clientOid": client_oid or f"sl-{sym}-{_now_ms()}",
            "executePrice": "0",
        }

        resp = await self._request_any_status("POST", "/api/v2/mix/order/place-plan-order", data=payload, auth=True)
        if resp.get("ok") is True:
            resp["qty"] = q_f
            resp["triggerPrice"] = tp_f
        return resp

    async def cancel_plan_orders(self, symbol: str, order_ids: list[str]) -> Dict[str, Any]:
        sym = (symbol or "").upper()
        payload = {
            "symbol": sym,
            "productType": self.product_type,
            "planType": "normal_plan",
            "orderIdList": order_ids,
        }
        return await self._request_any_status("POST", "/api/v2/mix/order/cancel-plan-order", data=payload, auth=True)

    async def get_order_detail(
        self,
        symbol: str,
        *,
        order_id: Optional[str] = None,
        client_oid: Optional[str] = None,
    ) -> Dict[str, Any]:
        sym = (symbol or "").upper()
        params: Dict[str, Any] = {"symbol": sym, "productType": self.product_type}
        if order_id:
            params["orderId"] = order_id
        if client_oid:
            params["clientOid"] = client_oid
        return await self._request_any_status("GET", "/api/v2/mix/order/detail", params=params, auth=True)

    @staticmethod
    def is_filled(order_detail_resp: Dict[str, Any]) -> bool:
        if not _is_ok(order_detail_resp):
            return False
        data = order_detail_resp.get("data") or {}
        state = str(data.get("state") or data.get("status") or "").lower()
        if state in {"filled", "full_fill", "fullfill", "completed", "success"}:
            return True
        filled = _safe_float(data.get("baseVolume") or data.get("filledQty") or data.get("filledSize"), 0.0)
        size = _safe_float(data.get("size") or data.get("quantity") or data.get("qty"), 0.0)
        return (size > 0) and (filled >= size * 0.999)
