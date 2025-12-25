# =====================================================================
# bitget_trader.py — Bitget Execution (Entry + TP1/TP2 + SL->BE)
# + DEBUG payload logs (safe)
# + Dynamic price formatting if meta is suspicious (fix 40020 price error)
# =====================================================================

from __future__ import annotations

import json
import logging
import time
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from settings import PRODUCT_TYPE, MARGIN_COIN

logger = logging.getLogger(__name__)

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

def _estimate_tick_from_price(price: float) -> float:
    p = abs(float(price))
    if p >= 10000:
        return 1.0
    elif p >= 1000:
        return 0.1
    elif p >= 100:
        return 0.01
    elif p >= 10:
        return 0.001
    elif p >= 1:
        return 0.0001
    elif p >= 0.1:
        return 0.00001
    elif p >= 0.01:
        return 0.000001
    else:
        return 0.0000001

def _decimals_from_tick(tick: float, cap: int = 12) -> int:
    t = abs(float(tick))
    if t <= 0:
        return 6
    # decimals ~ -log10(tick)
    d = int(round(-math.log10(t)))
    return max(0, min(cap, d))

def _fmt_decimal(price: float, decimals: int) -> str:
    # never scientific notation
    decimals = max(0, min(12, int(decimals)))
    return f"{float(price):.{decimals}f}"

@dataclass
class ContractMeta:
    symbol: str
    price_place: int
    price_tick: float
    qty_place: int
    qty_step: float
    min_qty: float
    raw: Dict[str, Any]

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

            # Robust tick extraction (Bitget fields vary)
            # Most contracts use:
            #   tick = priceEndStep / (10 ** pricePlace)
            # Example: pricePlace=1, priceEndStep=5 => tick=0.5
            pe = _safe_float(c.get("priceEndStep"), 0.0)
            if pe > 0:
                # If priceEndStep looks like an integer step, scale it by decimals.
                if price_place >= 0 and abs(pe - round(pe)) < 1e-9 and pe >= 1:
                    price_tick = float(pe) / float(10 ** price_place)
                else:
                    # Some symbols return the tick already as a decimal.
                    price_tick = float(pe)
            else:
                price_tick = (
                    _safe_float(c.get("priceStep"), 0.0)
                    or _safe_float(c.get("tickSize"), 0.0)
                    or _safe_float(c.get("priceTick"), 0.0)
                    or (10 ** (-price_place))
                )

            vs = _safe_float(c.get("volumeStep"), 0.0)
            if vs > 0 and qty_place >= 0 and abs(vs - round(vs)) < 1e-9 and vs >= 1 and qty_place > 0:
                qty_step = float(vs) / float(10 ** qty_place)
            else:
                qty_step = (
                    _safe_float(c.get("sizeMultiplier"), 0.0)
                    or float(vs)
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
                raw=c,
            )

            # Debug: show suspicious meta once
            if float(price_tick or 0) >= 1.0 and price_place == 0:
                # many memes will be <1 price => would format to "0"
                logger.warning(
                    "[META_SUS] %s pricePlace=%s priceTick=%s raw_tickSize=%s raw_priceEndStep=%s raw_priceStep=%s",
                    sym,
                    price_place,
                    price_tick,
                    c.get("tickSize"),
                    c.get("priceEndStep"),
                    c.get("priceStep"),
                )

        self._by_symbol = by_symbol
        self._ts = now
        logger.info("[META] refreshed contracts cache (%d symbols)", len(by_symbol))

    async def get(self, symbol: str) -> Optional[ContractMeta]:
        sym = (symbol or "").upper()
        await self.refresh()
        return self._by_symbol.get(sym)

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

    async def _quantize(self, symbol: str, price: float, qty: float, tick_hint: Optional[float] = None) -> Tuple[float, float, float]:
        meta = await self._meta.get(symbol)
        if not meta:
            tick = float(tick_hint or _estimate_tick_from_price(price))
            step = 1e-6
            return float(price), float(qty), tick

        meta_tick = float(meta.price_tick or (10 ** (-meta.price_place)))
        tick = float(tick_hint or meta_tick)
        step = float(meta.qty_step or (10 ** (-meta.qty_place)))

        # if meta is suspicious (tick=1, pricePlace=0) and price<1 => use hint/estimate
        if (meta.price_place == 0 and meta_tick >= 1.0 and float(price) < 1.0):
            tick = float(tick_hint or _estimate_tick_from_price(price))

        p = float(round(price / tick) * tick) if tick > 0 else float(price)
        q = float(int(qty / step) * step) if step > 0 else float(qty)

        if q < meta.min_qty:
            q = 0.0

        # clamp rounding noise
        # DO NOT force meta.price_place if suspicious; use dynamic decimals later
        if meta.qty_place >= 0:
            q = float(f"{q:.{max(0, meta.qty_place)}f}")

        return p, q, tick

    async def _format_price(self, symbol: str, price: float, tick_used: Optional[float] = None) -> str:
        meta = await self._meta.get(symbol)
        p = float(price)

        # Always avoid scientific notation
        if not meta:
            d = _decimals_from_tick(tick_used or _estimate_tick_from_price(p))
            return _fmt_decimal(p, d)

        meta_tick = float(meta.price_tick or (10 ** (-meta.price_place)))
        suspicious = (meta.price_place == 0 and meta_tick >= 1.0 and p < 1.0)

        if suspicious:
            # key fix: don't format to 0 for sub-1 prices
            d = _decimals_from_tick(tick_used or _estimate_tick_from_price(p))
            return _fmt_decimal(p, d)

        # normal case
        if meta.price_place >= 0:
            return _fmt_decimal(p, max(0, meta.price_place))

        # fallback
        d = _decimals_from_tick(tick_used or meta_tick)
        return _fmt_decimal(p, d)

    async def get_tick(self, symbol: str) -> float:
        meta = await self._meta.get(symbol)
        if not meta:
            return 1e-6
        return float(meta.price_tick or (10 ** (-meta.price_place)))

    async def debug_meta(self, symbol: str) -> Dict[str, Any]:
        meta = await self._meta.get(symbol)
        if not meta:
            return {"symbol": symbol, "meta": None}
        return {
            "symbol": meta.symbol,
            "pricePlace": meta.price_place,
            "priceTick": meta.price_tick,
            "qtyPlace": meta.qty_place,
            "qtyStep": meta.qty_step,
            "minQty": meta.min_qty,
            "raw": {
                "tickSize": meta.raw.get("tickSize"),
                "priceEndStep": meta.raw.get("priceEndStep"),
                "priceStep": meta.raw.get("priceStep"),
                "pricePlace": meta.raw.get("pricePlace"),
                "volumePlace": meta.raw.get("volumePlace"),
                "volumeStep": meta.raw.get("volumeStep"),
                "minTradeNum": meta.raw.get("minTradeNum"),
            },
        }

    async def place_limit(
        self,
        symbol: str,
        side: str,
        price: float,
        size: Optional[float] = None,
        client_oid: Optional[str] = None,
        trade_side: str = "open",
        reduce_only: bool = False,
        tick_hint: Optional[float] = None,  # <- important
        debug_tag: str = "ENTRY",
    ) -> Dict[str, Any]:
        sym = (symbol or "").upper()
        s = (side or "").lower()

        if size is None:
            notional = self.margin_usdt * self.leverage
            raw_qty = notional / max(1e-12, float(price))
        else:
            raw_qty = float(size)

        q_price, q_qty, tick_used = await self._quantize(sym, float(price), float(raw_qty), tick_hint=tick_hint)
        if q_qty <= 0:
            return {"ok": False, "code": "QTY0", "msg": "quantized qty is 0"}

        price_str = await self._format_price(sym, q_price, tick_used=tick_used)

        payload = {
            "symbol": sym,
            "productType": self.product_type,
            "marginCoin": self.margin_coin,
            "marginMode": self.margin_mode,
            "size": str(q_qty),
            "price": price_str,
            "side": s,
            "orderType": "limit",
            "timeInForceValue": "normal",
            "clientOid": client_oid or f"oid-{sym}-{_now_ms()}",
            "tradeSide": (trade_side or "open").lower(),
        }

        if reduce_only:
            payload["reduceOnly"] = "YES"

        # SAFE DEBUG (no secrets)
        logger.info(
            "[ORDER_%s] sym=%s side=%s tradeSide=%s reduceOnly=%s raw_price=%s q_price=%s price_str=%s raw_qty=%s q_qty=%s tick_used=%s",
            debug_tag, sym, s, payload["tradeSide"], payload.get("reduceOnly"), float(price), q_price, price_str, float(raw_qty), q_qty, tick_used
        )

        resp = await self._request_any_status("POST", "/api/v2/mix/order/place-order", data=payload, auth=True)

        if not _is_ok(resp):
            # attach debug block for scanner logs
            resp["_debug"] = {
                "debug_tag": debug_tag,
                "symbol": sym,
                "side": s,
                "tradeSide": payload["tradeSide"],
                "price_raw": float(price),
                "price_quant": q_price,
                "price_str": price_str,
                "qty_raw": float(raw_qty),
                "qty_quant": q_qty,
                "tick_used": tick_used,
            }
        else:
            resp["qty"] = q_qty
            resp["price"] = q_price
            resp["_debug"] = {
                "debug_tag": debug_tag,
                "price_str": price_str,
                "tick_used": tick_used,
            }
        return resp

    async def place_reduce_limit_tp(
        self,
        symbol: str,
        close_side: str,
        price: float,
        qty: float,
        client_oid: Optional[str] = None,
        *,
        trade_side: str = "close",
        reduce_only: bool = True,
        tick_hint: Optional[float] = None,
        debug_tag: str = "TP",
    ) -> Dict[str, Any]:
        return await self.place_limit(
            symbol=symbol,
            side=close_side,
            price=price,
            size=qty,
            client_oid=client_oid,
            trade_side=trade_side,
            reduce_only=reduce_only,
            tick_hint=tick_hint,
            debug_tag=debug_tag,
        )

    async def place_stop_market_sl(
        self,
        symbol: str,
        close_side: str,
        trigger_price: float,
        qty: float,
        *,
        reduce_only: bool = True,
        client_oid: Optional[str] = None,
        trigger_type: str = "mark_price",
        tick_hint: Optional[float] = None,
        debug_tag: str = "SL",
    ) -> Dict[str, Any]:
        sym = (symbol or "").upper()
        close_s = (close_side or "").lower()

        # quantize qty
        _, q_qty, tick_used = await self._quantize(sym, float(trigger_price), float(qty), tick_hint=tick_hint)
        if q_qty <= 0:
            return {"ok": False, "code": "QTY0", "msg": "quantized qty is 0"}

        trig_str = await self._format_price(sym, float(trigger_price), tick_used=tick_used)

        payload = {
            "symbol": sym,
            "productType": self.product_type,
            "marginCoin": self.margin_coin,
            "marginMode": self.margin_mode,
            "size": str(q_qty),
            "side": close_s,
            "orderType": "market",
            "triggerPrice": trig_str,
            "triggerType": trigger_type,
            "planType": "normal_plan",
            "tradeSide": "close",
            "clientOid": client_oid or f"sl-{sym}-{_now_ms()}",
            "executePrice": "0",
        }
        if reduce_only:
            payload["reduceOnly"] = "YES"


        logger.info(
            "[ORDER_%s] sym=%s close_side=%s trigger_type=%s trigger_raw=%s trigger_str=%s qty=%s tick_used=%s",
            debug_tag, sym, close_s, trigger_type, float(trigger_price), trig_str, q_qty, tick_used
        )

        resp = await self._request_any_status("POST", "/api/v2/mix/order/place-plan-order", data=payload, auth=True)

        if not _is_ok(resp):
            resp["_debug"] = {
                "debug_tag": debug_tag,
                "symbol": sym,
                "close_side": close_s,
                "trigger_raw": float(trigger_price),
                "trigger_str": trig_str,
                "qty_quant": q_qty,
                "tick_used": tick_used,
                "trigger_type": trigger_type,
            }
        else:
            resp["qty"] = q_qty
            resp["_debug"] = {"debug_tag": debug_tag, "trigger_str": trig_str, "tick_used": tick_used}
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

    async def get_order_detail(self, symbol: str, *, order_id: Optional[str] = None, client_oid: Optional[str] = None) -> Dict[str, Any]:
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
