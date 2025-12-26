# =====================================================================
# bitget_trader.py — Bitget Execution (Entry + TP1 + SL->BE)
# Desk-lead hardened:
# - Meta cache refresh locked + stable param ordering (signature-safe)
# - Quantization rules per order type (entry/TP vs SL trigger)
# - Dynamic price formatting fallback on 40020 (precision/price error)
# - Auto clamp + retry on 22047 (price band) for LIMIT / PLAN
# - Attach parsed hints for scanner (min_usdt / band min-max / debug payload)
# - Safer qty formatting (no sci notation) + float-noise resistant floor/ceil
# =====================================================================

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

from retry_utils import retry_async
from settings import PRODUCT_TYPE, MARGIN_COIN

logger = logging.getLogger(__name__)

_MIN_USDT_RE = re.compile(r"minimum amount\s*([0-9]*\.?[0-9]+)\s*USDT", re.IGNORECASE)
_MAX_RE = re.compile(r"maximum price limit:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_MIN_RE = re.compile(r"minimum price limit:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

# Bitget error codes we actively handle
_ERR_PRICE_PRECISION = {"40020"}  # price format/precision
_ERR_PRICE_BAND = {"22047"}       # price limit band
_ERR_MIN_USDT = {"45110"}         # minimum amount X USDT


# ---------------------------------------------------------------------
# small utils
# ---------------------------------------------------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
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
    if p >= 1000:
        return 0.1
    if p >= 100:
        return 0.01
    if p >= 10:
        return 0.001
    if p >= 1:
        return 0.0001
    if p >= 0.1:
        return 0.00001
    if p >= 0.01:
        return 0.000001
    return 0.0000001


def _decimals_from_step(step: float, cap: int = 12) -> int:
    t = abs(float(step))
    if t <= 0:
        return 6
    # step like 0.001 -> 3 decimals
    d = int(round(-math.log10(t)))
    return max(0, min(cap, d))


def _fmt_decimal(x: float, decimals: int) -> str:
    decimals = max(0, min(12, int(decimals)))
    # never sci notation
    return f"{float(x):.{decimals}f}"


def _floor_step(x: float, step: float) -> float:
    if step <= 0:
        return float(x)
    # float-noise resistant
    return float(math.floor((float(x) / step) + 1e-12) * step)


def _ceil_step(x: float, step: float) -> float:
    if step <= 0:
        return float(x)
    return float(math.ceil((float(x) / step) - 1e-12) * step)


def _parse_band(msg: str) -> Tuple[Optional[float], Optional[float]]:
    if not msg:
        return None, None
    mmax = _MAX_RE.search(msg)
    mmin = _MIN_RE.search(msg)
    mx = float(mmax.group(1)) if mmax else None
    mn = float(mmin.group(1)) if mmin else None
    return mn, mx


def _parse_min_usdt(msg: str) -> Optional[float]:
    if not msg:
        return None
    m = _MIN_USDT_RE.search(msg)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _clamp_band(price: float, tick: float, mn: Optional[float], mx: Optional[float]) -> Optional[float]:
    p = float(price)
    if p <= 0:
        return None
    if tick <= 0:
        if mx is not None:
            p = min(p, float(mx))
        if mn is not None:
            p = max(p, float(mn))
        return p if p > 0 else None

    # keep a small buffer away from band edges
    if mx is not None:
        p = min(p, float(mx) - 2.0 * tick)
    if mn is not None:
        p = max(p, float(mn) + 2.0 * tick)

    if p <= 0:
        return None
    return p


# ---------------------------------------------------------------------
# Contract meta
# ---------------------------------------------------------------------

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
        self._lock = asyncio.Lock()

    async def refresh(self, force: bool = False) -> None:
        async with self._lock:
            now = time.time()
            if (not force) and self._by_symbol and (now - self._ts) < self.ttl_s:
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

                price_place = _safe_int(c.get("pricePlace"), 6)
                qty_place = _safe_int(c.get("volumePlace"), 4)

                # Robust tick extraction
                price_tick = (
                    _safe_float(c.get("priceEndStep"), 0.0)
                    or _safe_float(c.get("priceStep"), 0.0)
                    or _safe_float(c.get("tickSize"), 0.0)
                    or _safe_float(c.get("priceTick"), 0.0)
                    or (10 ** (-max(0, price_place)))
                )

                qty_step = (
                    _safe_float(c.get("sizeMultiplier"), 0.0)
                    or _safe_float(c.get("volumeStep"), 0.0)
                    or _safe_float(c.get("minTradeNum"), 0.0)
                    or (10 ** (-max(0, qty_place)))
                )

                min_qty = _safe_float(c.get("minTradeNum"), 0.0) or qty_step

                by_symbol[sym] = ContractMeta(
                    symbol=sym,
                    price_place=int(price_place),
                    price_tick=float(price_tick),
                    qty_place=int(qty_place),
                    qty_step=float(qty_step),
                    min_qty=float(min_qty),
                    raw=c,
                )

                # one-time-ish warning when meta is clearly suspicious for sub-1 prices
                if float(price_tick or 0) >= 1.0 and int(price_place) == 0:
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


# ---------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------

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

    # ----------------------------
    # Low-level request (keeps body even on errors)
    # ----------------------------

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

        # stable ordering => stable signatures
        query = ""
        if params:
            items = sorted(params.items(), key=lambda kv: kv[0])
            query = "?" + "&".join(f"{k}={v}" for k, v in items)

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

        t0 = time.time()
        try:
            async with self.client.session.request(
                method.upper(),
                url,
                headers=headers,
                data=body if data else None,
                timeout=timeout,
            ) as resp:
                txt = await resp.text()
                status = resp.status
                latency_ms = int((time.time() - t0) * 1000)

                try:
                    js = json.loads(txt) if txt else {}
                except Exception:
                    return {
                        "ok": False,
                        "code": "NONJSON",
                        "msg": "non-json response",
                        "raw": txt,
                        "_http_status": status,
                        "_latency_ms": latency_ms,
                        "_path": path,
                    }

                code = str(js.get("code", ""))
                ok = (status < 400) and (code == "00000")
                js["ok"] = ok
                js["_http_status"] = status
                js["_latency_ms"] = latency_ms
                js["_path"] = path
                return js
        except Exception as e:
            latency_ms = int((time.time() - t0) * 1000)
            return {
                "ok": False,
                "code": "EXC",
                "msg": str(e),
                "_http_status": 0,
                "_latency_ms": latency_ms,
                "_path": path,
            }

    # ----------------------------
    # Meta helpers
    # ----------------------------

    async def get_tick(self, symbol: str) -> float:
        meta = await self._meta.get(symbol)
        if not meta:
            return 1e-6
        t = float(meta.price_tick or (10 ** (-max(0, meta.price_place))))
        return t if t > 0 else 1e-6

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

    # ----------------------------
    # Quantize + formatting (desk rules)
    # ----------------------------

    async def _quantize_price_qty(
        self,
        symbol: str,
        price: float,
        qty: float,
        *,
        side: Optional[str] = None,          # BUY/SELL (for LIMIT)
        close_side: Optional[str] = None,     # BUY/SELL (for SL trigger)
        is_trigger: bool = False,             # SL trigger quantization
        tick_hint: Optional[float] = None,
    ) -> Tuple[float, float, float, float, int, int]:
        """
        Returns: (q_price, q_qty, tick_used, step_used, price_place, qty_place)
        """
        meta = await self._meta.get(symbol)
        if not meta:
            tick = float(tick_hint or _estimate_tick_from_price(price))
            step = 1e-6
            q_price = float(price)
            q_qty = _floor_step(float(qty), step)
            return q_price, q_qty, tick, step, 6, 6

        meta_tick = float(meta.price_tick or (10 ** (-max(0, meta.price_place))))
        tick = float(tick_hint or meta_tick)
        step = float(meta.qty_step or (10 ** (-max(0, meta.qty_place))))

        # Suspicious meta: tick=1 with pricePlace=0 but price<1 => never format to 0
        if meta.price_place == 0 and meta_tick >= 1.0 and float(price) < 1.0:
            tick = float(tick_hint or _estimate_tick_from_price(price))

        # Price quantization rules:
        # - LIMIT: BUY -> floor, SELL -> ceil (more fill-prob)
        # - TRIGGER (SL): close_side SELL (long stop) -> ceil (trigger earlier),
        #                 close_side BUY (short stop) -> floor
        p = float(price)
        if tick > 0:
            if is_trigger:
                cs = (close_side or "").upper()
                if cs == "SELL":
                    q_price = _ceil_step(p, tick)
                elif cs == "BUY":
                    q_price = _floor_step(p, tick)
                else:
                    q_price = _floor_step(p, tick)
            else:
                s = (side or "").upper()
                if s == "BUY":
                    q_price = _floor_step(p, tick)
                elif s == "SELL":
                    q_price = _ceil_step(p, tick)
                else:
                    # fallback: nearest
                    q_price = float(round(p / tick) * tick)
        else:
            q_price = p

        # Qty quantization: always floor (avoid over-sizing)
        q = float(qty)
        q_qty = _floor_step(q, step) if step > 0 else q

        # min qty gate
        if q_qty < float(meta.min_qty or 0.0):
            q_qty = 0.0

        # clamp float noise with decimals
        q_dec = max(0, int(meta.qty_place))
        q_qty = float(_fmt_decimal(q_qty, q_dec)) if q_dec >= 0 else q_qty

        return float(q_price), float(q_qty), float(tick), float(step), int(meta.price_place), int(meta.qty_place)

    async def _format_price(
        self,
        symbol: str,
        price: float,
        *,
        tick_used: Optional[float] = None,
        force_decimals: Optional[int] = None,
    ) -> str:
        """
        - Normal: use meta.pricePlace
        - Fallback: decimals derived from tick_used to avoid Bitget 40020 price issues
        """
        meta = await self._meta.get(symbol)
        p = float(price)

        if force_decimals is not None:
            return _fmt_decimal(p, int(force_decimals))

        if not meta:
            d = _decimals_from_step(tick_used or _estimate_tick_from_price(p))
            return _fmt_decimal(p, d)

        meta_tick = float(meta.price_tick or (10 ** (-max(0, meta.price_place))))
        suspicious = (meta.price_place == 0 and meta_tick >= 1.0 and p < 1.0)
        if suspicious:
            d = _decimals_from_step(tick_used or _estimate_tick_from_price(p))
            return _fmt_decimal(p, d)

        # normal case
        if meta.price_place >= 0:
            # sometimes pricePlace is wrong; we handle via retry on 40020
            return _fmt_decimal(p, max(0, meta.price_place))

        # fallback
        d = _decimals_from_step(tick_used or meta_tick)
        return _fmt_decimal(p, d)

    def _format_qty_str(self, qty: float, qty_place: int) -> str:
        # never scientific notation
        d = max(0, min(12, int(qty_place)))
        return _fmt_decimal(float(qty), d)

    # ----------------------------
    # Orders
    # ----------------------------

    async def place_limit(
        self,
        symbol: str,
        side: str,
        price: float,
        size: Optional[float] = None,
        client_oid: Optional[str] = None,
        trade_side: str = "open",
        reduce_only: bool = False,
        tick_hint: Optional[float] = None,
        debug_tag: str = "ENTRY",
    ) -> Dict[str, Any]:
        """
        Universal LIMIT:
          - Entry: size=None -> notional from margin_usdt*leverage
          - TP1: size provided, reduce_only True
        Desk behavior:
          - quantize price BUY floor / SELL ceil
          - retry on 22047 (band clamp)
          - retry on 40020 (force decimals derived from tick)
        """
        sym = (symbol or "").upper()
        s = (side or "").lower()

        if size is None:
            notional = self.margin_usdt * self.leverage
            raw_qty = notional / max(1e-12, float(price))
        else:
            raw_qty = float(size)

        q_price, q_qty, tick_used, _step, _pp, qp = await self._quantize_price_qty(
            sym, float(price), float(raw_qty), side=s.upper(), is_trigger=False, tick_hint=tick_hint
        )
        if q_qty <= 0:
            return {"ok": False, "code": "QTY0", "msg": "quantized qty is 0"}

        # normal formatting
        price_str = await self._format_price(sym, q_price, tick_used=tick_used)
        size_str = self._format_qty_str(q_qty, qp)

        oid = client_oid or f"oid-{sym}-{_now_ms()}"
        payload = {
            "symbol": sym,
            "productType": self.product_type,
            "marginCoin": self.margin_coin,
            "marginMode": self.margin_mode,
            "size": size_str,
            "price": price_str,
            "side": s,
            "orderType": "limit",
            "timeInForceValue": "normal",
            "clientOid": oid,
            "tradeSide": (trade_side or "open").lower(),
        }
        if reduce_only:
            payload["reduceOnly"] = "YES"

        logger.info(
            "[ORDER_%s] sym=%s side=%s tradeSide=%s reduceOnly=%s raw_price=%s q_price=%s price_str=%s raw_qty=%s q_qty=%s tick_used=%s",
            debug_tag, sym, s, payload["tradeSide"], payload.get("reduceOnly"), float(price), q_price, price_str, float(raw_qty), q_qty, tick_used
        )

        async def _send(data_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            return await self._request_any_status("POST", "/api/v2/mix/order/place-order", data=(data_override or payload), auth=True)

        resp = await _send()

        # If band error => clamp and retry once
        code = str(resp.get("code", ""))
        msg = str(resp.get("msg") or "")
        if (not _is_ok(resp)) and code in _ERR_PRICE_BAND:
            mn, mx = _parse_band(msg)
            clamped = _clamp_band(q_price, tick_used, mn, mx)
            if clamped is not None:
                q_price2, q_qty2, tick2, _step2, _pp2, qp2 = await self._quantize_price_qty(
                    sym, float(clamped), float(raw_qty), side=s.upper(), is_trigger=False, tick_hint=tick_used
                )
                if q_qty2 > 0:
                    price_str2 = await self._format_price(sym, q_price2, tick_used=tick2)
                    payload2 = dict(payload)
                    payload2["price"] = price_str2
                    payload2["size"] = self._format_qty_str(q_qty2, qp2)
                    logger.warning("[ORDER_%s] band clamp retry mn=%s mx=%s before=%s after=%s", debug_tag, mn, mx, q_price, q_price2)
                    resp = await _send(payload2)
                    resp["_band"] = {"min": mn, "max": mx, "before": q_price, "after": q_price2}

        # If price precision error => force decimals derived from tick and retry once
        code = str(resp.get("code", ""))
        msg = str(resp.get("msg") or "")
        if (not _is_ok(resp)) and code in _ERR_PRICE_PRECISION:
            d = _decimals_from_step(tick_used or _estimate_tick_from_price(q_price))
            price_str2 = await self._format_price(sym, q_price, tick_used=tick_used, force_decimals=d)
            payload2 = dict(payload)
            payload2["price"] = price_str2
            logger.warning("[ORDER_%s] 40020 retry force_decimals=%s price_str=%s->%s", debug_tag, d, price_str, price_str2)
            resp = await _send(payload2)

        # Attach debug block for scanner
        if not _is_ok(resp):
            # Surface parsed hints
            if code in _ERR_MIN_USDT:
                mmin = _parse_min_usdt(msg)
                if mmin is not None:
                    resp["_min_usdt"] = float(mmin)
            if code in _ERR_PRICE_BAND:
                mn, mx = _parse_band(msg)
                resp["_band"] = {"min": mn, "max": mx}

            resp["_debug"] = {
                "debug_tag": debug_tag,
                "symbol": sym,
                "side": s,
                "tradeSide": payload["tradeSide"],
                "reduceOnly": payload.get("reduceOnly"),
                "price_raw": float(price),
                "price_quant": float(q_price),
                "price_str": str(payload.get("price")),
                "qty_raw": float(raw_qty),
                "qty_quant": float(q_qty),
                "tick_used": float(tick_used),
                "clientOid": oid,
                "msg": msg,
                "code": code,
            }
        else:
            resp["qty"] = float(q_qty)
            resp["price"] = float(q_price)
            resp["_debug"] = {
                "debug_tag": debug_tag,
                "price_str": str(payload.get("price")),
                "tick_used": float(tick_used),
                "clientOid": oid,
            }
        return resp

    async def place_reduce_limit_tp(
        self,
        symbol: str,
        close_side: str,
        price: float,
        qty: float,
        client_oid: Optional[str] = None,
        tick_hint: Optional[float] = None,
        debug_tag: str = "TP",
    ) -> Dict[str, Any]:
        # keep compatibility — calls unified place_limit
        return await self.place_limit(
            symbol=symbol,
            side=close_side,
            price=price,
            size=qty,
            client_oid=client_oid,
            trade_side="close",
            reduce_only=True,
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
        client_oid: Optional[str] = None,
        trigger_type: str = "mark_price",
        tick_hint: Optional[float] = None,
        debug_tag: str = "SL",
    ) -> Dict[str, Any]:
        """
        Plan order SL:
          - trigger quantization uses close_side (SELL => ceil, BUY => floor)
          - retry on 22047 band clamp
          - retry on 40020 by forcing decimals from tick
        """
        sym = (symbol or "").upper()
        close_s = (close_side or "").lower()
        close_u = close_s.upper()

        q_trig, q_qty, tick_used, _step, _pp, qp = await self._quantize_price_qty(
            sym,
            float(trigger_price),
            float(qty),
            close_side=close_u,
            is_trigger=True,
            tick_hint=tick_hint,
        )
        if q_qty <= 0:
            return {"ok": False, "code": "QTY0", "msg": "quantized qty is 0"}

        trig_str = await self._format_price(sym, q_trig, tick_used=tick_used)
        size_str = self._format_qty_str(q_qty, qp)
        oid = client_oid or f"sl-{sym}-{_now_ms()}"

        payload = {
            "symbol": sym,
            "productType": self.product_type,
            "marginCoin": self.margin_coin,
            "marginMode": self.margin_mode,
            "size": size_str,
            "side": close_s,
            "orderType": "market",
            "triggerPrice": trig_str,
            "triggerType": trigger_type,
            "planType": "normal_plan",
            "tradeSide": "close",
            "reduceOnly": "YES",
            "clientOid": oid,
            "executePrice": "0",
        }

        logger.info(
            "[ORDER_%s] sym=%s close_side=%s trigger_type=%s trigger_raw=%s trig_q=%s trig_str=%s qty=%s tick_used=%s",
            debug_tag, sym, close_s, trigger_type, float(trigger_price), q_trig, trig_str, q_qty, tick_used
        )

        async def _send(data_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            return await self._request_any_status("POST", "/api/v2/mix/order/place-plan-order", data=(data_override or payload), auth=True)

        resp = await _send()

        code = str(resp.get("code", ""))
        msg = str(resp.get("msg") or "")

        # band clamp retry
        if (not _is_ok(resp)) and code in _ERR_PRICE_BAND:
            mn, mx = _parse_band(msg)
            clamped = _clamp_band(q_trig, tick_used, mn, mx)
            if clamped is not None:
                q_trig2, q_qty2, tick2, _step2, _pp2, qp2 = await self._quantize_price_qty(
                    sym,
                    float(clamped),
                    float(qty),
                    close_side=close_u,
                    is_trigger=True,
                    tick_hint=tick_used,
                )
                if q_qty2 > 0:
                    trig_str2 = await self._format_price(sym, q_trig2, tick_used=tick2)
                    payload2 = dict(payload)
                    payload2["triggerPrice"] = trig_str2
                    payload2["size"] = self._format_qty_str(q_qty2, qp2)
                    logger.warning("[ORDER_%s] band clamp retry mn=%s mx=%s before=%s after=%s", debug_tag, mn, mx, q_trig, q_trig2)
                    resp = await _send(payload2)
                    resp["_band"] = {"min": mn, "max": mx, "before": q_trig, "after": q_trig2}

        # price precision retry
        code = str(resp.get("code", ""))
        msg = str(resp.get("msg") or "")
        if (not _is_ok(resp)) and code in _ERR_PRICE_PRECISION:
            d = _decimals_from_step(tick_used or _estimate_tick_from_price(q_trig))
            trig_str2 = await self._format_price(sym, q_trig, tick_used=tick_used, force_decimals=d)
            payload2 = dict(payload)
            payload2["triggerPrice"] = trig_str2
            logger.warning("[ORDER_%s] 40020 retry force_decimals=%s trig_str=%s->%s", debug_tag, d, trig_str, trig_str2)
            resp = await _send(payload2)

        if not _is_ok(resp):
            if code in _ERR_MIN_USDT:
                mmin = _parse_min_usdt(msg)
                if mmin is not None:
                    resp["_min_usdt"] = float(mmin)
            if code in _ERR_PRICE_BAND:
                mn, mx = _parse_band(msg)
                resp["_band"] = {"min": mn, "max": mx}

            resp["_debug"] = {
                "debug_tag": debug_tag,
                "symbol": sym,
                "close_side": close_s,
                "trigger_raw": float(trigger_price),
                "trigger_quant": float(q_trig),
                "trigger_str": trig_str,
                "qty_quant": float(q_qty),
                "tick_used": float(tick_used),
                "trigger_type": trigger_type,
                "clientOid": oid,
                "msg": msg,
                "code": code,
            }
        else:
            resp["qty"] = float(q_qty)
            resp["_debug"] = {"debug_tag": debug_tag, "trigger_str": trig_str, "tick_used": float(tick_used), "clientOid": oid}
        return resp

    # ----------------------------
    # Cancel / Detail
    # ----------------------------

    async def cancel_plan_orders(self, symbol: str, order_ids: List[str]) -> Dict[str, Any]:
        sym = (symbol or "").upper()
        ids = [str(x) for x in (order_ids or []) if str(x)]
        if not ids:
            return {"ok": False, "code": "NOIDS", "msg": "order_ids empty"}
        payload = {
            "symbol": sym,
            "productType": self.product_type,
            "planType": "normal_plan",
            "orderIdList": ids,
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
            params["orderId"] = str(order_id)
        if client_oid:
            params["clientOid"] = str(client_oid)
        return await self._request_any_status("GET", "/api/v2/mix/order/detail", params=params, auth=True)

    @staticmethod
    def is_filled(order_detail_resp: Dict[str, Any]) -> bool:
        """
        Desk-safe fill detection: handle variants of fields and partial states.
        """
        if not _is_ok(order_detail_resp):
            return False

        data = order_detail_resp.get("data") or {}
        state = str(data.get("state") or data.get("status") or "").lower()

        if state in {"filled", "full_fill", "fullfill", "completed", "success"}:
            return True

        # numerical fallback
        filled = _safe_float(
            data.get("baseVolume")
            or data.get("filledQty")
            or data.get("filledSize")
            or data.get("filledVolume")
            or data.get("dealSize"),
            0.0,
        )
        size = _safe_float(
            data.get("size")
            or data.get("quantity")
            or data.get("qty")
            or data.get("totalSize"),
            0.0,
        )

        if size <= 0:
            return False
        return filled >= size * 0.999
