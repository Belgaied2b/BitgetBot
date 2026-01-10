# =====================================================================
# bitget_trader.py — Bitget Execution (Entry + TP1 + SL->BE)
# Desk-lead hardened FINAL (ready to paste)
#
# Hardenings:
# - Contract meta parsing more robust (data can be list or dict->list)
# - Tick derivation: priceStep/priceEndStep interpreted with pricePlace
# - qty_step != minTradeNum (minTradeNum = minQty)
# - Stable params ordering for signatures (GET query sorted)
# - place_limit retry: 22047 band clamp, 40020 precision, 40762 downsize (ENTRY)
# - place_plan retry: 22047 band clamp, 40020 precision
# - cancel_order: safe payload + no crash
# - cancel_plan_orders: v2 payload orderIdList = list of objects (fix)
# - get_order_detail: safer params (includes marginCoin) + accepts orderId/clientOid
# - is_filled(): handles “filled” + numeric fallback; also treats partial fill as “filled enough”
#   so the watcher can arm protection when ANY position is open.
# - helpers: filled_qty(), order_size(), remaining_qty()
# - flash_close_position(): close-positions v2 helper
#
# Watcher additions:
# - get_all_positions(): /api/v2/mix/position/all-position
# - get_single_position(): /api/v2/mix/position/single-position
# - get_pending_orders(): /api/v2/mix/order/orders-pending
# - get_pending_plan_orders(): /api/v2/mix/order/orders-plan-pending
# - get_order_fills(): /api/v2/mix/order/fills
# - get_ticker(): /api/v2/mix/market/ticker
# - helpers to extract lists safely from v2 payload shapes
# =====================================================================

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Union

from retry_utils import retry_async
from settings import PRODUCT_TYPE, MARGIN_COIN, EXEC_LOG_ENABLE, EXEC_LOG_PATH

logger = logging.getLogger(__name__)

_MIN_USDT_RE = re.compile(r"minimum amount\s*([0-9]*\.?[0-9]+)\s*USDT", re.IGNORECASE)
_MAX_RE = re.compile(r"maximum price limit:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_MIN_RE = re.compile(r"minimum price limit:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

# Bitget error codes we actively handle
_ERR_PRICE_PRECISION = {"40020"}  # price format/precision
_ERR_PRICE_BAND = {"22047"}       # price limit band
_ERR_MIN_USDT = {"45110"}         # minimum amount X USDT
_ERR_BALANCE = {"40762"}          # The order amount exceeds the balance


# ---------------------------------------------------------------------
# small utils
# ---------------------------------------------------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
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
    try:
        d = int(round(-math.log10(t)))
    except Exception:
        d = 6
    return max(0, min(cap, d))


def _fmt_decimal(x: float, decimals: int) -> str:
    decimals = max(0, min(12, int(decimals)))
    return f"{float(x):.{decimals}f}"


def _floor_step(x: float, step: float) -> float:
    if step <= 0:
        return float(x)
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

    # keep a little buffer inside the band
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


def _tick_from_place_and_step(price_place: int, step_val: float) -> float:
    """
    Bitget often returns:
      pricePlace=5, priceEndStep="1"  -> tick = 1 * 10^-5
    If step_val < 1, it's already an absolute tick.
    """
    try:
        pp = int(price_place)
        sv = float(step_val)
        if sv <= 0:
            return 0.0
        if sv < 1.0:
            return sv
        base = (10.0 ** (-pp)) if pp > 0 else 1.0
        return float(sv * base)
    except Exception:
        return 0.0


class ContractMetaCache:
    def __init__(self, client, ttl_s: int = 600):
        self.client = client
        self.ttl_s = ttl_s
        self._ts = 0.0
        self._by_symbol: Dict[str, ContractMeta] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _extract_contracts_list(js: Dict[str, Any]) -> List[Dict[str, Any]]:
        data = js.get("data")
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            for k in ("list", "data", "rows"):
                v = data.get(k)
                if isinstance(v, list):
                    return [x for x in v if isinstance(x, dict)]
        return []

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

            if not isinstance(js, dict):
                logger.error("[META] bad contracts response (non-dict): %s", js)
                return

            if str(js.get("code", "")) not in ("", "00000"):
                logger.warning("[META] contracts response code=%s msg=%s", js.get("code"), js.get("msg"))

            contracts = self._extract_contracts_list(js)
            if not contracts:
                logger.error("[META] empty contracts list: %s", js)
                return

            by_symbol: Dict[str, ContractMeta] = {}

            for c in contracts:
                sym = (c.get("symbol") or "").upper()
                if not sym:
                    continue

                price_place = _safe_int(c.get("pricePlace"), 6)
                qty_place = _safe_int(c.get("volumePlace"), 4)

                # ---- PRICE TICK (FIX) ----
                pe = _safe_float(c.get("priceEndStep"), 0.0)
                ps = _safe_float(c.get("priceStep"), 0.0)
                ts = _safe_float(c.get("tickSize"), 0.0)
                pt = _safe_float(c.get("priceTick"), 0.0)

                tick = 0.0
                if ps > 0:
                    tick = _tick_from_place_and_step(price_place, ps)
                elif pe > 0:
                    tick = _tick_from_place_and_step(price_place, pe)
                elif ts > 0:
                    tick = float(ts)
                elif pt > 0:
                    # accept only if plausible
                    base = (10.0 ** (-price_place)) if price_place > 0 else 1.0
                    if pt < 1.0:
                        tick = float(pt)
                    else:
                        tick = float(base)

                if tick <= 0:
                    tick = float(10 ** (-max(0, price_place)))

                # ---- QTY STEP (FIX) ----
                vol_step = _safe_float(c.get("volumeStep"), 0.0)
                size_mult = _safe_float(c.get("sizeMultiplier"), 0.0)
                min_trade = _safe_float(c.get("minTradeNum"), 0.0)

                step = 0.0
                if vol_step > 0:
                    step = float(vol_step)
                elif size_mult > 0:
                    step = float(size_mult)
                else:
                    step = float(10 ** (-max(0, qty_place))) if qty_place > 0 else 1.0

                min_qty = float(min_trade) if min_trade > 0 else float(step)
                if min_qty < step:
                    min_qty = float(step)

                by_symbol[sym] = ContractMeta(
                    symbol=sym,
                    price_place=int(price_place),
                    price_tick=float(tick),
                    qty_place=int(qty_place),
                    qty_step=float(step),
                    min_qty=float(min_qty),
                    raw=c,
                )

            self._by_symbol = by_symbol
            self._ts = now
            logger.info("[META] refreshed contracts cache (%d symbols)", len(by_symbol))

    async def get(self, symbol: str) -> Optional[ContractMeta]:
        sym = (symbol or "").upper()
        await self.refresh(force=False)
        m = self._by_symbol.get(sym)
        if m is None:
            # force refresh once (new listing / cache stale)
            await self.refresh(force=True)
            m = self._by_symbol.get(sym)
        return m


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
        self._exec_log_enable = bool(EXEC_LOG_ENABLE)
        self._exec_log_path = str(EXEC_LOG_PATH or "exec_log.jsonl")

    # ----------------------------
    # Compat / helpers
    # ----------------------------

    @staticmethod
    def _symbol(symbol: str) -> str:
        return (symbol or "").upper().replace("-", "").replace("_", "")

    async def _call(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        auth: bool = True,
        timeout: int = 12,
    ) -> Dict[str, Any]:
        # Backward-compatible wrapper
        if (method or "GET").upper() == "GET":
            return await self._request_any_status("GET", path, params=params, timeout=timeout, auth=auth)
        return await self._request_any_status("POST", path, data=payload, timeout=timeout, auth=auth)

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
            items = sorted(params.items(), key=lambda kv: kv[0])
            query = "?" + "&".join(f"{k}={v}" for k, v in items)

        url = self.BASE + path + query
        body = json.dumps(data, separators=(",", ":")) if data else ""

        ts = str(_now_ms())
        headers: Dict[str, str] = {"Content-Type": "application/json"}

        if auth:
            # relies on your bitget_client._sign implementation
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
            return float(_estimate_tick_from_price(1.0))
        t = float(meta.price_tick or (10 ** (-max(0, meta.price_place))))
        return t if t > 0 else float(_estimate_tick_from_price(1.0))

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
                "sizeMultiplier": meta.raw.get("sizeMultiplier"),
            },
        }

    # ----------------------------
    # Quantize + formatting
    # ----------------------------

    async def _quantize_price_qty(
        self,
        symbol: str,
        price: float,
        qty: float,
        *,
        side: Optional[str] = None,
        close_side: Optional[str] = None,
        is_trigger: bool = False,
        tick_hint: Optional[float] = None,
    ) -> Tuple[float, float, float, float, int, int]:
        meta = await self._meta.get(symbol)
        if not meta:
            tick = float(tick_hint or _estimate_tick_from_price(price))
            step = 1.0
            q_price = float(price)
            q_qty = _floor_step(float(qty), step)
            return q_price, q_qty, tick, step, 6, 0

        meta_tick = float(meta.price_tick or (10 ** (-max(0, meta.price_place))))
        tick = float(tick_hint or meta_tick)
        step = float(meta.qty_step or 1.0)

        p = float(price)
        if tick > 0:
            if is_trigger:
                # for stop triggers, we quantize based on the close side (more conservative)
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
                    q_price = float(round(p / tick) * tick)
        else:
            q_price = p

        q = float(qty)
        q_qty = _floor_step(q, step) if step > 0 else q

        # hard min qty gate
        if q_qty < float(meta.min_qty or 0.0):
            q_qty = 0.0

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
        meta = await self._meta.get(symbol)
        p = float(price)

        if force_decimals is not None:
            return _fmt_decimal(p, int(force_decimals))

        if not meta:
            d = _decimals_from_step(tick_used or _estimate_tick_from_price(p))
            return _fmt_decimal(p, d)

        if meta.price_place >= 0:
            return _fmt_decimal(p, max(0, meta.price_place))

        d = _decimals_from_step(tick_used or meta.price_tick)
        return _fmt_decimal(p, d)

    def _format_qty_str(self, qty: float, qty_place: int) -> str:
        d = max(0, min(12, int(qty_place)))
        return _fmt_decimal(float(qty), d)

    async def _log_exec_event(
        self,
        event: str,
        *,
        payload: Dict[str, Any],
        resp: Dict[str, Any],
        symbol: Optional[str] = None,
    ) -> None:
        if not self._exec_log_enable or not self._exec_log_path:
            return

        record = {
            "ts_ms": _now_ms(),
            "event": str(event),
            "symbol": str(symbol or payload.get("symbol") or "").upper(),
            "ok": bool(resp.get("ok")),
            "code": str(resp.get("code", "")),
            "msg": str(resp.get("msg", "")),
            "latency_ms": int(resp.get("_latency_ms") or 0),
            "path": str(resp.get("_path") or ""),
            "payload": {
                "side": payload.get("side"),
                "orderType": payload.get("orderType"),
                "size": payload.get("size"),
                "price": payload.get("price"),
                "tradeSide": payload.get("tradeSide"),
                "reduceOnly": payload.get("reduceOnly"),
                "clientOid": payload.get("clientOid"),
                "triggerPrice": payload.get("triggerPrice"),
                "triggerType": payload.get("triggerType"),
                "planType": payload.get("planType"),
                "executePrice": payload.get("executePrice"),
                "force": payload.get("force"),
                "timeInForceValue": payload.get("timeInForceValue"),
            },
            "data": resp.get("data"),
        }

        def _write() -> None:
            try:
                with open(self._exec_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                return

        await asyncio.to_thread(_write)

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
        post_only: bool = False,
        tick_hint: Optional[float] = None,
        debug_tag: str = "ENTRY",
        # optional: attach SL/TP directly on entry
        preset_stop_loss: Optional[float] = None,
        preset_take_profit: Optional[float] = None,
    ) -> Dict[str, Any]:
        sym = self._symbol(symbol)
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

        price_str = await self._format_price(sym, q_price, tick_used=tick_used)
        size_str = self._format_qty_str(q_qty, qp)
        oid = client_oid or f"oid-{sym}-{_now_ms()}"

        payload: Dict[str, Any] = {
            "symbol": sym,
            "productType": self.product_type,
            "marginCoin": self.margin_coin,
            "marginMode": self.margin_mode,
            "size": size_str,
            "price": price_str,
            "side": s,
            "orderType": "limit",
            "force": "post_only" if post_only else "gtc",
            "timeInForceValue": "post_only" if post_only else "normal",
            "clientOid": oid,
            "tradeSide": (trade_side or "open").lower(),
        }
        payload_used = dict(payload)
        if reduce_only:
            payload["reduceOnly"] = "YES"
            payload_used["reduceOnly"] = "YES"

        if payload["tradeSide"] == "open":
            if preset_stop_loss is not None:
                payload["presetStopLossPrice"] = await self._format_price(sym, float(preset_stop_loss), tick_used=tick_used)
            if preset_take_profit is not None:
                payload["presetStopSurplusPrice"] = await self._format_price(sym, float(preset_take_profit), tick_used=tick_used)

        logger.info(
            "[ORDER_%s] sym=%s side=%s tradeSide=%s reduceOnly=%s raw_price=%s q_price=%s price_str=%s raw_qty=%s q_qty=%s tick_used=%s",
            debug_tag, sym, s, payload["tradeSide"], payload.get("reduceOnly"),
            float(price), q_price, price_str, float(raw_qty), q_qty, tick_used
        )

        async def _send(data_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            return await self._request_any_status("POST", "/api/v2/mix/order/place-order", data=(data_override or payload), auth=True)

        resp = await _send()

        # 22047 => clamp band retry
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
                    payload_used = dict(payload2)
                    resp["_band"] = {"min": mn, "max": mx, "before": q_price, "after": q_price2}

        # 40020 => force decimals from tick retry
        code = str(resp.get("code", ""))
        if (not _is_ok(resp)) and code in _ERR_PRICE_PRECISION:
            d = _decimals_from_step(tick_used or _estimate_tick_from_price(q_price))
            price_str2 = await self._format_price(sym, q_price, tick_used=tick_used, force_decimals=d)
            payload2 = dict(payload)
            payload2["price"] = price_str2
            logger.warning("[ORDER_%s] 40020 retry force_decimals=%s price_str=%s->%s", debug_tag, d, price_str, price_str2)
            resp = await _send(payload2)
            payload_used = dict(payload2)

        # 40762 => downsize retry (ENTRY only, when size=None and not reduce-only)
        code = str(resp.get("code", ""))
        if (not _is_ok(resp)) and code in _ERR_BALANCE and payload.get("tradeSide") == "open" and (not reduce_only) and size is None:
            downs = []
            cur_qty = float(raw_qty)
            for k in range(3):
                cur_qty *= 0.85
                q_price2, q_qty2, tick2, _step2, _pp2, qp2 = await self._quantize_price_qty(
                    sym, float(price), float(cur_qty), side=s.upper(), is_trigger=False, tick_hint=tick_used
                )
                if q_qty2 <= 0:
                    break
                payload2 = dict(payload)
                payload2["size"] = self._format_qty_str(q_qty2, qp2)
                downs.append({"try": k + 1, "qty": q_qty2})
                logger.warning("[ORDER_%s] 40762 downsize retry try=%s qty=%s", debug_tag, k + 1, q_qty2)
                resp = await _send(payload2)
                payload_used = dict(payload2)
                if _is_ok(resp):
                    resp["_downsized"] = downs
                    resp["qty"] = float(q_qty2)
                    resp["price"] = float(q_price2)
                    break
            if (not _is_ok(resp)) and downs:
                resp["_downsized"] = downs

        # Attach debug hints + min_usdt/band parsing
        if not _is_ok(resp):
            code = str(resp.get("code", ""))
            msg = str(resp.get("msg") or "")
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
                "tradeSide": payload.get("tradeSide"),
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
            resp["qty"] = float(resp.get("qty") or q_qty)
            resp["price"] = float(resp.get("price") or q_price)
            resp["_debug"] = {
                "debug_tag": debug_tag,
                "price_str": str(payload.get("price")),
                "tick_used": float(tick_used),
                "clientOid": oid,
            }

        await self._log_exec_event(
            "place_limit",
            payload=payload_used,
            resp=resp,
            symbol=sym,
        )
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
        sym = self._symbol(symbol)
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
        payload_used = dict(payload)

        logger.info(
            "[ORDER_%s] sym=%s close_side=%s trigger_type=%s trigger_raw=%s trig_q=%s trig_str=%s qty=%s tick_used=%s",
            debug_tag, sym, close_s, trigger_type, float(trigger_price), q_trig, trig_str, q_qty, tick_used
        )

        async def _send(data_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            return await self._request_any_status("POST", "/api/v2/mix/order/place-plan-order", data=(data_override or payload), auth=True)

        resp = await _send()

        # band clamp retry
        code = str(resp.get("code", ""))
        msg = str(resp.get("msg") or "")
        if (not _is_ok(resp)) and code in _ERR_PRICE_BAND:
            mn, mx = _parse_band(msg)
            clamped = _clamp_band(q_trig, tick_used, mn, mx)
            if clamped is not None:
                q_trig2, q_qty2, tick2, _step2, _pp2, qp2 = await self._quantize_price_qty(
                    sym, float(clamped), float(qty),
                    close_side=close_u, is_trigger=True, tick_hint=tick_used
                )
                if q_qty2 > 0:
                    trig_str2 = await self._format_price(sym, q_trig2, tick_used=tick2)
                    payload2 = dict(payload)
                    payload2["triggerPrice"] = trig_str2
                    payload2["size"] = self._format_qty_str(q_qty2, qp2)
                    logger.warning("[ORDER_%s] band clamp retry mn=%s mx=%s before=%s after=%s", debug_tag, mn, mx, q_trig, q_trig2)
                    resp = await _send(payload2)
                    payload_used = dict(payload2)
                    resp["_band"] = {"min": mn, "max": mx, "before": q_trig, "after": q_trig2}

        # price precision retry
        code = str(resp.get("code", ""))
        if (not _is_ok(resp)) and code in _ERR_PRICE_PRECISION:
            d = _decimals_from_step(tick_used or _estimate_tick_from_price(q_trig))
            trig_str2 = await self._format_price(sym, q_trig, tick_used=tick_used, force_decimals=d)
            payload2 = dict(payload)
            payload2["triggerPrice"] = trig_str2
            logger.warning("[ORDER_%s] 40020 retry force_decimals=%s trig_str=%s->%s", debug_tag, d, trig_str, trig_str2)
            resp = await _send(payload2)
            payload_used = dict(payload2)

        if not _is_ok(resp):
            code = str(resp.get("code", ""))
            msg = str(resp.get("msg") or "")
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

        await self._log_exec_event(
            "place_stop_market_sl",
            payload=payload_used,
            resp=resp,
            symbol=sym,
        )
        return resp

    # ----------------------------
    # Cancels (FIXED)
    # ----------------------------

    async def cancel_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_oid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Cancel a normal (non-plan) order by orderId or clientOid.
        """
        sym = self._symbol(symbol)
        payload: Dict[str, Any] = {
            "symbol": sym,
            "productType": self.product_type,
            "marginCoin": self.margin_coin,
        }
        if order_id:
            payload["orderId"] = str(order_id)
        if client_oid:
            payload["clientOid"] = str(client_oid)

        if not payload.get("orderId") and not payload.get("clientOid"):
            return {"ok": False, "code": "NOID", "msg": "order_id or client_oid required"}

        return await self._request_any_status("POST", "/api/v2/mix/order/cancel-order", data=payload, auth=True, timeout=6)

    async def cancel_plan_orders(
        self,
        symbol: str,
        order_ids: List[Union[str, Dict[str, Any]]],
        *,
        plan_type: str = "normal_plan",
    ) -> Dict[str, Any]:
        """
        Cancel trigger/plan orders.
        Bitget v2 requires orderIdList to be a list of objects.
        Each object: {"orderId":"...", "clientOid":"..."} (either one is ok).
        """
        sym = self._symbol(symbol)
        items: List[Dict[str, str]] = []

        for x in (order_ids or []):
            if isinstance(x, dict):
                oid = str(x.get("orderId") or "")
                coid = str(x.get("clientOid") or "")
                if oid or coid:
                    items.append({"orderId": oid, "clientOid": coid})
            else:
                s = str(x)
                if s:
                    items.append({"orderId": s, "clientOid": ""})

        if not items:
            return {"ok": False, "code": "NOIDS", "msg": "order_ids empty"}

        payload = {
            "symbol": sym,
            "productType": self.product_type,
            "planType": plan_type,
            "orderIdList": items,
        }
        return await self._request_any_status("POST", "/api/v2/mix/order/cancel-plan-order", data=payload, auth=True, timeout=8)

    # ----------------------------
    # Queries
    # ----------------------------

    async def get_order_detail(
        self,
        symbol: str,
        *,
        order_id: Optional[str] = None,
        client_oid: Optional[str] = None,
    ) -> Dict[str, Any]:
        sym = self._symbol(symbol)
        if not order_id and not client_oid:
            return {"ok": False, "code": "NOID", "msg": "order_id or client_oid required"}

        params: Dict[str, Any] = {
            "symbol": sym,
            "productType": self.product_type,
            "marginCoin": self.margin_coin,
        }
        if order_id:
            params["orderId"] = str(order_id)
        if client_oid:
            params["clientOid"] = str(client_oid)

        return await self._request_any_status("GET", "/api/v2/mix/order/detail", params=params, auth=True)

    @staticmethod
    def _data_row(order_detail_resp: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(order_detail_resp, dict):
            return {}
        data = order_detail_resp.get("data")
        if isinstance(data, dict):
            # sometimes nested lists exist, but detail is typically a dict
            return data
        if isinstance(data, list):
            for it in data:
                if isinstance(it, dict):
                    return it
        return {}

    @staticmethod
    def filled_qty(order_detail_resp: Dict[str, Any]) -> float:
        if not _is_ok(order_detail_resp):
            return 0.0
        data = BitgetTrader._data_row(order_detail_resp)
        return _safe_float(
            data.get("baseVolume")
            or data.get("filledQty")
            or data.get("filledSize")
            or data.get("filledVolume")
            or data.get("dealSize"),
            0.0,
        )

    @staticmethod
    def order_size(order_detail_resp: Dict[str, Any]) -> float:
        if not _is_ok(order_detail_resp):
            return 0.0
        data = BitgetTrader._data_row(order_detail_resp)
        return _safe_float(
            data.get("size")
            or data.get("quantity")
            or data.get("qty")
            or data.get("totalSize"),
            0.0,
        )

    @staticmethod
    def remaining_qty(order_detail_resp: Dict[str, Any]) -> float:
        size = BitgetTrader.order_size(order_detail_resp)
        filled = BitgetTrader.filled_qty(order_detail_resp)
        return max(0.0, float(size) - float(filled))

    @staticmethod
    def is_filled(order_detail_resp: Dict[str, Any]) -> bool:
        """
        IMPORTANT for your watcher:
        - returns True when fully filled
        - ALSO returns True when partially filled (filled > 0) so the watcher can arm SL/TP
          as soon as a position exists (prevents “partial fill unprotected”).
        """
        if not _is_ok(order_detail_resp):
            return False

        data = BitgetTrader._data_row(order_detail_resp)
        state = str(data.get("state") or data.get("status") or "").lower()

        if state in {"filled", "full_fill", "fullfill", "completed", "success"}:
            return True

        filled = BitgetTrader.filled_qty(order_detail_resp)
        size = BitgetTrader.order_size(order_detail_resp)

        # full numeric fill
        if size > 0 and filled >= size * 0.999:
            return True

        # partial fill -> treat as "filled enough" to arm protection
        if filled > 0:
            return True

        return False

    # ----------------------------
    # Emergency close
    # ----------------------------

    async def flash_close_position(
        self,
        symbol: str,
        *,
        hold_side: Optional[str] = None,  # "long" / "short" in hedge mode; None closes all sides
        timeout: int = 10,
    ) -> Dict[str, Any]:
        """
        Close position at market price (Bitget v2: /api/v2/mix/order/close-positions).
        """
        sym = self._symbol(symbol)
        payload: Dict[str, Any] = {"symbol": sym, "productType": self.product_type}
        if hold_side:
            payload["holdSide"] = str(hold_side).lower()
        return await self._request_any_status("POST", "/api/v2/mix/order/close-positions", data=payload, auth=True, timeout=timeout)

    # ----------------------------
    # Watcher / state snapshots (NEW)
    # ----------------------------

    @staticmethod
    def _extract_list(js: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Some endpoints return data=list, others return data={list:[...]}.
        We normalize to list[dict].
        """
        if not isinstance(js, dict):
            return []
        data = js.get("data")
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            for k in ("list", "data", "rows"):
                v = data.get(k)
                if isinstance(v, list):
                    return [x for x in v if isinstance(x, dict)]
        return []

    async def get_all_positions(self, *, timeout: int = 10) -> Dict[str, Any]:
        """
        GET /api/v2/mix/position/all-position
        """
        params = {"productType": self.product_type, "marginCoin": self.margin_coin}
        return await self._request_any_status("GET", "/api/v2/mix/position/all-position", params=params, auth=True, timeout=timeout)

    async def get_single_position(self, symbol: str, *, timeout: int = 10) -> Dict[str, Any]:
        """
        GET /api/v2/mix/position/single-position
        """
        sym = self._symbol(symbol)
        params = {"productType": self.product_type, "marginCoin": self.margin_coin, "symbol": sym}
        return await self._request_any_status("GET", "/api/v2/mix/position/single-position", params=params, auth=True, timeout=timeout)

    async def get_pending_orders(
        self,
        *,
        symbol: Optional[str] = None,
        limit: int = 100,
        id_less_than: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        timeout: int = 10,
    ) -> Dict[str, Any]:
        """
        GET /api/v2/mix/order/orders-pending
        """
        params: Dict[str, Any] = {"productType": self.product_type}
        if symbol:
            params["symbol"] = self._symbol(symbol)
        if limit:
            params["limit"] = int(limit)
        if id_less_than:
            params["idLessThan"] = str(id_less_than)
        if start_time:
            params["startTime"] = int(start_time)
        if end_time:
            params["endTime"] = int(end_time)
        return await self._request_any_status("GET", "/api/v2/mix/order/orders-pending", params=params, auth=True, timeout=timeout)

    async def get_pending_plan_orders(
        self,
        *,
        plan_type: str = "normal_plan",
        symbol: Optional[str] = None,
        limit: int = 100,
        id_less_than: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        timeout: int = 10,
    ) -> Dict[str, Any]:
        """
        GET /api/v2/mix/order/orders-plan-pending
        """
        params: Dict[str, Any] = {"productType": self.product_type, "planType": str(plan_type)}
        if symbol:
            params["symbol"] = self._symbol(symbol)
        if limit:
            params["limit"] = int(limit)
        if id_less_than:
            params["idLessThan"] = str(id_less_than)
        if start_time:
            params["startTime"] = int(start_time)
        if end_time:
            params["endTime"] = int(end_time)
        return await self._request_any_status("GET", "/api/v2/mix/order/orders-plan-pending", params=params, auth=True, timeout=timeout)

    async def get_order_fills(
        self,
        *,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        limit: int = 100,
        id_less_than: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        timeout: int = 10,
    ) -> Dict[str, Any]:
        """
        GET /api/v2/mix/order/fills
        """
        params: Dict[str, Any] = {"productType": self.product_type}
        if symbol:
            params["symbol"] = self._symbol(symbol)
        if order_id:
            params["orderId"] = str(order_id)
        if limit:
            params["limit"] = int(limit)
        if id_less_than:
            params["idLessThan"] = str(id_less_than)
        if start_time:
            params["startTime"] = int(start_time)
        if end_time:
            params["endTime"] = int(end_time)
        return await self._request_any_status("GET", "/api/v2/mix/order/fills", params=params, auth=True, timeout=timeout)

    async def get_ticker(
        self,
        symbol: str,
        *,
        timeout: int = 8,
    ) -> Dict[str, Any]:
        """
        GET /api/v2/mix/market/ticker (public)
        """
        params = {"productType": self.product_type, "symbol": self._symbol(symbol)}
        return await self._request_any_status("GET", "/api/v2/mix/market/ticker", params=params, auth=False, timeout=timeout)
