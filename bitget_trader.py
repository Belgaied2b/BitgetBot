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
#
# PATCH (2026-01):
# - Fix 45110 on ENTRY when size=None: qty quantization floors to step and can drop notional < MIN_ENTRY_USDT.
#   We bump q_qty using CEIL to qtyStep so q_price*q_qty >= MIN_ENTRY_USDT (default 5).
#   Only applies when size is None AND tradeSide=open AND not reduce-only.
# =====================================================================

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
import os
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
        self._ts: Dict[str, float] = {}
        self._cache: Dict[str, ContractMeta] = {}
        self._lock = asyncio.Lock()

    async def get(self, symbol: str) -> Optional[ContractMeta]:
        sym = (symbol or "").strip().upper()
        if not sym:
            return None

        now = time.time()
        try:
            if sym in self._cache and (now - float(self._ts.get(sym, 0.0))) < float(self.ttl_s):
                return self._cache.get(sym)
        except Exception:
            pass

        async with self._lock:
            now = time.time()
            try:
                if sym in self._cache and (now - float(self._ts.get(sym, 0.0))) < float(self.ttl_s):
                    return self._cache.get(sym)
            except Exception:
                pass

            try:
                resp = await self.client.get_contracts()
            except Exception:
                return self._cache.get(sym)

            data = None
            if isinstance(resp, dict):
                data = resp.get("data")
            contracts: List[Dict[str, Any]] = []

            # Bitget v2 can return: data=list or data={"list":[...]}
            if isinstance(data, list):
                contracts = [x for x in data if isinstance(x, dict)]
            elif isinstance(data, dict):
                lst = data.get("list") or data.get("data") or data.get("result") or []
                if isinstance(lst, list):
                    contracts = [x for x in lst if isinstance(x, dict)]

            found = None
            for c in contracts:
                s = str(c.get("symbol") or c.get("instId") or "").upper()
                if s == sym:
                    found = c
                    break

            if not found:
                return self._cache.get(sym)

            price_place = _safe_int(found.get("pricePlace"), 6)
            qty_place = _safe_int(found.get("volumePlace") or found.get("qtyPlace"), 0)

            # price tick: try priceEndStep/priceStep/tickSize etc.
            pe = found.get("priceEndStep")
            ps = found.get("priceStep")
            ts = found.get("tickSize")
            step_val = None
            for v in (pe, ps, ts):
                try:
                    if v is None:
                        continue
                    step_val = float(v)
                    break
                except Exception:
                    continue

            if step_val is None:
                step_val = 1.0

            price_tick = _tick_from_place_and_step(price_place, float(step_val))

            # volume step: try volumeStep; fallback 1
            vs = found.get("volumeStep") or found.get("qtyStep")
            qty_step = None
            try:
                qty_step = float(vs)
            except Exception:
                qty_step = 1.0
            if qty_step is None or qty_step <= 0:
                qty_step = 1.0

            # min qty: minTradeNum
            min_trade_num = found.get("minTradeNum") or found.get("minQty") or found.get("minSize")
            min_qty = 0.0
            try:
                min_qty = float(min_trade_num)
            except Exception:
                min_qty = 0.0

            meta = ContractMeta(
                symbol=sym,
                price_place=int(price_place),
                price_tick=float(price_tick),
                qty_place=int(qty_place),
                qty_step=float(qty_step),
                min_qty=float(min_qty),
                raw=dict(found),
            )
            self._cache[sym] = meta
            self._ts[sym] = float(now)
            return meta


# ---------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------

class BitgetTrader:
    def __init__(
        self,
        client,
        *,
        product_type: str = PRODUCT_TYPE,
        margin_coin: str = MARGIN_COIN,
        leverage: float = 10.0,
        margin_mode: str = "isolated",
        margin_usdt: float = 0.5,
    ):
        self.client = client
        self.product_type = str(product_type)
        self.margin_coin = str(margin_coin)
        self.leverage = float(leverage)
        self.margin_mode = str(margin_mode)
        self.margin_usdt = float(margin_usdt)

        self._meta = ContractMetaCache(client)
        self._exec_log_enable = bool(EXEC_LOG_ENABLE)
        self._exec_log_path = str(EXEC_LOG_PATH or "").strip()

    def _symbol(self, symbol: str) -> str:
        return (symbol or "").strip().upper()

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

    # ----------------------------
    # Low-level request wrapper (must exist on your client)
    # ----------------------------

    async def _request_any_status(self, method: str, path: str, *, params: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None, auth: bool = False) -> Dict[str, Any]:
        """
        This method expects your client to implement a request function.
        Your repo already uses this pattern; keep as-is.
        """
        return await self.client._request(method, path, params=params, data=data, auth=auth)  # type: ignore[attr-defined]

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

        # -----------------------------------------------------------------
        # Preflight (ENTRY only): avoid Bitget 45110 "less than the minimum amount X USDT"
        #
        # Root cause we observed in Railway logs:
        # - scanner targets notional ~= MIN_ENTRY_USDT (default 5)
        # - Bitget requires >= 5 USDT notional
        # - our qty quantization floors to qty_step (often 1 contract),
        #   which can push (q_price * q_qty) just below the minimum (e.g. 4.85 USDT).
        #
        # Fix:
        # - only when size is None (auto sizing) AND we are opening (trade_side=open) AND not reduce-only,
        #   we bump q_qty up using CEIL to qty_step so that q_price*q_qty >= MIN_ENTRY_USDT.
        # - We DO NOT do this when size is explicitly provided, because the caller may be intentionally downsizing
        #   (e.g., retries after 40762 balance error).
        # -----------------------------------------------------------------
        try:
            if (size is None) and (not reduce_only) and (str(trade_side or "open").lower() == "open"):
                min_usdt_env = float(os.getenv("MIN_ENTRY_USDT", "5") or "5")
                if min_usdt_env > 0 and (float(q_price) * float(q_qty) + 1e-12) < float(min_usdt_env):
                    meta2 = await self._meta.get(sym)
                    step2 = float(meta2.qty_step or 1.0) if meta2 else float(_step or 1.0)
                    min_qty2 = float(meta2.min_qty or 0.0) if meta2 else 0.0
                    qty_place2 = int(meta2.qty_place) if meta2 else int(qp)

                    req_qty = _ceil_step(float(min_usdt_env) / max(1e-12, float(q_price)), step2)
                    if min_qty2 > 0:
                        req_qty = max(float(req_qty), float(min_qty2))

                    req_qty = float(_fmt_decimal(float(req_qty), max(0, qty_place2)))

                    if req_qty > float(q_qty) + 1e-12:
                        q_qty = float(req_qty)
        except Exception:
            pass

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

        # 40020 => retry with forced decimals
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

    # =================================================================
    # The rest of the file (place_plan, cancel, positions, ticker, etc.)
    # is unchanged from your uploaded version.
    #
    # NOTE:
    # I’m keeping the remainder intact to avoid accidental regressions.
    # If you want, I can also paste the remainder verbatim, but it’s
    # already present in your repo and not related to 45110.
    # =================================================================
