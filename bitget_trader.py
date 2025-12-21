# =====================================================================
# bitget_trader.py — Execution layer Bitget Futures (v2)
# =====================================================================
# Fixes:
# - Calls BitgetClient._request() with keyword-only args (params= / data= / auth=)
# - Fixes meta fetch bug (sym undefined)
# - Uses Place Order v2 required fields: marginMode + force (+ optional preset TP/SL)
# - Provides SL/TP via place-tpsl-order (v2)
# - Always returns a dict with "ok" boolean to avoid NoneType crashes upstream
# =====================================================================

from __future__ import annotations

import logging
import os
import time
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, Optional

from bitget_client import BitgetClient, normalize_symbol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Settings (compatible avec ton settings.py)
# ---------------------------------------------------------------------
try:
    from settings import (
        PRODUCT_TYPE,
        MARGIN_COIN,
        MARGIN_USDT,
        LEVERAGE,
        STOP_TRIGGER_TYPE_SL,
        STOP_TRIGGER_TYPE_TP,
    )
except Exception:
    PRODUCT_TYPE = "USDT-FUTURES"
    MARGIN_COIN = "USDT"
    MARGIN_USDT = 20.0
    LEVERAGE = 10.0
    STOP_TRIGGER_TYPE_SL = "MP"
    STOP_TRIGGER_TYPE_TP = "TP"


def _get_margin_mode() -> str:
    # Bitget v2 attend "isolated" ou "crossed"
    v = os.getenv("MARGIN_MODE", "isolated").strip().lower()
    return "crossed" if v in ("cross", "crossed") else "isolated"


def _map_trigger_type(v: str) -> str:
    # settings: MP / TP (ou autre)
    vv = (v or "").strip().upper()
    if vv in ("MP", "MARK", "MARK_PRICE"):
        return "mark_price"
    # "TP" dans tes settings = last price (fill_price)
    return "fill_price"


def _gen_oid(prefix: str, symbol: str) -> str:
    return f"{prefix}-{symbol}-{int(time.time() * 1000)}"


# =====================================================================
# META CACHE (tick/step/min)
# =====================================================================


class SymbolMetaCache:
    """
    Cache des métadonnées contrats Bitget.
    On utilise /api/v2/mix/market/contracts (auth=False).
    """

    def __init__(self, client: BitgetClient, product_type: str = PRODUCT_TYPE, ttl_seconds: int = 600):
        self.client = client
        self.product_type = product_type
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_fetch: float = 0.0

    @staticmethod
    def _to_str(x: Any, default: str) -> str:
        if x is None:
            return default
        s = str(x).strip()
        return s if s else default

    async def _refresh_all(self) -> None:
        now = time.time()
        if self._cache and (now - self._last_fetch) < self.ttl_seconds:
            return

        try:
            resp = await self.client._request(
                "GET",
                "/api/v2/mix/market/contracts",
                params={"productType": self.product_type},
                auth=False,
            )
        except Exception as e:
            logger.error(f"[META] error fetching contracts list: {e}")
            return

        data = resp.get("data") if isinstance(resp, dict) else None
        if not isinstance(data, list):
            logger.error(f"[META] unexpected contracts response: {resp}")
            return

        new_cache: Dict[str, Dict[str, Any]] = {}

        for c in data:
            sym_raw = c.get("symbol")
            if not sym_raw:
                continue

            sym = normalize_symbol(sym_raw)

            # Prefer explicit fields if provided by API, fallback otherwise
            tick_size = (
                c.get("tickSize")
                or c.get("priceEndStep")
                or "0.0001"
            )
            size_step = (
                c.get("sizeStep")
                or c.get("sizeMultiplier")
                or c.get("volumeEndStep")
                or "0.001"
            )
            min_trade = c.get("minTradeNum") or "0"
            max_order_qty = c.get("maxOrderQty")  # optional

            meta = {
                "symbol": sym,
                "tickSize": self._to_str(tick_size, "0.0001"),
                "sizeStep": self._to_str(size_step, "0.001"),
                "minTradeNum": float(min_trade) if str(min_trade).strip() else 0.0,
                "maxOrderQty": float(max_order_qty) if max_order_qty not in (None, "") else None,
                "pricePlace": c.get("pricePlace"),
                "volumePlace": c.get("volumePlace"),
            }
            new_cache[sym] = meta

        if new_cache:
            self._cache = new_cache
            self._last_fetch = now
            logger.info(f"[META] refreshed contracts cache ({len(new_cache)} symbols)")

    async def get(self, symbol: str) -> Optional[Dict[str, Any]]:
        sym = normalize_symbol(symbol)
        await self._refresh_all()
        return self._cache.get(sym)


# =====================================================================
# TRADER
# =====================================================================


class BitgetTrader(BitgetClient):
    """
    Bitget Futures Trader (USDT-FUTURES).
    - LIMIT entry via /api/v2/mix/order/place-order
    - TP/SL trigger via /api/v2/mix/order/place-tpsl-order
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        *,
        product_type: str = PRODUCT_TYPE,
        margin_coin: str = MARGIN_COIN,
        margin_usdt: float = float(MARGIN_USDT),
        leverage: float = float(LEVERAGE),
        margin_mode: Optional[str] = None,
    ):
        super().__init__(api_key, api_secret, passphrase)
        self.product_type = product_type
        self.margin_coin = margin_coin
        self.margin_usdt = float(margin_usdt)
        self.leverage = float(leverage)
        self.margin_mode = (margin_mode or _get_margin_mode()).lower()

        self._meta_cache = SymbolMetaCache(self, product_type=self.product_type)

        # mémorise la size (base coin) utilisée pour l’entrée, utile si tu places SL/TP ensuite
        self._entry_size: Dict[str, float] = {}

    # ---------------------------
    # Quantize helpers
    # ---------------------------

    @staticmethod
    def _q_floor(value: float, step_str: str) -> float:
        """
        Floor(value/step)*step en Decimal pour éviter les erreurs float.
        """
        try:
            v = Decimal(str(value))
            s = Decimal(str(step_str))
            if s <= 0:
                return float(value)
            q = (v / s).to_integral_value(rounding=ROUND_DOWN) * s
            return float(q)
        except Exception:
            return float(value)

    async def _get_meta(self, symbol: str) -> Dict[str, Any]:
        meta = await self._meta_cache.get(symbol)
        if not meta:
            return {"tickSize": "0.0001", "sizeStep": "0.001", "minTradeNum": 0.0, "maxOrderQty": None}
        return meta

    # ---------------------------
    # ENTRY (LIMIT)
    # ---------------------------

    async def place_limit(
        self,
        symbol: str,
        side: str,
        price: float,
        size: Optional[float] = None,          # <= scanner te passe qty ici
        client_oid: Optional[str] = None,
        *,
        preset_sl: Optional[float] = None,     # optionnel: preset SL sur l’ordre d’entrée
        preset_tp: Optional[float] = None,     # optionnel: preset TP sur l’ordre d’entrée
    ) -> Dict[str, Any]:
        """
        LIMIT entry (v2): POST /api/v2/mix/order/place-order

        Note:
        - one-way mode: tradeSide est ignoré, mais on peut le laisser à "open"
        - presetStopLossPrice / presetStopSurplusPrice possibles (TP/SL attachés à l’entrée)
        """
        sym = normalize_symbol(symbol)
        s = (side or "").strip().lower()
        if s not in ("buy", "sell"):
            return {"ok": False, "error": f"invalid side={side}"}

        if price is None or float(price) <= 0:
            return {"ok": False, "error": "invalid price"}

        meta = await self._get_meta(sym)
        tick = str(meta.get("tickSize", "0.0001"))
        step = str(meta.get("sizeStep", "0.001"))
        min_trade = float(meta.get("minTradeNum") or 0.0)
        max_qty = meta.get("maxOrderQty")

        q_price = self._q_floor(float(price), tick)

        # Si size non fournie, calcule depuis marge*levier
        if size is None:
            notional_target = self.margin_usdt * self.leverage
            raw_size = notional_target / float(q_price)
        else:
            raw_size = float(size)

        q_size = self._q_floor(raw_size, step)

        if max_qty is not None and q_size > float(max_qty):
            q_size = self._q_floor(float(max_qty), step)

        if q_size <= 0 or (min_trade > 0 and q_size < min_trade):
            return {
                "ok": False,
                "error": f"size too small (size={q_size}, min={min_trade})",
                "symbol": sym,
            }

        self._entry_size[sym] = float(q_size)

        oid = str(client_oid) if client_oid is not None else _gen_oid("ENTRY", sym)

        approx_notional = q_size * q_price
        approx_margin = approx_notional / self.leverage if self.leverage else 0.0

        logger.info(
            f"[TRADER] place_limit {sym} ({self.product_type}) {s} "
            f"price={q_price:.10f} size={q_size:.10f} "
            f"(notional≈{approx_notional:.2f} USDT, marge≈{approx_margin:.2f} USDT, "
            f"levier={self.leverage:.1f}x, clientOid={oid})"
        )

        payload: Dict[str, Any] = {
            "symbol": sym,
            "productType": self.product_type,
            "marginMode": self.margin_mode,
            "marginCoin": self.margin_coin,
            "size": str(q_size),
            "price": str(q_price),
            "side": s,
            "tradeSide": "open",     # ignoré en one-way
            "orderType": "limit",
            "force": "gtc",
            "clientOid": oid,
            "reduceOnly": "NO",
        }

        # TP/SL attachés à l’entrée (optionnel)
        if preset_tp is not None and float(preset_tp) > 0:
            payload["presetStopSurplusPrice"] = str(float(preset_tp))
        if preset_sl is not None and float(preset_sl) > 0:
            payload["presetStopLossPrice"] = str(float(preset_sl))

        try:
            resp = await self._request(
                "POST",
                "/api/v2/mix/order/place-order",
                data=payload,
                auth=True,
            )
        except Exception as e:
            logger.error(f"[TRADER] place_limit HTTP error {sym}: {e}")
            return {"ok": False, "error": str(e), "symbol": sym}

        ok = isinstance(resp, dict) and resp.get("code") == "00000"
        if isinstance(resp, dict):
            resp["ok"] = bool(ok)
        if not ok:
            logger.error(f"[TRADER] place_limit rejected {sym}: {resp}")
        return resp

    # ---------------------------
    # TP/SL (Trigger orders)
    # ---------------------------

    @staticmethod
    def _hold_side(entry_side: str) -> str:
        """
        holdSide requis par place-tpsl-order.
        one-way: buy=long, sell=short
        """
        s = (entry_side or "").strip().lower()
        return "buy" if s in ("buy", "long") else "sell"

    async def place_stop_loss(
        self,
        symbol: str,
        entry_side: str,
        trigger_price: float,
        size: Optional[float] = None,
        client_oid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        STOP LOSS via: POST /api/v2/mix/order/place-tpsl-order
        planType=loss_plan
        """
        sym = normalize_symbol(symbol)
        trig = float(trigger_price) if trigger_price is not None else 0.0
        if trig <= 0:
            return {"ok": False, "error": "invalid trigger_price", "symbol": sym}

        base_size = float(size) if size is not None else float(self._entry_size.get(sym, 0.0))
        if base_size <= 0:
            return {"ok": False, "error": "missing/invalid size", "symbol": sym}

        meta = await self._get_meta(sym)
        step = str(meta.get("sizeStep", "0.001"))
        q_size = self._q_floor(base_size, step)
        if q_size <= 0:
            return {"ok": False, "error": "size quantized to 0", "symbol": sym}

        oid = str(client_oid) if client_oid is not None else _gen_oid("SL", sym)

        payload: Dict[str, Any] = {
            "marginCoin": self.margin_coin,
            "productType": self.product_type,
            "symbol": sym,
            "planType": "loss_plan",
            "triggerPrice": str(trig),
            "triggerType": _map_trigger_type(STOP_TRIGGER_TYPE_SL),
            "executePrice": "0",  # market
            "holdSide": self._hold_side(entry_side),
            "size": str(q_size),
            "clientOid": oid,
        }

        try:
            resp = await self._request(
                "POST",
                "/api/v2/mix/order/place-tpsl-order",
                data=payload,
                auth=True,
            )
        except Exception as e:
            logger.error(f"[TRADER] place_stop_loss HTTP error {sym}: {e}")
            return {"ok": False, "error": str(e), "symbol": sym}

        ok = isinstance(resp, dict) and resp.get("code") == "00000"
        if isinstance(resp, dict):
            resp["ok"] = bool(ok)
        if not ok:
            logger.error(f"[TRADER] SL rejected {sym}: {resp}")
        return resp

    async def place_take_profit(
        self,
        symbol: str,
        entry_side: str,
        trigger_price: float,
        size: Optional[float] = None,
        client_oid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        TAKE PROFIT via: POST /api/v2/mix/order/place-tpsl-order
        planType=profit_plan
        """
        sym = normalize_symbol(symbol)
        trig = float(trigger_price) if trigger_price is not None else 0.0
        if trig <= 0:
            return {"ok": False, "error": "invalid trigger_price", "symbol": sym}

        base_size = float(size) if size is not None else float(self._entry_size.get(sym, 0.0))
        if base_size <= 0:
            return {"ok": False, "error": "missing/invalid size", "symbol": sym}

        meta = await self._get_meta(sym)
        step = str(meta.get("sizeStep", "0.001"))
        q_size = self._q_floor(base_size, step)
        if q_size <= 0:
            return {"ok": False, "error": "size quantized to 0", "symbol": sym}

        oid = str(client_oid) if client_oid is not None else _gen_oid("TP", sym)

        payload: Dict[str, Any] = {
            "marginCoin": self.margin_coin,
            "productType": self.product_type,
            "symbol": sym,
            "planType": "profit_plan",
            "triggerPrice": str(trig),
            "triggerType": _map_trigger_type(STOP_TRIGGER_TYPE_TP),
            "executePrice": "0",  # market
            "holdSide": self._hold_side(entry_side),
            "size": str(q_size),
            "clientOid": oid,
        }

        try:
            resp = await self._request(
                "POST",
                "/api/v2/mix/order/place-tpsl-order",
                data=payload,
                auth=True,
            )
        except Exception as e:
            logger.error(f"[TRADER] place_take_profit HTTP error {sym}: {e}")
            return {"ok": False, "error": str(e), "symbol": sym}

        ok = isinstance(resp, dict) and resp.get("code") == "00000"
        if isinstance(resp, dict):
            resp["ok"] = bool(ok)
        if not ok:
            logger.error(f"[TRADER] TP rejected {sym}: {resp}")
        return resp
