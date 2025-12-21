# =====================================================================
# bitget_trader.py — Bitget Futures Trader (USDT-FUTURES) — 2025
# =====================================================================
# Objectif:
# - Calculer une taille stable (marge fixe * levier) + quantize via meta contrats
# - Placer un LIMIT d'entrée via /api/v2/mix/order/place-order
# - Corriger les erreurs courantes:
#     * marginMode manquant (400172)
#     * position mode mismatch (40774) -> fallback auto tradeSide
# - Ne dépend PAS d'un _request() qui raise sur HTTP 4xx: on fait une requête
#   "no-raise" dédiée pour lire code/msg sur les erreurs 4xx.
# =====================================================================

from __future__ import annotations

import json
import math
import time
import logging
from typing import Any, Dict, Optional

from bitget_client import BitgetClient

logger = logging.getLogger(__name__)

# Valeurs Bitget v2
DEFAULT_PRODUCT_TYPE = "USDT-FUTURES"
DEFAULT_MARGIN_COIN = "USDT"


class SymbolMetaCache:
    """
    Cache des métadonnées contrats Bitget.
    Endpoint: GET /api/v2/mix/market/contracts?productType=USDT-FUTURES

    On garde:
      - pricePlace / priceEndStep -> tick_size
      - volumePlace / sizeMultiplier -> size_step
      - minTradeNum / maxOrderQty
    """

    def __init__(self, client: BitgetClient, product_type: str = DEFAULT_PRODUCT_TYPE, ttl_seconds: int = 300):
        self.client = client
        self.product_type = product_type
        self.ttl_seconds = int(ttl_seconds)

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_fetch: float = 0.0

    async def _refresh_all(self) -> None:
        now = time.time()
        if self._cache and (now - self._last_fetch) < self.ttl_seconds:
            return

        resp = await self.client._request(
            "GET",
            "/api/v2/mix/market/contracts",
            params={"productType": self.product_type},
            auth=False,
        )

        data = resp.get("data") if isinstance(resp, dict) else None
        if not isinstance(data, list):
            return

        new_cache: Dict[str, Dict[str, Any]] = {}
        for c in data:
            try:
                sym = str(c.get("symbol", "")).upper().strip()
                if not sym:
                    continue

                price_place = int(c.get("pricePlace", "0") or 0)
                price_end_step = float(c.get("priceEndStep", "1") or 1)
                volume_place = int(c.get("volumePlace", "0") or 0)
                size_multiplier = float(c.get("sizeMultiplier", "1") or 1)
                min_trade_num = float(c.get("minTradeNum", "0") or 0)

                max_order_qty_raw = c.get("maxOrderQty")
                max_order_qty = float(max_order_qty_raw) if max_order_qty_raw not in (None, "") else None

                # tick = step * 10^(-pricePlace)
                tick_size = price_end_step * (10 ** -price_place)

                # Sanity guards (évite division by zero)
                if tick_size <= 0:
                    tick_size = None
                if size_multiplier <= 0:
                    size_multiplier = None

                new_cache[sym] = {
                    "symbol": sym,
                    "price_decimals": price_place,
                    "tick_size": tick_size,
                    "size_decimals": volume_place,
                    "size_step": size_multiplier,
                    "min_trade_num": min_trade_num,
                    "max_order_qty": max_order_qty,
                }
            except Exception:
                continue

        if new_cache:
            self._cache = new_cache
            self._last_fetch = now
            logger.info("[META] refreshed contracts cache (%s symbols)", len(new_cache))

    async def get(self, symbol: str) -> Optional[Dict[str, Any]]:
        symbol = str(symbol).upper().strip()
        await self._refresh_all()
        return self._cache.get(symbol)


class BitgetTrader:
    """
    Trader wrapper pour BitgetClient.
    """

    def __init__(
        self,
        client: BitgetClient,
        *,
        product_type: str = DEFAULT_PRODUCT_TYPE,
        margin_coin: str = DEFAULT_MARGIN_COIN,
        margin_mode: str = "isolated",   # "isolated" ou "crossed"
        leverage: float = 10.0,
        margin_usdt: float = 20.0,
        # compat:
        target_margin_usdt: Optional[float] = None,
    ):
        self.client = client
        self.product_type = str(product_type)
        self.margin_coin = str(margin_coin)

        mm = (margin_mode or "isolated").lower().strip()
        if mm not in ("isolated", "crossed"):
            mm = "isolated"
        self.margin_mode = mm

        if target_margin_usdt is not None:
            margin_usdt = float(target_margin_usdt)

        self.leverage = float(leverage)
        self.margin_usdt = float(margin_usdt)

        self._meta = SymbolMetaCache(client, product_type=self.product_type, ttl_seconds=300)

    # ------------------------------------------------------------------
    # Low-level request that DOES NOT raise on HTTP 4xx (for order placement)
    # ------------------------------------------------------------------
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
        Copie légère de BitgetClient._request, mais retourne le JSON même en HTTP 4xx.
        (Bitget renvoie souvent {code,msg,data} avec un status 400)
        """
        await self.client._ensure_session()

        params = params or {}
        data = data or {}

        query = ""
        if params:
            query = "?" + "&".join(f"{k}={v}" for k, v in params.items())

        url = self.client.BASE + path + query
        body = json.dumps(data, separators=(",", ":")) if data else ""

        ts = str(int(time.time() * 1000))
        headers: Dict[str, str] = {}

        if auth:
            sig = self.client._sign(ts, method.upper(), path, query, body)
            headers = {
                "ACCESS-KEY": self.client.api_key,
                "ACCESS-SIGN": sig,
                "ACCESS-TIMESTAMP": ts,
                "ACCESS-PASSPHRASE": self.client.api_passphrase,
                "Content-Type": "application/json",
            }

        async with self.client._session.request(method.upper(), url, data=body, headers=headers, timeout=12) as resp:
            txt = await resp.text()
            status = resp.status
            try:
                js = json.loads(txt) if txt else {}
            except Exception:
                js = {"code": "HTTP_ERROR", "msg": txt}

            if isinstance(js, dict):
                js.setdefault("http_status", status)
            return js

    # ------------------------------------------------------------------
    # Formatting / quantize
    # ------------------------------------------------------------------
    @staticmethod
    def _fmt(x: float, decimals: int) -> str:
        decimals = max(0, int(decimals))
        return f"{float(x):.{decimals}f}"

    def _quantize_price(self, meta: Optional[Dict[str, Any]], price: float) -> float:
        raw = float(price)
        if not meta:
            return raw

        tick = meta.get("tick_size")
        decimals = int(meta.get("price_decimals", 0) or 0)

        if not tick or float(tick) <= 0:
            return float(self._fmt(raw, decimals))

        tick = float(tick)
        q = math.floor(raw / tick) * tick
        return float(self._fmt(q, decimals))

    def _quantize_size(self, meta: Optional[Dict[str, Any]], size: float) -> float:
        raw = float(size)
        if raw <= 0:
            raise ValueError("size<=0")

        if not meta:
            return raw

        step = meta.get("size_step")
        decimals = int(meta.get("size_decimals", 0) or 0)
        min_trade = float(meta.get("min_trade_num", 0) or 0)
        max_qty = meta.get("max_order_qty")

        if not step or float(step) <= 0:
            # fallback uniquement par décimales
            sz = float(self._fmt(raw, decimals))
        else:
            step = float(step)
            sz = math.floor(raw / step) * step
            sz = float(self._fmt(sz, decimals))

        if max_qty not in (None, ""):
            try:
                max_qty_f = float(max_qty)
                if max_qty_f > 0 and sz > max_qty_f:
                    # clamp sur maxQty
                    if step and float(step) > 0:
                        sz = math.floor(max_qty_f / float(step)) * float(step)
                    else:
                        sz = max_qty_f
                    sz = float(self._fmt(sz, decimals))
            except Exception:
                pass

        # minTradeNum guard
        if min_trade and sz < min_trade:
            sz = float(self._fmt(min_trade, decimals))

        return float(sz)

    # ------------------------------------------------------------------
    # Public: place LIMIT
    # ------------------------------------------------------------------
    async def place_limit(
        self,
        *,
        symbol: str,
        side: str,            # "BUY"/"SELL" ou "buy"/"sell"
        price: float,
        size: Optional[float] = None,
        client_oid: Optional[str] = None,
        # compat params (si ton code les passe)
        preset_sl: Optional[float] = None,
        preset_tp: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Place un LIMIT d'entrée. Fallback automatique sur tradeSide si erreur 40774.

        - En one-way mode: ne PAS mettre tradeSide
        - En hedge mode: mettre tradeSide="open"
        """
        sym = str(symbol).upper().strip()
        sd = str(side).lower().strip()
        if sd in ("buy", "long"):
            sd = "buy"
        elif sd in ("sell", "short"):
            sd = "sell"
        else:
            raise ValueError(f"invalid side={side}")

        if client_oid is None:
            client_oid = f"entry-{sym}-{int(time.time() * 1000)}"
        client_oid = str(client_oid)

        meta = await self._meta.get(sym)
        q_price = self._quantize_price(meta, float(price))

        # Taille
        if size is None:
            # notional cible ~ marge * levier
            notional_target = max(0.0, self.margin_usdt) * max(1e-9, self.leverage)
            if q_price <= 0:
                raise ValueError("price<=0")
            raw_size = notional_target / q_price
        else:
            raw_size = float(size)

        q_size = self._quantize_size(meta, raw_size)

        # Logs
        approx_notional = q_size * q_price
        logger.info(
            "[EXEC] place LIMIT %s %s price=%s size=%s notional≈%.2f (marginMode=%s lev=%.1fx)",
            sym, sd,
            self._fmt(q_price, int(meta.get("price_decimals", 6) if meta else 6)),
            self._fmt(q_size, int(meta.get("size_decimals", 4) if meta else 4)),
            approx_notional,
            self.margin_mode,
            self.leverage,
        )

        base_payload: Dict[str, Any] = {
            "symbol": sym,
            "productType": self.product_type,
            "marginCoin": self.margin_coin,
            "marginMode": self.margin_mode,         # IMPORTANT
            "size": str(q_size),
            "price": str(q_price),
            "side": sd,
            "orderType": "limit",
            "timeInForceValue": "normal",
            "clientOid": client_oid,
        }

        # NOTE: preset_sl / preset_tp
        # Bitget v2 ne gère pas toujours un "preset" direct sur place-order selon configs.
        # On les ignore ici volontairement pour fiabiliser l'entrée.
        _ = preset_sl, preset_tp

        # 1) Try one-way style (no tradeSide)
        resp = await self._request_any_status(
            "POST",
            "/api/v2/mix/order/place-order",
            data=base_payload,
            auth=True,
        )

        code = str(resp.get("code", ""))
        if code == "00000":
            resp["ok"] = True
            return resp

        # 2) If mismatch, try hedge style
        if code == "40774":
            hedge_payload = dict(base_payload)
            hedge_payload["tradeSide"] = "open"
            resp2 = await self._request_any_status(
                "POST",
                "/api/v2/mix/order/place-order",
                data=hedge_payload,
                auth=True,
            )
            resp2["ok"] = str(resp2.get("code", "")) == "00000"
            if resp2["ok"]:
                return resp2

            logger.error("[EXEC] place-order failed (hedge retry) %s: %s", sym, resp2)
            return resp2

        resp["ok"] = False
        logger.error("[EXEC] place-order failed %s: %s", sym, resp)
        return resp
