import math
import time
import json
import hmac
import hashlib
import base64
import logging
from typing import Dict, Any, Optional

import aiohttp

try:
    # Optionnel : récup config globale si dispo
    from settings import TARGET_MARGIN_USDT, TARGET_LEVERAGE, BITGET_BASE_URL
except Exception:
    TARGET_MARGIN_USDT = 20.0
    TARGET_LEVERAGE = 10.0
    BITGET_BASE_URL = "https://api.bitget.com"

logger = logging.getLogger(__name__)

PRODUCT_TYPE = "USDT-FUTURES"
MARGIN_COIN = "USDT"
MARGIN_MODE = "isolated"


# =====================================================================
# Cache meta contrats (tick / sizeStep / minTrade / maxOrderQty)
# =====================================================================

class SymbolMetaCache:
    """
    Cache des métadonnées contrats Bitget pour USDT-FUTURES (V2).

    On interroge :
      GET /api/v2/mix/market/contracts?productType=USDT-FUTURES

    et on en tire :
      - pricePlace      -> nb de décimales du prix
      - priceEndStep    -> taille du pas de prix (en ticks)
      - volumePlace     -> nb de décimales de size
      - sizeMultiplier  -> step de taille
      - minTradeNum     -> taille minimale
      - maxOrderQty     -> taille max par ordre
    """

    def __init__(self, base_url: str, product_type: str = PRODUCT_TYPE, ttl_seconds: int = 300):
        self.base_url = base_url.rstrip("/")
        self.product_type = product_type
        self.ttl_seconds = ttl_seconds

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_fetch: float = 0.0

    async def _fetch_contracts(self, symbol: Optional[str] = None) -> Optional[Any]:
        """
        Appel public V2 pour récupérer les contrats.
        """
        url = f"{self.base_url}/api/v2/mix/market/contracts"
        params: Dict[str, Any] = {"productType": self.product_type}
        if symbol:
            params["symbol"] = symbol

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        logger.error(f"[META] HTTP {resp.status} GET {url} params={params} resp={text}")
                        return None
                    data = json.loads(text)
        except Exception as e:
            logger.error(f"[META] exception GET {url} params={params}: {e}")
            return None

        return data

    async def _refresh_all(self) -> None:
        """
        Rafraîchit le cache complet si TTL dépassé.
        """
        now = time.time()
        if self._cache and (now - self._last_fetch) < self.ttl_seconds:
            return

        data = await self._fetch_contracts(symbol=None)
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            logger.error(f"[META] unexpected contracts response: {data}")
            return

        new_cache: Dict[str, Dict[str, Any]] = {}
        for c in items:
            sym = c.get("symbol")
            if not sym:
                continue
            try:
                price_place = int(c.get("pricePlace", "0"))
                price_end_step = float(c.get("priceEndStep", "1"))
                volume_place = int(c.get("volumePlace", "0"))
                size_multiplier = float(c.get("sizeMultiplier", "1"))
                min_trade_num = float(c.get("minTradeNum", "0"))
                max_order_qty_raw = c.get("maxOrderQty")
                max_order_qty = float(max_order_qty_raw) if max_order_qty_raw is not None else None

                tick_size = price_end_step * (10 ** -price_place)

                new_cache[sym] = {
                    "symbol": sym,
                    "price_decimals": price_place,
                    "tick_size": tick_size,
                    "size_decimals": volume_place,
                    "size_step": size_multiplier,
                    "min_trade_num": min_trade_num,
                    "max_order_qty": max_order_qty,
                }
            except Exception as e:
                logger.warning(f"[META] skipping contract {c}: {e}")

        if new_cache:
            self._cache = new_cache
            self._last_fetch = now
            logger.info(f"[META] refreshed contracts cache ({len(new_cache)} symbols)")

    async def _fetch_single(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch meta pour un seul symbole si pas présent dans le cache global.
        """
        data = await self._fetch_contracts(symbol=symbol)
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list) or not items:
            logger.error(f"[META] empty contracts data for symbol={symbol}: {data}")
            return None

        c = items[0]
        try:
            price_place = int(c.get("pricePlace", "0"))
            price_end_step = float(c.get("priceEndStep", "1"))
            volume_place = int(c.get("volumePlace", "0"))
            size_multiplier = float(c.get("sizeMultiplier", "1"))
            min_trade_num = float(c.get("minTradeNum", "0"))
            max_order_qty_raw = c.get("maxOrderQty")
            max_order_qty = float(max_order_qty_raw) if max_order_qty_raw is not None else None

            tick_size = price_end_step * (10 ** -price_place)

            meta = {
                "symbol": symbol,
                "price_decimals": price_place,
                "tick_size": tick_size,
                "size_decimals": volume_place,
                "size_step": size_multiplier,
                "min_trade_num": min_trade_num,
                "max_order_qty": max_order_qty,
            }
            self._cache[symbol] = meta
            logger.info(f"[META] fetched single contract meta for {symbol}: {meta}")
            return meta
        except Exception as e:
            logger.error(f"[META] error parsing single contract {c} for symbol={symbol}: {e}")
            return None

    async def get(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Retourne la meta pour un symbole donné (ou None si échec).
        """
        await self._refresh_all()
        meta = self._cache.get(symbol)
        if meta is not None:
            return meta
        return await self._fetch_single(symbol)


# =====================================================================
# BitgetTrader V2 — client signé JSON pour /api/v2/mix
# =====================================================================

class BitgetTrader:
    """
    Trader Bitget Futures USDT (USDT-FUTURES) en mode desk institutionnel.

    - Utilise un cache de contrats pour respecter tick & sizeScale
    - Calcule la taille en fonction d'une marge fixe (TARGET_MARGIN_USDT)
      et d'un levier fixe (TARGET_LEVERAGE)
    - Place :
        * ordre LIMIT d'ouverture (V2 JSON)
        * plan orders STOP LOSS / TAKE PROFIT (V2 JSON)
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        *,
        product_type: str = PRODUCT_TYPE,
        margin_coin: str = MARGIN_COIN,
        margin_mode: str = MARGIN_MODE,
        target_margin_usdt: float = TARGET_MARGIN_USDT,
        leverage: float = TARGET_LEVERAGE,
        base_url: str = BITGET_BASE_URL,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase

        self.base_url = base_url.rstrip("/")
        self.PRODUCT_TYPE = product_type
        self.MARGIN_COIN = margin_coin
        self.MARGIN_MODE = margin_mode
        self.TARGET_MARGIN_USDT = float(target_margin_usdt)
        self.TARGET_LEVERAGE = float(leverage)

        # Cache des métas contrats
        self._meta_cache = SymbolMetaCache(self.base_url, product_type=self.PRODUCT_TYPE)

        # Taille d'entrée mémorisée par symbole pour SL/TP
        self._entry_size: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Helpers de signature V2
    # ------------------------------------------------------------------

    def _sign(self, timestamp: str, method: str, path: str, body: str) -> str:
        """
        Signature V2 : timestamp + method + path + body_json
        """
        pre_sign = f"{timestamp}{method.upper()}{path}{body}"
        mac = hmac.new(
            self.api_secret.encode("utf-8"),
            pre_sign.encode("utf-8"),
            hashlib.sha256,
        )
        return base64.b64encode(mac.digest()).decode("utf-8")

    async def _request_private(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Requête privée signée en JSON (V2).
        """
        method_up = method.upper()
        body = "" if payload is None else json.dumps(payload, separators=(",", ":"))
        ts = str(int(time.time() * 1000))

        sign = self._sign(ts, method_up, path, body)
        headers = {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": sign,
            "ACCESS-TIMESTAMP": ts,
            "ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}{path}"

        try:
            async with aiohttp.ClientSession() as session:
                if method_up == "GET":
                    async with session.get(url, headers=headers, timeout=10) as resp:
                        text = await resp.text()
                else:
                    async with session.post(url, headers=headers, data=body, timeout=10) as resp:
                        text = await resp.text()
        except Exception as e:
            logger.error(f"[HTTP] exception {method_up} {url} payload={payload}: {e}")
            raise

        try:
            data = json.loads(text)
        except Exception:
            logger.error(f"[HTTP] non-JSON response {method_up} {url}: {text}")
            raise RuntimeError(f"Non-JSON response: {text}")

        code = data.get("code")
        if code != "00000":
            logger.error(
                f"[HTTP] Bitget error {method_up} {url} payload={payload} "
                f"resp={text}"
            )
            raise RuntimeError(f"Bitget API error {code}: {data.get('msg')}")

        return data

    # ------------------------------------------------------------------
    # Helpers de quantization
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt(value: float, decimals: int) -> str:
        fmt = f"{{:.{decimals}f}}"
        return fmt.format(value)

    def _quantize_price_from_meta(self, meta: Optional[Dict[str, Any]], price: float) -> float:
        if not meta:
            return float(price)
        tick = float(meta["tick_size"])
        decimals = int(meta["price_decimals"])
        raw = float(price)
        q = math.floor(raw / tick) * tick
        return float(self._fmt(q, decimals))

    def _quantize_size_from_meta(self, meta: Optional[Dict[str, Any]], size: float) -> float:
        if not meta:
            return float(size)

        step = float(meta["size_step"])
        decimals = int(meta["size_decimals"])
        min_trade = float(meta["min_trade_num"])
        max_qty = meta.get("max_order_qty")

        raw = float(size)
        if raw <= 0:
            raise ValueError(f"invalid size {raw}")

        sz = math.floor(raw / step) * step

        if max_qty is not None:
            max_qty = float(max_qty)
            if sz > max_qty:
                sz = math.floor(max_qty / step) * step

        if sz < min_trade:
            raise ValueError(f"size {sz} < minTradeNum {min_trade}")

        return float(self._fmt(sz, decimals))

    # ------------------------------------------------------------------
    # PLACE LIMIT (entrée)
    # ------------------------------------------------------------------

    async def place_limit(
        self,
        symbol: str,
        side: str,
        price: float,
        client_oid: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Place un ordre LIMIT d'ouverture en respectant les contraintes :
          - tick de prix
          - sizeStep / minTradeNum

        Utilise : POST /api/v2/mix/order/place-order (V2 JSON)
        """
        side = side.lower()  # "buy" ou "sell"

        meta = await self._meta_cache.get(symbol)
        q_price = self._quantize_price_from_meta(meta, price)

        # notionnel cible ~ marge * levier
        notional_target = self.TARGET_MARGIN_USDT * self.TARGET_LEVERAGE
        raw_size = notional_target / max(q_price, 1e-12)

        try:
            q_size = self._quantize_size_from_meta(meta, raw_size)
        except ValueError as e:
            logger.error(f"[TRADER] {symbol} size too small for limit: {e}")
            return None

        # Mémoriser la taille d'entrée pour SL/TP
        self._entry_size[symbol] = q_size

        approx_notional = q_size * q_price
        approx_margin = approx_notional / self.TARGET_LEVERAGE

        if client_oid is None:
            client_oid = f"entry-{symbol}-{int(time.time() * 1000)}"

        if meta:
            price_str = self._fmt(q_price, meta["price_decimals"])
            size_str = self._fmt(q_size, meta["size_decimals"])
        else:
            price_str = f"{q_price:.10f}"
            size_str = f"{q_size:.4f}"

        logger.info(
            f"[TRADER] place_limit {symbol} {side} price={price_str} size={size_str} "
            f"(notional≈{approx_notional:.2f} USDT, marge≈{approx_margin:.2f} USDT, "
            f"levier={self.TARGET_LEVERAGE:.1f}x, clientOid={client_oid})"
        )

        body = {
            "symbol": symbol,
            "productType": self.PRODUCT_TYPE,
            "marginMode": self.MARGIN_MODE,
            "marginCoin": self.MARGIN_COIN,
            "size": size_str,
            "price": price_str,
            "side": side,           # "buy" ou "sell"
            "orderType": "limit",
            "force": "gtc",
            "clientOid": client_oid,
            "reduceOnly": "NO",
        }

        try:
            resp = await self._request_private(
                "POST",
                "/api/v2/mix/order/place-order",
                payload=body,
            )
            return resp
        except Exception as e:
            logger.error(f"[TRADER] place_limit error {symbol}: {e}")
            return None

    # ------------------------------------------------------------------
    # PLACE STOP LOSS (plan order)
    # ------------------------------------------------------------------

    async def place_stop_loss(
        self,
        symbol: str,
        open_side: str,        # "BUY"/"SELL" ou "LONG"/"SHORT"
        sl_price: float,
        size: float,
        fraction: float = 1.0,
        client_oid: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Place un plan order STOP LOSS (normal_plan) pour close la position.
        Utilise : POST /api/v2/mix/order/place-plan-order
        """
        meta = await self._meta_cache.get(symbol)

        open_side_up = open_side.upper()
        trigger_side = "sell" if open_side_up in ("BUY", "LONG") else "buy"

        base_size = self._entry_size.get(symbol, float(size))
        eff_size = base_size * float(fraction)

        try:
            q_size = self._quantize_size_from_meta(meta, eff_size)
        except ValueError as e:
            logger.error(f"[TRADER] SL {symbol} size too small: {e}")
            return None

        q_price = self._quantize_price_from_meta(meta, sl_price)

        if client_oid is None:
            client_oid = f"sl-{symbol}-{int(time.time() * 1000)}"

        if meta:
            price_str = self._fmt(q_price, meta["price_decimals"])
            size_str = self._fmt(q_size, meta["size_decimals"])
        else:
            price_str = f"{q_price:.10f}"
            size_str = f"{q_size:.4f}"

        logger.info(
            f"[TRADER] place_stop_loss {symbol} side(open)={open_side_up} "
            f"trigger_side={trigger_side} sl={price_str} size={size_str} "
            f"(fraction={fraction:.3f}, clientOid={client_oid})"
        )

        body = {
            "planType": "normal_plan",
            "symbol": symbol,
            "productType": self.PRODUCT_TYPE,
            "marginMode": self.MARGIN_MODE,
            "marginCoin": self.MARGIN_COIN,
            "size": size_str,
            "price": price_str,
            "triggerPrice": price_str,
            "triggerType": "mark_price",
            "orderType": "limit",
            "side": trigger_side,      # direction de l'ordre de clôture
            "tradeSide": "close",
            "reduceOnly": "YES",
            "clientOid": client_oid,
        }

        try:
            resp = await self._request_private(
                "POST",
                "/api/v2/mix/order/place-plan-order",
                payload=body,
            )
            return resp
        except Exception as e:
            logger.error(f"[TRADER] place_stop_loss error {symbol}: {e}")
            return None

    # ------------------------------------------------------------------
    # PLACE TAKE PROFIT (TP1 / TP2)
    # ------------------------------------------------------------------

    async def place_take_profit(
        self,
        symbol: str,
        open_side: str,
        tp_price: float,
        size: float,
        fraction: float = 0.5,
        client_oid: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Place un plan order TAKE PROFIT (TP1/TP2) pour close partiellement la position.
        Utilise : POST /api/v2/mix/order/place-plan-order
        """
        meta = await self._meta_cache.get(symbol)

        open_side_up = open_side.upper()
        trigger_side = "sell" if open_side_up in ("BUY", "LONG") else "buy"

        base_size = self._entry_size.get(symbol, float(size))
        eff_size = base_size * float(fraction)

        try:
            q_size = self._quantize_size_from_meta(meta, eff_size)
        except ValueError as e:
            logger.error(f"[TRADER] TP {symbol} size too small: {e}")
            return None

        q_price = self._quantize_price_from_meta(meta, tp_price)

        if client_oid is None:
            client_oid = f"tp-{symbol}-{int(time.time() * 1000)}"

        if meta:
            price_str = self._fmt(q_price, meta["price_decimals"])
            size_str = self._fmt(q_size, meta["size_decimals"])
        else:
            price_str = f"{q_price:.10f}"
            size_str = f"{q_size:.4f}"

        logger.info(
            f"[TRADER] place_take_profit {symbol} side(open)={open_side_up} "
            f"trigger_side={trigger_side} tp={price_str} size={size_str} "
            f"(fraction={fraction:.3f}, clientOid={client_oid})"
        )

        body = {
            "planType": "normal_plan",
            "symbol": symbol,
            "productType": self.PRODUCT_TYPE,
            "marginMode": self.MARGIN_MODE,
            "marginCoin": self.MARGIN_COIN,
            "size": size_str,
            "price": price_str,
            "triggerPrice": price_str,
            "triggerType": "mark_price",
            "orderType": "limit",
            "side": trigger_side,
            "tradeSide": "close",
            "reduceOnly": "YES",
            "clientOid": client_oid,
        }

        try:
            resp = await self._request_private(
                "POST",
                "/api/v2/mix/order/place-plan-order",
                payload=body,
            )
            return resp
        except Exception as e:
            logger.error(f"[TRADER] place_take_profit error {symbol}: {e}")
            return None
