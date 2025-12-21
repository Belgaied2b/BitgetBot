import math
import time
import logging
from typing import Dict, Any, Optional

from bitget_client import BitgetClient

try:
    # Optionnel : récup config globale si dispo
    from settings import TARGET_MARGIN_USDT, TARGET_LEVERAGE
except Exception:  # fallback si pas défini dans settings.py
    TARGET_MARGIN_USDT = 20.0
    TARGET_LEVERAGE = 10.0


logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Constantes Bitget Futures
# ----------------------------------------------------------------------

# productType pour /api/v2/mix/market/contracts
PRODUCT_TYPE = "USDT-FUTURES"
MARGIN_COIN = "USDT"


class SymbolMetaCache:
    """
    Cache des métadonnées contrats Bitget pour USDT-FUTURES (v2).

    On interroge :
      GET /api/v2/mix/market/contracts?productType=USDT-FUTURES

    et on en tire :
      - pricePlace      -> nb de décimales du prix
      - priceEndStep    -> taille du pas de prix (en ticks)
      - volumePlace     -> nb de décimales de size
      - minTradeNum     -> step de taille (min incrément autorisé)
      - sizeMultiplier  -> contract size (info, pas utilisée comme step)
      - maxOrderQty     -> taille max par ordre

    Meta retournée :
        {
          "symbol": "BTCUSDT",
          "price_decimals": 2,
          "tick_size": 0.5,
          "size_decimals": 3,
          "size_step": 0.001,        # dérivé de minTradeNum
          "min_trade_num": 0.001,
          "max_order_qty": 1000.0 or None,
        }
    """

    def __init__(self, client: BitgetClient, product_type: str = PRODUCT_TYPE, ttl_seconds: int = 300):
        self.client = client
        self.product_type = product_type
        self.ttl_seconds = ttl_seconds

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_fetch: float = 0.0

    async def _refresh_all(self) -> None:
        """
        Rafraîchit le cache complet si TTL dépassé.
        """
        now = time.time()
        if self._cache and (now - self._last_fetch) < self.ttl_seconds:
            return

        params = {"productType": self.product_type}
        try:
            resp = await self.client._request(
                "GET",
                "/api/v2/mix/market/contracts",
                params=params,
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
            sym = c.get("symbol")
            if not sym:
                continue
            try:
                price_place = int(c.get("pricePlace", "0"))
                price_end_step = float(c.get("priceEndStep", "1"))
                volume_place = int(c.get("volumePlace", "0"))

                # ATTENTION : sizeMultiplier = contract size (valeur faciale), pas le step de qty.
                size_multiplier = float(c.get("sizeMultiplier", "0") or 0)

                # Le vrai step de size est en pratique minTradeNum (ou un multiple).
                min_trade_num = float(c.get("minTradeNum", "0") or 0)
                if min_trade_num <= 0:
                    # fallback : si vraiment absent, on met 1.0 contrat
                    min_trade_num = 1.0

                size_step = min_trade_num

                max_order_qty_raw = c.get("maxOrderQty")
                max_order_qty = float(max_order_qty_raw) if max_order_qty_raw is not None else None

                # Tick de prix : step * 10^(-pricePlace)
                tick_size = price_end_step * (10 ** -price_place)

                new_cache[sym] = {
                    "symbol": sym,
                    "price_decimals": price_place,
                    "tick_size": tick_size,
                    "size_decimals": volume_place,
                    "size_step": size_step,
                    "min_trade_num": min_trade_num,
                    "size_multiplier": size_multiplier,  # info only
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
        params = {"productType": self.product_type, "symbol": symbol}
        try:
            resp = await self.client._request(
                "GET",
                "/api/v2/mix/market/contracts",
                params=params,
                auth=False,
            )
        except Exception as e:
            logger.error(f"[META] error fetching contract for {symbol}: {e}")
            return None

        data = resp.get("data") if isinstance(resp, dict) else None
        if not isinstance(data, list) or not data:
            logger.error(f"[META] empty contracts data for symbol={symbol}: {resp}")
            return None

        c = data[0]
        try:
            price_place = int(c.get("pricePlace", "0"))
            price_end_step = float(c.get("priceEndStep", "1"))
            volume_place = int(c.get("volumePlace", "0"))

            size_multiplier = float(c.get("sizeMultiplier", "0") or 0)
            min_trade_num = float(c.get("minTradeNum", "0") or 0)
            if min_trade_num <= 0:
                min_trade_num = 1.0

            size_step = min_trade_num

            max_order_qty_raw = c.get("maxOrderQty")
            max_order_qty = float(max_order_qty_raw) if max_order_qty_raw is not None else None

            tick_size = price_end_step * (10 ** -price_place)

            meta = {
                "symbol": symbol,
                "price_decimals": price_place,
                "tick_size": tick_size,
                "size_decimals": volume_place,
                "size_step": size_step,
                "min_trade_num": min_trade_num,
                "size_multiplier": size_multiplier,
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


class BitgetTrader(BitgetClient):
    """
    Trader Bitget Futures USDT (USDT-FUTURES) en mode desk institutionnel.

    - Utilise un cache de contrats pour respecter tick & sizeStep
    - Calcule la taille en fonction d'une marge fixe (TARGET_MARGIN_USDT)
      et d'un levier fixe (TARGET_LEVERAGE)
    - Place :
        * ordre LIMIT d'ouverture (v2: /api/v2/mix/order/place-order)
        * plan orders STOP LOSS / TAKE PROFIT (v2: /api/v2/mix/order/place-plan-order)
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        *,
        product_type: str = PRODUCT_TYPE,
        margin_coin: str = MARGIN_COIN,
        target_margin_usdt: float = TARGET_MARGIN_USDT,
        leverage: float = TARGET_LEVERAGE,
    ):
        super().__init__(api_key, api_secret, passphrase)
        self.PRODUCT_TYPE = product_type
        self.MARGIN_COIN = margin_coin
        self.TARGET_MARGIN_USDT = float(target_margin_usdt)
        self.TARGET_LEVERAGE = float(leverage)

        # Cache des métas contrats
        self._meta_cache = SymbolMetaCache(self, product_type=self.PRODUCT_TYPE)

        # Taille d'entrée mémorisée par symbole pour SL/TP
        self._entry_size: Dict[str, float] = {}

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

        # Align sur le step
        sz = math.floor(raw / step) * step

        # clamp min
        if sz < min_trade:
            raise ValueError(f"size {sz} < minTradeNum {min_trade}")

        # clamp max
        if max_qty is not None:
            max_qty = float(max_qty)
            if sz > max_qty:
                sz = math.floor(max_qty / step) * step

        return float(self._fmt(sz, decimals))

    # ------------------------------------------------------------------
    # PLACE LIMIT (entrée) — v2, calqué sur la V1 qui marchait
    # ------------------------------------------------------------------

    async def place_limit(
        self,
        symbol: str,
        side: str,   # "buy"/"sell" ou "long"/"short"
        price: float,
        client_oid: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Place un ordre LIMIT d'ouverture en respectant :
          - tick de prix
          - sizeStep / minTradeNum
          - sémantique V1 (open_long/open_short + timeInForceValue=normal)
            mais via endpoint v2 (/api/v2/mix/order/place-order)
        """
        meta = await self._meta_cache.get(symbol)
        q_price = self._quantize_price_from_meta(meta, price)

        # notionnel cible ~ marge * levier
        notional_target = self.TARGET_MARGIN_USDT * self.TARGET_LEVERAGE
        raw_size = notional_target / q_price

        try:
            q_size = self._quantize_size_from_meta(meta, raw_size)
        except ValueError as e:
            logger.error(f"[TRADER] {symbol} size too small for limit: {e}")
            raise

        # Mémoriser la taille d'entrée pour SL/TP
        self._entry_size[symbol] = q_size

        approx_notional = q_size * q_price
        approx_margin = approx_notional / self.TARGET_LEVERAGE

        # Normaliser side utilisateur -> side API Bitget (mode hedge-style)
        s = side.lower()
        if s in ("buy", "long"):
            api_side = "open_long"
        elif s in ("sell", "short"):
            api_side = "open_short"
        else:
            raise ValueError(f"invalid side {side}")

        if client_oid is None:
            client_oid = f"{int(time.time())}_0"
        client_oid_str = str(client_oid)

        if meta:
            price_str = self._fmt(q_price, meta["price_decimals"])
            size_str = self._fmt(q_size, meta["size_decimals"])
        else:
            price_str = f"{q_price:.10f}"
            size_str = f"{q_size:.4f}"

        logger.info(
            f"[TRADER] place_limit {symbol} ({self.PRODUCT_TYPE}) {s} price={price_str} "
            f"size={size_str} (notional≈{approx_notional:.2f} USDT, "
            f"marge≈{approx_margin:.2f} USDT, levier={self.TARGET_LEVERAGE:.1f}x, "
            f"clientOid={client_oid_str}, api_side={api_side})"
        )

        payload = {
            "symbol": symbol,
            "productType": self.PRODUCT_TYPE,
            "marginCoin": self.MARGIN_COIN,
            "size": size_str,
            "price": price_str,
            "side": api_side,                 # "open_long" / "open_short"
            "orderType": "limit",
            "timeInForceValue": "normal",
            "clientOid": client_oid_str,
        }

        try:
            resp = await self._request(
                "POST",
                "/api/v2/mix/order/place-order",
                params=payload,
                auth=True,
            )
        except Exception as e:
            logger.error(f"[TRADER] place_limit HTTP error {symbol}: {e}")
            raise

        # Vérif code business Bitget
        if isinstance(resp, dict) and resp.get("code") != "00000":
            code = resp.get("code")
            msg = resp.get("msg")
            logger.error(
                f"[TRADER] place_limit error {symbol}: Bitget API error {code}: {msg} "
                f"(payload={payload})"
            )
            raise Exception(f"Bitget API error {code}: {msg}")

        return resp

    # ------------------------------------------------------------------
    # PLACE STOP LOSS — v2 plan-order
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

        - size : taille d'entrée (brute, non quantisée)
        - fraction : part de cette taille à utiliser (1.0 = 100%)
        """
        meta = await self._meta_cache.get(symbol)

        open_side_up = open_side.upper()
        # En hedge-style : close_long si position long, close_short si position short
        if open_side_up in ("BUY", "LONG"):
            api_side = "close_long"
        else:
            api_side = "close_short"

        base_size = self._entry_size.get(symbol, float(size))
        eff_size = base_size * float(fraction)

        try:
            q_size = self._quantize_size_from_meta(meta, eff_size)
        except ValueError as e:
            logger.error(f"[TRADER] SL {symbol} size too small: {e}")
            raise

        q_price = self._quantize_price_from_meta(meta, sl_price)

        if client_oid is None:
            client_oid = f"{int(time.time())}_sl"
        client_oid_str = str(client_oid)

        if meta:
            price_str = self._fmt(q_price, meta["price_decimals"])
            size_str = self._fmt(q_size, meta["size_decimals"])
        else:
            price_str = f"{q_price:.10f}"
            size_str = f"{q_size:.4f}"

        logger.info(
            f"[TRADER] place_stop_loss {symbol} side(open)={open_side_up} "
            f"api_side={api_side} sl={price_str} size={size_str} (fraction={fraction:.3f})"
        )

        payload = {
            "planType": "normal_plan",
            "symbol": symbol,
            "productType": self.PRODUCT_TYPE,
            "marginCoin": self.MARGIN_COIN,
            "size": size_str,
            "price": price_str,
            "triggerPrice": price_str,
            "triggerType": "mark_price",
            "orderType": "limit",
            "side": api_side,               # "close_long"/"close_short"
            "timeInForceValue": "normal",
            "clientOid": client_oid_str,
        }

        try:
            resp = await self._request(
                "POST",
                "/api/v2/mix/order/place-plan-order",
                params=payload,
                auth=True,
            )
        except Exception as e:
            logger.error(f"[TRADER] place_stop_loss HTTP error {symbol}: {e}")
            raise

        if isinstance(resp, dict) and resp.get("code") != "00000":
            code = resp.get("code")
            msg = resp.get("msg")
            logger.error(
                f"[TRADER] place_stop_loss error {symbol}: Bitget API error {code}: {msg} "
                f"(payload={payload})"
            )
            raise Exception(f"Bitget API error {code}: {msg}")

        return resp

    # ------------------------------------------------------------------
    # PLACE TAKE PROFIT (TP1 / TP2) — v2 plan-order
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
        - size : taille d'entrée (brute)
        - fraction : fraction de cette taille pour ce TP (ex : 0.5)
        """
        meta = await self._meta_cache.get(symbol)

        open_side_up = open_side.upper()
        if open_side_up in ("BUY", "LONG"):
            api_side = "close_long"
        else:
            api_side = "close_short"

        base_size = self._entry_size.get(symbol, float(size))
        eff_size = base_size * float(fraction)

        try:
            q_size = self._quantize_size_from_meta(meta, eff_size)
        except ValueError as e:
            logger.error(f"[TRADER] TP {symbol} size too small: {e}")
            raise

        q_price = self._quantize_price_from_meta(meta, tp_price)

        if client_oid is None:
            client_oid = f"{int(time.time())}_tp"
        client_oid_str = str(client_oid)

        if meta:
            price_str = self._fmt(q_price, meta["price_decimals"])
            size_str = self._fmt(q_size, meta["size_decimals"])
        else:
            price_str = f"{q_price:.10f}"
            size_str = f"{q_size:.4f}"

        logger.info(
            f"[TRADER] place_take_profit {symbol} side(open)={open_side_up} "
            f"api_side={api_side} tp={price_str} size={size_str} (fraction={fraction:.3f})"
        )

        payload = {
            "planType": "normal_plan",
            "symbol": symbol,
            "productType": self.PRODUCT_TYPE,
            "marginCoin": self.MARGIN_COIN,
            "size": size_str,
            "price": price_str,
            "triggerPrice": price_str,
            "triggerType": "mark_price",
            "orderType": "limit",
            "side": api_side,
            "timeInForceValue": "normal",
            "clientOid": client_oid_str,
        }

        try:
            resp = await self._request(
                "POST",
                "/api/v2/mix/order/place-plan-order",
                params=payload,
                auth=True,
            )
        except Exception as e:
            logger.error(f"[TRADER] place_take_profit HTTP error {symbol}: {e}")
            raise

        if isinstance(resp, dict) and resp.get("code") != "00000":
            code = resp.get("code")
            msg = resp.get("msg")
            logger.error(
                f"[TRADER] place_take_profit error {symbol}: Bitget API error {code}: {msg} "
                f"(payload={payload})"
            )
            raise Exception(f"Bitget API error {code}: {msg}")

        return resp
