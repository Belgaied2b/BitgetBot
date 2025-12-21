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


# Pour le market v2 (contracts), on garde ce productType
PRODUCT_TYPE = "USDT-FUTURES"
MARGIN_COIN = "USDT"
MARGIN_MODE = "isolated"


class SymbolMetaCache:
    """
    Async cache des métadonnées contrats Bitget pour USDT-FUTURES (market v2).

    Endpoint utilisé :
      GET /api/v2/mix/market/contracts

    On en tire :
      - pricePlace      -> nb de décimales du prix
      - priceEndStep    -> taille du pas de prix (en ticks)
      - volumePlace     -> nb de décimales de size
      - sizeMultiplier  -> step de taille
      - minTradeNum     -> taille minimale
      - maxOrderQty     -> taille max par ordre

    Utilisation :
      meta = await cache.get("BTCUSDT")
      meta =>
        {
          "symbol": "BTCUSDT",
          "price_decimals": 2,
          "tick_size": 0.5,
          "size_decimals": 3,
          "size_step": 0.001,
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
            # IMPORTANT : BitgetClient._request NE PREND PAS 'body', on passe tout dans params
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
                size_multiplier = float(c.get("sizeMultiplier", "1"))
                min_trade_num = float(c.get("minTradeNum", "0"))
                max_order_qty_raw = c.get("maxOrderQty")
                max_order_qty = float(max_order_qty_raw) if max_order_qty_raw is not None else None

                # Tick = step * 10^(-pricePlace)
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


class BitgetTrader(BitgetClient):
    """
    Trader Bitget Futures USDT (USDT-FUTURES) en mode desk institutionnel.

    ⚠️ IMPORTANT :
      - Market : API v2 pour les contrats (contracts)
      - Trade  : API v1 officiel /api/mix/v1/order/placeOrder (beaucoup plus stable)

    - Utilise un cache de contrats pour respecter tick & sizeScale
    - Calcule la taille en fonction d'une marge fixe (TARGET_MARGIN_USDT)
      et d'un levier fixe (TARGET_LEVERAGE)
    - Place :
        * ordre LIMIT d'ouverture
        * plan orders STOP LOSS / TAKE PROFIT (v1 placePlan, best-effort)
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
        self.PRODUCT_TYPE = product_type       # utilisé pour le market v2
        self.MARGIN_COIN = margin_coin
        self.MARGIN_MODE = MARGIN_MODE
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
        # Format + cast pour éviter les flottants crades
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

        # clamp max
        if max_qty is not None:
            max_qty = float(max_qty)
            if sz > max_qty:
                sz = math.floor(max_qty / step) * step

        if sz < min_trade:
            raise ValueError(f"size {sz} < minTradeNum {min_trade}")

        return float(self._fmt(sz, decimals))

    @staticmethod
    def _to_contract_symbol(symbol: str) -> str:
        """
        Convertit 'ADAUSDT' -> 'ADAUSDT_UMCBL' pour l'API mix v1.
        Si le symbole est déjà un contractId (contient _UMCBL / _DMCBL / _CMCBL), on le laisse.
        """
        s = symbol.upper()
        if "_UMCBL" in s or "_DMCBL" in s or "_CMCBL" in s:
            return s
        return f"{s}_UMCBL"

    @staticmethod
    def _make_client_oid(prefix: str = "bot") -> str:
        """
        Génère un clientOid propre, uniquement [a-zA-Z0-9_], pas de '.'.
        """
        ts = int(time.time() * 1000)
        return f"{prefix}_{ts}"

    # ------------------------------------------------------------------
    # PLACE LIMIT (ENTRY) — API v1 OFFICIELLE
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

        Utilise :
          POST /api/mix/v1/order/placeOrder

        Params envoyés (doc) :
          symbol           ex. BTCUSDT_UMCBL
          marginCoin       "USDT"
          size             "0.01"
          price            "30145.5"
          side             "open_long" | "open_short"
          orderType        "limit"
          timeInForceValue "normal"
          clientOid        string
        """
        side_raw = side.lower().strip()
        if side_raw in ("buy", "long"):
            side_param = "open_long"
        elif side_raw in ("sell", "short"):
            side_param = "open_short"
        else:
            raise ValueError(f"invalid side={side}")

        # Meta & quantization
        meta = await self._meta_cache.get(symbol)
        q_price = self._quantize_price_from_meta(meta, price)

        # notionnel cible ~ marge * levier
        notional_target = self.TARGET_MARGIN_USDT * self.TARGET_LEVERAGE
        raw_size = notional_target / q_price

        try:
            q_size = self._quantize_size_from_meta(meta, raw_size)
        except ValueError as e:
            logger.error(f"[TRADER] {symbol} size too small for limit: {e}")
            return None

        # Mémoriser la taille d'entrée pour SL/TP
        self._entry_size[symbol] = q_size

        approx_notional = q_size * q_price
        approx_margin = approx_notional / self.TARGET_LEVERAGE

        # ContractId v1
        contract_symbol = self._to_contract_symbol(symbol)

        # clientOid propre (on ne laisse pas passer "1.0" ou autre)
        if client_oid is None:
            client_oid_str = self._make_client_oid(prefix="entry")
        else:
            # Nettoyage : string + remplace '.' par '_'
            client_oid_str = str(client_oid).replace(".", "_")

        # format price/size en string avec les bons décimales
        if meta:
            price_str = self._fmt(q_price, meta["price_decimals"])
            size_str = self._fmt(q_size, meta["size_decimals"])
        else:
            price_str = f"{q_price:.10f}"
            size_str = f"{q_size:.4f}"

        logger.info(
            f"[TRADER] place_limit {symbol} ({contract_symbol}) {side_raw} "
            f"price={price_str} size={size_str} "
            f"(notional≈{approx_notional:.2f} USDT, marge≈{approx_margin:.2f} USDT, "
            f"levier={self.TARGET_LEVERAGE:.1f}x, clientOid={client_oid_str})"
        )

        params = {
            "symbol": contract_symbol,
            "marginCoin": self.MARGIN_COIN,
            "size": size_str,
            "price": price_str,
            "side": side_param,              # "open_long" / "open_short"
            "orderType": "limit",
            "timeInForceValue": "normal",
            "clientOid": client_oid_str,
        }

        # BitgetClient._request : toutes les données dans 'params'
        resp = await self._request(
            "POST",
            "/api/mix/v1/order/placeOrder",
            params=params,
            auth=True,
        )
        return resp

    # ------------------------------------------------------------------
    # PLACE STOP LOSS — plan v1 (best-effort)
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
        Place un plan order STOP LOSS via /api/mix/v1/plan/placePlan.

        ⚠️ Best-effort :
          - planType : "loss_plan"
          - triggerType : "mark_price"
          - side : close_long / close_short
        """
        meta = await self._meta_cache.get(symbol)
        contract_symbol = self._to_contract_symbol(symbol)

        open_side_up = open_side.upper().strip()
        if open_side_up in ("BUY", "LONG"):
            side_param = "close_long"   # on ferme un long => close_long
        else:
            side_param = "close_short"  # on ferme un short => close_short

        # Utiliser la taille mémorisée si dispo
        base_size = self._entry_size.get(symbol, float(size))
        eff_size = base_size * float(fraction)

        try:
            q_size = self._quantize_size_from_meta(meta, eff_size)
        except ValueError as e:
            logger.error(f"[TRADER] SL {symbol} size too small: {e}")
            return None

        q_price = self._quantize_price_from_meta(meta, sl_price)

        if client_oid is None:
            client_oid_str = self._make_client_oid(prefix="sl")
        else:
            client_oid_str = str(client_oid).replace(".", "_")

        if meta:
            price_str = self._fmt(q_price, meta["price_decimals"])
            size_str = self._fmt(q_size, meta["size_decimals"])
        else:
            price_str = f"{q_price:.10f}"
            size_str = f"{q_size:.4f}"

        logger.info(
            f"[TRADER] place_stop_loss {symbol} ({contract_symbol}) side(open)={open_side_up} "
            f"side(plan)={side_param} sl={price_str} size={size_str} (fraction={fraction:.3f}, clientOid={client_oid_str})"
        )

        params = {
            "symbol": contract_symbol,
            "marginCoin": self.MARGIN_COIN,
            "size": size_str,
            "side": side_param,          # close_long / close_short
            "triggerPrice": price_str,
            "executePrice": price_str,
            "triggerType": "mark_price",
            "orderType": "limit",
            "planType": "loss_plan",
            "timeInForceValue": "normal",
            "clientOid": client_oid_str,
        }

        resp = await self._request(
            "POST",
            "/api/mix/v1/plan/placePlan",
            params=params,
            auth=True,
        )
        return resp

    # ------------------------------------------------------------------
    # PLACE TAKE PROFIT (TP1 / TP2) — plan v1 (best-effort)
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
        Place un plan order TAKE PROFIT (TP1/TP2) via /api/mix/v1/plan/placePlan.

        ⚠️ Best-effort :
          - planType : "profit_plan"
          - triggerType : "mark_price"
          - side : close_long / close_short
        """
        meta = await self._meta_cache.get(symbol)
        contract_symbol = self._to_contract_symbol(symbol)

        open_side_up = open_side.upper().strip()
        if open_side_up in ("BUY", "LONG"):
            side_param = "close_long"
        else:
            side_param = "close_short"

        base_size = self._entry_size.get(symbol, float(size))
        eff_size = base_size * float(fraction)

        try:
            q_size = self._quantize_size_from_meta(meta, eff_size)
        except ValueError as e:
            logger.error(f"[TRADER] TP {symbol} size too small: {e}")
            return None

        q_price = self._quantize_price_from_meta(meta, tp_price)

        if client_oid is None:
            client_oid_str = self._make_client_oid(prefix="tp")
        else:
            client_oid_str = str(client_oid).replace(".", "_")

        if meta:
            price_str = self._fmt(q_price, meta["price_decimals"])
            size_str = self._fmt(q_size, meta["size_decimals"])
        else:
            price_str = f"{q_price:.10f}"
            size_str = f"{q_size:.4f}"

        logger.info(
            f"[TRADER] place_take_profit {symbol} ({contract_symbol}) side(open)={open_side_up} "
            f"side(plan)={side_param} tp={price_str} size={size_str} "
            f"(fraction={fraction:.3f}, clientOid={client_oid_str})"
        )

        params = {
            "symbol": contract_symbol,
            "marginCoin": self.MARGIN_COIN,
            "size": size_str,
            "side": side_param,
            "triggerPrice": price_str,
            "executePrice": price_str,
            "triggerType": "mark_price",
            "orderType": "limit",
            "planType": "profit_plan",
            "timeInForceValue": "normal",
            "clientOid": client_oid_str,
        }

        resp = await self._request(
            "POST",
            "/api/mix/v1/plan/placePlan",
            params=params,
            auth=True,
        )
        return resp
