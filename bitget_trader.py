# =====================================================================
# bitget_trader.py — Bitget USDT-Futures execution layer (robuste, 2025)
# =====================================================================
# Objectif :
#   - Fix "float division by zero" (tick/step=0, quantize => price=0)
#   - Fix appels _request (BitgetClient._request est keyword-only)
#   - Meta cache safe (priceEndStep/pricePlace, sizeMultiplier/volumePlace)
#   - API v2 :
#       POST /api/v2/mix/order/place-order
#       POST /api/v2/mix/order/place-plan-order
#
# Notes :
#   - On calcule la size à partir d'une marge fixe (MARGIN_USDT) + LEVERAGE.
#   - SL/TP sont placés en "plan orders" MARKET reduceOnly (plus robuste).
# =====================================================================

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from bitget_client import BitgetClient, normalize_symbol

try:
    from settings import PRODUCT_TYPE, MARGIN_COIN, MARGIN_USDT, LEVERAGE
except Exception:
    PRODUCT_TYPE = "USDT-FUTURES"
    MARGIN_COIN = "USDT"
    MARGIN_USDT = 20.0
    LEVERAGE = 10.0

logger = logging.getLogger(__name__)


# =====================================================================
# Helpers
# =====================================================================

def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _to_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(float(x))
    except Exception:
        return default


def _now_ms() -> int:
    return int(time.time() * 1000)


def _gen_client_oid(prefix: str = "desk") -> str:
    return f"{prefix}-{_now_ms()}"


def _force_side(side: str) -> str:
    """
    Bitget mix side : 'buy' / 'sell'
    """
    s = (side or "").strip().lower()
    if s in ("buy", "long"):
        return "buy"
    if s in ("sell", "short"):
        return "sell"
    # fallback : assume buy
    return "buy"


def _close_side(open_side: str) -> str:
    return "sell" if _force_side(open_side) == "buy" else "buy"


def _tick_fallback(price: float) -> float:
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


def _fmt(v: float, decimals: int) -> str:
    decimals = max(0, int(decimals))
    return f"{float(v):.{decimals}f}"


# =====================================================================
# Meta cache
# =====================================================================

@dataclass
class ContractMeta:
    symbol: str
    price_decimals: int
    tick_size: float
    size_decimals: int
    size_step: float
    min_trade_num: float
    max_order_qty: Optional[float]


class SymbolMetaCache:
    def __init__(self, client: BitgetClient, product_type: str = PRODUCT_TYPE, ttl_seconds: int = 300):
        self.client = client
        self.product_type = product_type
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, ContractMeta] = {}
        self._last_fetch: float = 0.0

    def _parse_meta(self, raw: Dict[str, Any]) -> Optional[ContractMeta]:
        sym = normalize_symbol(str(raw.get("symbol", "")))
        if not sym:
            return None

        price_place = _to_int(raw.get("pricePlace"), 0)
        volume_place = _to_int(raw.get("volumePlace"), 0)

        price_end_step = _to_float(raw.get("priceEndStep"))
        size_multiplier = _to_float(raw.get("sizeMultiplier"))

        min_trade_num = _to_float(raw.get("minTradeNum")) or 0.0

        max_order_qty_raw = raw.get("maxOrderQty")
        max_order_qty = _to_float(max_order_qty_raw) if max_order_qty_raw is not None else None

        # --------------------------------------------------------------
        # Tick size
        # - Certains retours ont priceEndStep déjà "réel" (ex 0.0001)
        # - D'autres ont un step entier (ex 1) + pricePlace (ex 5)
        # --------------------------------------------------------------
        tick_size: Optional[float] = None
        if price_end_step is not None:
            if price_end_step < 1:
                tick_size = float(price_end_step)
            else:
                tick_size = float(price_end_step) * (10 ** -max(price_place, 0))

        if tick_size is None or tick_size <= 0:
            # fallback conservateur (évite division par zéro)
            tick_size = 10 ** (-max(price_place, 6)) if price_place > 0 else _tick_fallback(1.0)

        # --------------------------------------------------------------
        # Size step
        # - sizeMultiplier est souvent déjà la granularité de size
        # - si 0/None => fallback
        # --------------------------------------------------------------
        size_step: Optional[float] = None
        if size_multiplier is not None and size_multiplier > 0:
            size_step = float(size_multiplier)
        else:
            size_step = 10 ** (-max(volume_place, 0)) if volume_place > 0 else 1.0

        if size_step <= 0:
            size_step = 1.0

        return ContractMeta(
            symbol=sym,
            price_decimals=max(price_place, 0),
            tick_size=float(tick_size),
            size_decimals=max(volume_place, 0),
            size_step=float(size_step),
            min_trade_num=float(min_trade_num),
            max_order_qty=float(max_order_qty) if max_order_qty is not None else None,
        )

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
            logger.error("[META] error fetching contracts list: %s", e)
            return

        data = resp.get("data") if isinstance(resp, dict) else None
        if not isinstance(data, list):
            logger.error("[META] unexpected contracts response: %s", resp)
            return

        new_cache: Dict[str, ContractMeta] = {}
        for c in data:
            try:
                meta = self._parse_meta(c if isinstance(c, dict) else {})
                if meta:
                    new_cache[meta.symbol] = meta
            except Exception:
                continue

        if new_cache:
            self._cache = new_cache
            self._last_fetch = now
            logger.info("[META] refreshed contracts cache (%d symbols)", len(new_cache))

    async def get(self, symbol: str) -> Optional[ContractMeta]:
        symbol = normalize_symbol(symbol)
        await self._refresh_all()
        meta = self._cache.get(symbol)
        if meta:
            return meta

        # fallback single fetch
        try:
            resp = await self.client._request(
                "GET",
                "/api/v2/mix/market/contracts",
                params={"productType": self.product_type, "symbol": symbol},
                auth=False,
            )
        except Exception as e:
            logger.error("[META] error fetching contract for %s: %s", symbol, e)
            return None

        data = resp.get("data") if isinstance(resp, dict) else None
        if not isinstance(data, list) or not data:
            logger.error("[META] empty contracts data for %s: %s", symbol, resp)
            return None

        meta = self._parse_meta(data[0] if isinstance(data[0], dict) else {})
        if meta:
            self._cache[symbol] = meta
        return meta


# =====================================================================
# Trader
# =====================================================================

class BitgetTrader(BitgetClient):
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        *,
        product_type: str = PRODUCT_TYPE,
        margin_coin: str = MARGIN_COIN,
        margin_mode: str = "isolated",
        target_margin_usdt: float = float(MARGIN_USDT),
        leverage: float = float(LEVERAGE),
    ):
        super().__init__(api_key, api_secret, passphrase)

        self.PRODUCT_TYPE = product_type
        self.MARGIN_COIN = margin_coin
        self.MARGIN_MODE = margin_mode
        self.TARGET_MARGIN_USDT = float(target_margin_usdt)
        self.LEVERAGE = float(leverage)

        self._meta_cache = SymbolMetaCache(self, product_type=self.PRODUCT_TYPE, ttl_seconds=300)

        # mémoriser la size d'entrée par symbole (utile pour TP/SL fractions)
        self._entry_size: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Quantize safe
    # ------------------------------------------------------------------

    def _q_price(self, meta: Optional[ContractMeta], price: float) -> float:
        raw = float(price)
        if raw <= 0:
            return raw

        if not meta:
            return raw

        tick = float(meta.tick_size)
        if tick <= 0:
            return raw

        q = math.floor(raw / tick) * tick
        if q <= 0:
            # Meta suspect (tick trop grand). On évite 0 à tout prix.
            if tick <= raw * 2:
                q = tick
            else:
                q = raw

        return float(_fmt(q, meta.price_decimals))

    def _q_size(self, meta: Optional[ContractMeta], size: float) -> float:
        raw = float(size)
        if raw <= 0:
            raise ValueError(f"invalid size {raw}")

        if not meta:
            return raw

        step = float(meta.size_step)
        if step <= 0:
            step = 1.0

        q = math.floor(raw / step) * step
        if q <= 0:
            if step <= raw * 2:
                q = step
            else:
                q = raw

        # min trade
        min_trade = float(meta.min_trade_num or 0.0)
        if min_trade > 0 and q < min_trade:
            q = min_trade

        # max qty
        if meta.max_order_qty is not None and q > float(meta.max_order_qty):
            q = float(meta.max_order_qty)

        return float(_fmt(q, meta.size_decimals))

    # ------------------------------------------------------------------
    # Response wrapper
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap(resp: Any, *, extra: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {"ok": False, "resp": resp}
        if extra:
            out.update(extra)
        if error:
            out["error"] = error

        if isinstance(resp, dict):
            code = resp.get("code")
            if code == "00000":
                out["ok"] = True
        return out

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def place_limit(
        self,
        symbol: str,
        side: str,
        price: float,
        size: Optional[float] = None,
        *,
        client_oid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Place un LIMIT d'ouverture.
        - side: buy/sell (ou BUY/SELL/LONG/SHORT)
        - size: si None => size calculée à partir margin+levier (notional = margin*levier)
        """
        sym = normalize_symbol(symbol)
        side = _force_side(side)
        client_oid = client_oid or _gen_client_oid("entry")

        meta = await self._meta_cache.get(sym)
        price_q = self._q_price(meta, float(price))

        if price_q <= 0:
            return self._wrap(
                None,
                extra={"symbol": sym, "side": side, "price": price, "price_q": price_q},
                error="invalid_price_after_quantize",
            )

        # sizing
        if size is None:
            notional_target = float(self.TARGET_MARGIN_USDT * self.LEVERAGE)
            size_raw = notional_target / float(price_q)
        else:
            size_raw = float(size)

        try:
            size_q = self._q_size(meta, size_raw)
        except Exception as e:
            return self._wrap(
                None,
                extra={"symbol": sym, "side": side, "price_q": price_q, "size_raw": size_raw},
                error=f"invalid_size: {e}",
            )

        self._entry_size[sym] = float(size_q)

        payload = {
            "symbol": sym,
            "productType": self.PRODUCT_TYPE,
            "marginMode": self.MARGIN_MODE,
            "marginCoin": self.MARGIN_COIN,
            "size": str(size_q),
            "price": str(price_q),
            "side": side,
            "orderType": "limit",
            "force": "gtc",
            "clientOid": client_oid,
        }

        logger.info(
            "[TRADER] place_limit %s %s price=%s size=%s (notional≈%.2f USDT, margin≈%.2f USDT, levier=%.1fx, oid=%s)",
            sym,
            side,
            price_q,
            size_q,
            float(price_q) * float(size_q),
            float(price_q) * float(size_q) / max(self.LEVERAGE, 1e-9),
            self.LEVERAGE,
            client_oid,
        )

        try:
            resp = await self._request("POST", "/api/v2/mix/order/place-order", data=payload, auth=True)
            return self._wrap(resp, extra={"symbol": sym, "side": side, "price": price_q, "size": size_q, "clientOid": client_oid})
        except Exception as e:
            logger.error("[TRADER] place_limit HTTP error %s: %s", sym, e)
            return self._wrap(None, extra={"symbol": sym, "side": side, "price": price_q, "size": size_q}, error=str(e))

    async def _place_plan_market_close(
        self,
        symbol: str,
        open_side: str,
        trigger_price: float,
        size: float,
        *,
        client_oid: Optional[str] = None,
        tag: str = "plan",
    ) -> Dict[str, Any]:
        sym = normalize_symbol(symbol)
        open_side = _force_side(open_side)
        close_side = _close_side(open_side)
        client_oid = client_oid or _gen_client_oid(tag)

        meta = await self._meta_cache.get(sym)
        trigger_q = self._q_price(meta, float(trigger_price))
        if trigger_q <= 0:
            return self._wrap(None, extra={"symbol": sym, "trigger": trigger_price, "trigger_q": trigger_q}, error="invalid_trigger_price")

        try:
            size_q = self._q_size(meta, float(size))
        except Exception as e:
            return self._wrap(None, extra={"symbol": sym, "size_raw": size}, error=f"invalid_size: {e}")

        payload = {
            "planType": "normal_plan",
            "symbol": sym,
            "productType": self.PRODUCT_TYPE,
            "marginMode": self.MARGIN_MODE,
            "marginCoin": self.MARGIN_COIN,
            "size": str(size_q),
            "triggerPrice": str(trigger_q),
            "triggerType": "mark_price",
            "side": close_side,
            "orderType": "market",
            "reduceOnly": "yes",
            "clientOid": client_oid,
        }

        logger.info("[TRADER] %s %s close_side=%s trigger=%s size=%s oid=%s", tag, sym, close_side, trigger_q, size_q, client_oid)

        try:
            resp = await self._request("POST", "/api/v2/mix/order/place-plan-order", data=payload, auth=True)
            return self._wrap(resp, extra={"symbol": sym, "side": close_side, "trigger": trigger_q, "size": size_q, "clientOid": client_oid})
        except Exception as e:
            logger.error("[TRADER] %s HTTP error %s: %s", tag, sym, e)
            return self._wrap(None, extra={"symbol": sym, "side": close_side, "trigger": trigger_q, "size": size_q}, error=str(e))

    async def place_stop_loss(
        self,
        symbol: str,
        open_side: str,
        trigger_price: float,
        *,
        fraction: float = 1.0,
        client_oid: Optional[str] = None,
    ) -> Dict[str, Any]:
        sym = normalize_symbol(symbol)
        base_size = float(self._entry_size.get(sym, 0.0))
        if base_size <= 0:
            return self._wrap(None, extra={"symbol": sym}, error="missing_entry_size_for_sl")

        frac = max(0.0, min(1.0, float(fraction)))
        size = base_size * frac if frac > 0 else base_size

        return await self._place_plan_market_close(
            sym,
            open_side,
            trigger_price,
            size,
            client_oid=client_oid,
            tag="sl",
        )

    async def place_take_profit(
        self,
        symbol: str,
        open_side: str,
        trigger_price: float,
        *,
        fraction: float = 0.5,
        client_oid: Optional[str] = None,
        tag: str = "tp",
    ) -> Dict[str, Any]:
        sym = normalize_symbol(symbol)
        base_size = float(self._entry_size.get(sym, 0.0))
        if base_size <= 0:
            return self._wrap(None, extra={"symbol": sym}, error="missing_entry_size_for_tp")

        frac = max(0.0, min(1.0, float(fraction)))
        size = base_size * frac if frac > 0 else base_size

        return await self._place_plan_market_close(
            sym,
            open_side,
            trigger_price,
            size,
            client_oid=client_oid,
            tag=tag,
        )
