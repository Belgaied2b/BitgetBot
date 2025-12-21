# =====================================================================
# scanner.py — Bitget Desk Lead Scanner (Institutionnel H1 + Validation H4)
# =====================================================================

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import time
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from settings import (
    API_KEY,
    API_SECRET,
    API_PASSPHRASE,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    SCAN_INTERVAL_MIN,
    TOP_N_SYMBOLS,
    MAX_ORDERS_PER_SCAN,
    DRY_RUN,
    MARGIN_USDT,
    LEVERAGE,
    RISK_USDT,
)

# OPTIONNELS (si présents dans settings)
try:
    from settings import PRODUCT_TYPE, MARGIN_COIN
except Exception:
    PRODUCT_TYPE = "USDT-FUTURES"
    MARGIN_COIN = "USDT"

from bitget_client import get_client
from bitget_trader import BitgetTrader
from analyze_signal import SignalAnalyzer

from duplicate_guard import DuplicateGuard, fingerprint as make_fingerprint
from risk_manager import RiskManager
from retry_utils import retry_async

logger = logging.getLogger(__name__)

# =====================================================================
# GLOBALS
# =====================================================================

DUP_GUARD = DuplicateGuard(ttl_seconds=3600)
RISK = RiskManager()

TF_H1 = "1H"
TF_H4 = "4H"
CANDLE_LIMIT = 200

MAX_CONCURRENT_FETCH = 8

# Cache metas: symbol -> meta
_META_CACHE: Dict[str, Dict[str, Any]] = {}
_META_TS: Dict[str, float] = {}
_META_TTL = 300  # 5 min


# =====================================================================
# Telegram
# =====================================================================

async def send_telegram(msg: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    import requests  # local import

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }

    def _do():
        try:
            requests.post(url, json=payload, timeout=8)
        except Exception as e:
            logger.error("Telegram error: %s", e)

    await asyncio.to_thread(_do)


# =====================================================================
# Helpers
# =====================================================================

def _is_ok(resp: Any) -> bool:
    if not isinstance(resp, dict):
        return False
    if resp.get("ok") is True:
        return True
    return str(resp.get("code", "")) == "00000"


def _side_to_direction(side: str) -> str:
    s = (side or "").upper()
    return "LONG" if s == "BUY" else "SHORT"


def _to_buy_sell(side: str) -> str:
    s = (side or "").upper()
    return "buy" if s == "BUY" else "sell"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _estimate_tick(price: float) -> float:
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


def _quantize_floor(value: float, step: float) -> float:
    if step <= 0:
        return float(value)
    return math.floor(float(value) / step) * step


def _fmt(v: float, decimals: int) -> str:
    decimals = max(0, int(decimals))
    return f"{float(v):.{decimals}f}"


def _build_signal_message(result: Dict[str, Any]) -> str:
    symbol = result.get("symbol", "?")
    side = result.get("side", "?")
    entry = result.get("entry")
    sl = result.get("sl")
    tp1 = result.get("tp1")
    tp2 = result.get("tp2")
    rr = result.get("rr")
    setup = result.get("setup_type")

    inst = result.get("institutional") or {}
    inst_score = inst.get("institutional_score")
    flow_regime = inst.get("flow_regime")
    funding = inst.get("funding_rate")
    crowd = inst.get("crowding_regime")

    msg = (
        f"🎯 *SIGNAL {symbol}* → *{side}*\n"
        f"• Entrée: `{entry}`\n"
        f"• SL: `{sl}`\n"
    )
    if tp1 is not None:
        msg += f"• TP1: `{tp1}`\n"
    if tp2 is not None:
        msg += f"• TP2: `{tp2}`\n"
    if rr is not None:
        try:
            msg += f"• RR: `{round(float(rr), 3)}`\n"
        except Exception:
            msg += f"• RR: `{rr}`\n"
    if setup:
        msg += f"• Setup: `{setup}`\n"

    if inst_score is not None:
        msg += f"\n🏛 *Institutionnel*\n• Score: `{inst_score}`"
        if flow_regime:
            msg += f"\n• Flow: `{flow_regime}`"
        if crowd:
            msg += f"\n• Crowding: `{crowd}`"
        if funding is not None:
            msg += f"\n• Funding: `{funding}`"

    if DRY_RUN:
        msg += "\n\n🧪 *DRY_RUN=ON* (aucun ordre envoyé)"

    return msg


def _make_trader() -> BitgetTrader:
    """
    Compat avec tes différentes versions de BitgetTrader :
    - certaines acceptent margin_usdt/leverage
    - d’autres target_margin_usdt/leverage
    """
    sig = inspect.signature(BitgetTrader.__init__)
    kwargs: Dict[str, Any] = {}

    if "margin_usdt" in sig.parameters:
        kwargs["margin_usdt"] = float(MARGIN_USDT)
    if "target_margin_usdt" in sig.parameters:
        kwargs["target_margin_usdt"] = float(MARGIN_USDT)

    if "leverage" in sig.parameters:
        kwargs["leverage"] = float(LEVERAGE)

    return BitgetTrader(API_KEY, API_SECRET, API_PASSPHRASE, **kwargs)


# =====================================================================
# Meta fetch (safe)
# =====================================================================

async def _get_contract_meta(client, symbol: str) -> Optional[Dict[str, Any]]:
    now = time.time()
    ts = _META_TS.get(symbol, 0.0)
    if symbol in _META_CACHE and (now - ts) < _META_TTL:
        return _META_CACHE[symbol]

    params = {"productType": PRODUCT_TYPE, "symbol": symbol}
    try:
        js = await client._request(
            "GET",
            "/api/v2/mix/market/contracts",
            params=params,
            auth=False,
        )
    except Exception as e:
        logger.error("[META] fetch error %s: %s", symbol, e)
        return None

    data = js.get("data") if isinstance(js, dict) else None
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        logger.error("[META] empty/unexpected for %s: %s", symbol, js)
        return None

    c = data[0]
    try:
        price_place = int(c.get("pricePlace", 0))
        price_end_step = float(c.get("priceEndStep", 1))
        volume_place = int(c.get("volumePlace", 0))
        size_multiplier = float(c.get("sizeMultiplier", 1))
        min_trade_num = float(c.get("minTradeNum", 0))

        # Tick candidat
        tick = price_end_step * (10 ** -max(price_place, 0))

        meta = {
            "price_place": price_place,
            "volume_place": volume_place,
            "tick": float(tick),
            "step": float(size_multiplier),
            "min_trade_num": float(min_trade_num),
        }

        _META_CACHE[symbol] = meta
        _META_TS[symbol] = now
        return meta
    except Exception as e:
        logger.error("[META] parse error %s: %s | raw=%s", symbol, e, c)
        return None


# =====================================================================
# Execution (direct via _request keyword-only)
# =====================================================================

async def _place_limit_direct(
    trader: BitgetTrader,
    client,
    symbol: str,
    side_buy_sell: str,
    entry: float,
    notional: float,
    client_oid: str,
) -> Optional[Dict[str, Any]]:
    meta = await _get_contract_meta(client, symbol)

    # tick/step safe
    tick = float(meta["tick"]) if meta and meta.get("tick") else 0.0
    if tick <= 0 or tick >= entry:
        tick = _estimate_tick(entry)

    step = float(meta["step"]) if meta and meta.get("step") else 0.0
    if step <= 0:
        step = 1.0

    price_place = int(meta["price_place"]) if meta else 6
    volume_place = int(meta["volume_place"]) if meta else 4
    min_trade = float(meta["min_trade_num"]) if meta else 0.0

    q_price = _quantize_floor(entry, tick)
    if q_price <= 0:
        # dernier filet
        q_price = float(entry)

    raw_size = float(notional) / float(q_price)
    q_size = _quantize_floor(raw_size, step)

    if min_trade > 0 and q_size < min_trade:
        logger.error("[EXEC] %s size %.8f < minTradeNum %.8f", symbol, q_size, min_trade)
        return None

    payload = {
        "symbol": symbol,
        "productType": PRODUCT_TYPE,
        "marginCoin": MARGIN_COIN,
        "size": _fmt(q_size, volume_place),
        "price": _fmt(q_price, price_place),
        "side": side_buy_sell,              # "buy" / "sell"
        "orderType": "limit",
        "timeInForceValue": "normal",
        "clientOid": str(client_oid),
    }

    logger.info(
        "[EXEC] place LIMIT %s %s price=%s size=%s notional≈%.2f",
        symbol, side_buy_sell, payload["price"], payload["size"], q_price * q_size
    )

    try:
        resp = await trader._request(
            "POST",
            "/api/v2/mix/order/place-order",
            data=payload,
            auth=True,
        )
        return resp if isinstance(resp, dict) else None
    except Exception as e:
        logger.error("[EXEC] place-order HTTP error %s: %s", symbol, e)
        return None


async def _place_plan_direct(
    trader: BitgetTrader,
    client,
    symbol: str,
    open_side_buy_sell: str,
    trigger_price: float,
    size: float,
    client_oid: str,
) -> Optional[Dict[str, Any]]:
    meta = await _get_contract_meta(client, symbol)

    tick = float(meta["tick"]) if meta and meta.get("tick") else 0.0
    if tick <= 0 or tick >= trigger_price:
        tick = _estimate_tick(trigger_price)

    step = float(meta["step"]) if meta and meta.get("step") else 0.0
    if step <= 0:
        step = 1.0

    price_place = int(meta["price_place"]) if meta else 6
    volume_place = int(meta["volume_place"]) if meta else 4
    min_trade = float(meta["min_trade_num"]) if meta else 0.0

    # close side
    if open_side_buy_sell == "buy":
        close_side = "sell"
    else:
        close_side = "buy"

    q_trigger = _quantize_floor(trigger_price, tick)
    if q_trigger <= 0:
        q_trigger = float(trigger_price)

    q_size = _quantize_floor(size, step)
    if min_trade > 0 and q_size < min_trade:
        logger.error("[EXEC] %s plan size %.8f < minTradeNum %.8f", symbol, q_size, min_trade)
        return None

    payload = {
        "symbol": symbol,
        "productType": PRODUCT_TYPE,
        "marginCoin": MARGIN_COIN,
        "size": _fmt(q_size, volume_place),
        "side": close_side,
        "orderType": "limit",
        "timeInForceValue": "normal",
        "triggerType": "mark_price",
        "triggerPrice": _fmt(q_trigger, price_place),
        "executePrice": _fmt(q_trigger, price_place),
        "clientOid": str(client_oid),
    }

    try:
        resp = await trader._request(
            "POST",
            "/api/v2/mix/order/place-plan-order",
            data=payload,
            auth=True,
        )
        return resp if isinstance(resp, dict) else None
    except Exception as e:
        logger.error("[EXEC] place-plan-order HTTP error %s: %s", symbol, e)
        return None


# =====================================================================
# Market data
# =====================================================================

async def _fetch_dfs(client, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    async def _h1():
        return await client.get_klines_df(symbol, TF_H1, CANDLE_LIMIT)

    async def _h4():
        return await client.get_klines_df(symbol, TF_H4, CANDLE_LIMIT)

    df_h1 = await retry_async(_h1, retries=3, base_delay=0.4)
    df_h4 = await retry_async(_h4, retries=3, base_delay=0.4)
    return df_h1, df_h4


# =====================================================================
# Core processing
# =====================================================================

async def process_symbol(
    symbol: str,
    client,
    analyzer: SignalAnalyzer,
    trader: BitgetTrader,
    order_budget: asyncio.Semaphore,
    fetch_sem: asyncio.Semaphore,
) -> None:
    acquired_budget = False
    try:
        async with fetch_sem:
            df_h1, df_h4 = await _fetch_dfs(client, symbol)

        if df_h1 is None or df_h4 is None:
            return
        if getattr(df_h1, "empty", True) or getattr(df_h4, "empty", True):
            return
        if len(df_h1) < 80 or len(df_h4) < 80:
            return

        macro = {}
        result = await analyzer.analyze(symbol, df_h1, df_h4, macro)

        if not result or not result.get("valid"):
            return

        side = str(result.get("side", "")).upper()
        if side not in ("BUY", "SELL"):
            return

        direction = _side_to_direction(side)

        entry = _safe_float(result.get("entry"), 0.0)
        sl = _safe_float(result.get("sl"), 0.0)
        tp1 = _safe_float(result.get("tp1"), 0.0)
        tp2 = _safe_float(result.get("tp2"), 0.0)

        if entry <= 0:
            return

        rr = result.get("rr")
        setup = result.get("setup_type")

        inst = result.get("institutional") or {}
        inst_score = inst.get("institutional_score", 0)

        # anti-doublons
        fp = make_fingerprint(symbol, side, entry, sl, tp1, extra=setup, precision=6)
        if DUP_GUARD.is_duplicate(fp):
            logger.info("[DUP] skip %s %s (déjà envoyé)", symbol, side)
            return

        notional = float(MARGIN_USDT) * float(LEVERAGE)
        allowed, reason = RISK.can_trade(
            symbol=symbol,
            side=direction,
            notional=notional,
            rr=_safe_float(rr, 0.0) if rr is not None else None,
            inst_score=int(inst_score) if inst_score is not None else 0,
            commitment=None,
        )
        if not allowed:
            logger.info("[RISK] reject %s %s → %s", symbol, direction, reason)
            return

        await send_telegram(_build_signal_message(result))
        DUP_GUARD.mark(fp)

        if DRY_RUN:
            return

        # budget ordres
        try:
            await asyncio.wait_for(order_budget.acquire(), timeout=0.01)
            acquired_budget = True
        except asyncio.TimeoutError:
            logger.info("[BUDGET] max orders per scan atteint → skip %s", symbol)
            return

        side_bs = _to_buy_sell(side)
        client_oid_entry = f"entry-{symbol}-{int(time.time() * 1000)}"

        entry_res = await _place_limit_direct(
            trader=trader,
            client=client,
            symbol=symbol,
            side_buy_sell=side_bs,
            entry=entry,
            notional=notional,
            client_oid=client_oid_entry,
        )

        if not _is_ok(entry_res):
            logger.error("[ENTRY] FAILED %s → %s", symbol, entry_res)
            await send_telegram(f"❌ *ENTRY FAILED* {symbol} {side} @ `{entry}`\n`{entry_res}`")
            if acquired_budget:
                order_budget.release()
            return

        # register open
        RISK.register_open(symbol=symbol, side=direction, notional=notional, risk=float(RISK_USDT))
        logger.info("[ENTRY] OK %s %s @ %s (oid=%s)", symbol, side, entry, client_oid_entry)

        # SL / TP (optionnels)
        # On ferme 50% sur TP1 et 50% sur TP2 si dispo (sinon tout sur TP1)
        # Taille estimée: notional / entry (approx), re-quantifiée au moment de l'ordre plan
        approx_size = notional / entry

        if sl > 0:
            client_oid_sl = f"sl-{symbol}-{int(time.time() * 1000)}"
            sl_res = await _place_plan_direct(
                trader=trader,
                client=client,
                symbol=symbol,
                open_side_buy_sell=side_bs,
                trigger_price=sl,
                size=approx_size,
                client_oid=client_oid_sl,
            )
            if not _is_ok(sl_res):
                logger.error("[SL] FAILED %s → %s", symbol, sl_res)

        if tp1 > 0:
            client_oid_tp1 = f"tp1-{symbol}-{int(time.time() * 1000)}"
            tp1_res = await _place_plan_direct(
                trader=trader,
                client=client,
                symbol=symbol,
                open_side_buy_sell=side_bs,
                trigger_price=tp1,
                size=approx_size * (0.5 if tp2 > 0 else 1.0),
                client_oid=client_oid_tp1,
            )
            if not _is_ok(tp1_res):
                logger.error("[TP1] FAILED %s → %s", symbol, tp1_res)

        if tp2 > 0:
            client_oid_tp2 = f"tp2-{symbol}-{int(time.time() * 1000)}"
            tp2_res = await _place_plan_direct(
                trader=trader,
                client=client,
                symbol=symbol,
                open_side_buy_sell=side_bs,
                trigger_price=tp2,
                size=approx_size * 0.5,
                client_oid=client_oid_tp2,
            )
            if not _is_ok(tp2_res):
                logger.error("[TP2] FAILED %s → %s", symbol, tp2_res)

    except Exception:
        logger.exception("[%s] process_symbol error", symbol)
        if acquired_budget:
            try:
                order_budget.release()
            except Exception:
                pass


# =====================================================================
# Scan loop
# =====================================================================

async def scan_once(client, analyzer: SignalAnalyzer, trader: BitgetTrader) -> None:
    symbols = await retry_async(client.get_contracts_list, retries=3, base_delay=0.6)
    if not symbols:
        logger.warning("⚠️ get_contracts_list() vide")
        return

    symbols = list(symbols)[: int(TOP_N_SYMBOLS)]
    logger.info("=== START SCAN (%d symbols) ===", len(symbols))

    fetch_sem = asyncio.Semaphore(MAX_CONCURRENT_FETCH)
    order_budget = asyncio.Semaphore(int(MAX_ORDERS_PER_SCAN))

    async def _worker(sym: str):
        await process_symbol(sym, client, analyzer, trader, order_budget, fetch_sem)

    await asyncio.gather(*[_worker(sym) for sym in symbols])

    logger.info("=== END SCAN ===")


async def start_scanner() -> None:
    logging.basicConfig(level=logging.INFO)

    client = await get_client(API_KEY, API_SECRET, API_PASSPHRASE)
    trader = _make_trader()
    analyzer = SignalAnalyzer()

    logger.info("🚀 Scanner started | interval=%s min | dry_run=%s", SCAN_INTERVAL_MIN, DRY_RUN)

    while True:
        t0 = time.time()
        try:
            await scan_once(client, analyzer, trader)
        except Exception:
            logger.exception("SCAN ERROR")

        dt = time.time() - t0
        sleep_s = max(1, int(float(SCAN_INTERVAL_MIN) * 60 - dt))
        await asyncio.sleep(sleep_s)


if __name__ == "__main__":
    try:
        asyncio.run(start_scanner())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.create_task(start_scanner())
        loop.run_forever()
