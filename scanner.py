# =====================================================================
# scanner.py — Bitget Desk Lead Scanner (Institutionnel H1 + Validation H4)
# =====================================================================

from __future__ import annotations

import asyncio
import inspect
import logging
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
    PRODUCT_TYPE,
    MARGIN_COIN,
)

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


# =====================================================================
# Telegram (async-safe via to_thread)
# =====================================================================

async def send_telegram(msg: str) -> None:
    """Envoi Telegram sans bloquer l'event loop."""
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
    """Bitget success: code == '00000' ou resp['ok']==True."""
    if not isinstance(resp, dict):
        return False
    if resp.get("ok") is True:
        return True
    return str(resp.get("code", "")) == "00000"


def _side_to_direction(side: str) -> str:
    s = (side or "").upper()
    return "LONG" if s == "BUY" else "SHORT"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


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


def _make_trader(client) -> Optional[BitgetTrader]:
    """
    Crée BitgetTrader sans jamais passer un kwarg non supporté.
    Essaie:
      1) BitgetTrader(client, **kwargs)
      2) BitgetTrader(API_KEY, API_SECRET, API_PASSPHRASE, **kwargs)
    Si ça échoue => None (le scan continue sans exécution).
    """
    kwargs: Dict[str, Any] = {}
    try:
        sig = inspect.signature(BitgetTrader.__init__)
        params = sig.parameters

        # kwargs optionnels (uniquement si existants)
        if "product_type" in params:
            kwargs["product_type"] = PRODUCT_TYPE
        if "margin_coin" in params:
            kwargs["margin_coin"] = MARGIN_COIN
        if "margin_mode" in params:
            kwargs["margin_mode"] = "isolated"

        # sizing (selon impl)
        if "target_margin_usdt" in params:
            kwargs["target_margin_usdt"] = float(MARGIN_USDT)
        if "margin_usdt" in params:
            kwargs["margin_usdt"] = float(MARGIN_USDT)
        if "leverage" in params:
            kwargs["leverage"] = float(LEVERAGE)

    except Exception:
        # si inspect foire, on tente minimal
        kwargs = {}

    # 1) style: BitgetTrader(client, ...)
    try:
        return BitgetTrader(client, **kwargs)
    except TypeError:
        pass
    except Exception as e:
        logger.error("[TRADER] init with client failed: %s", e)

    # 2) style: BitgetTrader(API_KEY, API_SECRET, API_PASSPHRASE, ...)
    try:
        return BitgetTrader(API_KEY, API_SECRET, API_PASSPHRASE, **kwargs)
    except TypeError:
        pass
    except Exception as e:
        logger.error("[TRADER] init with keys failed: %s", e)

    # 3) dernier essai minimal avec keys
    try:
        return BitgetTrader(API_KEY, API_SECRET, API_PASSPHRASE)
    except Exception as e:
        logger.error("[TRADER] init minimal failed: %s", e)
        return None


async def _place_limit_compat(
    trader: BitgetTrader,
    *,
    symbol: str,
    side: str,
    entry: float,
    notional: float,
    client_oid: str,
    sl: float,
    tp1: float,
) -> Any:
    """
    Appel place_limit compatible signatures variées.
    - size / qty / amount
    - client_oid / clientOid
    - preset_sl/preset_tp si dispo
    """
    qty = (float(notional) / float(entry)) if entry > 0 else 0.0

    try:
        sig = inspect.signature(trader.place_limit)
        params = sig.parameters
    except Exception:
        params = {}

    kwargs: Dict[str, Any] = {}

    # base fields
    if "symbol" in params:
        kwargs["symbol"] = symbol
    if "side" in params:
        kwargs["side"] = side
    if "price" in params:
        kwargs["price"] = entry
    elif "entry" in params:
        kwargs["entry"] = entry

    # qty/size
    if "size" in params:
        kwargs["size"] = qty
    elif "qty" in params:
        kwargs["qty"] = qty
    elif "amount" in params:
        kwargs["amount"] = qty

    # client oid name
    if "client_oid" in params:
        kwargs["client_oid"] = client_oid
    elif "clientOid" in params:
        kwargs["clientOid"] = client_oid

    # optional presets
    if "preset_sl" in params:
        kwargs["preset_sl"] = sl
    if "preset_tp" in params:
        kwargs["preset_tp"] = (tp1 if tp1 > 0 else None)

    # Try kwargs call
    try:
        return await trader.place_limit(**kwargs)
    except TypeError:
        # Fallback positional (différents ordres possibles selon impl)
        try:
            return await trader.place_limit(symbol, side, entry, qty, client_oid)
        except TypeError:
            try:
                return await trader.place_limit(symbol, side, entry, qty)
            except TypeError:
                return await trader.place_limit(symbol, side, entry)


# =====================================================================
# Core processing
# =====================================================================

async def _fetch_dfs(client, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    async def _h1():
        return await client.get_klines_df(symbol, TF_H1, CANDLE_LIMIT)

    async def _h4():
        return await client.get_klines_df(symbol, TF_H4, CANDLE_LIMIT)

    df_h1 = await retry_async(_h1, retries=3, base_delay=0.4)
    df_h4 = await retry_async(_h4, retries=3, base_delay=0.4)
    return df_h1, df_h4


async def process_symbol(
    symbol: str,
    client,
    analyzer: SignalAnalyzer,
    trader: Optional[BitgetTrader],
    order_budget: asyncio.Semaphore,
    fetch_sem: asyncio.Semaphore,
) -> None:
    """
    Pipeline complet :
    - fetch H1/H4
    - analyze
    - duplicate guard
    - risk checks
    - telegram
    - place limit
    """
    try:
        async with fetch_sem:
            df_h1, df_h4 = await _fetch_dfs(client, symbol)

        if df_h1 is None or df_h4 is None:
            return
        if getattr(df_h1, "empty", True) or getattr(df_h4, "empty", True):
            return
        if len(df_h1) < 80 or len(df_h4) < 80:
            return

        logger.info("[SCAN] %s analyzing...", symbol)

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
        if entry <= 0 or sl <= 0 or abs(entry - sl) < 1e-12:
            logger.warning("[SKIP] %s invalid entry/sl entry=%s sl=%s", symbol, entry, sl)
            return

        tp1_val = _safe_float(result.get("tp1"), 0.0)

        rr = result.get("rr")
        setup = result.get("setup_type")
        inst = result.get("institutional") or {}
        inst_score = inst.get("institutional_score", 0)

        fp = make_fingerprint(symbol, side, entry, sl, tp1_val, extra=setup, precision=6)
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

        # Telegram + mark duplicate
        await send_telegram(_build_signal_message(result))
        DUP_GUARD.mark(fp)

        # Si DRY_RUN ou trader absent => on ne place pas d'ordre, mais le scan continue
        if DRY_RUN or trader is None:
            if trader is None:
                logger.warning("[EXEC] trader not initialized -> skip order placement (scan continues)")
            return

        # Budget ordres par scan
        try:
            await asyncio.wait_for(order_budget.acquire(), timeout=0.01)
        except asyncio.TimeoutError:
            logger.info("[BUDGET] max orders per scan atteint → skip %s", symbol)
            return

        client_oid = f"{symbol}-{int(time.time() * 1000)}"

        entry_res = await _place_limit_compat(
            trader,
            symbol=symbol,
            side=side,
            entry=entry,
            notional=notional,
            client_oid=client_oid,
            sl=sl,
            tp1=tp1_val,
        )

        if not _is_ok(entry_res):
            logger.error("[ENTRY] FAILED %s → %s", symbol, entry_res)
            await send_telegram(f"❌ *ENTRY FAILED* {symbol} {side} @ `{entry}`\n`{entry_res}`")
            # si échec, on rend le slot budget (sinon tu perds 1 ordre possible)
            try:
                order_budget.release()
            except Exception:
                pass
            return

        RISK.register_open(symbol=symbol, side=direction, notional=notional, risk=float(RISK_USDT))
        logger.info("[ENTRY] OK %s %s @ %s (oid=%s)", symbol, side, entry, client_oid)

    except Exception:
        logger.exception("[%s] process_symbol error", symbol)


# =====================================================================
# Scan loop
# =====================================================================

async def scan_once(client, analyzer: SignalAnalyzer, trader: Optional[BitgetTrader]) -> None:
    symbols = await retry_async(client.get_contracts_list, retries=3, base_delay=0.6)
    if not symbols:
        logger.warning("⚠️ get_contracts_list() vide")
        return

    symbols = sorted(set(map(str.upper, symbols)))
    symbols = symbols[: int(TOP_N_SYMBOLS)]

    logger.info("📊 Scan %d symboles (TOP_N_SYMBOLS=%s)", len(symbols), TOP_N_SYMBOLS)

    fetch_sem = asyncio.Semaphore(MAX_CONCURRENT_FETCH)
    order_budget = asyncio.Semaphore(int(MAX_ORDERS_PER_SCAN))

    async def _worker(sym: str):
        await process_symbol(sym, client, analyzer, trader, order_budget, fetch_sem)

    await asyncio.gather(*[_worker(sym) for sym in symbols])


async def start_scanner() -> None:
    """Démarre le scanner en boucle infinie."""
    logging.basicConfig(level=logging.INFO)

    client = await get_client(API_KEY, API_SECRET, API_PASSPHRASE)

    trader = _make_trader(client)
    if trader is None:
        logger.error("❌ Trader init failed -> bot will run scan/telegram only (no orders).")

    analyzer = SignalAnalyzer()

    logger.info(
        "🚀 Scanner started | interval=%s min | dry_run=%s | trader=%s",
        SCAN_INTERVAL_MIN,
        DRY_RUN,
        "OK" if trader is not None else "OFF",
    )

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
