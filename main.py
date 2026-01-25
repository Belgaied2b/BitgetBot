# =====================================================================
# main.py — Entry point (Railway / Docker / local)
# =====================================================================

from __future__ import annotations

import asyncio
import logging

from logger import setup_logging
from scanner import start_scanner


async def main() -> None:
    setup_logging()
    log = logging.getLogger("main")

    log.info("Bot starting...")

    try:
        await start_scanner()
    except asyncio.CancelledError:
        log.warning("Bot cancelled (shutdown).")
        raise
    except Exception:
        log.exception("Unhandled exception in main()")
        raise


if __name__ == "__main__":
    # Standard path
    try:
        asyncio.run(main())
    except RuntimeError:
        # Fallback when an event loop already exists (some PaaS / notebooks)
        loop = asyncio.get_event_loop()
        loop.create_task(main())
        loop.run_forever()
