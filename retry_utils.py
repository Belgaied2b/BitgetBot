# =====================================================================
# retry_utils.py — Retry automatique pour les appels API
# =====================================================================
import asyncio
import random


async def retry_async(fn, retries: int = 3, base_delay: float = 0.5):
    """
    Exécute fn() avec retry automatique.
    fn doit être une fonction async.
    """
    for i in range(retries):
        try:
            return await fn()
        except Exception:
            if i == retries - 1:
                raise
            await asyncio.sleep(base_delay * (2 ** i) + random.random() * 0.1)
