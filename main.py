# =====================================================================
# main.py — Entry point Railway Bitget Bot
# =====================================================================

import asyncio
import logging
from scanner import start_scanner

logging.basicConfig(level=logging.INFO)

print("🚀 Bot Bitget Institutionnel — Démarrage...")

async def main():
    try:
        await start_scanner()   # ASYNC SAFE — OK !!
    except Exception as e:
        print(f"❌ ERREUR GLOBALE : {e}")

if __name__ == "__main__":
    try:
        # Tentative normale
        asyncio.run(main())
    except RuntimeError:
        # Si une boucle event existe déjà (cas Railway, PTB)
        loop = asyncio.get_event_loop()
        loop.create_task(main())
        loop.run_forever()
