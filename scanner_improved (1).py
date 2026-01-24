"""
scanner_improved.py – Scanner asynchrone avec contrôle de concurrence et découplage des responsabilités

Ce module sépare la détection de signaux, l’exécution des ordres et la surveillance en
composants distincts. Il utilise un sémaphore pour limiter le nombre de tâches
concurrentes et ajoute des timeouts explicites sur la prise d’ordres. Les
paramètres sont tirés de settings_improved.py.
"""
import asyncio
from typing import List, Optional

from settings_improved import get_settings

class SignalDetector:
    async def detect_signals(self, symbols: List[str]) -> List[str]:
        """Détecte les symboles présentant un setup de trading (mock)."""
        await asyncio.sleep(0.1)  # simulation d’analyse
        return symbols  # retourne tous les symboles en exemple

class OrderExecutor:
    async def place_order(self, symbol: str, side: str, notional: float) -> bool:
        """Place un ordre sur l’échange. Retourne True si succès, False sinon."""
        settings = get_settings().scanner
        try:
            await asyncio.wait_for(self._send_order(symbol, side, notional), timeout=settings.order_timeout_sec)
            return True
        except asyncio.TimeoutError:
            print(f"[OrderExecutor] Timeout lors de la prise d’ordre pour {symbol}")
            return False

    async def _send_order(self, symbol: str, side: str, notional: float) -> None:
        # Ici on implémente l’appel réel à l’API de l’exchange
        await asyncio.sleep(0.5)  # mock de latence réseau
        print(f"[OrderExecutor] Ordre {side} {symbol} notional={notional} exécuté")

class Watchdog:
    async def monitor_positions(self) -> None:
        """Surveille les positions ouvertes et applique des stops/timeouts."""
        while True:
            await asyncio.sleep(10)
            print("[Watchdog] Vérification des positions…")

class Scanner:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.detector = SignalDetector()
        self.executor = OrderExecutor()
        self.watchdog = Watchdog()
        self._semaphore = asyncio.Semaphore(get_settings().scanner.max_concurrent_tasks)

    async def _handle_symbol(self, symbol: str) -> None:
        async with self._semaphore:
            signals = await self.detector.detect_signals([symbol])
            for sym in signals:
                # exemple : passe systématiquement un ordre d’achat pour chaque symbole signalé
                success = await self.executor.place_order(sym, "BUY", 1.0)
                if success:
                    print(f"[Scanner] Ordre placé pour {sym}")

    async def start(self) -> None:
        # démarrage du watchdog en tâche de fond
        asyncio.create_task(self.watchdog.monitor_positions())
        # boucle principale de scan
        while True:
            tasks = [self._handle_symbol(sym) for sym in self.symbols]
            await asyncio.gather(*tasks)
            # intervalle entre les scans (exemple : 60 s)
            await asyncio.sleep(60)

if __name__ == "__main__":
    # exemple d’utilisation
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    scanner = Scanner(symbols)
    asyncio.run(scanner.start())
