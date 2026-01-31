"""
logger_improved.py – Utilise le module standard `logging` avec support des couleurs et un format cohérent
pour l'ensemble du projet BitgetBot.  La fonction `get_logger` permet de créer un logger
compatibilité ascendante avec l'ancienne interface (info, warning, warn, error, debug).
"""
import logging
import sys
from typing import Any, Tuple

# Codes ANSI pour la coloration des messages suivant le niveau
_COLOR_CODES = {
    "DEBUG": "\033[90m",   # gris clair
    "INFO": "\033[94m",    # bleu
    "SUCCESS": "\033[92m",  # vert
    "WARNING": "\033[93m",  # jaune
    "ERROR": "\033[91m",    # rouge
    "END": "\033[0m",
}

class ColoredFormatter(logging.Formatter):
    """Formatteur qui applique des couleurs ANSI en fonction du niveau."""

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        msg = super().format(record)
        color = _COLOR_CODES.get(levelname, "")
        end = _COLOR_CODES["END"] if color else ""
        return f"{color}{msg}{end}"


def _setup_root_logger() -> None:
    """Configure le logger racine une seule fois."""
    root = logging.getLogger()
    if root.handlers:
        return  # déjà configuré
    handler = logging.StreamHandler(sys.stdout)
    formatter = ColoredFormatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    root.setLevel(logging.INFO)
    root.addHandler(handler)


# fonction utilitaire pour formater comme l'ancien logger (_fmt)
def _fmt(msg: Any, args: Tuple[Any, ...]) -> str:
    if not args:
        return str(msg)
    try:
        return str(msg) % args
    except Exception:
        return f"{msg} " + " ".join(map(str, args))


def get_logger(name: str = "app"):
    """
    Retourne un logger nommé avec des méthodes compatibles avec l'ancienne interface.  Chaque message
    est préfixé par le nom du composant pour faciliter le traçage.  Les méthodes `info`,
    `warning`/`warn`, `error` et `debug` sont disponibles.
    """
    _setup_root_logger()

    class _Compat:
        def __init__(self, n: str):
            self._logger = logging.getLogger(n)
            # Ajouter un niveau SUCCESS intermédiaire (entre INFO et WARNING)
            logging.addLevelName(25, "SUCCESS")

        def info(self, msg: Any, *args: Any, **kwargs: Any) -> None:
            self._logger.info(f"[{name}] {_fmt(msg, args)}")

        def success(self, msg: Any, *args: Any, **kwargs: Any) -> None:
            # journalise au niveau 25 (SUCCESS)
            self._logger.log(25, f"[{name}] {_fmt(msg, args)}")

        def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:
            self._logger.warning(f"[{name}] {_fmt(msg, args)}")

        def warn(self, msg: Any, *args: Any, **kwargs: Any) -> None:
            self.warning(msg, *args, **kwargs)

        def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
            self._logger.error(f"[{name}] {_fmt(msg, args)}")

        def debug(self, msg: Any, *args: Any, **kwargs: Any) -> None:
            self._logger.debug(f"[{name}] {_fmt(msg, args)}")

    return _Compat(name)
