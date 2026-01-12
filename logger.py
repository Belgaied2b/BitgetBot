# =====================================================================
# logger.py — Logger couleur avec timestamps, safe async
# =====================================================================
import datetime
from typing import Any, Tuple


class Logger:
    COLORS = {
        "INFO": "\033[94m",
        "SUCCESS": "\033[92m",
        "WARN": "\033[93m",
        "ERROR": "\033[91m",
        "END": "\033[0m",
    }

    @staticmethod
    def _ts() -> str:
        return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def info(cls, msg: str) -> None:
        print(f"{cls.COLORS['INFO']}[{cls._ts()}] INFO: {msg}{cls.COLORS['END']}")

    @classmethod
    def success(cls, msg: str) -> None:
        print(f"{cls.COLORS['SUCCESS']}[{cls._ts()}] SUCCESS: {msg}{cls.COLORS['END']}")

    @classmethod
    def warn(cls, msg: str) -> None:
        print(f"{cls.COLORS['WARN']}[{cls._ts()}] WARNING: {msg}{cls.COLORS['END']}")

    @classmethod
    def error(cls, msg: str) -> None:
        print(f"{cls.COLORS['ERROR']}[{cls._ts()}] ERROR: {msg}{cls.COLORS['END']}")


def _fmt(msg: Any, args: Tuple[Any, ...]) -> str:
    """
    Supporte les appels façon logging:
      log.info("x=%s y=%s", x, y)
    et les appels simples:
      log.info(f"...")
    """
    if not args:
        return str(msg)
    try:
        return str(msg) % args
    except Exception:
        # fallback si placeholders incohérents
        return f"{msg} " + " ".join(map(str, args))


def get_logger(name: str = "app"):
    """
    Logger compatible avec institutional_ws_hub.py:
      - info(...)
      - warning(...) (mappé sur Logger.warn)
      - warn(...)    (alias)
      - error(...)
      - debug(...)   (no-op)
    """

    class _Compat:
        def __init__(self, n: str):
            self._name = n

        def info(self, msg: Any, *args: Any, **kwargs: Any) -> None:
            Logger.info(f"[{self._name}] {_fmt(msg, args)}")

        def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:
            Logger.warn(f"[{self._name}] {_fmt(msg, args)}")

        def warn(self, msg: Any, *args: Any, **kwargs: Any) -> None:
            self.warning(msg, *args, **kwargs)

        def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
            Logger.error(f"[{self._name}] {_fmt(msg, args)}")

        def debug(self, msg: Any, *args: Any, **kwargs: Any) -> None:
            # no-op par défaut (changez en Logger.info si vous voulez)
            return

    return _Compat(name)
