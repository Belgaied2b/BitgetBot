# =====================================================================
# logger.py — Logger couleur avec timestamps, safe async
# =====================================================================
import datetime


class Logger:
    COLORS = {
        "INFO": "\033[94m",
        "SUCCESS": "\033[92m",
        "WARN": "\033[93m",
        "ERROR": "\033[91m",
        "END": "\033[0m",
    }

    @staticmethod
    def _ts():
        return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def info(cls, msg: str):
        print(f"{cls.COLORS['INFO']}[{cls._ts()}] INFO: {msg}{cls.COLORS['END']}")

    @classmethod
    def success(cls, msg: str):
        print(f"{cls.COLORS['SUCCESS']}[{cls._ts()}] SUCCESS: {msg}{cls.COLORS['END']}")

    @classmethod
    def warn(cls, msg: str):
        print(f"{cls.COLORS['WARN']}[{cls._ts()}] WARNING: {msg}{cls.COLORS['END']}")

    @classmethod
    def error(cls, msg: str):
        print(f"{cls.COLORS['ERROR']}[{cls._ts()}] ERROR: {msg}{cls.COLORS['END']}")
