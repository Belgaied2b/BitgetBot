"""
settings_improved.py — Chargement de configuration depuis un fichier TOML avec rechargement dynamique

Ce module utilise le format TOML pour stocker la configuration du desk et expose
un objet Settings accessible dans tout le projet. Un simple appel à
`reload_settings()` mettra à jour les valeurs à partir du fichier.
"""
import tomllib
from dataclasses import dataclass, field
from typing import Optional
import threading
import os

_DEFAULT_CONFIG_PATH = os.getenv("BITGETBOT_CONFIG_PATH", "settings.toml")
_lock = threading.Lock()

@dataclass
class RiskSettings:
    risk_usdt: float = 50.0
    equity_usdt: float = 10000.0
    max_gross_exposure: float = 2.0
    max_symbol_exposure: float = 0.5

@dataclass
class ScannerSettings:
    max_concurrent_tasks: int = 20
    order_timeout_sec: float = 5.0

@dataclass
class TPClampSettings:
    momentum_coeff: float = 0.5
    atr_coeff: float = 1.0
    liquidity_coeff: float = 0.5

@dataclass
class MacroSettings:
    use_interest_rates: bool = False
    use_inflation: bool = False
    use_stablecoin_volume: bool = False
    use_social_sentiment: bool = False
    smoothing_window: int = 5

@dataclass
class OptionsSettings:
    vol_smoothing_window: int = 3

@dataclass
class Settings:
    risk: RiskSettings = field(default_factory=RiskSettings)
    scanner: ScannerSettings = field(default_factory=ScannerSettings)
    tp_clamp: TPClampSettings = field(default_factory=TPClampSettings)
    macro: MacroSettings = field(default_factory=MacroSettings)
    options: OptionsSettings = field(default_factory=OptionsSettings)

# instance globale
_current_settings = Settings()


def _merge_dataclass(dc, data: dict) -> None:
    """Merge un dictionnaire dans un dataclass en convertissant les types."""
    for k, v in data.items():
        if hasattr(dc, k):
            attr = getattr(dc, k)
            # récursion pour les sous dataclasses
            if hasattr(attr, '__dataclass_fields__') and isinstance(v, dict):
                _merge_dataclass(attr, v)
            else:
                try:
                    setattr(dc, k, type(attr)(v))
                except Exception:
                    setattr(dc, k, v)


def load_settings(path: Optional[str] = None) -> Settings:
    """
    Charge la configuration à partir d'un fichier TOML et retourne un objet Settings.
    Si le fichier n'existe pas, les valeurs par défaut sont utilisées.
    """
    global _current_settings
    cfg_path = path or _DEFAULT_CONFIG_PATH
    data = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "rb") as f:
            try:
                data = tomllib.load(f)
            except Exception as e:
                print(f"[Settings] Erreur de parsing du fichier {cfg_path}: {e}")
    settings = Settings()
    for section, sec_data in data.items():
        if hasattr(settings, section):
            _merge_dataclass(getattr(settings, section), sec_data)
    return settings


def reload_settings(path: Optional[str] = None) -> None:
    """
    Recharge la configuration globalement. Utiliser cette fonction pour mettre à jour
    les paramètres à chaud dans l'application.
    """
    global _current_settings
    with _lock:
        _current_settings = load_settings(path)
        print(f"[Settings] Configuration rechargée depuis {path or _DEFAULT_CONFIG_PATH}")


def get_settings() -> Settings:
    """Retourne la configuration actuelle (lecture seule)."""
    return _current_settings

# Charger la configuration au premier import
def _init() -> None:
    global _current_settings
    _current_settings = load_settings()

_init()
