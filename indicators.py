# =====================================================================
# indicators.py — Core + Institutional Indicators (Desk)
# =====================================================================

from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

# Essaye de récupérer quelques params depuis settings.py, sinon valeurs par défaut
try:
    from settings import (
        VOL_REGIME_ATR_PCT_LOW,
        VOL_REGIME_ATR_PCT_HIGH,
    )
except Exception:
    VOL_REGIME_ATR_PCT_LOW = 0.015
    VOL_REGIME_ATR_PCT_HIGH = 0.035


# =====================================================================
# Helpers de base
# =====================================================================

def _to_close_series(x: Any) -> pd.Series:
    """
    Accepte soit un DataFrame OHLC (avec colonne 'close'),
    soit une Series déjà prête.
    """
    if isinstance(x, pd.Series):
        return x.astype(float)
    if isinstance(x, pd.DataFrame):
        return x["close"].astype(float)
    raise TypeError("Expected DataFrame with 'close' or Series for price input")


# =====================================================================
# Moyennes mobiles (EMA / SMA)
# =====================================================================

def ema(series_or_df: Any, length: int = 20) -> pd.Series:
    """
    EMA classique sur la clôture (ou sur la Series fournie).
    """
    c = _to_close_series(series_or_df)
    return c.ewm(span=length, adjust=False).mean()


def sma(series_or_df: Any, length: int = 20) -> pd.Series:
    """
    SMA classique.
    """
    c = _to_close_series(series_or_df)
    return c.rolling(window=length, min_periods=1).mean()


# =====================================================================
# RSI
# =====================================================================

def rsi(series_or_df: Any, length: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI), 0-100.
    """
    c = _to_close_series(series_or_df)
    delta = c.diff()

    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain_series = pd.Series(gain, index=c.index)
    loss_series = pd.Series(loss, index=c.index)

    avg_gain = gain_series.ewm(alpha=1.0 / length, adjust=False).mean()
    avg_loss = loss_series.ewm(alpha=1.0 / length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    rsi_val = rsi_val.fillna(50.0)

    return rsi_val


# =====================================================================
# MACD
# =====================================================================

def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD standard :
        - macd_line = EMA(fast) - EMA(slow)
        - signal_line = EMA(macd_line, signal)
        - hist = macd_line - signal_line

    Retourne DataFrame avec colonnes ['macd', 'signal', 'hist'].
    """
    close = df["close"].astype(float)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line

    out = pd.DataFrame(
        {
            "macd": macd_line,
            "signal": signal_line,
            "hist": hist,
        },
        index=df.index,
    )
    return out


# =====================================================================
# ATR (corrigé) + alias true_atr
# =====================================================================

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Average True Range (ATR) simple, basé sur :
      TR = max(
        high - low,
        |high - prev_close|,
        |low - prev_close|
      )
      ATR = moyenne mobile simple sur 'length' périodes.
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_raw = tr.rolling(window=length, min_periods=1).mean()
    # Correction du FutureWarning : on utilise .bfill() au lieu de fillna(method="bfill")
    return atr_raw.bfill().fillna(0.0)


def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Alias pour atr(), pour compatibilité éventuelle.
    """
    return atr(df, length=length)


def true_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Alias backward-compat pour les modules (stops.py) qui importent true_atr.
    Utilise la même logique que atr().
    """
    return atr(df, length=length)


# =====================================================================
# OTE (Optimal Trade Entry) — approx
# =====================================================================

def compute_ote(
    df: pd.DataFrame,
    bias: str,
    lookback: int = 50,
) -> Dict[str, Any]:
    """
    Approximation "OTE ICT" :
      - on prend le plus bas / plus haut sur les N dernières bougies,
      - on calcule les niveaux 0.62 / 0.705 (golden zone),
      - on vérifie si le prix actuel est dans la zone "OTE" du côté cohérent.

    Retour :
      {
        "in_ote": bool,
        "ote_low": float,
        "ote_high": float,
      }
    """
    if df is None or len(df) < 10:
        return {"in_ote": False, "ote_low": None, "ote_high": None}

    sub = df.tail(lookback)
    high = float(sub["high"].max())
    low = float(sub["low"].min())
    last = float(sub["close"].iloc[-1])

    if high <= low:
        return {"in_ote": False, "ote_low": None, "ote_high": None}

    # Fibonacci retracements
    diff = high - low
    fib_62 = high - 0.62 * diff
    fib_705 = high - 0.705 * diff

    ote_low = min(fib_62, fib_705)
    ote_high = max(fib_62, fib_705)

    in_ote = ote_low <= last <= ote_high

    return {
        "in_ote": bool(in_ote),
        "ote_low": float(ote_low),
        "ote_high": float(ote_high),
    }


# =====================================================================
# Volatility regime (basé sur ATR%)
# =====================================================================

def volatility_regime(df: pd.DataFrame, atr_length: int = 14) -> str:
    """
    Classe la volatilité en fonction de ATR% sur les dernières bougies.

    ATR% = ATR / close
    Règles par défaut (modifiables via settings) :
      - ATR% < VOL_REGIME_ATR_PCT_LOW  -> "LOW"
      - ATR% > VOL_REGIME_ATR_PCT_HIGH -> "HIGH"
      - sinon                           -> "MEDIUM"
    """
    if df is None or len(df) < atr_length + 5:
        return "UNKNOWN"

    atr_series = atr(df, length=atr_length)
    close = df["close"].astype(float)
    atr_pct = (atr_series / close).replace([np.inf, -np.inf], np.nan)

    last = float(atr_pct.iloc[-1]) if not atr_pct.empty else np.nan
    if not np.isfinite(last):
        return "UNKNOWN"

    if last < VOL_REGIME_ATR_PCT_LOW:
        return "LOW"
    if last > VOL_REGIME_ATR_PCT_HIGH:
        return "HIGH"
    return "MEDIUM"


# =====================================================================
# Extension signal (sur-extension prix / RSI)
# =====================================================================

def extension_signal(df: pd.DataFrame, ema_fast_len: int = 20, ema_slow_len: int = 50) -> Optional[str]:
    """
    Détecte une sur-extension simple :
      - distance du prix par rapport à EMA50
      - RSI extrême

    Retourne :
      - "OVEREXTENDED_LONG"  : prix très au-dessus, RSI > 70
      - "OVEREXTENDED_SHORT" : prix très en-dessous, RSI < 30
      - None                 : pas de signal d'extension fort
    """
    if df is None or len(df) < max(ema_slow_len, 30):
        return None

    close = df["close"].astype(float)
    ema_fast = ema(close, ema_fast_len)
    ema_slow = ema(close, ema_slow_len)
    r = rsi(close)

    last_close = float(close.iloc[-1])
    last_ema_slow = float(ema_slow.iloc[-1])
    last_rsi = float(r.iloc[-1])

    if last_ema_slow <= 0:
        return None

    dist_pct = (last_close - last_ema_slow) / last_ema_slow

    # seuils assez génériques, pas trop agressifs
    if dist_pct > 0.06 and last_rsi > 70:
        return "OVEREXTENDED_LONG"
    if dist_pct < -0.06 and last_rsi < 30:
        return "OVEREXTENDED_SHORT"

    return None


# =====================================================================
# Momentum institutionnel (EMA spread + MACD + RSI + volume)
# =====================================================================

def institutional_momentum(df: pd.DataFrame) -> str:
    """
    Synthèse momentum orientée desk :

      - EMA20 vs EMA50 (trend short-term)
      - MACD (direction / force)
      - RSI (position dans le range 0-100)
      - Volume relatif

    Retour :
      - "STRONG_BULLISH"
      - "BULLISH"
      - "NEUTRAL"
      - "BEARISH"
      - "STRONG_BEARISH"
    """
    if df is None or len(df) < 60:
        return "NEUTRAL"

    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    ema_fast = ema(close, 20)
    ema_slow = ema(close, 50)
    m = macd(df)
    r = rsi(close)

    # Spread EMA
    ema_spread = ema_fast - ema_slow
    ema_spread_last = float(ema_spread.iloc[-1])

    # MACD
    macd_last = float(m["macd"].iloc[-1])
    macd_hist_last = float(m["hist"].iloc[-1])

    # RSI
    r_last = float(r.iloc[-1])

    # Volume relatif
    vol_sub = volume.tail(40)
    vol_avg = float(vol_sub.mean()) if not vol_sub.empty else 0.0
    vol_last = float(volume.iloc[-1])
    vol_factor = vol_last / vol_avg if vol_avg > 0 else 1.0

    score = 0.0

    # EMA spread
    if ema_spread_last > 0:
        score += 1.0
    elif ema_spread_last < 0:
        score -= 1.0

    # MACD + hist
    if macd_last > 0 and macd_hist_last > 0:
        score += 1.0
    elif macd_last < 0 and macd_hist_last < 0:
        score -= 1.0

    # RSI
    if r_last > 60:
        score += 0.5
    elif r_last < 40:
        score -= 0.5

    # Volume
    if vol_factor > 1.5:
        score += 0.5
    elif vol_factor < 0.7:
        score -= 0.5

    # classification finale
    if score >= 2.0:
        return "STRONG_BULLISH"
    if score >= 0.5:
        return "BULLISH"
    if score <= -2.0:
        return "STRONG_BEARISH"
    if score <= -0.5:
        return "BEARISH"
    return "NEUTRAL"


# =====================================================================
# Composite momentum (score 0-100 + label)
# =====================================================================

def composite_momentum(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Retourne un score "momentum composite" dans [0, 100] et un label textuel.

    Composants :
      - trend_score : basé sur l'EMA spread (direction)
      - macd_score  : basé sur MACD/hist
      - rsi_score   : récompense RSI bull / bear raisonnable
      - vol_score   : volume relatif
      - ext_score   : pénalité si extension forte (overextended)
    """
    if df is None or len(df) < 60:
        return {
            "score": 50.0,
            "label": "NEUTRAL",
            "components": {},
        }

    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    ema_fast = ema(close, 20)
    ema_slow = ema(close, 50)
    ema_spread = ema_fast - ema_slow
    ema_spread_last = float(ema_spread.iloc[-1])

    m = macd(df)
    macd_last = float(m["macd"].iloc[-1])
    macd_hist_last = float(m["hist"].iloc[-1])

    r = rsi(close)
    r_last = float(r.iloc[-1])

    vol_sub = volume.tail(40)
    vol_avg = float(vol_sub.mean()) if not vol_sub.empty else 0.0
    vol_last = float(volume.iloc[-1])
    vol_factor = vol_last / vol_avg if vol_avg > 0 else 1.0

    ext = extension_signal(df)

    # Trend score in [-1, +1]
    if ema_spread_last > 0:
        trend_score = 1.0
    elif ema_spread_last < 0:
        trend_score = -1.0
    else:
        trend_score = 0.0

    # MACD score approx in [-1, +1]
    macd_raw = macd_last + 0.5 * macd_hist_last
    macd_score = float(np.tanh(macd_raw))

    # RSI score : centré sur 50, normalisé
    rsi_score = (r_last - 50.0) / 25.0  # ~ [-2, 2]
    rsi_score = float(np.clip(rsi_score, -2.0, 2.0))

    # Volume score : >1.5 -> +1, <0.7 -> -1
    if vol_factor > 1.5:
        vol_score = 1.0
    elif vol_factor < 0.7:
        vol_score = -1.0
    else:
        vol_score = 0.0

    # Extension penalty
    if ext == "OVEREXTENDED_LONG" or ext == "OVEREXTENDED_SHORT":
        ext_score = -0.7
    else:
        ext_score = 0.0

    # Combine (pondérations arbitraires mais raisonnables)
    raw_score = (
        1.0 * trend_score
        + 0.8 * macd_score
        + 0.8 * rsi_score
        + 0.7 * vol_score
        + 1.0 * ext_score
    )

    # compress & normalise to [0, 100]
    norm = 50.0 + 25.0 * float(np.tanh(raw_score))  # ~[25,75] mais lissé
    norm = float(np.clip(norm, 0.0, 100.0))

    # Label
    if norm >= 70:
        label = "BULLISH"
    elif norm >= 55:
        label = "SLIGHT_BULLISH"
    elif norm <= 30:
        label = "BEARISH"
    elif norm <= 45:
        label = "SLIGHT_BEARISH"
    else:
        label = "NEUTRAL"

    return {
        "score": norm,
        "label": label,
        "components": {
            "trend_score": trend_score,
            "macd_score": macd_score,
            "rsi_score": rsi_score,
            "vol_score": vol_score,
            "ext_score": ext_score,
        },
    }
