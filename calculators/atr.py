# calculators/atr.py
"""
ATR (Average True Range) calculator for trailing stops.
"""
import pandas as pd
import numpy as np


def calculate_atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate ATR (Average True Range).
    
    Args:
        bars: DataFrame with 'high', 'low', 'close' columns
        period: ATR period (default 14)
    
    Returns:
        Series with ATR values
    """
    if bars.empty or len(bars) < period + 1:
        return pd.Series(dtype=float)
    
    high = bars["high"]
    low = bars["low"]
    close = bars["close"]
    
    # True Range components
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    # True Range is the maximum of the three
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR is the moving average of True Range
    atr = tr.rolling(window=period).mean()
    
    return atr


def calculate_trailing_stop(
    bars: pd.DataFrame,
    entry_price: float,
    direction: str,
    atr_multiplier: float = 2.0,
    atr_period: int = 14
) -> float:
    """
    Calculate trailing stop based on ATR.
    
    Args:
        bars: DataFrame with price data
        entry_price: Entry price of the position
        direction: "long" or "short"
        atr_multiplier: Multiplier for ATR (default 2.0)
        atr_period: ATR period (default 14)
    
    Returns:
        Trailing stop price
    """
    if bars.empty:
        return entry_price
    
    atr = calculate_atr(bars, atr_period)
    if atr.empty:
        return entry_price
    
    current_atr = atr.iloc[-1]
    current_price = float(bars.iloc[-1]["close"])
    
    if direction == "long":
        # Trailing stop below current price
        trailing_stop = current_price - (current_atr * atr_multiplier)
        # Never move stop down (only up)
        return max(trailing_stop, entry_price)
    else:  # short
        # Trailing stop above current price
        trailing_stop = current_price + (current_atr * atr_multiplier)
        # Never move stop up (only down)
        return min(trailing_stop, entry_price)

