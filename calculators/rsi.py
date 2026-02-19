# calculators/rsi.py
"""
RSI (Relative Strength Index) calculator.
"""
import pandas as pd
import numpy as np


def calculate_rsi(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate RSI indicator.
    
    Args:
        bars: DataFrame with 'close' column
        period: RSI period (default 14)
    
    Returns:
        Series with RSI values
    """
    if bars.empty or len(bars) < period + 1:
        return pd.Series(dtype=float)
    
    closes = bars["close"]
    deltas = closes.diff()
    
    gains = deltas.where(deltas > 0, 0)
    losses = -deltas.where(deltas < 0, 0)
    
    avg_gains = gains.rolling(window=period).mean()
    avg_losses = losses.rolling(window=period).mean()
    
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def detect_rsi_divergence(
    bars: pd.DataFrame,
    rsi: pd.Series,
    direction: str,
    lookback_bars: int = 10
) -> bool:
    """
    Detect RSI divergence.
    
    For longs: bearish divergence (price makes higher high, RSI makes lower high)
    For shorts: bullish divergence (price makes lower low, RSI makes higher low)
    
    Args:
        bars: DataFrame with price data
        rsi: RSI series
        direction: "long" or "short"
        lookback_bars: Number of bars to look back for divergence
    
    Returns:
        True if divergence detected, False otherwise
    """
    if len(bars) < lookback_bars or len(rsi) < lookback_bars:
        return False
    
    recent_bars = bars.iloc[-lookback_bars:]
    recent_rsi = rsi.iloc[-lookback_bars:]
    
    # Use integer positions instead of index to avoid MultiIndex issues
    if direction == "long":
        # Bearish divergence: price higher high, RSI lower high
        high_values = recent_bars["high"].values
        price_high_pos = int(np.argmax(high_values))
        
        if price_high_pos < 0 or price_high_pos >= len(recent_rsi):
            return False
        
        rsi_at_price_high = recent_rsi.iloc[price_high_pos]
        
        # Check if current RSI is lower than RSI at price high
        # Only check if the high was not the most recent bar
        current_rsi = recent_rsi.iloc[-1]
        if current_rsi < rsi_at_price_high and price_high_pos < len(recent_bars) - 1:
            return True
    else:  # short
        # Bullish divergence: price lower low, RSI higher low
        low_values = recent_bars["low"].values
        price_low_pos = int(np.argmin(low_values))
        
        if price_low_pos < 0 or price_low_pos >= len(recent_rsi):
            return False
        
        rsi_at_price_low = recent_rsi.iloc[price_low_pos]
        
        # Check if current RSI is higher than RSI at price low
        # Only check if the low was not the most recent bar
        current_rsi = recent_rsi.iloc[-1]
        if current_rsi > rsi_at_price_low and price_low_pos < len(recent_bars) - 1:
            return True
    
    return False

