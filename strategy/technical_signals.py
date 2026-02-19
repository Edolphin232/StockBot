# strategy/technical_signals.py
"""
Technical signal monitor - detects support/resistance breaks and other technical signals.
Sends notifications when important levels are broken.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np
import pytz
from calculators.vwap import calculate_vwap

EASTERN = pytz.timezone("US/Eastern")


@dataclass
class TechnicalSignal:
    """Represents a technical signal."""
    signal_type: str  # "support_break", "resistance_break", "ma20_cross", etc.
    level: float
    price: float
    time: datetime
    direction: str  # "bullish" or "bearish"
    description: str


def calculate_ma(bars: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate moving average.
    
    Args:
        bars: DataFrame with 'close' column
        period: MA period (default 20)
    
    Returns:
        Series with MA values
    """
    if bars.empty or len(bars) < period:
        return pd.Series(dtype=float)
    
    return bars["close"].rolling(window=period).mean()


def detect_ma20_cross(bars: pd.DataFrame, ma20: pd.Series) -> Optional[TechnicalSignal]:
    """
    Detect MA20 crossover signals.
    
    Bullish: price crosses above MA20
    Bearish: price crosses below MA20
    
    Returns TechnicalSignal if crossover detected, None otherwise.
    """
    if len(bars) < 2 or len(ma20) < 2:
        return None
    
    current_price = float(bars.iloc[-1]["close"])
    prev_price = float(bars.iloc[-2]["close"])
    current_ma = float(ma20.iloc[-1])
    prev_ma = float(ma20.iloc[-2])
    
    # Bullish crossover: price was below MA, now above
    if prev_price <= prev_ma and current_price > current_ma:
        return TechnicalSignal(
            signal_type="ma20_cross",
            level=current_ma,
            price=current_price,
            time=datetime.now(EASTERN),
            direction="bullish",
            description=f"Price crossed above MA20 (${current_ma:.2f})"
        )
    
    # Bearish crossover: price was above MA, now below
    if prev_price >= prev_ma and current_price < current_ma:
        return TechnicalSignal(
            signal_type="ma20_cross",
            level=current_ma,
            price=current_price,
            time=datetime.now(EASTERN),
            direction="bearish",
            description=f"Price crossed below MA20 (${current_ma:.2f})"
        )
    
    return None


def find_support_resistance_levels(
    bars: pd.DataFrame,
    prev_close: float,
    orb_range: Optional[object] = None,
    lookback_bars: int = 60
) -> dict:
    """
    Find key support and resistance levels.
    
    Returns dict with:
    - support_levels: list of support prices
    - resistance_levels: list of resistance prices
    """
    if bars.empty:
        return {"support_levels": [], "resistance_levels": []}
    
    recent_bars = bars.iloc[-lookback_bars:] if len(bars) > lookback_bars else bars
    
    support_levels = []
    resistance_levels = []
    
    # Previous day's close
    if prev_close:
        if prev_close < float(recent_bars["close"].iloc[-1]):
            support_levels.append(prev_close)
        else:
            resistance_levels.append(prev_close)
    
    # ORB levels
    if orb_range:
        support_levels.append(orb_range.low)
        resistance_levels.append(orb_range.high)
    
    # Round numbers (e.g., 680, 685, 690, 695, 700)
    current_price = float(recent_bars["close"].iloc[-1])
    round_base = int(current_price // 5) * 5
    
    for i in range(-5, 10):  # Check levels from 25 below to 45 above
        level = round_base + (i * 5)
        if level > 0:
            if level < current_price:
                support_levels.append(level)
            else:
                resistance_levels.append(level)
    
    # Recent highs and lows (local extrema)
    if len(recent_bars) >= 10:
        # Find local highs (resistance)
        highs = recent_bars["high"].rolling(window=5, center=True).max()
        local_highs = recent_bars[recent_bars["high"] == highs]["high"].unique()
        for high in local_highs:
            if high > current_price and high not in resistance_levels:
                resistance_levels.append(float(high))
        
        # Find local lows (support)
        lows = recent_bars["low"].rolling(window=5, center=True).min()
        local_lows = recent_bars[recent_bars["low"] == lows]["low"].unique()
        for low in local_lows:
            if low < current_price and low not in support_levels:
                support_levels.append(float(low))
    
    # VWAP
    vwap = calculate_vwap(bars)
    if not vwap.empty:
        current_vwap = float(vwap.iloc[-1])
        if current_vwap < current_price:
            support_levels.append(current_vwap)
        else:
            resistance_levels.append(current_vwap)
    
    # Sort and remove duplicates
    support_levels = sorted(set(support_levels), reverse=True)  # Highest to lowest
    resistance_levels = sorted(set(resistance_levels))  # Lowest to highest
    
    return {
        "support_levels": support_levels,
        "resistance_levels": resistance_levels
    }


def check_support_resistance_breaks(
    bars: pd.DataFrame,
    prev_close: float,
    orb_range: Optional[object] = None,
    tolerance: float = 0.1  # Price must break by at least this amount
) -> list[TechnicalSignal]:
    """
    Check if price has broken through support or resistance levels.
    
    Returns list of TechnicalSignal objects for any breaks detected.
    """
    if bars.empty or len(bars) < 2:
        return []
    
    current_price = float(bars.iloc[-1]["close"])
    prev_price = float(bars.iloc[-2]["close"])
    
    levels = find_support_resistance_levels(bars, prev_close, orb_range)
    signals = []
    
    # Check resistance breaks (bullish)
    for resistance in levels["resistance_levels"]:
        # Price broke above resistance
        if prev_price <= resistance and current_price > resistance + tolerance:
            signals.append(TechnicalSignal(
                signal_type="resistance_break",
                level=resistance,
                price=current_price,
                time=datetime.now(EASTERN),
                direction="bullish",
                description=f"Resistance broken: ${resistance:.2f}"
            ))
    
    # Check support breaks (bearish)
    for support in levels["support_levels"]:
        # Price broke below support
        if prev_price >= support and current_price < support - tolerance:
            signals.append(TechnicalSignal(
                signal_type="support_break",
                level=support,
                price=current_price,
                time=datetime.now(EASTERN),
                direction="bearish",
                description=f"Support broken: ${support:.2f}"
            ))
    
    return signals


def check_all_signals(
    bars: pd.DataFrame,
    prev_close: float,
    orb_range: Optional[object] = None
) -> list[TechnicalSignal]:
    """
    Check all technical signals and return list of detected signals.
    
    Returns:
        List of TechnicalSignal objects
    """
    if bars.empty:
        return []
    
    signals = []
    
    # Check MA20 crossover
    ma20 = calculate_ma(bars, period=20)
    if not ma20.empty and len(ma20) >= 2:
        ma20_signal = detect_ma20_cross(bars, ma20)
        if ma20_signal:
            signals.append(ma20_signal)
    
    # Check support/resistance breaks
    break_signals = check_support_resistance_breaks(bars, prev_close, orb_range)
    signals.extend(break_signals)
    
    return signals

