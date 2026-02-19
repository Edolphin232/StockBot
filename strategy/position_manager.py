# strategy/position_manager.py
"""
Tiered exit strategy for trade positions.

Entry: ORB breakout confirmed
Tier 1 (33%): First key support/resistance level
Tier 2 (33%): VWAP reclaim OR RSI divergence (whichever comes first)
Tier 3 (34%): Trailing stop (2x ATR) or hard stop at 12:00 PM
Stop Loss: ORB high/low (invalidation level)
"""
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Optional
import pandas as pd
import pytz
import numpy as np
from calculators.vwap import calculate_vwap
from calculators.rsi import calculate_rsi, detect_rsi_divergence
from calculators.atr import calculate_atr, calculate_trailing_stop

EASTERN = pytz.timezone("US/Eastern")


@dataclass
class Position:
    """Represents an active trade position."""
    entry_price: float
    stop_loss: float  # ORB high/low (invalidation level)
    direction: str  # "long" or "short"
    entry_time: datetime
    date: str
    orb_range: object  # ORBRange object
    prev_close: float  # Previous day's close for support/resistance
    tier1_target: Optional[float] = None  # First key level
    tier1_exited: bool = False
    tier2_exited: bool = False
    tier3_exited: bool = False
    trailing_stop: Optional[float] = None  # Current trailing stop level
    
    @property
    def risk(self) -> float:
        """Distance to stop loss (ORB invalidation level)."""
        if self.direction == "long":
            return self.entry_price - self.stop_loss
        else:  # short
            return self.stop_loss - self.entry_price
    
    def current_pnl_pct(self, current_price: float) -> float:
        """Current P&L as percentage."""
        if self.direction == "long":
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:  # short
            return ((self.entry_price - current_price) / self.entry_price) * 100


@dataclass
class ExitSignal:
    """Represents an exit condition."""
    tier: int  # 1, 2, or 3
    reason: str
    price: float
    time: datetime
    exit_all: bool = False  # True for Tier 3, False for partial exits


def find_key_level(
    entry_price: float,
    direction: str,
    prev_close: float,
    orb_range: object,
    vwap: float
) -> Optional[float]:
    """
    Find first key support/resistance level.
    Checks: previous day's low/high, round numbers, VWAP distance.
    
    Returns the first level that price would hit in the direction of the trade.
    """
    if direction == "long":
        # For longs, look for resistance levels above entry
        levels = []
        
        # Previous day's high (if available and above entry)
        # We'll use prev_close as proxy for previous day's range
        if prev_close and prev_close > entry_price:
            levels.append(prev_close)
        
        # Round numbers above entry (e.g., 690, 695, 700)
        round_base = int(entry_price // 5) * 5  # Round down to nearest 5
        for i in range(1, 10):  # Check next 9 round numbers
            round_level = round_base + (i * 5)
            if round_level > entry_price:
                levels.append(round_level)
        
        # VWAP distance (if VWAP is above entry)
        if vwap > entry_price:
            vwap_distance = vwap - entry_price
            # Add level at VWAP or at a reasonable distance
            levels.append(vwap)
        
        # ORB high (if above entry)
        if orb_range and orb_range.high > entry_price:
            levels.append(orb_range.high)
        
        # Return the lowest level (first one price would hit)
        if levels:
            return min(levels)
    
    else:  # short
        # For shorts, look for support levels below entry
        levels = []
        
        # Previous day's low (if available and below entry)
        if prev_close and prev_close < entry_price:
            levels.append(prev_close)
        
        # Round numbers below entry
        round_base = int(entry_price // 5) * 5  # Round down to nearest 5
        for i in range(1, 10):
            round_level = round_base - (i * 5)
            if round_level < entry_price and round_level > 0:
                levels.append(round_level)
        
        # VWAP distance (if VWAP is below entry)
        if vwap < entry_price:
            levels.append(vwap)
        
        # ORB low (if below entry)
        if orb_range and orb_range.low < entry_price:
            levels.append(orb_range.low)
        
        # Return the highest level (first one price would hit going down)
        if levels:
            return max(levels)
    
    return None


def check_tier1_exit(
    position: Position,
    current_price: float
) -> Optional[ExitSignal]:
    """
    Check if Tier 1 exit condition is met (33% at first key level).
    """
    if position.tier1_exited or position.tier1_target is None:
        return None
    
    if position.direction == "long":
        if current_price >= position.tier1_target:
            return ExitSignal(
                tier=1,
                reason=f"Tier 1: Key resistance level hit (${position.tier1_target:.2f})",
                price=current_price,
                time=datetime.now(EASTERN),
                exit_all=False
            )
    else:  # short
        if current_price <= position.tier1_target:
            return ExitSignal(
                tier=1,
                reason=f"Tier 1: Key support level hit (${position.tier1_target:.2f})",
                price=current_price,
                time=datetime.now(EASTERN),
                exit_all=False
            )
    
    return None


def check_tier2_exit(
    position: Position,
    bars: pd.DataFrame,
    current_price: float
) -> Optional[ExitSignal]:
    """
    Check if Tier 2 exit condition is met.
    Conditions: VWAP reclaim OR RSI divergence (whichever comes first).
    
    Only checks if Tier 1 has been hit.
    """
    if not position.tier1_exited or position.tier2_exited:
        return None
    
    if bars.empty or len(bars) < 15:  # Need enough bars for RSI
        return None
    
    # Check VWAP reclaim
    vwap = calculate_vwap(bars)
    if not vwap.empty:
        current_vwap = vwap.iloc[-1]
        vwap_reclaimed = False
        
        if position.direction == "long":
            vwap_reclaimed = current_price < current_vwap
        else:  # short
            vwap_reclaimed = current_price > current_vwap
        
        if vwap_reclaimed:
            return ExitSignal(
                tier=2,
                reason="Tier 2: VWAP reclaim",
                price=current_price,
                time=datetime.now(EASTERN),
                exit_all=False
            )
    
    # Check RSI divergence
    rsi = calculate_rsi(bars, period=14)
    if not rsi.empty and len(rsi) >= 10:
        rsi_divergence = detect_rsi_divergence(bars, rsi, position.direction, lookback_bars=10)
        if rsi_divergence:
            return ExitSignal(
                tier=2,
                reason="Tier 2: RSI divergence",
                price=current_price,
                time=datetime.now(EASTERN),
                exit_all=False
            )
    
    return None


def check_tier3_exit(
    position: Position,
    bars: pd.DataFrame,
    current_price: float
) -> Optional[ExitSignal]:
    """
    Check if Tier 3 exit condition is met.
    Conditions: Trailing stop (2x ATR) OR hard time stop at 12:00 PM.
    """
    if position.tier3_exited:
        return None
    
    # Check time stop first
    now = datetime.now(EASTERN)
    cutoff_time = now.replace(hour=12, minute=0, second=0, microsecond=0)
    
    if now >= cutoff_time:
        return ExitSignal(
            tier=3,
            reason="Tier 3: Hard time stop (12:00 PM ET)",
            price=current_price,
            time=now,
            exit_all=True
        )
    
    # Check trailing stop (2x ATR)
    if bars.empty or len(bars) < 15:
        return None
    
    trailing_stop = calculate_trailing_stop(
        bars,
        position.entry_price,
        position.direction,
        atr_multiplier=2.0,
        atr_period=14
    )
    
    # Update position's trailing stop (only moves in favorable direction)
    if position.trailing_stop is None:
        position.trailing_stop = trailing_stop
    else:
        if position.direction == "long":
            # For longs, trailing stop only moves up
            position.trailing_stop = max(position.trailing_stop, trailing_stop)
        else:  # short
            # For shorts, trailing stop only moves down
            position.trailing_stop = min(position.trailing_stop, trailing_stop)
    
    # Check if price hit trailing stop
    if position.direction == "long":
        if current_price <= position.trailing_stop:
            return ExitSignal(
                tier=3,
                reason=f"Tier 3: Trailing stop hit (${position.trailing_stop:.2f}, 2x ATR)",
                price=current_price,
                time=datetime.now(EASTERN),
                exit_all=True
            )
    else:  # short
        if current_price >= position.trailing_stop:
            return ExitSignal(
                tier=3,
                reason=f"Tier 3: Trailing stop hit (${position.trailing_stop:.2f}, 2x ATR)",
                price=current_price,
                time=datetime.now(EASTERN),
                exit_all=True
            )
    
    return None


def check_stop_loss(position: Position, current_price: float) -> Optional[ExitSignal]:
    """
    Check if stop loss is hit (ORB high/low - invalidation level).
    """
    if position.direction == "long":
        if current_price <= position.stop_loss:
            return ExitSignal(
                tier=0,  # Stop loss
                reason=f"Stop loss hit - ORB invalidated (${position.stop_loss:.2f})",
                price=current_price,
                time=datetime.now(EASTERN),
                exit_all=True
            )
    else:  # short
        if current_price >= position.stop_loss:
            return ExitSignal(
                tier=0,  # Stop loss
                reason=f"Stop loss hit - ORB invalidated (${position.stop_loss:.2f})",
                price=current_price,
                time=datetime.now(EASTERN),
                exit_all=True
            )
    
    return None


def check_exits(
    position: Position,
    bars: pd.DataFrame,
    vwap: Optional[float] = None
) -> list[ExitSignal]:
    """
    Check all exit conditions and return list of exit signals.
    
    Returns list of ExitSignal objects (can be multiple if conditions overlap).
    """
    if bars.empty:
        return []
    
    current_price = float(bars.iloc[-1]["close"])
    exits = []
    
    # Check stop loss first (highest priority)
    stop_exit = check_stop_loss(position, current_price)
    if stop_exit:
        exits.append(stop_exit)
        return exits  # Stop loss exits everything
    
    # Check Tier 3 (trailing stop or time stop)
    tier3_exit = check_tier3_exit(position, bars, current_price)
    if tier3_exit:
        exits.append(tier3_exit)
        return exits  # Tier 3 exits everything
    
    # Check Tier 1 (key level)
    tier1_exit = check_tier1_exit(position, current_price)
    if tier1_exit:
        exits.append(tier1_exit)
        position.tier1_exited = True
    
    # Check Tier 2 (VWAP/RSI) - only if Tier 1 hit
    if position.tier1_exited:
        tier2_exit = check_tier2_exit(position, bars, current_price)
        if tier2_exit:
            exits.append(tier2_exit)
            position.tier2_exited = True
    
    return exits


def create_position_from_signal(
    signal,
    orb_range: object,
    prev_close: float,
    bars: pd.DataFrame
) -> Position:
    """
    Create a Position from a TradeSignal.
    
    Args:
        signal: TradeSignal object
        orb_range: ORBRange object (for stop loss)
        prev_close: Previous day's close (for support/resistance)
        bars: Current bars (for VWAP calculation)
    
    Returns:
        Position object
    """
    # Stop loss is ORB high/low (invalidation level)
    if signal.direction == "long":
        stop_loss = orb_range.low  # Invalidate if price goes back below ORB low
    else:  # short
        stop_loss = orb_range.high  # Invalidate if price goes back above ORB high
    
    # Calculate VWAP for key level detection
    vwap_series = calculate_vwap(bars)
    current_vwap = float(vwap_series.iloc[-1]) if not vwap_series.empty else signal.entry_price
    
    # Find Tier 1 target (first key level)
    tier1_target = find_key_level(
        signal.entry_price,
        signal.direction,
        prev_close,
        orb_range,
        current_vwap
    )
    
    position = Position(
        entry_price=signal.entry_price,
        stop_loss=stop_loss,
        direction=signal.direction,
        entry_time=datetime.now(EASTERN),
        date=signal.date,
        orb_range=orb_range,
        prev_close=prev_close,
        tier1_target=tier1_target
    )
    
    return position
