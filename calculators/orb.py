# calculators/orb.py
"""
Opening Range Breakout (ORB) calculator.
Inputs: full day bars DataFrame
Outputs: opening range levels + breakout detection
"""
from dataclasses import dataclass, field
from datetime import timedelta
import pandas as pd


@dataclass
class ORBRange:
    high: float = 0.0
    low: float = 0.0
    mid: float = 0.0
    range_pct: float = 0.0


@dataclass
class ORBResult:
    range: ORBRange = None
    breakout: bool = False
    direction: str = "none"       # "long", "short", "none"
    trigger_bar_idx: int = -1
    trigger_time: pd.Timestamp = None
    trigger_price: float = 0.0


def calculate_opening_range(bars: pd.DataFrame, orb_period_minutes: int = 30) -> ORBRange:
    """
    Compute the high/low/mid of the opening range period.

    Args:
        bars:                Full day 1-min bars
        orb_period_minutes:  How many minutes define the opening range
    """
    if bars.empty:
        return None

    market_open = bars.index.get_level_values("timestamp")[0].replace(hour=9, minute=30)
    orb_end = market_open + timedelta(minutes=orb_period_minutes)
    timestamps = bars.index.get_level_values("timestamp")
    orb_bars = bars[(timestamps >= market_open) & (timestamps < orb_end)]

    if orb_bars.empty:
        return None

    high = orb_bars["high"].max()
    low = orb_bars["low"].min()
    mid = (high + low) / 2
    range_pct = (high - low) / low * 100

    return ORBRange(high=high, low=low, mid=mid, range_pct=round(range_pct, 4))


def detect_breakout(
    bars: pd.DataFrame,
    orb: ORBRange,
    orb_period_minutes: int = 30,
    buffer_pct: float = 0.05,
    confirm_bars: int = 1,
    entry_cutoff_minutes: int = 390,
    min_range_pct: float = 0.1,
    max_range_pct: float = 2.0,
) -> ORBResult:
    """
    Detect if price breaks out of the opening range after ORB period ends.

    Args:
        bars:                   Full day bars
        orb:                    ORBRange from calculate_opening_range()
        orb_period_minutes:     Must match what was used to compute ORB
        buffer_pct:             % buffer above/below ORB to avoid false breaks
        confirm_bars:           Consecutive bars needed to confirm breakout
        entry_cutoff_minutes:   Stop looking for entries after this many minutes
        min_range_pct:          Skip if ORB is too narrow (choppy)
        max_range_pct:          Skip if ORB is too wide (gappy)
    """
    if orb is None:
        return ORBResult()

    if orb.range_pct < min_range_pct or orb.range_pct > max_range_pct:
        return ORBResult(range=orb)

    buffer = orb.mid * (buffer_pct / 100)
    market_open = bars.index.get_level_values("timestamp")[0].replace(hour=9, minute=30)
    orb_end = market_open + timedelta(minutes=orb_period_minutes)
    entry_cutoff = market_open + timedelta(minutes=entry_cutoff_minutes)
    timestamps = bars.index.get_level_values("timestamp")
    post_orb = bars[(timestamps >= orb_end) & (timestamps <= entry_cutoff)]

    long_streak = 0
    short_streak = 0

    for idx, bar in post_orb.iterrows():
        ts = idx[1] if isinstance(idx, tuple) else idx
        if bar["close"] > orb.high + buffer:
            long_streak += 1
            short_streak = 0
        elif bar["close"] < orb.low - buffer:
            short_streak += 1
            long_streak = 0
        else:
            long_streak = 0
            short_streak = 0

        if long_streak >= confirm_bars:
            trigger_idx = bars.index.get_loc(idx)
            return ORBResult(
                range=orb,
                breakout=True,
                direction="long",
                trigger_bar_idx=trigger_idx,
                trigger_time=ts,
                trigger_price=bar["close"],
            )
        if short_streak >= confirm_bars:
            trigger_idx = bars.index.get_loc(idx)
            return ORBResult(
                range=orb,
                breakout=True,
                direction="short",
                trigger_bar_idx=trigger_idx,
                trigger_time=ts,
                trigger_price=bar["close"],
            )

    return ORBResult(range=orb)