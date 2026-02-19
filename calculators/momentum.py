# calculators/momentum.py
"""
Momentum confirmation: counts consecutive directional bars
around the breakout trigger.
"""
import pandas as pd


def count_consecutive_bars(
    bars: pd.DataFrame,
    trigger_bar_idx: int,
    direction: str,
    lookback: int = 5,
    lookahead: int = 3,
) -> int:
    """
    Count consecutive bars moving in the signal direction
    around the trigger bar.

    Args:
        bars:             Full day bars
        trigger_bar_idx:  Index of the breakout bar
        direction:        "long" or "short"
        lookback:         Bars before trigger to include
        lookahead:        Bars after trigger to include
    """
    if trigger_bar_idx < 0 or trigger_bar_idx >= len(bars):
        return 0

    start = max(0, trigger_bar_idx - lookback)
    end = min(len(bars), trigger_bar_idx + lookahead)
    count = 0

    for i in range(start, end):
        bar = bars.iloc[i]
        if direction == "long" and bar["close"] > bar["open"]:
            count += 1
        elif direction == "short" and bar["close"] < bar["open"]:
            count += 1
        else:
            count = 0  # reset on opposing bar

    return count