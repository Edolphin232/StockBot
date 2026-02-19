# calculators/volume.py
"""
Volume confirmation calculator.
Checks if breakout bar has above-average volume.
"""
import pandas as pd


def confirm_volume(
    bars: pd.DataFrame,
    trigger_bar_idx: int,
    multiplier: float = 1.5,
) -> bool:
    """
    Check if the breakout bar volume exceeds average prior volume.

    Args:
        bars:             Full day bars
        trigger_bar_idx:  Index of the breakout bar
        multiplier:       How much above average the bar must be
    """
    if trigger_bar_idx < 1 or trigger_bar_idx >= len(bars):
        return False

    prior_avg = bars["volume"].iloc[:trigger_bar_idx].mean()
    trigger_vol = bars["volume"].iloc[trigger_bar_idx]

    return trigger_vol >= prior_avg * multiplier