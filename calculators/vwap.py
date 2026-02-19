# calculators/vwap.py
"""
VWAP calculator and trend analyzer.
Inputs: full day bars DataFrame
Outputs: VWAP series, side, crosses, trending bool
"""
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class VWAPResult:
    series: pd.Series = field(default_factory=pd.Series)
    side: str = "mixed"           # "above", "below", "mixed"
    crosses: int = 0
    trending: bool = False


def calculate_vwap(bars: pd.DataFrame) -> pd.Series:
    """
    Compute cumulative intraday VWAP.
    Returns a Series aligned to bars index.
    """
    typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
    cumulative_tp_vol = (typical_price * bars["volume"]).cumsum()
    cumulative_vol = bars["volume"].cumsum()
    return cumulative_tp_vol / cumulative_vol


def analyze_vwap(
    bars: pd.DataFrame,
    trigger_bar_idx: int = -1,
    confirm_bars: int = 5,
    max_crosses: int = 3,
) -> VWAPResult:
    """
    Analyze VWAP trend behavior up to the trigger bar.

    Args:
        bars:             Full day bars
        trigger_bar_idx:  Look at bars up to this index (-1 = full day)
        confirm_bars:     Last N bars must all be on same VWAP side
        max_crosses:      Max VWAP crosses allowed before flagging chop
    """
    if bars.empty:
        return VWAPResult()

    vwap = calculate_vwap(bars)
    closes = bars["close"]

    end_idx = trigger_bar_idx if trigger_bar_idx > 0 else len(bars)
    subset_close = closes.iloc[:end_idx]
    subset_vwap = vwap.iloc[:end_idx]

    # Count crosses
    above = subset_close > subset_vwap
    crosses = max(0, int((above != above.shift(1)).sum()) - 1)

    # Check last N bars are on same side
    all_above = all_below = False
    if len(subset_close) >= confirm_bars:
        last_close = subset_close.iloc[-confirm_bars:]
        last_vwap = subset_vwap.iloc[-confirm_bars:]
        all_above = (last_close > last_vwap).all()
        all_below = (last_close < last_vwap).all()

    trending = crosses <= max_crosses and (all_above or all_below)
    side = "above" if all_above else "below" if all_below else "mixed"

    return VWAPResult(series=vwap, side=side, crosses=crosses, trending=trending)