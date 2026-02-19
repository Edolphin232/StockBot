# calculators/gap.py
"""
Gap analysis: overnight gap size and direction.
Inputs: current open, previous close
Outputs: gap_pct, gap_direction, gap_qualifies
"""
from dataclasses import dataclass


@dataclass
class GapResult:
    gap_pct: float = 0.0
    gap_direction: str = "flat"   # "up", "down", "flat"
    gap_qualifies: bool = False


def calculate_gap(
    current_open: float,
    prev_close: float,
    min_gap_pct: float = 0.1,
    max_gap_pct: float = 3.0,
) -> GapResult:
    """
    Analyze the overnight gap.

    Args:
        current_open:  Today's opening price
        prev_close:    Previous day's closing price
        min_gap_pct:   Minimum gap % to qualify
        max_gap_pct:   Maximum gap % before reversal risk flag
    """
    if not prev_close or prev_close == 0:
        return GapResult()

    gap_pct = (current_open - prev_close) / prev_close * 100
    direction = "up" if gap_pct > 0 else "down" if gap_pct < 0 else "flat"
    qualifies = abs(gap_pct) >= min_gap_pct  # no upper limit — just flag it

    return GapResult(
        gap_pct=round(gap_pct, 4),
        gap_direction=direction,
        gap_qualifies=qualifies,
    )