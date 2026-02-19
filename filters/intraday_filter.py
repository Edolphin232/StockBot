# filters/intraday_filter.py
"""
Intraday filter — uses calculators, no data fetching.
"""
from dataclasses import dataclass, field
import pandas as pd
from calculators.orb import calculate_opening_range, detect_breakout, ORBResult
from calculators.vwap import analyze_vwap, VWAPResult
from calculators.volume import confirm_volume
from calculators.momentum import count_consecutive_bars


@dataclass
class IntradayResult:
    orb: ORBResult = None
    vwap: VWAPResult = None
    volume_confirmed: bool = False
    consecutive_bars: int = 0
    gap_held: bool = False
    score: float = 0.0
    signal_direction: str = "none"
    trigger_bar_idx: int = -1

    @property
    def has_signal(self) -> bool:
        return self.orb is not None and self.orb.breakout and self.signal_direction != "none"


def check_gap_hold(
    bars: pd.DataFrame,
    prev_close: float,
    gap_direction: str,
    orb_period_minutes: int = 30,
) -> bool:
    if not prev_close or gap_direction == "flat":
        return False
    from datetime import timedelta
    market_open = bars.index.get_level_values("timestamp")[0].replace(hour=9, minute=30)
    orb_end = market_open + timedelta(minutes=orb_period_minutes)
    timestamps = bars.index.get_level_values("timestamp")
    orb_bars = bars[timestamps < orb_end]
    if orb_bars.empty:
        return False
    if gap_direction == "up":
        return orb_bars["low"].min() > prev_close
    else:
        return orb_bars["high"].max() < prev_close


def run_intraday_filter(
    bars: pd.DataFrame,
    prev_close: float = None,
    gap_direction: str = "flat",
    # ORB settings
    orb_period_minutes: int = 30,
    orb_buffer_pct: float = 0.05,
    orb_confirm_bars: int = 1,
    orb_min_range_pct: float = 0.1,
    orb_max_range_pct: float = 2.0,
    entry_cutoff_minutes: int = 390,
    # VWAP settings
    vwap_confirm_bars: int = 5,
    vwap_max_crosses: int = 3,
    # Volume settings
    volume_multiplier: float = 1.5,
    # Momentum settings
    min_consecutive_bars: int = 2,
    # Scoring weights
    score_orb: float = 2.0,
    score_vwap: float = 1.0,
    score_volume: float = 1.0,
    score_momentum: float = 0.5,
    score_gap_hold: float = 1.0,
) -> IntradayResult:

    if bars.empty or len(bars) < 10:
        return IntradayResult()

    # Step 1: ORB
    orb_range = calculate_opening_range(bars, orb_period_minutes)
    if orb_range is None:
        return IntradayResult()

    orb = detect_breakout(
        bars, orb_range,
        orb_period_minutes=orb_period_minutes,
        buffer_pct=orb_buffer_pct,
        confirm_bars=orb_confirm_bars,
        entry_cutoff_minutes=entry_cutoff_minutes,
        min_range_pct=orb_min_range_pct,
        max_range_pct=orb_max_range_pct,
    )

    # Step 2: VWAP
    vwap = analyze_vwap(
        bars,
        trigger_bar_idx=orb.trigger_bar_idx,
        confirm_bars=vwap_confirm_bars,
        max_crosses=vwap_max_crosses,
    )

    # Step 3: Volume
    vol_confirmed = False
    if orb.breakout:
        vol_confirmed = confirm_volume(bars, orb.trigger_bar_idx, volume_multiplier)

    # Step 4: Momentum
    consec = 0
    if orb.breakout:
        consec = count_consecutive_bars(bars, orb.trigger_bar_idx, orb.direction)

    # Step 5: Gap hold
    gap_held = check_gap_hold(bars, prev_close, gap_direction, orb_period_minutes)

    # Score
    score = 0.0
    if orb.breakout:
        score += score_orb
    if vwap.trending:
        score += score_vwap
    if vol_confirmed:
        score += score_volume
    if consec >= min_consecutive_bars:
        score += score_momentum
    if gap_held:
        score += score_gap_hold

    # Signal direction — require VWAP alignment
    signal_dir = "none"
    if orb.breakout:
        if orb.direction == "long" and vwap.side == "above":
            signal_dir = "long"
        elif orb.direction == "short" and vwap.side == "below":
            signal_dir = "short"

    return IntradayResult(
        orb=orb,
        vwap=vwap,
        volume_confirmed=vol_confirmed,
        consecutive_bars=consec,
        gap_held=gap_held,
        score=score,
        signal_direction=signal_dir,
        trigger_bar_idx=orb.trigger_bar_idx,
    )