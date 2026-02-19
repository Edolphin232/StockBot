# strategy/signal_generator.py
"""
Signal generator — combines premarket and intraday filters
into a single actionable trade signal.
No data fetching, no config imports — everything passed explicitly.
"""
from dataclasses import dataclass
from filters.premarket_filter import PremarketResult, run_premarket_filter
from filters.intraday_filter import IntradayResult, run_intraday_filter
import pandas as pd


@dataclass
class TradeSignal:
    date: str
    direction: str = "none"        # "long", "short", "none"
    entry_price: float = 0.0
    total_score: float = 0.0
    triggered: bool = False
    trigger_bar_idx: int = -1
    premarket: PremarketResult = None
    intraday: IntradayResult = None
    reason: str = ""

    @property
    def confidence(self) -> str:
        if self.total_score >= 6.0:
            return "HIGH"
        elif self.total_score >= 4.0:
            return "MODERATE"
        else:
            return "LOW"


def generate_signal(
    date: str,
    bars: pd.DataFrame,
    prev_close: float,
    vix_level: float,
    catalyst_events: list = None,
    # Gate — premarket must pass to run intraday
    premarket_hard_filter: bool = True,
    # Minimum total score to trigger
    signal_threshold: float = 4.0,
    # Premarket settings
    min_gap_pct: float = 0.1,
    max_gap_pct: float = 3.0,
    vix_min: float = 14.0,
    vix_max: float = 35.0,
    score_gap: float = 1.0,
    score_vix: float = 1.0,
    score_catalyst: float = 1.5,
    # Intraday settings
    orb_period_minutes: int = 30,
    orb_buffer_pct: float = 0.05,
    orb_confirm_bars: int = 1,
    orb_min_range_pct: float = 0.1,
    orb_max_range_pct: float = 2.0,
    entry_cutoff_minutes: int = 390,
    vwap_confirm_bars: int = 5,
    vwap_max_crosses: int = 3,
    volume_multiplier: float = 1.5,
    min_consecutive_bars: int = 2,
    score_orb: float = 2.0,
    score_vwap: float = 1.0,
    score_volume: float = 1.0,
    score_momentum: float = 0.5,
    score_gap_hold: float = 1.0,
) -> TradeSignal:
    """
    Generate a trade signal for a single trading day.

    Args:
        date:                 "YYYY-MM-DD"
        bars:                 Full day 1-min bars from DataStore
        prev_close:           Previous day's close from DataStore
        vix_level:            VIX open for the day from DataStore
        catalyst_events:      Optional list of catalyst dicts
        premarket_hard_filter: If True, skip intraday when premarket fails
        signal_threshold:     Minimum combined score to trigger signal

    Returns:
        TradeSignal with full context
    """
    signal = TradeSignal(date=date)

    if bars is None or bars.empty or len(bars) < 10:
        signal.reason = "Insufficient bars"
        return signal

    if prev_close is None:
        signal.reason = "No prev close available"
        return signal

    # Step 1: Premarket filter
    current_open = float(bars.iloc[0]["open"])

    premarket = run_premarket_filter(
        current_open=current_open,
        prev_close=prev_close,
        vix_level=vix_level or 0.0,
        date=date,
        catalyst_events=catalyst_events,
        score_gap=score_gap,
        score_vix=score_vix,
        score_catalyst=score_catalyst,
        min_gap_pct=min_gap_pct,
        max_gap_pct=max_gap_pct,
        vix_min=vix_min,
        vix_max=vix_max,
    )
    signal.premarket = premarket

    if premarket_hard_filter and not premarket.pass_filter:
        signal.reason = "Failed premarket filter"
        return signal

    # Step 2: Intraday filter
    intraday = run_intraday_filter(
        bars=bars,
        prev_close=prev_close,
        gap_direction=premarket.gap.gap_direction,
        orb_period_minutes=orb_period_minutes,
        orb_buffer_pct=orb_buffer_pct,
        orb_confirm_bars=orb_confirm_bars,
        orb_min_range_pct=orb_min_range_pct,
        orb_max_range_pct=orb_max_range_pct,
        entry_cutoff_minutes=entry_cutoff_minutes,
        vwap_confirm_bars=vwap_confirm_bars,
        vwap_max_crosses=vwap_max_crosses,
        volume_multiplier=volume_multiplier,
        min_consecutive_bars=min_consecutive_bars,
        score_orb=score_orb,
        score_vwap=score_vwap,
        score_volume=score_volume,
        score_momentum=score_momentum,
        score_gap_hold=score_gap_hold,
    )
    signal.intraday = intraday

    if not intraday.has_signal:
        signal.reason = "No intraday signal"
        return signal

    # Step 3: Combine scores and check threshold
    total_score = premarket.score + intraday.score
    signal.total_score = total_score
    signal.direction = intraday.signal_direction
    signal.trigger_bar_idx = intraday.trigger_bar_idx

    if intraday.trigger_bar_idx >= 0:
        signal.entry_price = float(bars.iloc[intraday.trigger_bar_idx]["close"])

    if total_score >= signal_threshold:
        signal.triggered = True
        signal.reason = f"Signal triggered: score {total_score:.1f} >= {signal_threshold}"
    else:
        signal.reason = f"Score {total_score:.1f} below threshold {signal_threshold}"

    return signal