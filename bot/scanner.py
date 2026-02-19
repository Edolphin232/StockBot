# bot/scanner.py
"""
Core scanning logic — fetches bars, runs signal pipeline.
Stateless: called every minute by the scheduler.
"""
import pytz
from datetime import datetime, date, time
from data.store import DataStore
from fetchers.alpaca_client import fetch_bars as alpaca_fetch
from fetchers.yahoo_client import fetch_bars as yf_fetch
from fetchers.catalyst_client import fetch_catalysts
from filters.premarket_filter import run_premarket_filter
from calculators.orb import calculate_opening_range, detect_breakout
from strategy.signal_generator import generate_signal, TradeSignal
from strategy.technical_signals import calculate_ma

EASTERN = pytz.timezone("US/Eastern")


def get_today_store() -> tuple[DataStore, str, float, float]:
    """
    Fetch today's data and return a loaded DataStore.
    Called once at startup, refreshed each minute with new bars.
    """
    today     = date.today().strftime("%Y-%m-%d")
    yesterday = _prev_trading_day()

    start = EASTERN.localize(datetime.combine(date.today(), time(9, 30)))
    end   = EASTERN.localize(datetime.now())

    prev_start = EASTERN.localize(datetime.strptime(yesterday, "%Y-%m-%d").replace(hour=9, minute=30))
    prev_end   = EASTERN.localize(datetime.strptime(yesterday, "%Y-%m-%d").replace(hour=16, minute=0))

    store = DataStore()

    import pandas as pd
    prev_bars = alpaca_fetch("SPY", prev_start, prev_end, timeframe="1m")
    today_bars = alpaca_fetch("SPY", start, end, timeframe="1m")

    if not prev_bars.empty and not today_bars.empty:
        store.load_spy(pd.concat([prev_bars, today_bars]))
    elif not today_bars.empty:
        store.load_spy(today_bars)

    store.load_vix(yf_fetch("^VIX", yesterday, today))

    prev_close = store.get_prev_close(today)
    vix        = store.get_vix(today) or store.get_vix(yesterday)

    return store, today, prev_close, vix


def run_premarket_scan(store: DataStore, date: str, prev_close: float, vix: float):
    """Run and return premarket filter result."""
    bars = store.get_day_bars(date)
    if bars.empty:
        return None
    current_open = float(bars.iloc[0]["open"])
    
    # Fetch catalysts for this date
    catalyst_events = fetch_catalysts(target_date=date)
    
    return run_premarket_filter(
        current_open=current_open,
        prev_close=prev_close,
        vix_level=vix or 0.0,
        date=date,
        catalyst_events=catalyst_events,
    )


def run_orb_scan(store: DataStore, date: str):
    """Run and return ORB result after 30-min window closes."""
    bars = store.get_day_bars(date)
    if bars.empty:
        return None, None
    orb_range  = calculate_opening_range(bars)
    orb_result = detect_breakout(bars, orb_range) if orb_range else None
    return orb_range, orb_result


def run_signal_scan(store: DataStore, date: str, prev_close: float, vix: float):
    """Run full signal generator — called every minute."""
    bars = store.get_day_bars(date)
    if bars is None or bars.empty:
        return None
    
    # Fetch catalysts for this date
    catalyst_events = fetch_catalysts(target_date=date)
    
    return generate_signal(
        date=date,
        bars=bars,
        prev_close=prev_close,
        vix_level=vix or 0.0,
        catalyst_events=catalyst_events,
    )


def run_early_signal_scan(store: DataStore, date: str) -> TradeSignal:
    """
    Check for early signal at 9:45 AM.
    Conditions:
    1. Price >0.3% from open
    2. Price on one side of MA20 for last 10 bars (uses MA10 if not enough bars for MA20)
    3. Volume above average
    
    Returns TradeSignal if all conditions met, None otherwise.
    """
    bars = store.get_day_bars(date)
    if bars.empty or len(bars) < 10:  # Need at least 10 bars
        return None
    
    # Get opening price (first bar's open)
    open_price = float(bars.iloc[0]["open"])
    current_price = float(bars.iloc[-1]["close"])
    
    # Condition 1: Price >0.3% from open
    price_change_pct = ((current_price - open_price) / open_price) * 100
    if abs(price_change_pct) < 0.3:
        return None  # Not enough movement
    
    # Determine direction based on price movement
    direction = "long" if price_change_pct > 0 else "short"
    
    # Condition 2: Price on one side of MA for last 10 bars
    # Use MA20 if we have enough bars, otherwise use MA10
    ma_period = 20 if len(bars) >= 20 else 10
    ma = calculate_ma(bars, period=ma_period)
    if ma.empty or len(ma) < 10:
        return None
    
    # Check last 10 bars (or all available bars if less than 10)
    check_bars_count = min(10, len(bars))
    last_bars = bars.iloc[-check_bars_count:]
    last_ma = ma.iloc[-check_bars_count:]
    
    # Check if all checked bars are on the same side of MA
    if direction == "long":
        # For long: all prices should be above MA
        all_above = all(
            float(last_bars.iloc[i]["close"]) > float(last_ma.iloc[i])
            for i in range(len(last_bars))
        )
        if not all_above:
            return None
    else:  # short
        # For short: all prices should be below MA
        all_below = all(
            float(last_bars.iloc[i]["close"]) < float(last_ma.iloc[i])
            for i in range(len(last_bars))
        )
        if not all_below:
            return None
    
    # Condition 3: Volume above average
    if len(bars) < 2:
        return None
    
    avg_volume = bars["volume"].iloc[:-1].mean()  # Average of all bars except last
    current_volume = float(bars["volume"].iloc[-1])
    
    if current_volume <= avg_volume:
        return None  # Volume not above average
    
    # All conditions met - create early signal
    signal = TradeSignal(
        date=date,
        direction=direction,
        entry_price=current_price,
        total_score=5.0,  # Early signal gets fixed score
        triggered=True,
        trigger_bar_idx=len(bars) - 1,
        reason=f"Early signal: {direction.upper()} - Price {abs(price_change_pct):.2f}% from open, MA20 aligned, volume confirmed"
    )
    
    return signal


def _prev_trading_day() -> str:
    from datetime import timedelta
    d = date.today() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")