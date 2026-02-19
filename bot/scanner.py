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
from filters.premarket_filter import run_premarket_filter
from calculators.orb import calculate_opening_range, detect_breakout
from strategy.signal_generator import generate_signal

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
    prev_bars = alpaca_fetch("SPY", prev_start, prev_end, timeframe="1min")
    today_bars = alpaca_fetch("SPY", start, end, timeframe="1min")

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
    return run_premarket_filter(
        current_open=current_open,
        prev_close=prev_close,
        vix_level=vix or 0.0,
        date=date,
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
    return generate_signal(
        date=date,
        bars=bars,
        prev_close=prev_close,
        vix_level=vix or 0.0,
    )


def _prev_trading_day() -> str:
    from datetime import timedelta
    d = date.today() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")