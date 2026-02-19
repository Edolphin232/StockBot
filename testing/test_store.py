# testing/test_store.py
"""
Tests for DataStore using real Alpaca API calls.
"""
import sys
import os

import pytz
from datetime import datetime
from data.store import DataStore
from fetchers.alpaca_client import fetch_bars
from fetchers.yahoo_client import fetch_bars as yf_fetch

EASTERN = pytz.timezone("US/Eastern")

# A known recent date range with good data
START = EASTERN.localize(datetime(2025, 1, 2, 9, 30))
END   = EASTERN.localize(datetime(2025, 1, 7, 16, 0))
DATES = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]


def setup() -> DataStore:
    """Fetch real data once, reuse across all tests."""
    print("Fetching data from Alpaca...")
    store = DataStore()
    store.load_spy(fetch_bars("SPY", START, END, timeframe="1m"))
    store.load_vix(yf_fetch("^VIX", START, END, timeframe="1d"))
    print("Done.\n")
    return store


def test_is_ready(store: DataStore):
    assert store.is_ready(), "Store should be ready after loading"
    print("✅ test_is_ready passed")


def test_get_trading_dates(store: DataStore):
    dates = store.get_trading_dates()
    assert len(dates) > 0, "Should have trading dates"
    assert dates == sorted(dates), "Dates should be sorted"
    print(f"✅ test_get_trading_dates passed — got {dates}")


def test_get_day_bars(store: DataStore):
    bars = store.get_day_bars("2025-01-02")
    assert not bars.empty, "Should return bars for 2025-01-02"
    assert "close" in bars.columns
    assert "volume" in bars.columns
    print(f"✅ test_get_day_bars passed — {len(bars)} bars")


def test_get_day_bars_missing(store: DataStore):
    bars = store.get_day_bars("2020-01-01")
    assert bars.empty, "Should return empty for date outside range"
    print("✅ test_get_day_bars_missing passed")


def test_get_prev_close(store: DataStore):
    prev = store.get_prev_close("2025-01-03")
    assert prev is not None, "Should have prev close for Jan 3"
    assert isinstance(prev, float)
    assert 400 < prev < 700, f"SPY prev close looks wrong: {prev}"
    print(f"✅ test_get_prev_close passed — prev close: {prev:.2f}")


def test_get_prev_close_first_date(store: DataStore):
    dates = store.get_trading_dates()
    first = dates[0]
    result = store.get_prev_close(first)
    assert result is None, "First date should have no prev close"
    print("✅ test_get_prev_close_first_date passed")


def test_get_vix(store: DataStore):
    vix = store.get_vix("2025-01-02")
    assert vix is not None, "Should return VIX for valid date"
    assert isinstance(vix, float)
    assert 5 < vix < 80, f"VIX value looks wrong: {vix}"
    print(f"✅ test_get_vix passed — VIX: {vix:.2f}")


def test_get_vix_missing(store: DataStore):
    result = store.get_vix("2020-01-01")
    assert result is None, "Should return None for missing date"
    print("✅ test_get_vix_missing passed")


if __name__ == "__main__":
    print("Running DataStore tests...\n")
    store = setup()

    test_is_ready(store)
    test_get_trading_dates(store)
    test_get_day_bars(store)
    test_get_day_bars_missing(store)
    test_get_prev_close(store)
    test_get_prev_close_first_date(store)
    test_get_vix(store)
    test_get_vix_missing(store)

    print("\n✅ All tests passed")