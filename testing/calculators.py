# testing/test_calculators.py
"""
Quick test of all calculators using real Alpaca data for today.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytz
import pandas as pd
from datetime import datetime, date, timedelta, time
from data.store import DataStore
from fetchers.alpaca_client import fetch_bars as alpaca_fetch
from fetchers.yahoo_client import fetch_bars as yf_fetch
from calculators.gap import calculate_gap
from calculators.orb import calculate_opening_range, detect_breakout
from calculators.vwap import calculate_vwap, analyze_vwap
from calculators.volume import confirm_volume
from calculators.momentum import count_consecutive_bars

EASTERN = pytz.timezone("US/Eastern")

# Use yesterday (2-18) for trading day, and 2-17 for previous close
LAST_TRADING_DAY = "2026-02-18"
PREV_CLOSE_DAY   = "2026-02-17"

target_date = datetime.strptime(LAST_TRADING_DAY, "%Y-%m-%d")
start = EASTERN.localize(datetime.combine(target_date.date(), time(9, 30)))
end   = EASTERN.localize(datetime.combine(target_date.date(), time(16, 0)))


def setup() -> tuple:
    print(f"Fetching data for {LAST_TRADING_DAY} and {PREV_CLOSE_DAY}...")
    store = DataStore()
    
    # Fetch SPY bars for both days (needed for prev_close calculation)
    prev_date = datetime.strptime(PREV_CLOSE_DAY, "%Y-%m-%d")
    prev_start = EASTERN.localize(datetime.combine(prev_date.date(), time(9, 30)))
    prev_end = EASTERN.localize(datetime.combine(prev_date.date(), time(16, 0)))
    
    # Fetch previous day bars
    prev_spy_data = alpaca_fetch("SPY", prev_start, prev_end, timeframe="1m")
    if prev_spy_data is None or prev_spy_data.empty:
        print(f"⚠️  No SPY bars returned for {PREV_CLOSE_DAY}")
    else:
        store.load_spy(prev_spy_data)
    
    # Fetch current day bars
    spy_data = alpaca_fetch("SPY", start, end, timeframe="1m")
    if spy_data is None:
        print("❌ Alpaca client returned None - check API keys")
        sys.exit(1)
    if spy_data.empty:
        print(f"❌ No SPY bars returned for {LAST_TRADING_DAY}")
        sys.exit(1)
    
    # Append current day bars to store (combine with previous if exists)
    if not store._spy_bars.empty:
        store._spy_bars = pd.concat([store._spy_bars, spy_data])
    else:
        store.load_spy(spy_data)
    
    # Fetch VIX
    vix_data = yf_fetch("^VIX", PREV_CLOSE_DAY, LAST_TRADING_DAY)
    store.load_vix(vix_data)
    
    bars       = store.get_day_bars(LAST_TRADING_DAY)
    prev_close = store.get_prev_close(LAST_TRADING_DAY)
    vix        = store.get_vix(LAST_TRADING_DAY)
    print(f"Bars loaded: {len(bars)} | Prev close: {prev_close} | VIX: {vix}\n")
    return bars, prev_close, vix


def test_gap(prev_close):
    result = calculate_gap(
        current_open=prev_close * 1.005,  # simulate 0.5% gap up
        prev_close=prev_close
    )
    print(f"GAP:    pct={result.gap_pct:+.3f}% | dir={result.gap_direction} | qualifies={result.gap_qualifies}")
    assert result.gap_direction == "up"
    print("✅ test_gap passed\n")


def test_orb(bars):
    orb = calculate_opening_range(bars)
    print(f"ORB:    high={orb.high:.2f} | low={orb.low:.2f} | range={orb.range_pct:.3f}%")

    result = detect_breakout(bars, orb)
    print(f"        breakout={result.breakout} | dir={result.direction} | trigger_idx={result.trigger_bar_idx}")
    if result.breakout:
        print(f"        trigger_time={result.trigger_time} | price={result.trigger_price:.2f}")
    print("✅ test_orb passed\n")
    return result


def test_vwap(bars, orb_result):
    vwap = calculate_vwap(bars)
    print(f"VWAP:   current={vwap.iloc[-1]:.2f} | bars={len(vwap)}")

    result = analyze_vwap(bars, trigger_bar_idx=orb_result.trigger_bar_idx)
    print(f"        side={result.side} | crosses={result.crosses} | trending={result.trending}")
    assert not result.series.empty
    print("✅ test_vwap passed\n")


def test_volume(bars, orb_result):
    if not orb_result.breakout:
        print("⏭️  test_volume skipped — no breakout detected\n")
        return
    confirmed = confirm_volume(bars, orb_result.trigger_bar_idx)
    trigger_vol = bars["volume"].iloc[orb_result.trigger_bar_idx]
    avg_vol = bars["volume"].iloc[:orb_result.trigger_bar_idx].mean()
    print(f"VOLUME: trigger={trigger_vol:.0f} | avg={avg_vol:.0f} | confirmed={confirmed}")
    print("✅ test_volume passed\n")


def test_momentum(bars, orb_result):
    if not orb_result.breakout:
        print("⏭️  test_momentum skipped — no breakout detected\n")
        return
    count = count_consecutive_bars(bars, orb_result.trigger_bar_idx, orb_result.direction)
    print(f"MOMENTUM: consecutive {orb_result.direction} bars = {count}")
    assert count >= 0
    print("✅ test_momentum passed\n")


if __name__ == "__main__":
    bars, prev_close, vix = setup()

    if bars.empty:
        print("❌ No bars returned — is market open today?")
        sys.exit(1)

    test_gap(prev_close)
    orb_result = test_orb(bars)
    test_vwap(bars, orb_result)
    test_volume(bars, orb_result)
    test_momentum(bars, orb_result)

    print("✅ All calculator tests passed")