import os
import pytz
from datetime import datetime, time

import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.requests import StockBarsRequest

from dotenv import load_dotenv
load_dotenv(override=True)

EASTERN = pytz.timezone("US/Eastern")

ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")

client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

# Convenient mapping so callers can pass a string like "1min", "5min", "1hour", "1day"
TIMEFRAME_MAP = {
    "1min":   TimeFrame(1,  TimeFrameUnit.Minute),
    "2min":   TimeFrame(2,  TimeFrameUnit.Minute),
    "5min":   TimeFrame(5,  TimeFrameUnit.Minute),
    "15min":  TimeFrame(15, TimeFrameUnit.Minute),
    "30min":  TimeFrame(30, TimeFrameUnit.Minute),
    "1hour":  TimeFrame(1,  TimeFrameUnit.Hour),
    "2hour":  TimeFrame(2,  TimeFrameUnit.Hour),
    "4hour":  TimeFrame(4,  TimeFrameUnit.Hour),
    "1day":   TimeFrame(1,  TimeFrameUnit.Day),
    "1week":  TimeFrame(1,  TimeFrameUnit.Week),
    "1month": TimeFrame(1,  TimeFrameUnit.Month),
}

def to_eastern(dt):
    return EASTERN.localize(dt).astimezone(EASTERN)

def fetch_bars(symbols, start_dt, end_dt, timeframe="1min", feed="iex"):
    """
    Batch-fetch bars for 1 or multiple symbols.
    
    Args:
        symbols:    str or list of str
        start_dt:   tz-aware datetime
        end_dt:     tz-aware datetime
        timeframe:  string key from TIMEFRAME_MAP (e.g. "5min", "1hour", "1day")
                    OR a raw TimeFrame object if you need custom intervals
        feed:       "iex" (free) or "sip" (paid)
    
    Returns:
        Multi-index DataFrame (symbol, timestamp) or empty DataFrame on failure
    """
    if client is None:
        return None

    if isinstance(symbols, str):
        symbols = [symbols]

    # Accept either a string shorthand or a raw TimeFrame object
    if isinstance(timeframe, str):
        if timeframe not in TIMEFRAME_MAP:
            raise ValueError(f"Unknown timeframe '{timeframe}'. Choose from: {list(TIMEFRAME_MAP.keys())}")
        tf = TIMEFRAME_MAP[timeframe]
    elif isinstance(timeframe, TimeFrame):
        tf = timeframe
    else:
        raise TypeError("timeframe must be a string key or a TimeFrame object")

    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=tf,
        start=start_dt.astimezone(pytz.UTC),
        end=end_dt.astimezone(pytz.UTC),
        feed=feed
    )

    try:
        df = client.get_stock_bars(req).df
        if df.empty:
            return df

        if isinstance(df.index, pd.MultiIndex):
            df = df.tz_convert(EASTERN, level="timestamp")
        else:
            df = df.tz_convert(EASTERN)

        return df

    except Exception as e:
        print(f"[Alpaca ERROR] {e}")
        return pd.DataFrame()