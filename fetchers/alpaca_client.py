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


def to_eastern(dt):
    return EASTERN.localize(dt).astimezone(EASTERN)


def fetch_bars(symbols, start_dt, end_dt):
    """
    Batch-fetch IEX minute bars for 1 or multiple symbols.
    Returns a multi-index dataframe: (symbol, timestamp)
    """
    if client is None:
        return None

    if isinstance(symbols, str):
        symbols = [symbols]

    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start=start_dt.astimezone(pytz.UTC),
        end=end_dt.astimezone(pytz.UTC),
        feed="iex"
    )

    try:
        df = client.get_stock_bars(req).df
        if df.empty:
            return df

        # Convert timezone only
        # When using the Alpaca .df helper, the index is a MultiIndex:
        #   (symbol, timestamp)
        # and only the timestamp level is tz-aware. tz_convert() on a
        # MultiIndex must specify which level holds datetimes.
        if isinstance(df.index, pd.MultiIndex):
            # Alpaca names the datetime level "timestamp"
            df = df.tz_convert(EASTERN, level="timestamp")
        else:
            df = df.tz_convert(EASTERN)

        return df
    except Exception as e:
        print(f"[Alpaca ERROR] {e}")
        return pd.DataFrame()
