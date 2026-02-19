# fetchers/yfinance_client.py
import yfinance as yf
import pandas as pd


VALID_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "1d", "1wk", "1mo"]


def fetch_bars(
    symbol: str,
    start: str,
    end: str,
    timeframe: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV bars from Yahoo Finance.

    Args:
        symbol:   Ticker string e.g. "SPY", "^VIX", "AAPL"
        start:    "YYYY-MM-DD"
        end:      "YYYY-MM-DD"
        interval: Bar size — one of 1m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo

    Returns:
        DataFrame with lowercase columns: open, high, low, close, volume
        Empty DataFrame on failure.

    Notes:
        - yfinance only supports intraday data for the last 60 days
        - 1m data is limited to the last 7 days
    """
    if timeframe not in VALID_INTERVALS:
        raise ValueError(f"Invalid interval '{timeframe}'. Choose from: {VALID_INTERVALS}")

    try:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval=timeframe,
            auto_adjust=True,
            progress=False
        )
    except Exception as e:
        print(f"[YF ERROR] {symbol} — {e}")
        return pd.DataFrame()

    if df.empty:
        print(f"[YF WARNING] No data returned for {symbol} {start} → {end}")
        return pd.DataFrame()

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    # Keep only standard OHLCV columns that exist
    expected = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in expected if c in df.columns]]
    df = df.dropna(how="all")

    return df