import yfinance as yf
import pandas as pd


def normalize_daily_df(df):
    """
    Fixes yfinance's inconsistent daily OHLCV structure.
    Output:
        index = datetime
        columns = open, high, low, close, volume (lowercase)
    Returns empty DataFrame if unusable.
    """

    if df is None or len(df) == 0:
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # CASE 1 — MultiIndex columns (batch download)
    # ------------------------------------------------------------------
    if isinstance(df.columns, pd.MultiIndex):
        # flatten ('AAPL','Close') -> 'close'
        df.columns = [c[-1].lower() for c in df.columns]
    else:
        # CASE 2 — Single-level: normalize to lowercase
        df.columns = [str(c).lower() for c in df.columns]

    # ------------------------------------------------------------------
    # CASE 3 — If "close" missing, use "adj close"
    # ------------------------------------------------------------------
    if "close" not in df.columns:
        if "adj close" in df.columns:
            df["close"] = df["adj close"]
        else:
            # unusable data
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Filter to our expected columns
    # ------------------------------------------------------------------
    expected = ["open", "high", "low", "close", "volume"]
    available = [c for c in expected if c in df.columns]
    df = df[available]

    # ------------------------------------------------------------------
    # Drop rows with no data
    # ------------------------------------------------------------------
    df = df.dropna(how="all")

    return df


def fetch_daily_history(symbol):
    """
    Fetch daily OHLCV for single ticker.
    Ensures output format is normalized by calling normalize_daily_df().
    """
    try:
        df = yf.download(
            symbol,
            period="90d",
            interval="1d",
            auto_adjust=True,
            progress=False
        )
    except Exception as e:
        print(f"[YFIN ERROR] {symbol} {e}")
        return pd.DataFrame()

    return normalize_daily_df(df)


def fetch_daily_history_batch(symbols):
    """
    Batch download daily OHLCV for all tickers in one request.
    Returns a dict:
        { "AAPL": df, "MSFT": df, ... }
    Each df is normalized via normalize_daily_df().
    """
    if not symbols:
        return {}

    try:
        df = yf.download(
            " ".join(symbols),
            period="90d",
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            progress=False
        )
    except Exception as e:
        print(f"[YFIN BATCH ERROR] {e}")
        return {}

    results = {}

    # daily_df[ticker] => its OHLC
    for sym in symbols:
        if sym not in df.columns.levels[0]:
            results[sym] = pd.DataFrame()
            continue

        sub = df[sym]
        sub = normalize_daily_df(sub)
        results[sym] = sub

    return results
