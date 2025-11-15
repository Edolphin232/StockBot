import numpy as np

def calc_rsi(series, period=14):
    if len(series) < period + 1:
        return np.nan
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs)).iloc[-1]

def calc_atr(df, period=14):
    if len(df) < period + 1:
        return np.nan
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = high_low.combine(high_close, max).combine(low_close, max)
    return tr.rolling(period).mean().iloc[-1]

def calc_daily_technicals(df):
    if df.empty:
        return {
            "rsi14_daily": np.nan,
            "sma20": np.nan,
            "sma50": np.nan,
            "atr14": np.nan
        }

    return {
        "rsi14_daily": calc_rsi(df["close"], 14),
        "sma20": df["close"].rolling(20).mean().iloc[-1],
        "sma50": df["close"].rolling(50).mean().iloc[-1],
        "atr14": calc_atr(df, 14)
    }
