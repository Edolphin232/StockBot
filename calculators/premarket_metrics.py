import numpy as np
import pandas as pd
from datetime import datetime, date


def price_near(df, t="09:20"):
    """
    Return the close price at the timestamp nearest to time `t`.
    Falls back to NaN if the dataframe is empty.
    """
    if df is None or df.empty:
        return np.nan

    target = pd.to_datetime(t).time()
    times = df.index.time
    if len(times) == 0:
        return np.nan

    # Find index whose time is closest to the target time
    nearest_idx = min(
        range(len(times)),
        key=lambda i: abs(
            datetime.combine(date.today(), times[i])
            - datetime.combine(date.today(), target)
        ),
    )
    return df.iloc[nearest_idx].close


def calc_premarket_metrics(df):
    M = {}
    if df is None or df.empty:
        return {k: np.nan for k in [
            "premarket_pct_change","premarket_range_pct","premarket_last_price",
            "premarket_high","premarket_low","premarket_vwap",
            "distance_from_premarket_vwap_pct_920","distance_from_premarket_vwap_pct_930",
            "pm_trend_slope_pct_per_hour","pm_pullback_depth_pct"
        ]}

    # Work on a copy to avoid pandas SettingWithCopyWarning when df is a slice
    df = df.copy()

    o = df["open"].iloc[0]
    c = df["close"].iloc[-1]
    h = df["high"].max()
    l = df["low"].min()

    M["premarket_pct_change"] = (c - o) / o * 100
    M["premarket_range_pct"] = (h - l) / o * 100
    M["premarket_last_price"] = c
    M["premarket_high"] = h
    M["premarket_low"] = l

    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    M["premarket_vwap"] = df["vwap"].iloc[-1]

    # Distances from premarket VWAP at times nearest to 09:20 and 09:30
    p_920 = price_near(df, "09:20")
    p_930 = price_near(df, "09:30")

    M["distance_from_premarket_vwap_pct_920"] = (
        (p_920 - M["premarket_vwap"]) / M["premarket_vwap"] * 100
        if not np.isnan(p_920) and M["premarket_vwap"] != 0
        else np.nan
    )
    M["distance_from_premarket_vwap_pct_930"] = (
        (p_930 - M["premarket_vwap"]) / M["premarket_vwap"] * 100
        if not np.isnan(p_930) and M["premarket_vwap"] != 0
        else np.nan
    )

    # Trend slope — guard against degenerate/NaN data that can break polyfit
    df["idx"] = np.arange(len(df))
    # Drop rows where close is NaN
    mask = df["close"].notna()
    x = df.loc[mask, "idx"].values
    y = df.loc[mask, "close"].values
    if len(x) > 1:
        try:
            slope = np.polyfit(x, y, 1)[0]
            M["pm_trend_slope_pct_per_hour"] = (slope / o) * 60 * 100
        except Exception:
            M["pm_trend_slope_pct_per_hour"] = np.nan
    else:
        M["pm_trend_slope_pct_per_hour"] = np.nan

    # Pullback depth
    peak = df["close"].cummax()
    dd = (df["close"] - peak) / peak
    M["pm_pullback_depth_pct"] = dd.min() * 100

    return M
