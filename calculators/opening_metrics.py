import numpy as np

def calc_opening_metrics(df, premarket_vwap):
    M = {}
    for k in [
        "price_930","price_1000","move_930_to_1000_pct",
        "opening_range_high","opening_range_low","opening_vwap_930_1000",
        "price_1000_vs_orb","price_1000_vs_opening_vwap",
        "price_1000_vs_premarket_vwap","higher_highs_9_30_to_10_00",
        "higher_lows_9_30_to_10_00"
    ]:
        M[k] = np.nan

    if df is None or df.empty:
        return M

    # Work on a copy to avoid pandas SettingWithCopyWarning when df is a slice
    df = df.copy()

    o = df.iloc[0].open
    c = df.iloc[-1].close

    M["price_930"] = o
    M["price_1000"] = c
    M["move_930_to_1000_pct"] = (c - o) / o * 100

    M["opening_range_high"] = df["high"].max()
    M["opening_range_low"] = df["low"].min()

    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    M["opening_vwap_930_1000"] = df["vwap"].iloc[-1]

    orb = M["opening_range_high"] if c > M["opening_range_high"] else M["opening_range_low"]
    M["price_1000_vs_orb"] = c - orb
    M["price_1000_vs_opening_vwap"] = (c - M["opening_vwap_930_1000"]) / M["opening_vwap_930_1000"] * 100
    M["price_1000_vs_premarket_vwap"] = (c - premarket_vwap) / premarket_vwap * 100 if premarket_vwap else np.nan

    highs = df["high"].values
    lows = df["low"].values

    M["higher_highs_9_30_to_10_00"] = np.sum(np.diff(highs) > 0)
    M["higher_lows_9_30_to_10_00"] = np.sum(np.diff(lows) > 0)

    return M
