#!/usr/bin/env python3
import os
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import pytz

from fetchers.alpaca_client import fetch_bars

ET = pytz.timezone("US/Eastern")

# location where build_metrics outputs the TSV
METRICS_PATH = "data/metrics/premarket_metrics.tsv"


def dt(day, h, m):
    return ET.localize(datetime.combine(day, datetime.min.time())).replace(hour=h, minute=m)


def forward_return(symbol, day):
    t1015  = dt(day, 10, 15)
    t1100  = dt(day, 11, 0)
    t1200  = dt(day, 12, 0)
    tclose = dt(day, 16, 0)

    df = fetch_bars([symbol], t1015, tclose)
    if df is None or df.empty or symbol not in df.index.get_level_values(0):
        return np.nan, np.nan, np.nan

    df = df.loc[symbol].sort_index()
    p1015 = df["close"].iloc[0]

    def get_ret(t):
        sub = df[df.index >= t]
        if sub.empty:
            return np.nan
        p = sub["close"].iloc[0]
        return (p - p1015) / p1015 * 100

    return get_ret(t1100), get_ret(t1200), get_ret(tclose)


def run_day(symbol, day):
    """
    Runs your existing build_metrics.py for a single symbol+day
    then returns the row and adds realized returns.
    """

    # Run CLI
    cmd = [
        "python", "build_metrics.py",
        "--s", symbol,
        "--d", str(day)
    ]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=False)

    # Load metrics TSV
    if not os.path.exists(METRICS_PATH):
        print("[WARN] No TSV produced.")
        return None

    df = pd.read_csv(METRICS_PATH, sep="\t")

    if df.empty:
        return None

    row = df.iloc[0].to_dict()

    # Compute forward returns
    r1100, r1200, rclose = forward_return(symbol, day)

    row.update({
        "ret_11": r1100,
        "ret_12": r1200,
        "ret_close": rclose,
    })

    return row


def main():
    symbol = "TSLA"
    today = date.today()

    all_rows = []

    for i in range(1, 30):   # past 30 calendar days
        day = today - timedelta(days=i)
        if day.weekday() >= 5:
            continue  # skip weekends

        print(f"\n=== Testing {symbol} on {day} ===")
        row = run_day(symbol, day)

        if row is not None:
            all_rows.append(row)

    if not all_rows:
        print("No data.")
        return

    df = pd.DataFrame(all_rows)
    out_path = "data/backtest/tsla_backtest.tsv"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)

    print("\n[RESULTS] Saved:", out_path)

    # summary
    breakpoint()
    if "score" in df.columns:
        df["pred"] = df["score"].apply(lambda s: "CALL" if s>0 else ("PUT" if s<0 else "NEUTRAL"))

        correct_11 = np.mean((df["pred"]=="CALL") & (df["ret_11"]>0)) + \
                     np.mean((df["pred"]=="PUT") & (df["ret_11"]<0))

        correct_close = np.mean((df["pred"]=="CALL") & (df["ret_close"]>0)) + \
                        np.mean((df["pred"]=="PUT") & (df["ret_close"]<0))

        print("\nPrediction Stats:")
        print("CALL %:", np.mean(df["pred"]=="CALL"))
        print("PUT % :", np.mean(df["pred"]=="PUT"))

        print("\nDirection accuracy:")
        print("10:15→11:00:", correct_11)
        print("10:15→Close: ", correct_close)


if __name__ == "__main__":
    main()
