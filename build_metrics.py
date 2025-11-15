#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
from datetime import datetime, date, time
import pytz

# -------------------------------------------------------------------
# Ensure project path
# -------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

from fetchers.alpaca_client import fetch_bars
from fetchers.yahoo_client import fetch_daily_history_batch
from calculators.premarket_metrics import calc_premarket_metrics
from calculators.volume_metrics import calc_volume_metrics
from calculators.opening_metrics import calc_opening_metrics
from calculators.technicals import calc_daily_technicals


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

SP500_PATH = "data/indices/sp500.csv"
OUTPUT_PATH = "data/metrics/premarket_metrics.tsv"
ET = pytz.timezone("US/Eastern")


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def load_sp500(path):
    df = pd.read_csv(path)
    return df["symbol"].dropna().unique().tolist()


def dt(d: date, h: int, m: int):
    return ET.localize(datetime.combine(d, time(h, m)))


def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main(target_date=None, symbols=None):
    """
    Run metric build for a given date and list of symbols.

    :param target_date: datetime.date to use (defaults to today)
    :param symbols: iterable of ticker symbols (defaults to S&P500 list)
    """
    if target_date is None:
        target_date = date.today()

    if symbols is None:
        print("[INIT] Loading S&P500...")
        symbols = load_sp500(SP500_PATH)
    else:
        # Allow a single string or any iterable
        if isinstance(symbols, str):
            symbols = [symbols]
        print(f"[INIT] Using {len(symbols)} provided symbol(s)...")

    pm_start = dt(target_date, 4, 0)
    pm_end   = dt(target_date, 9, 30)
    op_start = dt(target_date, 9, 30)
    op_end   = dt(target_date, 10, 0)

    # -------------------------------------------------------------------
    # 1) BATCH PREMARKET FETCH
    # -------------------------------------------------------------------
    print("[FETCH] Premarket (batch)...")
    pm_chunks = []
    for chunk in chunk_list(symbols, 150):
        df = fetch_bars(chunk, pm_start, pm_end)
        if df is not None and not df.empty:
            pm_chunks.append(df)
    pm_df = pd.concat(pm_chunks) if pm_chunks else pd.DataFrame()
    pm_index = pm_df.index.get_level_values(0) if not pm_df.empty else []


    # -------------------------------------------------------------------
    # 2) BATCH OPENING FETCH
    # -------------------------------------------------------------------
    print("[FETCH] Opening 9:30–10:00 (batch)...")
    op_chunks = []
    for chunk in chunk_list(symbols, 150):
        df = fetch_bars(chunk, op_start, op_end)
        if df is not None and not df.empty:
            op_chunks.append(df)
    op_df = pd.concat(op_chunks) if op_chunks else pd.DataFrame()
    op_index = op_df.index.get_level_values(0) if not op_df.empty else []


    # -------------------------------------------------------------------
    # 3) ONE BATCH CALL FOR ALL DAILY HISTORY (normalized helper)
    # -------------------------------------------------------------------
    print("[FETCH] yfinance daily (batch helper)...")
    daily_dict = fetch_daily_history_batch(symbols)


    # -------------------------------------------------------------------
    # 4) Compute metrics
    # -------------------------------------------------------------------
    print("[PROCESS] Computing metrics per symbol...")
    results = []

    for sym in symbols:

        df_pm = pm_df.loc[sym] if sym in pm_index else pd.DataFrame()
        df_op = op_df.loc[sym] if sym in op_index else pd.DataFrame()
        df_daily = daily_dict.get(sym, pd.DataFrame())

        # Metric categories
        pre = calc_premarket_metrics(df_pm)
        vol = calc_volume_metrics(df_pm, df_daily)
        tech = calc_daily_technicals(df_daily)
        openm = calc_opening_metrics(df_op, pre.get("premarket_vwap"))

        row = {
            "symbol": sym,
            **pre,
            **vol,
            **openm,
            **tech
        }
        results.append(row)

    # -------------------------------------------------------------------
    # 5) Save output
    # -------------------------------------------------------------------
    print(f"[SAVE] → {OUTPUT_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_PATH, sep="\t", index=False)

    print("[DONE] Complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Build premarket metrics.")
    parser.add_argument(
        "-d",
        "--date",
        help="Target date in YYYY-MM-DD (default: today).",
    )
    parser.add_argument(
        "-s",
        "--symbol",
        action="append",
        help="Symbol to process (can be given multiple times). Defaults to full S&P500 list.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print(f"[ERROR] Invalid date format: {args.date}. Expected YYYY-MM-DD.")
            sys.exit(1)
    else:
        target_date = date.today()

    symbols = args.symbol if args.symbol else None
    main(target_date=target_date, symbols=symbols)
