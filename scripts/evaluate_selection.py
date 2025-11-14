#!/usr/bin/env python3
import os
import sys
import argparse
from datetime import datetime, timedelta
from typing import Optional, List

import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()
import pytz
import yfinance as yf

# Ensure project root is on sys.path when running directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _default_date_str() -> str:
    eastern = pytz.timezone("US/Eastern")
    return datetime.now(tz=eastern).strftime("%Y-%m-%d")


def _default_topbottom_path(date_str: str) -> str:
    return os.path.join(PROJECT_ROOT, "data", "premarket", f"{date_str}_topbottom.tsv")


def _default_out_path(date_str: str) -> str:
    return os.path.join(PROJECT_ROOT, "data", "premarket", f"{date_str}_evaluation.tsv")


def _ensure_datetime_utc_naive(date_str: str) -> datetime:
    # yfinance uses date-only (naive midnight) index for daily data
    return datetime.strptime(date_str, "%Y-%m-%d")


def _download_daily_history(symbols: List[str], start_dt: datetime, end_dt_exclusive: datetime) -> pd.DataFrame:
    # Use yf.download for batch efficiency; returns dataframe with MultiIndex columns:
    # level 0: [Adj Close, Close, High, Low, Open, Volume], level 1: ticker
    return yf.download(
        tickers=symbols,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt_exclusive.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )


def _extract_returns_for_symbol(hist_df: pd.DataFrame, symbol: str, day: datetime) -> tuple[float, float]:
    """
    Returns:
      (cc_return_pct, oc_return_pct)
      - cc: (Close[t] / Close[t-1] - 1) * 100
      - oc: (Close[t] / Open[t] - 1) * 100
      NaNs if unavailable.
    """
    cc = np.nan
    oc = np.nan
    try:
        # Support both single and multi-ticker shapes
        if isinstance(hist_df.columns, pd.MultiIndex):
            close_series = hist_df["Close"][symbol]
            open_series = hist_df["Open"][symbol]
        else:
            close_series = hist_df["Close"]
            open_series = hist_df["Open"]
        # Align to date
        close_today = close_series.loc[day]
        open_today = open_series.loc[day]
        prev_close = close_series.shift(1).loc[day]
        if pd.notna(close_today) and pd.notna(prev_close) and prev_close != 0:
            cc = float((close_today / prev_close - 1.0) * 100.0)
        if pd.notna(close_today) and pd.notna(open_today) and open_today != 0:
            oc = float((close_today / open_today - 1.0) * 100.0)
    except KeyError:
        # Missing data for this symbol/date; leave NaNs
        pass
    except Exception:
        # Unexpected shape; leave NaNs
        pass
    return (cc, oc)


def main():
    parser = argparse.ArgumentParser(description="Evaluate top/bottom selection by checking same-day returns after market close.")
    parser.add_argument("--date", default=None, help="Date (YYYY-MM-DD). Defaults to today (US/Eastern).")
    parser.add_argument("--topbottom", default=None, help="Path to <date>_topbottom.tsv. Defaults to data/premarket/<date>_topbottom.tsv.")
    parser.add_argument("--out", default=None, help="Path to write evaluation TSV. Defaults to data/premarket/<date>_evaluation.tsv.")
    args = parser.parse_args()

    # Resolve date and paths
    date_str = args.date or _default_date_str()
    day = _ensure_datetime_utc_naive(date_str)
    topbottom_path = args.topbottom or _default_topbottom_path(date_str)
    out_path = args.out or _default_out_path(date_str)

    if not os.path.exists(topbottom_path):
        raise FileNotFoundError(f"Top/bottom TSV not found: {topbottom_path}")

    sel = pd.read_csv(topbottom_path, sep="\t")
    if sel.empty or "symbol" not in sel.columns:
        raise RuntimeError("Top/bottom TSV must contain a 'symbol' column.")

    # Unique symbols in the selection
    symbols = sel["symbol"].astype(str).dropna().unique().tolist()
    if not symbols:
        raise RuntimeError("No symbols found in selection TSV.")

    # Download recent history to ensure prev-close is present
    start_dt = day - timedelta(days=10)
    end_dt_exclusive = day + timedelta(days=1)
    hist = _download_daily_history(symbols, start_dt, end_dt_exclusive)

    # Compute returns per symbol
    cc_list = []
    oc_list = []
    for symbol in sel["symbol"].astype(str).tolist():
        cc, oc = _extract_returns_for_symbol(hist, symbol, day)
        cc_list.append(cc)
        oc_list.append(oc)

    result = sel.copy()
    result["cc_return_pct"] = cc_list
    result["oc_return_pct"] = oc_list

    # The selection file is sorted by signal_score desc; first half = top, last half = bottom
    n = len(result)
    half = n // 2
    top_df = result.head(half)
    bot_df = result.tail(half)

    # Aggregate metrics
    def safe_mean(series: pd.Series) -> float:
        vals = pd.to_numeric(series, errors="coerce")
        if vals.notna().any():
            return float(vals.mean())
        return float("nan")

    avg_top_cc = safe_mean(top_df["cc_return_pct"])
    avg_bot_cc = safe_mean(bot_df["cc_return_pct"])
    avg_top_oc = safe_mean(top_df["oc_return_pct"])
    avg_bot_oc = safe_mean(bot_df["oc_return_pct"])

    # Save detailed results
    result.to_csv(out_path, index=False, sep="\t", float_format="%.2f")

    # Print concise summary
    print(f"Evaluated selection for {date_str} (rows={n})")
    print(f"Saved detailed report: {out_path}")
    print("Group averages:")
    print(f"  Top  cc_return_pct: {avg_top_cc:.2f}%")
    print(f"  Bottom cc_return_pct: {avg_bot_cc:.2f}%")
    print(f"  Top  oc_return_pct: {avg_top_oc:.2f}%")
    print(f"  Bottom oc_return_pct: {avg_bot_oc:.2f}%")


if __name__ == "__main__":
    main()


