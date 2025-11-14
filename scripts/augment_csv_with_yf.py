#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
import pytz

# Ensure project root is on sys.path when running directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ranker.premarket import fetch_technical_indicators_yf

# Expected technical columns returned by fetch_technical_indicators_yf
TECH_COLS = [
    "rsi14", "macd", "macd_signal", "macd_hist",
    "sma20", "sma50",
    "bb_upper", "bb_middle", "bb_lower",
    "avg_volume_20d", "prev_volume_1d",
    "yf_last_close", "yf_last_volume",
]

def _default_premarket_csv_path() -> str:
    eastern = pytz.timezone("US/Eastern")
    date_str = datetime.now(tz=eastern).strftime("%Y-%m-%d")
    return os.path.join(PROJECT_ROOT, "data", "premarket", f"{date_str}.csv")


def _normalize_after_merge(df: pd.DataFrame) -> pd.DataFrame:
    # Coalesce *_y over *_x for our known columns
    for base in TECH_COLS:
        if base not in df.columns:
            if f"{base}_y" in df.columns:
                df[base] = df[f"{base}_y"]
            elif f"{base}_x" in df.columns:
                df[base] = df[f"{base}_x"]
    # Drop duplicates
    drop_cols = []
    for base in TECH_COLS:
        for suf in ("_x", "_y"):
            col = f"{base}{suf}"
            if col in df.columns:
                drop_cols.append(col)
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def main():
    parser = argparse.ArgumentParser(description="Augment an existing premarket CSV with Yahoo Finance technical indicators.")
    parser.add_argument("--out", default=None, help="Output CSV path (default: overwrite input).")
    # Intraday-friendly defaults: last 14 days, 15-minute candles
    parser.add_argument("--period", default="6mo", help="Historical period (e.g., '6mo', '1y')")
    parser.add_argument("--interval", default="1d", help="Candle interval (use '1d' for daily indicators)")
    args = parser.parse_args()

    in_path = _default_premarket_csv_path()
    df = pd.read_csv(in_path)
    if df.empty or "symbol" not in df.columns:
        raise RuntimeError("Input CSV must contain a 'symbol' column.")
    symbols = df["symbol"].dropna().astype(str).unique().tolist()
    tech = fetch_technical_indicators_yf(tickers=symbols, period=args.period, interval=args.interval)
    if not tech.empty:
        df = df.merge(tech, on="symbol", how="left")
        df = _normalize_after_merge(df)
    out_path = args.out or in_path
    df.to_csv(out_path, index=False)
    print(f"Updated {out_path} with technical indicators.")


if __name__ == "__main__":
    main()


