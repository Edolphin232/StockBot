#!/usr/bin/env python3
import os
import sys
import argparse
from datetime import datetime
import pandas as pd
import pytz
import time
from finnhub.exceptions import FinnhubAPIException
from dotenv import load_dotenv
load_dotenv()
from tqdm import tqdm

# Ensure project root is on sys.path when running directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ranker.premarket import (
    get_finnhub_client,
    get_universe_tickers,
)

def fetch_premarket_snapshot(tickers, client, sleep_sec: float = 0.0, max_retries: int = 3) -> pd.DataFrame:
    """
    Pull basic premarket snapshot from Finnhub quote endpoint.
    """
    rows = []

    for symbol in tqdm(tickers, desc="Finnhub quotes", unit="stk"):
        attempt = 0
        while True:
            try:
                q = client.quote(symbol)
                break
            except FinnhubAPIException as e:
                # Handle rate limit: 429
                if hasattr(e, "status_code") and int(getattr(e, "status_code", 0)) == 429:
                    attempt += 1
                    backoff = min(10.0, max(1.0, sleep_sec)) * attempt
                    time.sleep(backoff)
                    if attempt <= max_retries:
                        continue
                # Other errors or exhausted retries: skip this symbol
                q = {}
                break
        c = q.get("c")
        pc = q.get("pc")
        v = q.get("v") or 0
        if c is None or pc is None:
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            continue
        pct = (c - pc) / pc * 100.0 if pc else None
        rows.append(
            {
                "symbol": symbol,
                "premarket_price": c,
                "previous_close": pc,
                "premarket_pct_change": pct,
                "premarket_volume": v,
                "gap_from_close": (c - pc) if c is not None and pc is not None else None,
            }
        )
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    return pd.DataFrame(rows)


def _default_out_path() -> str:
    eastern = pytz.timezone("US/Eastern")
    date_str = datetime.now(tz=eastern).strftime("%Y-%m-%d")
    out_dir = os.path.join(PROJECT_ROOT, "data", "premarket")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{date_str}.csv")


def main():
    parser = argparse.ArgumentParser(description="Build premarket CSV from Finnhub quotes.")
    parser.add_argument("--universe", default="sp500")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=str, default=None, help="Output CSV path (default: data/premarket/YYYY-MM-DD.csv ET)")
    parser.add_argument("--sleep", type=float, default=1.02, help="Seconds to sleep between quote requests (respect API limits).")
    args = parser.parse_args()

    client = get_finnhub_client()
    tickers = get_universe_tickers(args.universe, client)
    if args.limit:
        tickers = tickers[: args.limit]

    df = fetch_premarket_snapshot(tickers, client, sleep_sec=max(0.0, float(args.sleep)))
    if df.empty:
        raise RuntimeError("No premarket data fetched.")

    out_path = args.out or _default_out_path()
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()


