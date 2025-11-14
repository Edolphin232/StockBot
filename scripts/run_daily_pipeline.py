#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
from datetime import datetime
import pytz
from dotenv import load_dotenv
load_dotenv()

# Ensure project root is on sys.path when running directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def default_premarket_out_path() -> str:
    eastern = pytz.timezone("US/Eastern")
    date_str = datetime.now(tz=eastern).strftime("%Y-%m-%d")
    out_dir = os.path.join(PROJECT_ROOT, "data", "premarket")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{date_str}.csv")


def run_cmd(cmd: list[str]):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run daily pipeline: Finnhub CSV -> YF technicals -> analyze all (news + score) -> select top/bottom.")
    parser.add_argument("--universe", default="sp500")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None, help="Output CSV path (default: data/premarket/YYYY-MM-DD.csv)")
    parser.add_argument("--quote-sleep", type=float, default=1.02, help="Seconds to sleep between Finnhub quote requests.")
    parser.add_argument("--top", type=int, default=20, help="Top and bottom N to select at the end.")
    parser.add_argument("--topbottom-out", default=None, help="Output path for top/bottom TSV (default: auto-suffixed next to --out).")
    # Intraday-friendly defaults: last 30d of 30m candles
    parser.add_argument("--period", default="6mo", help="Historical period (e.g., '6mo', '1y')")
    parser.add_argument("--interval", default="1d", help="Candle interval (use '1d' for daily indicators)")
    parser.add_argument("--tech-out", default=None, help="Output path for technicals CSV (default: auto-suffixed next to --out).")
    # News: bias toward recent, limit volume
    parser.add_argument("--news-days", type=int, default=1, help="Lookback days for company news (default: 3).")
    parser.add_argument("--news-max", type=int, default=8, help="Max articles per symbol (default: 8).")
    # Extreme tech thresholds passthrough
    parser.add_argument("--rsi-low", type=float, default=30.0)
    parser.add_argument("--rsi-high", type=float, default=70.0)
    parser.add_argument("--macd-abs", type=float, default=0.5)
    parser.add_argument("--bb-low-pos", type=float, default=0.2)
    parser.add_argument("--bb-high-pos", type=float, default=0.8)
    args = parser.parse_args()

    out_csv = args.out or default_premarket_out_path()

    # 1) Build premarket snapshot
    cmd1 = [
        sys.executable,
        os.path.join(CURRENT_DIR, "build_finnhub_premarket_csv.py"),
        "--universe", args.universe,
        "--out", out_csv,
        "--sleep", str(args.quote_sleep),
    ]
    if args.limit:
        cmd1.extend(["--limit", str(args.limit)])
    run_cmd(cmd1)

    # 2) Augment with YF technicals
    cmd2 = [
        sys.executable,
        os.path.join(CURRENT_DIR, "augment_csv_with_yf.py"),
        "--period", args.period,
        "--interval", args.interval,
    ]
    run_cmd(cmd2)

    # 3) Analyze all (news + score) and select top/bottom
    cmd3 = [
        sys.executable,
        os.path.join(CURRENT_DIR, "analyze_and_select.py"),
        "--top", str(args.top),
    ]
    if args.news_days:
        cmd3.extend(["--days", str(args.news_days)])
    if args.news_max:
        cmd3.extend(["--max-articles", str(args.news_max)])
    if args.topbottom_out:
        cmd3.extend(["--out-top", args.topbottom_out])
    # Let analyze script write full analysis next to input by default
    run_cmd(cmd3)

    print(f"Pipeline complete: {out_csv}")


if __name__ == "__main__":
    main()


