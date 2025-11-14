import os
#!/usr/bin/env python3
import os
import sys
import argparse
from ranker.premarket import (
    get_finnhub_client,
    get_universe_tickers,
    premarket_window_et,
    fetch_premarket_metrics,
    rank_top_movers,
)

# Ensure project root is on sys.path when running directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Compute and display premarket movers.")
    parser.add_argument("--universe", choices=["sp500", "nasdaq100", "dji"], default="sp500", help="Universe to use")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tickers processed")
    parser.add_argument("--top", type=int, default=20, help="Top N movers to display")
    parser.add_argument("--sleep", type=float, default=0.4, help="Delay between API calls to avoid rate limits")
    args = parser.parse_args()

    client = get_finnhub_client()
    tickers = get_universe_tickers(args.universe, client)

    from_ts, to_ts = premarket_window_et()
    data = fetch_premarket_metrics(
        client,
        tickers,
        from_ts,
        to_ts,
        sleep_sec=args.sleep,
        max_tickers=args.limit,
    )

    gainers, losers = rank_top_movers(data, top_n=args.top)
    print("Top Premarket Gainers:")
    print(gainers)
    print("\nTop Premarket Losers:")
    print(losers)


if __name__ == "__main__":
    main()

