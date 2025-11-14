import os
from ranker.premarket import (
    get_finnhub_client,
    get_universe_tickers,
    premarket_window_et,
    fetch_premarket_metrics,
    rank_top_movers,
)

# Config via env
MAX_TICKERS = int(os.getenv("PREMARKET_MAX_TICKERS", "150"))
SLEEP_SEC = float(os.getenv("PREMARKET_SLEEP_SEC", "0.4"))
TOP_N = int(os.getenv("PREMARKET_TOP_N", "20"))

client = get_finnhub_client()

# Universe (prefer local CSV if present)
tickers = get_universe_tickers("sp500", client)

# Premarket window (ET 4:00-9:30 for last trading day)
from_timestamp, to_timestamp = premarket_window_et()

# Fetch metrics
data = fetch_premarket_metrics(
    client,
    tickers,
    from_timestamp,
    to_timestamp,
    sleep_sec=SLEEP_SEC,
    max_tickers=MAX_TICKERS,
)

# Rank and print
gainers, losers = rank_top_movers(data, top_n=TOP_N)

print("Top Premarket Gainers:")
print(gainers)
print("\nTop Premarket Losers:")
print(losers)
