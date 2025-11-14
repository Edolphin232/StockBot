#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import time
from datetime import datetime
import pytz
from dotenv import load_dotenv
load_dotenv()
from tqdm import tqdm

# Ensure project root is on sys.path when running directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ranker.news_fetcher import fetch_news
from ranker.scorer import SentimentScorer
from ranker.config import NEWS_LOOKBACK_DAYS, NEWS_MAX_ARTICLES


def score_news(symbol: str, scorer: SentimentScorer, days: int, max_articles: int):
    """
    Returns (sentiment, count)
    """
    articles = fetch_news(symbol, days=days, max_articles=max_articles)
    if not articles:
        return (0.0, 0)
    scores = scorer.score_articles(articles)
    if scores:
        abs_threshold = 0.05
        significant = [(s, abs(s)) for s in scores if abs(s) >= abs_threshold]
        if significant:
            num = sum(s * w for s, w in significant)
            den = sum(w for _, w in significant) or 1.0
            sentiment = float(num / den)
        else:
            sentiment = float(sum(scores) / len(scores))
    else:
        sentiment = 0.0
    return (sentiment, len(articles))


def _default_premarket_csv_path() -> str:
    eastern = pytz.timezone("US/Eastern")
    date_str = datetime.now(tz=eastern).strftime("%Y-%m-%d")
    return os.path.join(PROJECT_ROOT, "data", "premarket", f"{date_str}.csv")


def main():
    parser = argparse.ArgumentParser(description="Augment an existing premarket CSV with news sentiment.")
    parser.add_argument("--out", default=None, help="Output CSV path (default: overwrite input).")
    # Intraday-friendly defaults: very recent and light
    parser.add_argument("--days", type=int, default=1, help="Lookback window (days) for news (default: 1).")
    parser.add_argument("--max-articles", type=int, default=2, help="Max number of articles per ticker to fetch (default: 2).")
    args = parser.parse_args()

    in_path = _default_premarket_csv_path()
    df = pd.read_csv(in_path)
    if df.empty or "symbol" not in df.columns:
        raise RuntimeError("Input CSV must contain a 'symbol' column.")

    scorer = SentimentScorer()

    sentiments = []
    counts = []
    symbols = df["symbol"].astype(str).tolist()
    total = len(symbols)
    for idx, sym in enumerate(tqdm(symbols, desc="News", unit="stk")):
        sentiment, count = score_news(sym, scorer, days=args.days, max_articles=args.max_articles)
        sentiments.append(sentiment)
        counts.append(count)
        # Respect Finnhub company-news rate limit (<=60 requests/minute)
        if idx < total - 1:
            time.sleep(1.02)  # slight buffer over 1s

    df["news_sentiment"] = sentiments
    df["news_count"] = counts

    out_path = args.out or in_path
    df.to_csv(out_path, index=False)
    print(f"Updated {out_path} with news sentiment.")


if __name__ == "__main__":
    main()


