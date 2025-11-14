import os
import argparse
from typing import List

from ranker.pipeline import StockRanker
from ranker.news_fetcher import fetch_news
from ranker.scorer import SentimentScorer


def smoke_test() -> None:
    scorer = SentimentScorer()
    texts = [
        "The company announced record profits and strong guidance.",
        "The company missed earnings expectations and issued a weak outlook.",
    ]
    scores = [scorer.score_article(t) for t in texts]
    print("Smoke test scores:")
    for t, s in zip(texts, scores):
        print(f"- {s:+.4f} :: {t}")


def run_rank(
    tickers: List[str],
    days: int,
    max_articles: int,
) -> None:
    ranker = StockRanker()
    results = ranker.rank(
        tickers,
        days=days,
        max_articles=max_articles,
    )
    if not results:
        print("No results (possibly no news found or missing API key).")
        return
    print("Ranked tickers (higher is more positive):")
    for t, score in results.items():
        print(f"- {t}: {score:+.4f}")


def inspect_articles(
    tickers: List[str],
    days: int,
    max_articles: int,
) -> None:
    scorer = SentimentScorer()
    for t in tickers:
        print(f"\nInspecting articles for {t} (days={days}, max_articles={max_articles})")
        articles = fetch_news(t, days=days, max_articles=max_articles)
        if not articles:
            print("  No articles found.")
            continue
        scores = scorer.score_articles(articles)
        # Show top 10 by absolute magnitude
        ranked = sorted(zip(articles, scores), key=lambda x: abs(x[1]), reverse=True)[:10]
        for art, s in ranked:
            snippet = (art[:180] + "…") if len(art) > 180 else art
            print(f"  {s:+.4f} :: {snippet}")


def main():
    parser = argparse.ArgumentParser(description="Rank tickers by recent news sentiment.")
    parser.add_argument("tickers", nargs="*", help="Ticker symbols to rank (e.g., AAPL TSLA)")
    parser.add_argument("--days", type=int, default=None, help="Lookback window in days")
    parser.add_argument("--max-articles", type=int, default=None, help="Max articles per ticker")
    parser.add_argument("--smoke", action="store_true", help="Run local embedding/scoring smoke test")
    parser.add_argument("--inspect", action="store_true", help="Print per-article sentiment for each ticker")
    args = parser.parse_args()

    if args.smoke:
        smoke_test()

    if args.tickers:
        api_key = os.getenv("FINNHUB_API_KEY", "")
        if not api_key:
            print("Warning: FINNHUB_API_KEY not set; Finnhub calls will return no news.")
        if args.inspect:
            # Default sensible values if not provided
            days = args.days if args.days is not None else 7
            max_articles = args.max_articles if args.max_articles is not None else 50
            inspect_articles(
                args.tickers,
                days=days,
                max_articles=max_articles,
            )
        else:
            run_rank(
                args.tickers,
                days=args.days,
                max_articles=args.max_articles,
            )
    elif not args.smoke:
        parser.print_help()


if __name__ == "__main__":
    main()

