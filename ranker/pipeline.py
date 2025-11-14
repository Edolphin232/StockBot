from .news_fetcher import fetch_news
from .scorer import SentimentScorer
import numpy as np

class StockRanker:
    def __init__(self):
        self.scorer = SentimentScorer()

    def score_stock(self, ticker: str, days: int | None = None, max_articles: int | None = None):
        # Normalize optional arguments to valid integers expected by fetch_news
        from .config import NEWS_LOOKBACK_DAYS, NEWS_MAX_ARTICLES
        days_to_use = days if isinstance(days, int) and days > 0 else NEWS_LOOKBACK_DAYS
        max_articles_to_use = (
            max_articles if isinstance(max_articles, int) and max_articles > 0 else NEWS_MAX_ARTICLES
        )
        articles = fetch_news(
            ticker,
            days=days_to_use,
            max_articles=max_articles_to_use,
        )
        if not articles:
            return None  # No data
        scores = self.scorer.score_articles(articles)
        # Emphasize strongly polarized articles; de-emphasize near-neutral noise
        abs_threshold = 0.05
        significant = [(s, abs(s)) for s in scores if abs(s) >= abs_threshold]
        if not significant:
            # If everything is near-neutral, fall back to simple mean
            return float(np.mean(scores))
        vals = [s for s, _ in significant]
        weights = [w for _, w in significant]
        weighted = float(np.average(vals, weights=weights))
        return weighted

    def rank(self, tickers, days: int | None = None, max_articles: int | None = None):
        results = {}
        for t in tickers:
            s = self.score_stock(t, days=days, max_articles=max_articles)
            if s is not None:
                results[t] = s
        # sort highest → lowest score
        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
