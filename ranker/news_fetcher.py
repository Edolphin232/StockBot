from datetime import datetime, timedelta
from typing import List
import logging
import os
import requests

from .config import (
    FINNHUB_API_KEY,
    FINNHUB_BASE_URL,
    NEWS_LOOKBACK_DAYS,
    NEWS_MAX_ARTICLES,
)

_logger = logging.getLogger(__name__)


def _format_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def fetch_news(
    ticker: str,
    days: int = NEWS_LOOKBACK_DAYS,
    max_articles: int = NEWS_MAX_ARTICLES,
) -> List[str]:
    """
    Fetch company news articles for a given ticker from Finnhub and return
    a list of combined headline + summary texts.
    """
    api_key = os.getenv("FINNHUB_API_KEY", FINNHUB_API_KEY)
    if not api_key:
        _logger.warning("FINNHUB_API_KEY is not set; returning empty news for %s", ticker)
        return []

    to_date = datetime.utcnow().date()
    from_date = to_date - timedelta(days=max(1, days))
    params = {
        "symbol": ticker.upper(),
        "from": _format_date(from_date),
        "to": _format_date(to_date),
        "token": api_key,
    }
    url = f"{FINNHUB_BASE_URL}/company-news"
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        _logger.error("Failed to fetch news for %s: %s", ticker, exc)
        return []
    except ValueError:
        _logger.error("Invalid JSON response from Finnhub for %s", ticker)
        return []

    if not isinstance(data, list):
        _logger.error("Unexpected Finnhub response type for %s: %s", ticker, type(data))
        return []

    articles: List[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        headline = (item.get("headline") or "").strip()
        summary = (item.get("summary") or "").strip()

        if not headline and not summary:
            continue

        combined = headline
        if summary:
            combined = f"{headline}. {summary}" if headline else summary
        if not combined:
            continue
        articles.append(combined)

        if len(articles) >= max_articles:
            break

    return articles
