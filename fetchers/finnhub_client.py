# fetchers/finnhub_client.py
"""
Finnhub API client — thin HTTP wrapper, no business logic.
All filtering, scoring, and formatting happens in the caller.
"""
import os
import requests
from typing import Optional

API_KEY = os.getenv("FINNHUB_API_KEY")
BASE = "https://finnhub.io/api/v1"


def _get(endpoint: str, params: Optional[dict] = None) -> dict:
    p = dict(params or {})
    p["token"] = API_KEY
    r = requests.get(BASE + endpoint, params=p, timeout=10)
    r.raise_for_status()
    return r.json()


def get_earnings_calendar(start_date: str, end_date: str) -> list:
    """
    Fetch raw earnings calendar from Finnhub.

    Args:
        start_date: "YYYY-MM-DD"
        end_date:   "YYYY-MM-DD"

    Returns:
        List of raw dicts from Finnhub earningsCalendar endpoint.
        Fields include: symbol, date, hour, revenueEstimate, epsEstimate, epsActual, revenueActual.
        Empty list on failure or missing key.
    """
    if not API_KEY:
        print("[Finnhub] FINNHUB_API_KEY not set")
        return []
    try:
        data = _get("/calendar/earnings", {"from": start_date, "to": end_date})
        return data.get("earningsCalendar", [])
    except Exception as e:
        print(f"[Finnhub] get_earnings_calendar failed: {e}")
        return []


def get_quote(symbol: str) -> dict:
    """
    Fetch real-time quote for a symbol.

    Returns:
        Dict with keys: c (current price), h, l, o, pc (prev close).
        Empty dict on failure.
    """
    if not API_KEY:
        return {}
    try:
        return _get("/quote", {"symbol": symbol})
    except Exception as e:
        print(f"[Finnhub] get_quote({symbol}) failed: {e}")
        return {}
