"""
sunday_report/next_earnings.py
Single-ticker earnings intel fetcher.

fetch_next_earnings(ticker) returns a structured dict with:
  - next earnings date + session (BMO/AMC)
  - EPS estimate
  - Historical beat/miss record (last 8 quarters)
  - Options-implied move

Works for ANY ticker regardless of revenue size.
Uses yfinance as the data source.
"""
import sys
from math import sqrt
from typing import Optional

import pandas as pd
import yfinance as yf


def fetch_next_earnings(ticker: str) -> Optional[dict]:
    """
    Fetch earnings intel for a single ticker.

    Args:
        ticker: Stock symbol e.g. "AAPL"

    Returns:
        Dict with keys:
            ticker          str
            date            str  "YYYY-MM-DD"
            session         str  "BMO" | "AMC"
            eps_estimate    float | None
            beat_count      int | None   (quarters beaten out of total_quarters)
            total_quarters  int | None
            avg_beat_pct    float | None (average EPS surprise %, positive = beat)
            implied_move_pct    float | None
            implied_move_dollar float | None
        Returns None if no upcoming earnings found.
    """
    stock = yf.Ticker(ticker.upper())

    # ------------------------------------------------------------------
    # Next earnings date
    # ------------------------------------------------------------------
    cal = stock.earnings_dates
    if cal is None or cal.empty:
        return None

    now = pd.Timestamp.now(tz=cal.index.tz)
    future = cal[cal.index > now]
    if future.empty:
        return None

    next_ts = future.index[0]
    next_row = future.iloc[0]
    date_str = next_ts.strftime("%Y-%m-%d")
    session = "BMO" if next_ts.hour < 12 else "AMC"
    eps_estimate = _safe_float(next_row.get("EPS Estimate"))

    # ------------------------------------------------------------------
    # Historical beats
    # ------------------------------------------------------------------
    beat_count = None
    total_quarters = None
    avg_beat_pct = None

    history = stock.earnings_history
    if history is not None and not history.empty:
        h = history.tail(8).copy()
        h["actual"] = pd.to_numeric(h.get("epsActual", pd.Series(dtype=float)), errors="coerce")
        h["estimate"] = pd.to_numeric(h.get("epsEstimate", pd.Series(dtype=float)), errors="coerce")
        h = h.dropna(subset=["actual", "estimate"])

        if not h.empty:
            h["surprise_pct"] = (h["actual"] - h["estimate"]) / h["estimate"].abs() * 100
            beat_count = int((h["actual"] > h["estimate"]).sum())
            total_quarters = len(h)
            avg_beat_pct = round(float(h["surprise_pct"].mean()), 2)

    # ------------------------------------------------------------------
    # Options-implied move
    # ------------------------------------------------------------------
    implied_move_pct = None
    implied_move_dollar = None

    try:
        price_hist = stock.history(period="2d")
        if not price_hist.empty:
            price = float(price_hist["Close"].iloc[-1])
            options = stock.option_chain()
            calls = options.calls.copy()
            calls["dist"] = (calls["strike"] - price).abs()
            atm = calls.sort_values("dist").iloc[0]
            iv = float(atm["impliedVolatility"])
            dte = max((next_ts - now).days, 1) / 365
            move = price * iv * sqrt(dte)
            implied_move_dollar = round(move, 2)
            implied_move_pct = round(move / price * 100, 2)
    except Exception:
        pass

    return {
        "ticker": ticker.upper(),
        "date": date_str,
        "session": session,
        "eps_estimate": eps_estimate,
        "beat_count": beat_count,
        "total_quarters": total_quarters,
        "avg_beat_pct": avg_beat_pct,
        "implied_move_pct": implied_move_pct,
        "implied_move_dollar": implied_move_dollar,
    }


def _safe_float(val) -> Optional[float]:
    try:
        f = float(val)
        return None if pd.isna(f) else round(f, 2)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python next_earnings.py TICKER")
        sys.exit(1)

    result = fetch_next_earnings(sys.argv[1])
    if result is None:
        print("No upcoming earnings found.")
        sys.exit(1)

    print(f"\n📊 {result['ticker']} Earnings Intel")
    print("──────────────────────────────────")
    print(f"Next earnings : {result['date']}  ({result['session']})")
    if result["eps_estimate"] is not None:
        print(f"EPS Estimate  : {result['eps_estimate']}")
    if result["beat_count"] is not None:
        print(f"Beat history  : {result['beat_count']}/{result['total_quarters']}  avg {result['avg_beat_pct']:+.1f}%")
    if result["implied_move_pct"] is not None:
        print(f"Implied move  : ±${result['implied_move_dollar']}  ({result['implied_move_pct']:.1f}%)")
