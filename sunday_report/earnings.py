"""
sunday_report/earnings.py
Business logic for the weekly earnings batch:
  - fetch_and_score_earnings()  →  fetch + $20B filter + ImpactScore
  - format_earnings_table()     →  ASCII table string for Discord
"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from fetchers.finnhub_client import get_earnings_calendar

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Fetch + score (used by Sunday batch and the ImpactScore filter command)
# ---------------------------------------------------------------------------

def fetch_and_score_earnings(start_date, end_date) -> pd.DataFrame:
    """
    Fetch earnings calendar and apply the big-company filter + ImpactScore.

    Only companies with revenueEstimate > $20B are included.
    Results are sorted by date then ImpactScore descending.

    Args:
        start_date: date or "YYYY-MM-DD" string
        end_date:   date or "YYYY-MM-DD" string

    Returns:
        DataFrame with columns: Ticker, Date, Time, RevenueEst, ImpactScore
        Empty DataFrame if nothing qualifies.
    """
    raw = get_earnings_calendar(str(start_date), str(end_date))
    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw)
    df = df.drop_duplicates("symbol")

    df["revenueEstimate"] = pd.to_numeric(df["revenueEstimate"], errors="coerce").fillna(0)
    df = df[df["revenueEstimate"] > 20_000_000_000]

    if df.empty:
        return pd.DataFrame()

    df["ImpactScore"] = np.log10(df["revenueEstimate"] + 1)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    out = df[["symbol", "date", "hour", "revenueEstimate", "ImpactScore"]].rename(columns={
        "symbol": "Ticker",
        "date": "Date",
        "hour": "Time",
        "revenueEstimate": "RevenueEst",
    })

    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out = out.sort_values(["Date", "ImpactScore"], ascending=[True, False]).head(40)

    path = DATA_DIR / "weekly_earnings.csv"
    out.to_csv(path, index=False)

    return out


# ---------------------------------------------------------------------------
# Format for Discord
# ---------------------------------------------------------------------------

def _fmt_revenue(val) -> str:
    if pd.isna(val) or val == 0:
        return "N/A"
    if val >= 1_000_000_000:
        return f"${val / 1_000_000_000:.1f}B"
    return f"${val / 1_000_000:.0f}M"


def format_earnings_table(df: pd.DataFrame, title: str = "📊 **Weekly Earnings Report**") -> str:
    """
    Format a scored earnings DataFrame as a Discord-ready ASCII table string.

    Args:
        df:    Output of fetch_and_score_earnings()
        title: Header line for the message

    Returns:
        Formatted string ready to send to Discord.
    """
    if df.empty:
        return f"{title}\n\nNo high-impact earnings found for the selected period."

    top = df.head(35)
    lines = [title, f"**Top {len(top)} of {len(df)} high-impact earnings:**\n", "```"]
    lines.append(f"{'Ticker':<8} {'Date':<12} {'Time':<6} {'Revenue Est':<14}")
    lines.append("-" * 44)

    for _, row in top.iterrows():
        ticker = str(row["Ticker"])[:7]
        date_str = str(row["Date"])[:10] if pd.notna(row["Date"]) else "TBD"
        time_str = str(row["Time"])[:5] if pd.notna(row["Time"]) else "TBD"
        rev_str = _fmt_revenue(row["RevenueEst"])
        lines.append(f"{ticker:<8} {date_str:<12} {time_str:<6} {rev_str:<14}")

    lines.append("```")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point (standalone use only)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import date
    target = None
    if len(sys.argv) >= 2:
        try:
            target = datetime.fromisoformat(sys.argv[1]).date()
        except ValueError:
            print("Invalid date format. Use YYYY-MM-DD.")
            sys.exit(1)

    start = target or date.today()
    end = start + timedelta(days=7)
    df = fetch_and_score_earnings(start, end)
    print(format_earnings_table(df))
