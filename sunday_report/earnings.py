import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import requests

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

API_KEY = os.getenv("FINNHUB_API_KEY")
BASE = "https://finnhub.io/api/v1"

def finnhub(endpoint, params=None):
    params = params or {}
    params["token"] = API_KEY
    r = requests.get(BASE + endpoint, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def get_earnings_report(target_date=None):
    """
    Fetches earnings calendar for the next 7 days (or a specific day) and returns a formatted message.
    Returns tuple: (success: bool, message: str, csv_path: Path or None)
    """
    try:
        print("Fetching earnings calendar...")
        if target_date:
            if isinstance(target_date, str):
                target_date = datetime.fromisoformat(target_date).date()
            start = target_date
            end = target_date
            date_scope_label = f"{start.isoformat()}"
            report_title = f"📊 **Earnings Report** ({date_scope_label})\n"
        else:
            start = datetime.utcnow().date()
            end = start + timedelta(days=7)
            report_title = "📊 **Weekly Earnings Report**\n"

        earnings = finnhub("/calendar/earnings", {
            "from": start.isoformat(),
            "to": end.isoformat()
        })["earningsCalendar"]

        if not earnings:
            return (True, f"{report_title}\nNo high-impact earnings found for the selected period.", None)

        df = pd.DataFrame(earnings)

        # Deduplicate
        df = df.drop_duplicates("symbol")

        # Use revenue estimate as importance filter
        df["revenueEstimate"] = pd.to_numeric(df["revenueEstimate"], errors="coerce").fillna(0)

        # Filter to companies that actually matter
        df = df[df["revenueEstimate"] > 20_000_000_000]   # > $1B expected revenue

        if df.empty:
            return (True, f"{report_title}\nNo high-impact earnings found for the selected period.", None)

        # Impact score based on size of revenue
        df["ImpactScore"] = np.log10(df["revenueEstimate"] + 1)

        # Sort by day/date first, then by impact within each day
        df = df.sort_values(["date", "ImpactScore"], ascending=[True, False]).head(40)

        # Convert date to datetime and add weekday
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["weekday"] = df["date"].dt.strftime("%a")  # Mon, Tue, Wed, etc.

        out = df[["symbol", "date", "weekday", "hour", "revenueEstimate", "ImpactScore"]].rename(columns={
            "symbol": "Ticker",
            "date": "Date",
            "weekday": "Day",
            "hour": "Time",
            "revenueEstimate": "RevenueEst"
        })

        path = DATA_DIR / "weekly_earnings.csv"
        out.to_csv(path, index=False)

        # Format for Discord (limit to top 35 for readability)
        top_rows = out.head(35)
        
        # Format revenue estimates nicely
        def format_revenue(val):
            if pd.isna(val) or val == 0:
                return "N/A"
            if val >= 3_000_000_000:
                return f"${val/1_000_000_000:.2f}B"
            return f"${val/1_000_000:.1f}M"
        
        lines = [report_title]
        if target_date:
            lines.append(f"**High-impact earnings on {date_scope_label}** (Top 35 of {len(out)}):\n")
        else:
            lines.append(f"**High-impact earnings next week** (Top 35 of {len(out)}):\n")
        lines.append("```")
        lines.append(f"{'Ticker':<8} {'Day':<4} {'Date':<12} {'Time':<6} {'Revenue Est':<15}")
        lines.append("-" * 50)
        
        for _, row in top_rows.iterrows():
            ticker = str(row["Ticker"])[:7]
            day_str = str(row["Day"])[:3] if pd.notna(row["Day"]) else "N/A"
            date_str = str(row["Date"])[:10] if pd.notna(row["Date"]) else "TBD"
            time_str = str(row["Time"])[:5] if pd.notna(row["Time"]) else "TBD"
            revenue_str = format_revenue(row["RevenueEst"])
            lines.append(f"{ticker:<8} {day_str:<4} {date_str:<12} {time_str:<6} {revenue_str:<15}")
        
        lines.append("```")
        
        message = "\n".join(lines)
        print("\nHigh-impact earnings next week:")
        print(out.to_string(index=False))
        print(f"\nSaved to {path}")
        
        return (True, message, path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (False, f"❌ Error fetching earnings: {str(e)}", None)


if __name__ == "__main__":
    target_date = None
    if len(sys.argv) >= 2:
        try:
            target_date = datetime.fromisoformat(sys.argv[1]).date()
        except ValueError:
            print("Invalid date format. Use YYYY-MM-DD.")
            exit(1)
    success, message, path = get_earnings_report(target_date)
    print("\n" + message)
    if not success:
        exit(1)
