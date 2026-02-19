#!/usr/bin/env python3
"""
Sunday Weekly Risk Report
Combines:
- Macro catalysts (TradingEconomics)
- Earnings (Finnhub)
- Options positioning (weekly_positioning.csv)
- Volatility + cross-asset stress (Yahoo Finance)
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv
from bs4 import BeautifulSoup


# -----------------------------
# Setup
# -----------------------------
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("sunday")

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

TE_KEY = os.getenv("TE_API_KEY")
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
FINNHUB_BASE = "https://finnhub.io/api/v1"

# -----------------------------
# Finnhub
# -----------------------------
def finnhub_get(endpoint, params=None):
    params = params or {}
    params["token"] = FINNHUB_KEY
    r = requests.get(FINNHUB_BASE + endpoint, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def fetch_earnings(start_date, end_date):
    if not FINNHUB_KEY:
        log.warning("FINNHUB_API_KEY not set")
        return pd.DataFrame()

    data = finnhub_get("/calendar/earnings", {
        "from": str(start_date),
        "to": str(end_date)
    })

    cal = data.get("earningsCalendar", [])
    if not cal:
        return pd.DataFrame()

    df = pd.DataFrame(cal)
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
        "revenueEstimate": "RevenueEst"
    })

    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    return out.sort_values(["Date", "ImpactScore"], ascending=[True, False])

def filter_earnings_for_window(df, start_date, end_date):
    if df is None or df.empty:
        return pd.DataFrame()

    e = df.copy()
    e["Date"] = pd.to_datetime(e["Date"], errors="coerce").dt.date
    e = e.dropna(subset=["Date"])
    return e[(e["Date"] >= start_date) & (e["Date"] < end_date)]

# -----------------------------
# TradingEconomics
# -----------------------------
def fetch_macro_calendar(start_date, end_date):
    if not TE_KEY:
        return pd.DataFrame()

    r = requests.get("https://api.tradingeconomics.com/calendar", params={
        "c": TE_KEY,
        "d1": start_date,
        "d2": end_date
    }, timeout=20)

    data = r.json()
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df = df.rename(columns={"Event":"event","Country":"country","Date":"date","Time":"time","Importance":"importance"})
    return df[df["country"]=="United States"]

BIG_WORDS = ["fed","fomc","powell","cpi","pce","inflation","payroll","gdp","pmi","retail"]

def filter_big_macro(df):
    if df.empty:
        return df

    def is_big(r):
        txt = (r.get("event") or "").lower()
        return any(k in txt for k in BIG_WORDS) or int(r.get("importance",0)) >= 2

    df["big"] = df.apply(is_big, axis=1)
    return df[df["big"]].sort_values("date")

# -----------------------------
# Yahoo
# -----------------------------
YAHOO_SYMBOLS = {
    "SPY":"SPY","QQQ":"QQQ","IWM":"IWM","VIX":"^VIX","VIX9D":"^VIX9D",
    "VIX1D":"^VIX1D","10Y":"^TNX","DXY":"DX-Y.NYB","WTI":"CL=F","BTC":"BTC-USD"
}

def pct_change(s, days=5):
    s = s.dropna()
    return None if len(s)<days+1 else (s.iloc[-1]/s.iloc[-(days+1)]-1)*100

def fetch_yahoo_dashboard():
    data = yf.download(list(YAHOO_SYMBOLS.values()), period="14d", auto_adjust=True, progress=False)
    rows=[]
    for name,tkr in YAHOO_SYMBOLS.items():
        try:
            close = data["Close"][tkr]
            rows.append({"Asset":name,"Last":float(close.iloc[-1]),"Chg_5D_%":round(pct_change(close),2)})
        except:
            rows.append({"Asset":name,"Last":None,"Chg_5D_%":None})
    return pd.DataFrame(rows)

# -----------------------------
# Discord
# -----------------------------
def format_for_discord(macro, earnings, dashboard):
    lines=["📅 **Weekly Market Risk**\n"]

    if macro.empty:
        lines.append("🧭 **Macro:** None → Earnings dominate")
    else:
        lines.append("🧭 **Macro Catalysts**")
        for _, r in macro.head(6).iterrows():
            lines.append(f"• {r.get('date','')} {r.get('time','')} — {r.get('event','')}")

    lines.append("\n💣 **Top Earnings**")
    if earnings.empty:
        lines.append("• _(none this week)_")
    else:
        for _,r in earnings.iterrows():
            lines.append(f"• {r['Ticker']} — {r['Date']} ({r.get('Time','')}) — ${r['RevenueEst']/1e9:.0f}B")

    vix = dashboard[dashboard.Asset=="VIX"].iloc[0]
    lines.append(f"\n📈 **VIX:** {vix['Last']:.2f}")
    lines.append("\n🧠 **Regime:** Earnings + Macro → High-vol directional")

    return "\n".join(lines)


# -----------------------------
def fetch_fomc_calendar_from_fed(start_date, end_date):
    """
    Scrape scheduled FOMC meetings from federalreserve.gov
    No news. No speeches. Calendar risk only.
    """
    try:
        url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        # Try multiple methods to find the year
        year = None
        
        # Method 1: Look for the year in the page title or h1
        title = soup.find("h1")
        if title:
            title_text = title.get_text()
            import re
            year_match = re.search(r'20\d{2}', title_text)
            if year_match:
                year = int(year_match.group())
        
        # Method 2: Just use current year if not found
        if year is None:
            year = datetime.now().year
            log.info(f"Using current year: {year}")

        events = []

        for row in soup.find_all("div", class_="fomc-meeting"):
            month_div = row.find("div", class_="fomc-meeting__month")
            date_div = row.find("div", class_="fomc-meeting__date")

            if not month_div or not date_div:
                continue

            month = month_div.get_text(strip=True)
            day_text = date_div.get_text(strip=True)

            # Remove * from unscheduled meetings
            day_text = day_text.replace("*", "").strip()

            # Examples: "28-29", "17-18", "30"
            try:
                if "-" in day_text:
                    # Multi-day meeting - take the LAST day (statement day)
                    start_day, end_day = day_text.split("-")
                    statement_day_num = int(end_day.strip())
                else:
                    # Single day meeting
                    statement_day_num = int(day_text.strip())

                # Parse the statement date
                statement_date = pd.to_datetime(f"{month} {statement_day_num} {year}").date()

                # Only include if within our window
                if start_date <= statement_date <= end_date:
                    events.append({
                        "date": str(statement_date),
                        "time": "14:00",
                        "event": "FOMC Statement",
                        "importance": 3
                    })

            except Exception as e:
                log.debug(f"Failed to parse FOMC date: {month} {day_text} - {e}")
                continue

        df = pd.DataFrame(events)
        if df.empty:
            return df
        
        return df.drop_duplicates("date").sort_values("date")

    except Exception as e:
        log.warning(f"FOMC scraping failed: {e}")
        return pd.DataFrame()



# -----------------------------
# Main
# -----------------------------
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--date",default=None)
    args=parser.parse_args()

    start = datetime.strptime(args.date,"%Y-%m-%d").date() if args.date else datetime.now(timezone.utc).date()
    end = start + timedelta(days=7)
    lookback = start - timedelta(days=7)

    earnings_all = fetch_earnings(start,end)
    earnings = filter_earnings_for_window(earnings_all,start,end)

    fed_schedule = fetch_fomc_calendar_from_fed(lookback, end)

    macro = filter_big_macro(fetch_macro_calendar(lookback.isoformat(), end.isoformat()))

    
    if not fed_schedule.empty:
        fed_schedule["importance"] = 3
        macro = pd.concat([macro, fed_schedule], ignore_index=True)

    dashboard = fetch_yahoo_dashboard()

    print("\n=== DISCORD MESSAGE ===\n")
    print(format_for_discord(macro,earnings,dashboard))

if __name__=="__main__":
    main()
