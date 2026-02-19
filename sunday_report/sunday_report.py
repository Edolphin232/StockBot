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

from sunday_report.earnings import fetch_and_score_earnings


# -----------------------------
# Setup
# -----------------------------
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("sunday")

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

TE_KEY = os.getenv("TE_API_KEY")

# Alias so existing callers of fetch_earnings() still work
fetch_earnings = fetch_and_score_earnings

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

    return "\n".join(lines)


def create_sunday_report_embed(macro, earnings, dashboard):
    """
    Create a Discord embed for the Sunday weekly risk report.
    Returns discord.Embed or None if discord is not available.
    """
    try:
        import discord
    except ImportError:
        log.error("discord.py not installed")
        return None
    
    embed = discord.Embed(
        title="📅 Weekly Market Risk Report",
        color=discord.Color.blue()
    )
    
    # Macro Catalysts
    if macro.empty:
        embed.add_field(
            name="🧭 Macro Catalysts",
            value="None → Earnings dominate",
            inline=False
        )
    else:
        macro_text = []
        for _, r in macro.head(6).iterrows():
            date_str = r.get('date', '')
            time_str = r.get('time', '')
            event_str = r.get('event', '')
            macro_text.append(f"• {date_str} {time_str} — {event_str}")
        
        embed.add_field(
            name="🧭 Macro Catalysts",
            value="\n".join(macro_text) if macro_text else "None",
            inline=False
        )
    
    # Top Earnings
    earnings_text = []
    if earnings.empty:
        earnings_text.append("• _(none this week)_")
    else:
        for _, r in earnings.iterrows():
            ticker = r['Ticker']
            date = r['Date']
            time_str = r.get('Time', '')
            revenue = r['RevenueEst'] / 1e9
            earnings_text.append(f"• {ticker} — {date} ({time_str}) — ${revenue:.0f}B")
    
    embed.add_field(
        name="💣 Top Earnings",
        value="\n".join(earnings_text),
        inline=False
    )
    
    return embed


async def send_sunday_report_to_discord(macro, earnings, dashboard):
    """
    Send the Sunday report to Discord using the bot's client.
    """
    try:
        import discord
        import asyncio
    except ImportError:
        log.error("discord.py not installed")
        return False
    
    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
    DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")
    
    if not DISCORD_TOKEN or not DISCORD_CHANNEL_ID:
        log.error("DISCORD_TOKEN or DISCORD_CHANNEL_ID not set")
        return False
    
    try:
        channel_id = int(DISCORD_CHANNEL_ID)
    except ValueError:
        log.error(f"Invalid DISCORD_CHANNEL_ID: {DISCORD_CHANNEL_ID}")
        return False
    
    # Create a temporary client for sending the message
    intents = discord.Intents.default()
    client = discord.Client(intents=intents)
    
    message_sent = False
    
    @client.event
    async def on_ready():
        nonlocal message_sent
        try:
            channel = client.get_channel(channel_id)
            if channel is None:
                log.error(f"Channel {channel_id} not found")
                await client.close()
                return
            
            # Create and send embed
            embed = create_sunday_report_embed(macro, earnings, dashboard)
            if embed:
                await channel.send(embed=embed)
                log.info("Sunday report sent to Discord")
            else:
                # Fallback to plain text
                text = format_for_discord(macro, earnings, dashboard)
                await channel.send(text)
                log.info("Sunday report sent to Discord (plain text)")
            
            message_sent = True
            await client.close()
        except Exception as e:
            log.error(f"Error sending Sunday report: {e}")
            try:
                await client.close()
            except:
                pass
    
    try:
        # Start client (async call that runs until client.close() is called)
        await client.start(DISCORD_TOKEN)
        return message_sent
    except Exception as e:
        log.error(f"Failed to connect to Discord: {e}")
        try:
            await client.close()
        except:
            pass
        return False


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
# Report Generation
# -----------------------------
def generate_sunday_report(target_date: str = None) -> tuple:
    """
    Generate the Sunday report data.
    
    Args:
        target_date: Optional "YYYY-MM-DD" string. If None, uses today's date.
    
    Returns:
        Tuple of (macro, earnings, dashboard) DataFrames
    """
    start = datetime.strptime(target_date,"%Y-%m-%d").date() if target_date else datetime.now(timezone.utc).date()
    end = start + timedelta(days=7)

    earnings_all = fetch_earnings(start,end)
    earnings = filter_earnings_for_window(earnings_all,start,end)

    # Use the same catalyst fetcher as the bot
    from fetchers.catalyst_client import fetch_catalysts
    
    # Fetch catalysts for the next 7 days
    catalyst_events = fetch_catalysts(days_ahead=7)
    
    # Convert catalyst events to DataFrame format matching the old structure
    macro_rows = []
    for event in catalyst_events:
        event_date = event.get("date", "")
        event_name = event.get("event", "")
        # Catalyst events don't have time, so we'll leave it empty or use a default
        macro_rows.append({
            "date": event_date,
            "time": "",  # Catalyst calendar doesn't include time
            "event": event_name,
            "importance": 3 if event.get("impact", "").lower() == "high" else 2
        })
    
    macro = pd.DataFrame(macro_rows) if macro_rows else pd.DataFrame()
    
    # Also fetch FOMC schedule and merge
    lookback = start - timedelta(days=7)
    fed_schedule = fetch_fomc_calendar_from_fed(lookback, end)
    if not fed_schedule.empty:
        fed_schedule["importance"] = 3
        if macro.empty:
            macro = fed_schedule
        else:
            macro = pd.concat([macro, fed_schedule], ignore_index=True)
    
    # Sort by date
    if not macro.empty:
        macro = macro.sort_values("date")
    
    dashboard = fetch_yahoo_dashboard()

    return macro, earnings, dashboard


# -----------------------------
# Main
# -----------------------------
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--date",default=None)
    args=parser.parse_args()

    macro, earnings, dashboard = generate_sunday_report(args.date)

    print("\n=== DISCORD MESSAGE ===\n")
    print(format_for_discord(macro,earnings,dashboard))
    
    # Send to Discord
    import asyncio
    try:
        success = asyncio.run(send_sunday_report_to_discord(macro, earnings, dashboard))
        if success:
            print("\n✅ Sunday report sent to Discord")
        else:
            print("\n❌ Failed to send Sunday report to Discord")
    except Exception as e:
        log.error(f"Error sending to Discord: {e}")
        print(f"\n❌ Error sending to Discord: {e}")

if __name__=="__main__":
    main()
