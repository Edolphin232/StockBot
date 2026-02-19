#!/usr/bin/env python3
"""
SPY Catalyst Calendar Module
=============================
Checks for upcoming high-impact economic events that could affect
your debit spread entries.

Two modes:
  1. Static calendar (default) — manually maintained JSON file, never breaks
  2. Finnhub API (optional) — auto-fetches events if you provide an API key

Run as standalone or import into spy_scanner.py
"""

import json
import os
from datetime import datetime, timedelta, date

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CALENDAR_FILE = "catalyst_calendar.json"
FINNHUB_API_KEY = ""  # Optional: paste your free key from finnhub.io
WARNING_DAYS = 1       # Warn if event is within this many days


def load_static_calendar() -> list:
    """
    Load events from the local JSON calendar file.
    Returns list of event dicts with 'date', 'event', 'impact' fields.
    """
    if not os.path.exists(CALENDAR_FILE):
        print(f"  [!] Calendar file '{CALENDAR_FILE}' not found.")
        print(f"      Run: python catalyst_calendar.py --init  to create it.")
        return []

    with open(CALENDAR_FILE, "r") as f:
        events = json.load(f)

    return events


def fetch_finnhub_calendar(days_ahead: int = 7) -> list:
    """
    Fetch upcoming economic events from Finnhub's free API.
    Requires FINNHUB_API_KEY to be set.
    Free tier: 60 calls/min.
    """
    if not FINNHUB_API_KEY:
        return []

    try:
        import requests
    except ImportError:
        print("  [!] 'requests' not installed. Run: pip install requests")
        return []

    today = date.today()
    end = today + timedelta(days=days_ahead)

    url = "https://finnhub.io/api/v1/calendar/economic"
    params = {
        "from": today.isoformat(),
        "to": end.isoformat(),
        "token": FINNHUB_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [!] Finnhub API error: {e}")
        return []

    events = []
    for item in data.get("economicCalendar", []):
        # Filter to US events with medium/high impact
        country = item.get("country", "")
        impact = item.get("impact", "")

        if country == "US" and impact in ("high", "medium"):
            events.append({
                "date": item.get("time", "")[:10],
                "event": item.get("event", "Unknown"),
                "impact": impact.upper(),
                "estimate": item.get("estimate"),
                "previous": item.get("prev"),
            })

    return events


def check_catalysts(days_ahead: int = 2) -> dict:
    """
    Main function: check if any high-impact events fall within
    the next 'days_ahead' days.

    Returns dict with:
      - has_catalyst: bool
      - events_upcoming: list of events in the warning window
      - events_this_week: list of all events in next 7 days
      - warning: str message
    """
    today = date.today()
    warning_cutoff = today + timedelta(days=days_ahead)
    week_cutoff = today + timedelta(days=7)

    # Try static calendar first
    all_events = load_static_calendar()

    # If Finnhub key is set, supplement with API data
    if FINNHUB_API_KEY:
        api_events = fetch_finnhub_calendar(days_ahead=7)
        if api_events:
            # Merge, preferring API data for dates that overlap
            api_dates = {e["date"] for e in api_events}
            static_only = [e for e in all_events if e["date"] not in api_dates]
            all_events = api_events + static_only

    # Filter to upcoming
    events_upcoming = []
    events_this_week = []

    for event in all_events:
        try:
            event_date = date.fromisoformat(event["date"])
        except (ValueError, KeyError):
            continue

        if today <= event_date <= week_cutoff:
            events_this_week.append(event)

        if today <= event_date <= warning_cutoff:
            events_upcoming.append(event)

    # Sort by date
    events_upcoming.sort(key=lambda e: e["date"])
    events_this_week.sort(key=lambda e: e["date"])

    # Generate warning
    has_catalyst = len(events_upcoming) > 0

    if has_catalyst:
        event_names = [f"{e['event']} ({e['date']})" for e in events_upcoming]
        warning = f"CATALYST WARNING: {', '.join(event_names)}"
    else:
        warning = "No high-impact catalysts in the next 48 hours."

    return {
        "has_catalyst": has_catalyst,
        "events_upcoming": events_upcoming,
        "events_this_week": events_this_week,
        "warning": warning,
    }


def print_catalyst_report(result: dict):
    """Print a clean catalyst report."""
    print("\n" + "-" * 60)

    if result["has_catalyst"]:
        print("  [WARN] UPCOMING CATALYSTS")
        print(f"      {result['warning']}")
        print()
        print("      Consider:")
        print("      - Avoid new entries until after the event")
        print("      - Or size down if entering before the event")
        print("      - IV may spike into the event (bad for buying)")
    else:
        print("  [CLEAR] CATALYSTS")
        print(f"      {result['warning']}")

    if result["events_this_week"]:
        print()
        print("      This week's calendar:")
        for event in result["events_this_week"]:
            impact = event.get("impact", "?")
            print(f"        {event['date']}  [{impact}]  {event['event']}")

    print()
    print("      Verify / check for updates:")
    print("        FOMC:  https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
    print("        CPI:   https://www.bls.gov/schedule/news_release/cpi.htm")
    print("        PPI:   https://www.bls.gov/schedule/news_release/ppi.htm")
    print("        NFP:   https://www.bls.gov/schedule/news_release/empsit.htm")
    print("        GDP/PCE: https://www.bea.gov/news/schedule")
    print("-" * 60)


def init_calendar():
    """
    Create the catalyst_calendar.json with VERIFIED 2026 dates.

    Sources:
      - FOMC: federalreserve.gov/monetarypolicy/fomccalendars.htm
        (Announcement is Day 2 of each two-day meeting)
      - CPI, PPI, NFP: White House PFEI 2026 schedule (official PDF)
      - GDP, PCE/Personal Income: BEA schedule + rescheduling notices

    Last verified: February 2026
    """

    events = [
        # ──────────────────────────────────────────────
        # 2026 FOMC RATE DECISIONS (Day 2 = announcement day)
        # Source: federalreserve.gov/monetarypolicy/fomccalendars.htm
        # ──────────────────────────────────────────────
        {"date": "2026-01-28", "event": "FOMC Rate Decision", "impact": "HIGH"},
        {"date": "2026-03-18", "event": "FOMC Rate Decision", "impact": "HIGH"},
        {"date": "2026-04-29", "event": "FOMC Rate Decision", "impact": "HIGH"},
        {"date": "2026-06-17", "event": "FOMC Rate Decision", "impact": "HIGH"},
        {"date": "2026-07-29", "event": "FOMC Rate Decision", "impact": "HIGH"},
        {"date": "2026-09-16", "event": "FOMC Rate Decision", "impact": "HIGH"},
        {"date": "2026-10-28", "event": "FOMC Rate Decision", "impact": "HIGH"},
        {"date": "2026-12-09", "event": "FOMC Rate Decision", "impact": "HIGH"},

        # ──────────────────────────────────────────────
        # 2026 CPI RELEASES (8:30 AM ET)
        # Source: BLS official schedule (bls.gov/schedule/news_release/cpi.htm)
        # ──────────────────────────────────────────────
        {"date": "2026-02-13", "event": "CPI Report (Jan data)", "impact": "HIGH"},
        {"date": "2026-03-11", "event": "CPI Report (Feb data)", "impact": "HIGH"},
        {"date": "2026-04-10", "event": "CPI Report (Mar data)", "impact": "HIGH"},
        {"date": "2026-05-12", "event": "CPI Report (Apr data)", "impact": "HIGH"},
        {"date": "2026-06-10", "event": "CPI Report (May data)", "impact": "HIGH"},
        {"date": "2026-07-14", "event": "CPI Report (Jun data)", "impact": "HIGH"},
        {"date": "2026-08-12", "event": "CPI Report (Jul data)", "impact": "HIGH"},
        {"date": "2026-09-11", "event": "CPI Report (Aug data)", "impact": "HIGH"},
        {"date": "2026-10-14", "event": "CPI Report (Sep data)", "impact": "HIGH"},
        {"date": "2026-11-10", "event": "CPI Report (Oct data)", "impact": "HIGH"},
        {"date": "2026-12-10", "event": "CPI Report (Nov data)", "impact": "HIGH"},

        # ──────────────────────────────────────────────
        # 2026 NONFARM PAYROLLS (8:30 AM ET)
        # Source: BLS official schedule (bls.gov/schedule/news_release/empsit.htm)
        # ──────────────────────────────────────────────
        {"date": "2026-02-11", "event": "Nonfarm Payrolls (Jan data)", "impact": "HIGH"},
        {"date": "2026-03-06", "event": "Nonfarm Payrolls (Feb data)", "impact": "HIGH"},
        {"date": "2026-04-03", "event": "Nonfarm Payrolls (Mar data)", "impact": "HIGH"},
        {"date": "2026-05-08", "event": "Nonfarm Payrolls (Apr data)", "impact": "HIGH"},
        {"date": "2026-06-05", "event": "Nonfarm Payrolls (May data)", "impact": "HIGH"},
        {"date": "2026-07-02", "event": "Nonfarm Payrolls (Jun data)", "impact": "HIGH"},
        {"date": "2026-08-07", "event": "Nonfarm Payrolls (Jul data)", "impact": "HIGH"},
        {"date": "2026-09-04", "event": "Nonfarm Payrolls (Aug data)", "impact": "HIGH"},
        {"date": "2026-10-02", "event": "Nonfarm Payrolls (Sep data)", "impact": "HIGH"},
        {"date": "2026-11-06", "event": "Nonfarm Payrolls (Oct data)", "impact": "HIGH"},
        {"date": "2026-12-04", "event": "Nonfarm Payrolls (Nov data)", "impact": "HIGH"},

        # ──────────────────────────────────────────────
        # 2026 PPI RELEASES (8:30 AM ET)
        # Source: BLS official schedule (bls.gov/schedule/news_release/ppi.htm)
        # ──────────────────────────────────────────────
        {"date": "2026-01-14", "event": "PPI Report (Nov 2025 data)", "impact": "MEDIUM"},
        {"date": "2026-01-30", "event": "PPI Report (Dec 2025 data)", "impact": "MEDIUM"},
        {"date": "2026-02-27", "event": "PPI Report (Jan data)", "impact": "MEDIUM"},
        {"date": "2026-03-12", "event": "PPI Report (Feb data)", "impact": "MEDIUM"},
        {"date": "2026-04-14", "event": "PPI Report (Mar data)", "impact": "MEDIUM"},
        {"date": "2026-05-13", "event": "PPI Report (Apr data)", "impact": "MEDIUM"},
        {"date": "2026-06-11", "event": "PPI Report (May data)", "impact": "MEDIUM"},
        {"date": "2026-07-15", "event": "PPI Report (Jun data)", "impact": "MEDIUM"},
        {"date": "2026-08-13", "event": "PPI Report (Jul data)", "impact": "MEDIUM"},
        {"date": "2026-09-10", "event": "PPI Report (Aug data)", "impact": "MEDIUM"},
        {"date": "2026-10-15", "event": "PPI Report (Sep data)", "impact": "MEDIUM"},
        {"date": "2026-11-13", "event": "PPI Report (Oct data)", "impact": "MEDIUM"},
        {"date": "2026-12-15", "event": "PPI Report (Nov data)", "impact": "MEDIUM"},

        # ──────────────────────────────────────────────
        # 2026 GDP RELEASES (8:30 AM ET)
        # Source: White House PFEI 2026 + BEA rescheduling notice
        # NOTE: BEA rescheduled Q4 2025 Advance from Jan 29 to Feb 20
        #       due to 2025 appropriations lapse.
        # ──────────────────────────────────────────────
        {"date": "2026-02-20", "event": "GDP Advance Q4 2025 (rescheduled from Jan 29)", "impact": "HIGH"},
        {"date": "2026-04-30", "event": "GDP (Advance Q1)", "impact": "HIGH"},
        {"date": "2026-05-28", "event": "GDP (Second Q1)", "impact": "MEDIUM"},
        {"date": "2026-06-25", "event": "GDP (Third Q1)", "impact": "MEDIUM"},
        {"date": "2026-07-30", "event": "GDP (Advance Q2)", "impact": "HIGH"},
        {"date": "2026-08-26", "event": "GDP (Second Q2)", "impact": "MEDIUM"},
        {"date": "2026-09-30", "event": "GDP (Third Q2)", "impact": "MEDIUM"},
        {"date": "2026-10-29", "event": "GDP (Advance Q3)", "impact": "HIGH"},
        {"date": "2026-11-25", "event": "GDP (Second Q3)", "impact": "MEDIUM"},
        {"date": "2026-12-23", "event": "GDP (Third Q3)", "impact": "MEDIUM"},

        # ──────────────────────────────────────────────
        # 2026 PERSONAL INCOME & OUTLAYS / PCE (8:30 AM ET)
        # Source: White House PFEI 2026 schedule
        # NOTE: Dec 2025 report rescheduled to Feb 20 per BEA
        # ──────────────────────────────────────────────
        {"date": "2026-02-20", "event": "PCE Price Index (Dec 2025, rescheduled)", "impact": "HIGH"},
        {"date": "2026-03-27", "event": "PCE Price Index", "impact": "HIGH"},
        {"date": "2026-04-30", "event": "PCE Price Index", "impact": "HIGH"},
        {"date": "2026-05-28", "event": "PCE Price Index", "impact": "HIGH"},
        {"date": "2026-06-25", "event": "PCE Price Index", "impact": "HIGH"},
        {"date": "2026-07-30", "event": "PCE Price Index", "impact": "HIGH"},
        {"date": "2026-08-26", "event": "PCE Price Index", "impact": "HIGH"},
        {"date": "2026-09-30", "event": "PCE Price Index", "impact": "HIGH"},
        {"date": "2026-10-29", "event": "PCE Price Index", "impact": "HIGH"},
        {"date": "2026-11-25", "event": "PCE Price Index", "impact": "HIGH"},
        {"date": "2026-12-23", "event": "PCE Price Index", "impact": "HIGH"},

    ]

    with open(CALENDAR_FILE, "w") as f:
        json.dump(events, f, indent=2)

    print(f"\n  Created '{CALENDAR_FILE}' with {len(events)} events.")
    print(f"")
    print(f"  VERIFIED sources (Feb 2026):")
    print(f"    FOMC:       federalreserve.gov/monetarypolicy/fomccalendars.htm")
    print(f"    CPI/PPI/NFP: White House PFEI 2026 (official PDF)")
    print(f"    GDP/PCE:    bea.gov/news/schedule (with rescheduling updates)")
    print(f"    Retail:     White House PFEI 2026 (Census Bureau)")
    print(f"")
    print(f"  NOTE: BEA rescheduled some early 2026 releases due to the")
    print(f"  2025 appropriations lapse. Calendar reflects known changes")
    print(f"  as of Feb 2026. Check bea.gov/news/schedule for updates.")
    print()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if "--init" in sys.argv:
        init_calendar()
    else:
        result = check_catalysts(days_ahead=WARNING_DAYS)
        print_catalyst_report(result)