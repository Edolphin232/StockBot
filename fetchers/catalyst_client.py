# fetchers/catalyst_client.py
"""
Catalyst event fetcher — retrieves high-impact economic events.
Uses static JSON calendar file.
"""
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, List, Dict

# Configuration - JSON file is in the same directory as this file
CALENDAR_FILE = Path(__file__).parent / "catalyst_calendar.json"


def load_static_calendar() -> List[Dict]:
    """
    Load events from the local JSON calendar file.
    
    Returns:
        List of event dicts with 'date', 'event', 'impact' fields.
        Empty list if no file exists.
    """
    if not CALENDAR_FILE.exists():
        print(f"[Catalyst] Calendar file not found: {CALENDAR_FILE}")
        return []
    
    try:
        with open(CALENDAR_FILE, "r") as f:
            data = json.load(f)
        
        # Handle nested format: {"events": [...]}
        if isinstance(data, dict) and "events" in data:
            return data["events"]
        # Handle direct array format: [...]
        elif isinstance(data, list):
            return data
        else:
            print(f"[Catalyst] Unexpected JSON format in {CALENDAR_FILE}")
            return []
    except Exception as e:
        print(f"[Catalyst] Error loading calendar file: {e}")
        return []




def fetch_catalysts(
    target_date: Optional[str] = None,
    days_ahead: int = 7
) -> List[Dict]:
    """
    Fetch catalyst events for a specific date or upcoming dates.
    
    Args:
        target_date: Optional "YYYY-MM-DD" string. If provided, returns events for that date.
                    If None, returns all events in the next days_ahead days.
        days_ahead: How many days ahead to fetch (default 7, only used if target_date is None)
    
    Returns:
        List of event dicts with 'date', 'event', 'impact' fields.
        Format: [{"date": "2025-02-17", "event": "CPI", "impact": "HIGH"}, ...]
    
    Notes:
        - Loads events from static calendar (catalyst_calendar.json)
        - File should be in the fetchers/ directory
    """
    # Load static calendar
    all_events = load_static_calendar()
    
    # Filter by target_date if provided
    if target_date:
        try:
            target = date.fromisoformat(target_date)
            filtered = [
                e for e in all_events
                if e.get("date") == target_date
            ]
            return filtered
        except ValueError:
            print(f"[Catalyst] Invalid date format: {target_date}")
            return []
    
    # Otherwise return events in the next days_ahead days
    today = date.today()
    cutoff = today + timedelta(days=days_ahead)
    
    upcoming = []
    for event in all_events:
        try:
            event_date = date.fromisoformat(event["date"])
            if today <= event_date <= cutoff:
                upcoming.append(event)
        except (ValueError, KeyError):
            continue
    
    # Sort by date
    upcoming.sort(key=lambda e: e.get("date", ""))
    return upcoming

