import requests
from datetime import datetime, timedelta
import pytz
import os

import dotenv

dotenv.load_dotenv()

API_KEY = os.getenv("API_KEY")

BASE_URL = "https://api.tradingeconomics.com/calendar"

def get_high_impact_us_events(days_ahead=7):
    today = datetime.utcnow()
    end_date = today + timedelta(days=days_ahead)

    url = "https://api.tradingeconomics.com/calendar?c=guest:guest"


    response = requests.get(url)
    print(response.status_code)
    print(response.text)

    data = response.json()
    print(data)

    events = []

    for event in data:
        event_time = datetime.strptime(
            event["Date"], "%Y-%m-%dT%H:%M:%S"
        )

        if today <= event_time <= end_date:
            events.append({
                "event": event["Event"],
                "date": event_time,
                "actual": event.get("Actual"),
                "forecast": event.get("Forecast"),
                "previous": event.get("Previous"),
            })

    return sorted(events, key=lambda x: x["date"])


if __name__ == "__main__":
    events = get_high_impact_us_events(7)

    print("\n📅 High Impact US Macro Events (Next 7 Days)\n")

    for e in events:
        print(
            f"{e['date']} — {e['event']}"
            f" | Forecast: {e['forecast']}"
            f" | Previous: {e['previous']}"
        )
